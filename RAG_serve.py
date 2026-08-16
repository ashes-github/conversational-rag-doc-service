"""FastAPI backend for conversational RAG over uploaded PDF documents."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from langchain_chroma import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from transformers.utils import logging

logging.set_verbosity_error()
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# In-memory prototype state. Each session has an isolated index and history.
session_vectorstores: dict[str, Chroma] = {}
session_documents: dict[str, list[dict[str, int | str]]] = {}
history_store: dict[str, ChatMessageHistory] = {}
state_lock = RLock()


def validate_session_id(session_id: str) -> str:
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id must not be empty")
    if len(session_id) > 128:
        raise HTTPException(status_code=422, detail="session_id is too long")
    return session_id


def collection_name(session_id: str) -> str:
    """Create a stable Chroma-safe name without exposing the session ID."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    return f"session-{digest}"


def get_or_create_vectorstore(session_id: str) -> Chroma:
    with state_lock:
        vectorstore = session_vectorstores.get(session_id)
        if vectorstore is None:
            vectorstore = Chroma(
                collection_name=collection_name(session_id),
                embedding_function=embeddings,
            )
            session_vectorstores[session_id] = vectorstore
        return vectorstore


def retrieve_for_session(query: str, config: RunnableConfig) -> list[Document]:
    session_id = str(config.get("configurable", {}).get("session_id", "")).strip()
    if not session_id:
        return []

    with state_lock:
        vectorstore = session_vectorstores.get(session_id)
        if vectorstore is None:
            return []
        return vectorstore.similarity_search(query, k=4)


session_retriever = RunnableLambda(retrieve_for_session)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, formulate a "
            "standalone question that can be understood without the chat history. "
            "Do not answer the question. Return it unchanged if reformulation is "
            "not needed.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, session_retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions only from the retrieved document context below. "
            "If the context does not contain the answer, say that you couldn't find "
            "the answer in the uploaded documents. Use at most three concise "
            "sentences.\n\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    with state_lock:
        if session_id not in history_store:
            history_store[session_id] = ChatMessageHistory()
        return history_store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

app = FastAPI(
    title="Conversational RAG API",
    version="1.1.0",
    description="Conversational question answering over session-scoped PDF uploads",
)
add_routes(app, conversational_rag_chain, path="/chain")


## Indexing Functions
def index_pdf_bytes(session_id: str, filename: str, content: bytes) -> int:
    safe_filename = Path(filename).name
    with tempfile.TemporaryDirectory(prefix="rag-upload-") as temp_dir:
        pdf_path = Path(temp_dir) / safe_filename
        pdf_path.write_bytes(content)
        documents = PyPDFLoader(str(pdf_path)).load()

    for document in documents:
        document.metadata["filename"] = safe_filename
        document.metadata["source"] = safe_filename

    splits = text_splitter.split_documents(documents)
    if not splits:
        raise ValueError("No extractable text was found in the PDF")

    vectorstore = get_or_create_vectorstore(session_id)
    with state_lock:
        vectorstore.add_documents(splits)
        session_documents.setdefault(session_id, []).append(
            {"filename": safe_filename, "chunks": len(splits)}
        )
    return len(splits)


@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
):
    session_id = validate_session_id(session_id)
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF is required")

    indexed_files = []
    total_chunks = 0
    for uploaded_file in files:
        filename = Path(uploaded_file.filename or "").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported: {filename or 'unnamed file'}",
            )

        content = await uploaded_file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{filename} is empty")

        try:
            chunk_count = await run_in_threadpool(
                index_pdf_bytes, session_id, filename, content
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not process {filename}: {exc}",
            ) from exc

        indexed_files.append({"filename": filename, "chunks": chunk_count})
        total_chunks += chunk_count

    return {
        "session_id": session_id,
        "files_uploaded": len(indexed_files),
        "chunks_indexed": total_chunks,
        "documents": indexed_files,
    }


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    with state_lock:
        history = history_store.get(session_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = [
            {"type": message.type, "content": message.content}
            for message in history.messages
        ]
    return {"session_id": session_id, "messages": messages}


@app.get("/documents/{session_id}")
async def get_documents(session_id: str):
    with state_lock:
        documents = list(session_documents.get(session_id, []))
    return {"session_id": session_id, "documents": documents}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
