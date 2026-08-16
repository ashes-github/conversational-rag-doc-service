## RAG Q&A Conversation With PDF Including Chat History
# import streamlit as st
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from langserve import add_routes
from fastapi import FastAPI, HTTPException

from dotenv import load_dotenv

from transformers.utils import logging

logging.set_verbosity_error()

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## Input the Groq API Key
api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

## chat interface

## statefully manage chat history

# if 'store' not in st.session_state:
# st.session_state.store={}

## Process uploaded  PDF's
documents = []
temppdf = f"./temp.pdf"
# with open(temppdf,"wb") as file:
# file.write(uploaded_file.getvalue())
# file_name=uploaded_file.name

loader = PyPDFLoader(temppdf)
docs = loader.load()
documents.extend(docs)

# Split and create embeddings for the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

## Answer question

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


## 6. App definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces",
    # openapi_url=None   # disables /openapi.json
    # docs_url=None,      # disables Swagger UI
    # redoc_url=None      # disables ReDoc
)

add_routes(app, conversational_rag_chain, path="/chain")

from fastapi import UploadFile, File
from typing import List


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    # process PDFs

    # session_id = str(uuid.uuid4())
    for file in files:
        file_path = f"./{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

    return {
        # "session_id": session_id,
        "files_uploaded": len(files)
    }


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")

    history = store[session_id]

    messages = []

    for msg in history.messages:
        messages.append({"type": msg.type, "content": msg.content})

    return {"session_id": session_id, "messages": messages}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
