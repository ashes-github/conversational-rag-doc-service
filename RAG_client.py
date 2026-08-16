"""Streamlit client for the conversational RAG API."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 120


def response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("detail", payload))
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def get_groq_response(input_text: str, session_id: str) -> str:
    body = {
        "input": {"input": input_text},
        "config": {"configurable": {"session_id": session_id}},
    }
    response = requests.post(
        f"{API_BASE_URL}/chain/invoke",
        json=body,
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(response_detail(response))
    return response.json()["output"]["answer"]


def submit_pdfs(uploaded_files, session_id: str) -> dict:
    files = [
        ("files", (pdf.name, pdf.getvalue(), "application/pdf"))
        for pdf in uploaded_files
    ]
    response = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        data={"session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(response_detail(response))
    return response.json()


st.set_page_config(page_title="Document intelligence assistant", page_icon=":material/docs:")
st.title("Document intelligence assistant")
st.caption("Upload PDFs and ask grounded questions using conversational context.")

st.session_state.setdefault("uploaded_sessions", set())
st.session_state.setdefault("messages", {})
st.session_state.setdefault("session_id", "default_session")

with st.sidebar:
    st.header("Documents")
    session_id = st.text_input(
        "Session ID",
        key="session_id",
        max_chars=128,
    ).strip()
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploads",
    )

    upload_clicked = st.button(
        "Index documents",
        type="primary",
        icon=":material/upload_file:",
        width="stretch",
    )

    if upload_clicked:
        if not session_id:
            st.error("Enter a session ID before uploading documents.")
        elif not uploaded_files:
            st.warning("Choose at least one PDF.")
        else:
            try:
                with st.spinner("Parsing and indexing PDFs..."):
                    upload_result = submit_pdfs(uploaded_files, session_id)
                st.session_state.uploaded_sessions.add(session_id)
                st.success(
                    f"Indexed {upload_result['files_uploaded']} file(s) into "
                    f"{upload_result['chunks_indexed']} chunks."
                )
                for document in upload_result["documents"]:
                    st.caption(
                        f":material/check_circle: {document['filename']} | "
                        f"{document['chunks']} chunks"
                    )
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Upload failed: {exc}")

if session_id in st.session_state.uploaded_sessions:
    session_messages = st.session_state.messages.setdefault(session_id, [])
    for message in session_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input(
        "Ask a question about the uploaded PDFs",
        submit_mode="disable",
    ):
        session_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching the documents..."):
                    answer = get_groq_response(question, session_id)
                st.markdown(answer)
            session_messages.append({"role": "assistant", "content": answer})
        except (requests.RequestException, RuntimeError, KeyError) as exc:
            st.error(f"Question failed: {exc}")
else:
    with st.container(border=True):
        st.subheader("Start with your documents")
        st.write("Choose one or more PDFs in the sidebar, then index them to begin.")
