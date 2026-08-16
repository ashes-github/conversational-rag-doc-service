import requests
import streamlit as st
from transformers.utils import logging

logging.set_verbosity_error()

def get_groq_response(input_text, session_id):
    json_body = {
        "input": {"input": input_text},
        "config": {"configurable": {"session_id": session_id}},
        "kwargs": {"additionalProp1": {}},
    }
    print(json_body)
    response = requests.post("http://127.0.0.1:8000/chain/invoke", json=json_body)

    print(response.json())

    return response.json()["output"]["answer"]

def submit_pdfs(uploaded_files):

    files = []

    for pdf in uploaded_files:
        files.append(
            (
                "files",
                (
                    pdf.name,
                    pdf.getvalue(),
                    "application/pdf"
                )
            )
        )
    response = requests.post(
        "http://127.0.0.1:8000/upload",
        files=files
    )
    return response


## Streamlit app
## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")

## chat interface
session_id=st.text_input("Session ID",value="default_session")

st.write("Upload Pdf's and chat with their content")
uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

if "upload_response" not in st.session_state:
    st.session_state.upload_response = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Submit Documents") and uploaded_files:
    response = submit_pdfs(uploaded_files)
    st.session_state.upload_response = response.json()
    st.session_state.documents_uploaded = True

    #print(response.json())
    st.write(response.json())

if st.session_state.documents_uploaded:
    input_text = st.text_input("Enter text you want to have conversation about:")

    if input_text:
        answer = get_groq_response(input_text, session_id)
        st.session_state.messages.append(answer)
        st.write(f"Answer:\n\n{answer}")
        #st.write("Chat History: ", st.session_state.messages)
