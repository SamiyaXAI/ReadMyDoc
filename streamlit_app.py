# streamlit_app.py

import streamlit as st
from dotenv import load_dotenv
import os
import PyPDF2

from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# --- Load Hugging Face Token ---
load_dotenv()  # loads the .env file
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # from .env

# --- Initialize Embeddings and LLM ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
    huggingfacehub_api_token=HF_TOKEN
)

# --- Function to Extract PDF Text ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# --- Function to Process and Answer Query ---
def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        raw_text = extract_text_from_pdf(uploaded_file)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents([raw_text])

        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

# --- Streamlit UI ---
st.set_page_config(page_title='🧠 Ask the Document App (HuggingFace Edition)')
st.title('🧠 Ask the Document App (No OpenAI Required)')

st.markdown("""
Upload a PDF document and ask a question about its content.  
Powered by free Hugging Face models (Mistral-7B + MiniLM), no OpenAI API key needed.
""")

# Upload + Query UI
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")
query_text = st.text_input("🔍 Enter your question:", placeholder="e.g. Summarize the document", disabled=not uploaded_file)

result = []
with st.form("query_form", clear_on_submit=True):
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and query_text))
    if submitted:
        with st.spinner("🤖 Thinking..."):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if result:
    st.subheader("📌 Answer:")
    st.info(result[0])

with st.expander("ℹ️ Debug Info"):
    st.write("Hugging Face model: `Mistral-7B-Instruct-v0.1`")
    st.write("Embedding model: `all-MiniLM-L6-v2`")
