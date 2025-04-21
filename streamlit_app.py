# app.py
import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import PyPDF2

# Load .env file
load_dotenv()


# Initialize LLM and embeddings
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # You can replace with another free model
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Core QA generation function
def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        raw_text = extract_text_from_pdf(uploaded_file)
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents([raw_text])
        # Vectorize and create retriever
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        # Run QA chain
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

# Page title
st.set_page_config(page_title='📄 Ask the Document (OpenAI-Free)')
st.title('📄 Ask the Document App (No OpenAI API Needed)')

# Description
st.header('About the App')
st.write("""
This app allows you to upload any `.pdf` document and ask questions about its content using a Retrieval-Augmented Generation (RAG) system — completely OpenAI-free!

### How It Works
- Upload a PDF document
- Ask a question about the content
- Get a smart answer powered by open-source LLMs and Hugging Face embeddings
""")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
query_text = st.text_input("Enter your question:", placeholder="e.g. Summarize this article.", disabled=not uploaded_file)

# Form
result = []
with st.form("query_form", clear_on_submit=True):
    submitted = st.form_submit_button("Submit", disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner("Thinking..."):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

# Show result
if len(result):
    st.subheader("Answer:")
    st.info(result[0])

# Debug or logs (optional)
with st.expander("Show Debug Info"):
    st.write("HuggingFaceHub LLM used: `Mistral-7B-Instruct-v0.1`")
    st.write("Embeddings: `all-MiniLM-L6-v2`")
