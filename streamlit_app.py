import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# --- Load token from .env file ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Embedding model (Free + Fast) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Language model (LLM) ---
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-rw-1b",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
    huggingfacehub_api_token=HF_TOKEN
)

# --- Extract PDF Text ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# --- Generate answer using RAG ---
def generate_response(uploaded_file, query_text):
    raw_text = extract_text_from_pdf(uploaded_file)

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents([raw_text])

    # Create vector DB
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # Build QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.run(query_text)

# --- Streamlit UI ---
st.set_page_config(page_title="📚 ReadMyDoc (HuggingFace Only)")
st.title("📚 Ask Your PDF (No OpenAI Needed)")

st.markdown("""
Upload a PDF and ask questions about it.  
Your data stays local. Powered by free, open Hugging Face models.
""")

uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")
query_text = st.text_input("💬 Ask a question:", placeholder="e.g. Summarize the document")

result = []
with st.form("query_form", clear_on_submit=True):
    submitted = st.form_submit_button("Get Answer", disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner("Thinking..."):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if result:
    st.subheader("📌 Answer")
    st.success(result[0])

with st.expander("ℹ️ Details"):
    st.write("Embedding model: `all-MiniLM-L6-v2`")
    st.write("LLM model: `tiiuae/falcon-rw-1b`")
    st.write("No OpenAI used. 100% free and open-source stack.")
