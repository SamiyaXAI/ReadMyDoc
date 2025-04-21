
import os
os.environ["STREAMLIT_WATCH_TOC"] = "false"
import streamlit as st

import PyPDF2
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use stable, meta-error-free embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

# Use a supported, lightweight Hugging Face model
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-rw-1b",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
    huggingfacehub_api_token=HF_TOKEN
)

# Extract raw text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# Generate answer using RAG
def generate_response(uploaded_file, query_text):
    text = extract_text_from_pdf(uploaded_file)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = splitter.create_documents([text])

    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.run(query_text)

# Streamlit UI
st.set_page_config(page_title="📚 Ask My PDF", layout="centered")
st.title("📚 Ask My PDF (Hugging Face Only)")

st.markdown("Upload a PDF file and ask questions about it. Powered by open-source models.")

uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")
query_text = st.text_input("💬 Ask your question:", placeholder="e.g. Summarize the document")

result = []
with st.form("qa_form", clear_on_submit=True):
    submitted = st.form_submit_button("Get Answer", disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner("🤖 Thinking..."):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if result:
    st.subheader("📌 Answer")
    st.success(result[0])

with st.expander("ℹ️ Debug Info"):
    st.markdown("- **Embeddings**: `paraphrase-MiniLM-L6-v2`")
    st.markdown("- **LLM**: `falcon-rw-1b`")
    st.markdown("- **OpenAI**: ❌ Not used")
