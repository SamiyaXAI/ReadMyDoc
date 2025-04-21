import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# --- Load environment variables from .env file ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Set up Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Set up LLM using a lightweight, free model ---
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
    huggingfacehub_api_token=HF_TOKEN
)

# --- Function to extract text from PDF ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# --- Generate Answer Function ---
def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        raw_text = extract_text_from_pdf(uploaded_file)

        # Split the text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents([raw_text])

        # Create vector store
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return qa.run(query_text)

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Ask the Document App (Hugging Face Edition)")
st.title("🧠 Ask the Document App (No OpenAI Needed)")

st.markdown("""
Upload a PDF file and ask questions about its content.  
Answers are generated using **open-source Hugging Face models**, not OpenAI.
""")

# Upload
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")
query_text = st.text_input("🔍 Ask your question:", placeholder="e.g. What is this article about?", disabled=not uploaded_file)

# Form + Processing
result = []
with st.form("query_form", clear_on_submit=True):
    submitted = st.form_submit_button("Submit", disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner("🤖 Thinking..."):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

# Display Result
if result:
    st.subheader("📌 Answer:")
    st.success(result[0])

# Debug Info
with st.expander("ℹ️ Debug Info"):
    st.write("Embeddings: `all-MiniLM-L6-v2`")
    st.write("LLM: `google/flan-t5-base`")
