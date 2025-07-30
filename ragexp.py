import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from openai import OpenAI

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# Step 1: Load and chunk PDFs
def load_and_chunk_pdfs(file_paths):
    all_chunks = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = file_path
        all_chunks.extend(chunks)
    return all_chunks

# Step 2a: Dense embedding with FAISS
def embed_and_store_dense(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    return db

# Step 2b: Sparse retrieval with BM25
def create_sparse_retriever(chunks):
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 5
    return retriever

# Step 3: RAG chain for both retrieval types
def create_rag_chains(dense_db, sparse_retriever):
    def answer_question(user_query):
        # Dense Retrieval
        start_dense = time.time()
        dense_docs = dense_db.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(user_query)
        dense_context = "\n\n".join([
            f"{doc.page_content}\n(Source: {doc.metadata['source']})" for doc in dense_docs
        ])
        dense_prompt = (
            "Answer the question based on the context below. Be concise.\n\n"
            f"Context:\n{dense_context}\n\nQuestion: {user_query}\nAnswer:"
        )
        dense_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": dense_prompt}],
            temperature=0.5
        )
        end_dense = time.time()

        # Sparse Retrieval
        start_sparse = time.time()
        sparse_docs = sparse_retriever.get_relevant_documents(user_query)
        sparse_context = "\n\n".join([
            f"{doc.page_content}\n(Source: {doc.metadata['source']})" for doc in sparse_docs
        ])
        sparse_prompt = (
            "Answer the question based on the context below. Be concise.\n\n"
            f"Context:\n{sparse_context}\n\nQuestion: {user_query}\nAnswer:"
        )
        sparse_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": sparse_prompt}],
            temperature=0.5
        )
        end_sparse = time.time()

        return {
            "dense": {
                "answer": dense_resp.choices[0].message.content,
                "context": dense_context,
                "time": round(end_dense - start_dense, 2)
            },
            "sparse": {
                "answer": sparse_resp.choices[0].message.content,
                "context": sparse_context,
                "time": round(end_sparse - start_sparse, 2)
            }
        }

    return answer_question

# Step 4: Streamlit UI
st.set_page_config(page_title="🔍 Dense vs Sparse RAG Chatbot", layout="centered")
st.title("🔍 Dense vs Sparse RAG Chatbot")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)



query = st.text_input("Ask a question based on the uploaded PDF(s):")

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_paths.append(tmp_file.name)

    with st.spinner("Processing PDFs and creating retrievers..."):
        chunks = load_and_chunk_pdfs(file_paths)
        dense_db = embed_and_store_dense(chunks)
        sparse_retriever = create_sparse_retriever(chunks)
        rag_chain = create_rag_chains(dense_db, sparse_retriever)



    if query:
        with st.spinner("Generating answers using both retrieval techniques..."):
            results = rag_chain(query)

            st.markdown("## 🔵 Dense Retrieval (FAISS + OpenAI Embeddings)")
            st.write(results['dense']['answer'])
            st.markdown(f"**⏱️ Response Time:** {results['dense']['time']} seconds")
            st.markdown("**📖 Source Chunks:**")
            st.code(results['dense']['context'])

            st.markdown("---")

            st.markdown("## 🟠 Sparse Retrieval (BM25)")
            st.write(results['sparse']['answer'])
            st.markdown(f"**⏱️ Response Time:** {results['sparse']['time']} seconds")
            st.markdown("**📖 Source Chunks:**")
            st.code(results['sparse']['context'])
