
## 📘 Project Description: Dense vs Sparse RAG PDF Chatbot

This project is an interactive **Retrieval-Augmented Generation (RAG) chatbot** built using **Streamlit** that allows users to upload PDF documents and ask questions about their content. It implements and compares **two retrieval techniques**—**Dense Retrieval** and **Sparse Retrieval**—to evaluate their **answer quality** and **response time** side by side.

---

### 🔍 Key Features

* **PDF Upload & Parsing**: Users can upload one or more PDFs. The app extracts, chunks, and indexes the content for question-answering.
* **Dense Retrieval (FAISS + OpenAI Embeddings)**:

  * Uses `OpenAIEmbeddings` to create dense vector representations of PDF chunks.
  * Retrieves top-k semantically similar chunks using `FAISS`.
* **Sparse Retrieval (BM25)**:

  * Uses `BM25Retriever`, a classic lexical approach based on term frequency and inverse document frequency (TF-IDF).
  * Fetches top-k relevant chunks based on exact keyword matches.
* **Dual Answer Generation**:

  * Both retrievers feed their respective context into OpenAI's `gpt-3.5-turbo` model to generate answers.
  * Response time for both methods is measured independently.
* **Side-by-Side Comparison**:

  * Displays answers, sources, and response times from both dense and sparse retrieval pipelines, helping evaluate their trade-offs.

---

### 🧠 Use Cases

* Benchmarking dense vs. sparse retrieval for educational or research purposes.
* Extracting insights from technical documents, assignments, or research papers.
* Demonstrating how retrieval mechanisms affect the performance and accuracy of RAG-based LLM pipelines.

---

### 🛠️ Tech Stack

* **Streamlit** – Frontend and UI
* **LangChain** – Document loading, chunking, and retrievers
* **FAISS** – Dense vector search
* **BM25Retriever** – Sparse keyword-based retrieval
* **OpenAI API** – LLM for response generation
* **PyMuPDF** – PDF parsing
* **dotenv** – Environment variable management

---

### 💡 Sample Questions to Try

* *What are the submission requirements for the facial detection project?*
* *What dates are required for stock price prediction?*
* *How should the model output be structured?*
* *What are the constraints for the image dataset?*

