# Context-Isolated PDF RAG Chatbot using Python, LangChain & Streamlit

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![LangChain](https://img.shields.io/badge/LangChain-Modern_LCEL-green)
![Gemini API](https://img.shields.io/badge/AI-Google_Gemini-orange)

An open-source, lightweight **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and interactively query their contents. Built with Python, this app ensures privacy and accuracy by utilizing a local vector database.

##  Key Features & SEO Highlights

* **Zero AI Hallucinations (Strict Context Isolation):** The generative model is constrained to answer *only* based on the provided PDF context. It will not generate false information using external pre-trained data.
* **Local Vector Storage:** Utilises **FAISS** (Facebook AI Similarity Search) to store document embeddings locally in-memory, ensuring your document vectors aren't permanently stored on third-party servers.
* **Modern LangChain Architecture:** Built using LangChain Expression Language (LCEL), replacing deprecated chain wrappers for faster execution.
* **Google Gemini Integration:** Powered by the cutting-edge `gemini-1.5-flash` model for high-speed inference and `embedding-001` for deep semantic vectorization.

##  Application Preview
![App Screenshot](<my .png>)

##  Architecture & Tech Stack

1. **Document Ingestion:** `PyPDF` extracts raw text from user-uploaded PDFs.
2. **Semantic Chunking:** `RecursiveCharacterTextSplitter` divides the text into manageable 1000-character chunks with a 200-character overlap to preserve contextual flow.
3. **Embeddings Generation:** Google Gemini's Embedding API transforms text chunks into high-dimensional vectors.
4. **Vector Database:** Vectors are indexed into a local **FAISS** store.
5. **Retrieval & QA:** Modern LCEL constructs retrieve the top `k=4` most relevant chunks to answer user queries with `temperature=0.0` for maximum factual accuracy.

##  Installation & Setup Guide

Run this RAG pipeline on your local machine by following these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]
