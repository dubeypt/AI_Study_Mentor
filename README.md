# AI Study Mentor — Context-Isolated PDF RAG Chatbot

**Python · LangChain · Streamlit · FAISS · Google Gemini (Embeddings) · Groq (Generation)**

An open-source, lightweight Retrieval-Augmented Generation (RAG) application that lets users upload PDF study material and interactively ask questions about its contents. Answers are generated strictly from the uploaded document — not from the model's general knowledge — reducing hallucinations for study and reference use cases.

---

## App Preview

![App Preview](my%20.png)


---

## Key Features

- **Context-Isolated Answers** — The prompt explicitly instructs the model to answer only from the retrieved document context, and to say so plainly if the answer isn't present, rather than guessing.
- **Local Vector Storage** — Uses FAISS to store document embeddings locally (saved to a `faiss_index` folder on disk), so document vectors aren't sent to or stored on third-party servers.
- **Split-Model Architecture** — Uses **Google Gemini** (`models/gemini-embedding-001`) purely for generating embeddings, and **Groq** (`llama-3.3-70b-versatile`) purely for answer generation — combining Gemini's embedding quality with Groq's fast inference speed.
- **Modern LangChain Architecture** — Built using LangChain Expression Language (LCEL) for the retrieval-and-answer chain.
- **Bring-Your-Own-API-Key** — No hardcoded credentials; users enter their own free Gemini and Groq API keys directly in the app sidebar, so anyone can run it without the maintainer's keys.
- **Built-in Error Handling** — Gracefully handles corrupted/password-protected PDFs, missing API keys, and missing vector index (prompts the user to process a document first).

---

## Architecture & Tech Stack

| Step | Component | Details |
|---|---|---|
| 1. Document Ingestion | `PyPDF2` | Extracts raw text from one or more uploaded PDFs |
| 2. Semantic Chunking | `RecursiveCharacterTextSplitter` | Splits text into 10,000-character chunks with 1,000-character overlap |
| 3. Embedding Generation | Google Gemini (`models/gemini-embedding-001`) | Converts chunks into vectors — **embeddings only** |
| 4. Vector Database | FAISS | Stores and indexes vectors locally (`faiss_index/`) |
| 5. Retrieval | FAISS similarity search | Retrieves the most relevant chunks for a given question |
| 6. Answer Generation | Groq (`llama-3.3-70b-versatile`), temperature 0.3 | Generates the final answer via an LCEL chain (`prompt \| model \| StrOutputParser`), strictly grounded in retrieved context |

---

## Installation & Setup Guide

**1. Clone the repository**
```bash
git clone https://github.com/dubeypt/AI_Study_Mentor.git
cd AI_Study_Mentor
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `requirements.txt` currently lists `streamlit`, `PyPDF2`, `langchain`, `langchain-google-genai`, `google-generativeai`, `faiss-cpu` — but `app.py` also imports `langchain_groq` (for `ChatGroq`). Add `langchain-groq` to `requirements.txt` before deploying, or installs elsewhere will fail with a `ModuleNotFoundError`.

**4. Run the app**
```bash
streamlit run app.py
```

**5. Add your API keys in the app**

No `.env` file needed — when the app opens, enter your keys directly in the sidebar:
- **Gemini API Key** (for embeddings) — [get a free key](https://aistudio.google.com/app/apikey)
- **Groq API Key** (for answer generation) — [get a free key](https://console.groq.com/keys)

---

## Usage

1. Enter your Gemini and Groq API keys in the sidebar.
2. Upload one or more PDF study documents.
3. Click **"Process Documents"** and wait for chunking + embedding to complete.
4. Type a question in the text box and click **"Generate Answer"**.
5. If the answer isn't in your uploaded material, the AI Mentor will say so directly instead of guessing.

---

## Project Structure

```
AI_Study_Mentor/
├── app.py             # Main Streamlit app — UI, RAG pipeline, chat logic
├── check.py           # [Add a one-line description of what this script does]
├── requirements.txt
├── my .png            # App preview screenshot (consider renaming — see note above)
└── README.md
```

---

## Roadmap / Future Improvements

- Add `langchain-groq` to `requirements.txt`
- Support persistent multi-session FAISS indexes (currently overwritten per new upload)
- Add source-chunk citations alongside answers
- Rename the preview image to remove the space in its filename

---

## Author

**Aditya Dubey**
[GitHub](https://github.com/dubeypt) · [LinkedIn](https://linkedin.com/in/pt-aditya-dubey)
