import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- UI Configuration ---
st.set_page_config(page_title="AI Study Mentor", layout="wide")
st.header("AI Study Mentor (RAG Agent)")

# Sidebar for API Keys
with st.sidebar:
    st.title("Configuration")
    api_key = st.text_input("Enter your Google Gemini API Key (for embeddings)", type="password")
    groq_api_key = st.text_input("Enter your Groq API Key (for answers)", type="password")
    st.markdown("[Get free Gemini key](https://aistudio.google.com/app/apikey) · [Get free Groq key](https://console.groq.com/keys)")

# --- Core RAG Functions with Error Handling ---

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files with error handling for corrupted files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f" Error reading file '{pdf.name}'. It might be corrupted, empty, or password-protected. Please upload a valid PDF.")
            return None  # Stop processing if error occurs
    return text

def get_text_chunks(text):
    """Splits the large text into smaller manageable chunks for the LLM."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Creates embeddings and stores them in a local FAISS database with network error handling."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f" Failed to process embeddings. Please check your Gemini API Key or Internet connection. Details: {e}")
        return False

def get_conversational_chain(groq_api_key):
    """Sets up the LangChain QA pipeline using Groq for fast answer generation."""
    prompt_template = """
    You are a helpful AI Study Mentor. Answer the question as comprehensively as possible using the provided context. 
    If the answer is not in the provided context, just say, "I am sorry, this information is not available in the uploaded study material."
    Do not guess the answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=groq_api_key)
    prompt = PromptTemplate.from_template(prompt_template)

    # Modern LCEL Pipeline
    chain = prompt | model | StrOutputParser()
    return chain

def user_input(user_question, api_key, groq_api_key):
    """Handles user queries and handles missing database or API errors."""
    try:
        if not os.path.exists("faiss_index"):
            st.warning(" Please upload and process a PDF document first before asking questions.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        chain = get_conversational_chain(groq_api_key)

        with st.spinner("Generating answer..."):
            response = chain.invoke({"context": context_text, "question": user_question})
            st.write(" **AI Mentor:** ", response)

    except Exception as e:
        st.error(f" An error occurred while generating the answer. Make sure your API keys are valid. Details: {e}")

# --- Main App Logic ---

if not api_key or not groq_api_key:
    st.warning("Please enter both your Gemini API Key and Groq API Key in the sidebar to proceed.")
else:
    # File Uploader
    pdf_docs = st.file_uploader("Upload your Study Material (PDF) and click 'Process'", accept_multiple_files=True)

    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing your study material..."):
                raw_text = get_pdf_text(pdf_docs)

                # Only proceed if text extraction was successful
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    is_success = get_vector_store(text_chunks, api_key)

                    if is_success:
                        st.success(" Processing Complete! You can now ask questions.")
        else:
            st.error(" Please upload a PDF first.")

    st.divider()

    # Chat Interface
    user_question = st.text_input("Ask a question from your study material:")
    if st.button("Generate Answer"):
        if user_question:
            user_input(user_question, api_key, groq_api_key)
        else:
            st.warning(" Please type a question first.")