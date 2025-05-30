import os
import base64
import pytesseract
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set the path to Tesseract (for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Apply custom dark theme CSS
st.markdown("""
    <style>
    body, .stApp, .block-container {
        background-color: #0f0f0f;
        color: #f5f5f5;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stFileUploader > div {
        background-color: #1f1f1f;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #272727;
        color: #f5f5f5;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #3a3a3a;
    }
    .uploadedFileName {
        color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #00ffe1;'>🧠 PDF Chatbot with Ollama & OCR</h1>", unsafe_allow_html=True)

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF (camera-scan or text-based)", type="pdf")

# Show PDF preview
if uploaded_file:
    st.markdown("<h3 style='color: #00ffe1;'>📘 PDF Preview</h3>", unsafe_allow_html=True)
    base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)
    uploaded_file.seek(0)

# Persistent DB path
chroma_db = "chroma_db"
if not os.path.exists(chroma_db):
    os.makedirs(chroma_db)

# LLM and Embeddings
llm = Ollama(model="llama3.2")
embeddings = OllamaEmbeddings(model="llama3.2")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)

# Extract text using OCR if necessary
def extract_text_ocr(file):
    pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(image)
    return text

def extract_pdf_text(uploaded_file):
    try:
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        extracted_text = ""
        for page in pdf_doc:
            text = page.get_text()
            extracted_text += text
        if len(extracted_text.strip()) < 100:
            st.warning("Low text detected, switching to OCR mode...")
            uploaded_file.seek(0)
            return extract_text_ocr(uploaded_file)
        return extracted_text
    except Exception as e:
        st.error("Error reading PDF: " + str(e))
        return ""

def process_pdf(text):
    chunks = text_splitter.split_text(text)
    vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=chroma_db)
    vectordb.persist()
    return vectordb

# Main logic
if uploaded_file:
    uploaded_file.seek(0)
    pdf_text = extract_pdf_text(uploaded_file)

    user_question = st.text_input("💬 Ask a question about the PDF:")
    if user_question:
        vectordb = process_pdf(pdf_text)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        response = conversation.run({"question": user_question})
        st.markdown("<h3 style='color: #00ffe1;'>🧠 Answer</h3>", unsafe_allow_html=True)
        st.write(response)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #555;'>Built with ❤️ using Ollama, LangChain, Streamlit & Tesseract OCR</div>", unsafe_allow_html=True)
