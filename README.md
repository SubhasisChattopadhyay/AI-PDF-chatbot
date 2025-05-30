AI - PDF Chatbot
This is an AI-powered PDF chatbot designed to facilitate efficient document interaction and information retrieval. This project integrates various technologies to enable users to query PDF documents, extract relevant information, and receive concise answers in real-time.

Features
Document Upload: Upload PDF documents for real-time interaction.

Text Extraction: Utilizes OCR (Optical Character Recognition) for extracting text from scanned PDFs.

Question Answering: Provides answers to user queries based on the content of the PDF.

Natural Language Processing (NLP): Implements NLP techniques for document understanding and query processing.

Interactive Interface: Responsive web interface built with React and Django for seamless user interaction.

Technologies Used

AI & NLP: LangChain, Ollama, PyMuPDF, Tesseract OCR

Web Development: Streamlit

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/smartgov-ai-pdf-chatbot.git
cd smartgov-ai-pdf-chatbot
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
npm install  # if using npm for frontend dependencies
Set up Tesseract OCR:

Download and install Tesseract from Tesseract OCR GitHub page.

Set the tesseract_cmd path in your environment or directly in the code (pytesseract.pytesseract.tesseract_cmd).

Run the application:

bash
Copy
Edit
streamlit run app.py
Usage
Upload a PDF file using the file uploader.

Ask questions related to the uploaded PDF content.

Receive answers based on document analysis and NLP processing.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please create a GitHub issue or fork the repository and submit a pull request.


Acknowledgments
Inspired by the need for efficient document interaction and information retrieval in smart city applications.

Built using state-of-the-art AI and NLP technologies to enhance user accessibility and productivity.

Feel free to customize and expand upon this template based on additional features, specific installation instructions, or any other pertinent details about your project.
