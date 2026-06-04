import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Load the environment variables to get GOOGLE_API_KEY
load_dotenv()

# 2. Load the PDF
pdf_path = "Deepak_Kumar_Resume.pdf"  # <-- Update this to your exact filename
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Loaded {len(pages)} page(s) from resume.")

# 3. Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50 
)
chunks = text_splitter.split_documents(pages)
print(f"Split resume into {len(chunks)} chunks.")

# 4. Initialize Google's Cloud Embedding Model
print("Connecting to Gemini API for embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 5. Save to Chroma Vector Database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Success! Your resume has been embedded via Gemini and stored in ChromaDB.")