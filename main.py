import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Setup & Environment ---
load_dotenv()
app = FastAPI(title="Resume Chatbot API")

## middleware for deployment (for CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-react-portfolio-url.com"], # <-- Put your actual deployed React URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow your React app to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change "*" to your portfolio's URL (e.g., "https://deepak-portfolio.com")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Initialize Models & Database ---
print("Waking up Gemini and loading ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

# --- 3. Build the Memory Pipeline ---
# Step A: Rewrite the question using history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Step B: Answer the rewritten question using the database
qa_system_prompt = (
    "You are a professional AI assistant representing Deepak Kumar. "
    "Use the following retrieved context from his resume to answer the question. "
    "If you don't know the answer, say 'I am not sure based on the provided resume.'\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step C: Combine into final chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. Define API Data Structures ---
class Message(BaseModel):
    role: str  # "user" or "ai"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []

# --- 5. The API Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert frontend history into LangChain message objects
        langchain_history = []
        for msg in request.history:
            if msg.role == "user":
                langchain_history.append(HumanMessage(content=msg.content))
            else:
                langchain_history.append(AIMessage(content=msg.content))

        # Ask Gemini
        response = rag_chain.invoke({
            "input": request.question,
            "chat_history": langchain_history
        })
        
        return {"answer": response['answer']}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload