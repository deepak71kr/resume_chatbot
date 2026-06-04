import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables
load_dotenv()

print("Initializing Gemini API and loading database...")

# 4. Initialize Google's Cloud Embedding Model
print("Connecting to Gemini API for embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 3. Connect to the existing ChromaDB folder
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
# Fetch the top 3 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

# 4. Initialize the free Gemini LLM over the internet
# gemini-2.5-flash is extremely fast and perfect for RAG tasks
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2  
)

# 5. Define the prompt structure
system_prompt = (
    "You are a professional, helpful AI assistant representing me (the candidate). "
    "Use the following pieces of retrieved context from my resume to answer the question. in very crisp bullet points short and to the point (max 30 words as response)."
    "Be concise, confident, and professional. If you don't know the answer, say "
    "'I am not sure about that based on the provided resume.'\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. Create the RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. Continuous Interactive Chat Loop
print("\n=== Chatbot Ready! ===")
print("Type your question below. Type 'exit' or 'quit' to end the chat.\n")

while True:
    # Get question from user
    user_question = input("You: ")
    
    # Check for exit command
    if user_question.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    # Skip empty inputs
    if not user_question.strip():
        continue
        
    print("Bot is thinking...")
    
    try:
        # Run the RAG chain
        response = rag_chain.invoke({"input": user_question})
        print(f"\nAI: {response['answer']}\n")
        print("-" * 40)
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")