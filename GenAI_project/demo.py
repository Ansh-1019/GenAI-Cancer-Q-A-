import os
from dotenv import load_dotenv
import time 

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Use Modern LangChain Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_classic.chains import RetrievalQA 
from langchain_google_genai import ChatGoogleGenerativeAI


# --- 3. Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
vector_db_path = "./chroma_knowledge_base"


# --- 4. Load the Components ---
print("--- 1. Loading Vector Store and Embeddings ---")
try:
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    print("SUCCESS: Vector store and embeddings loaded.")
except Exception as e:
    print(f"ERROR loading components. Check the path and model: {e}")
    exit()


# --- 5. Initialize the LLM (Gemini) ---
print("--- 2. Initializing LLM (Gemini) ---")
try:
    LLM_MODEL_NAME = "gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME, 
        temperature=0.0
    )
    print(f"SUCCESS: Gemini LLM ('{LLM_MODEL_NAME}') initialized.")
except Exception as e:
    print(f"ERROR: Could not initialize Gemini LLM. Ensure GOOGLE_API_KEY is set in your .env file. Error: {e}")
    exit()


# --- 6. Setup the Retrieval-Augmentation Chain ---
print("--- 3. Setting up Retrieval Chain ---")
# Using MMR search
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20}) 

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
print("SUCCESS: RetrievalQA Chain ready.")


# --- 7. Define the Query Function (MODIFIED FOR FILE OUTPUT) ---

def run_rag_query_to_file(query: str, filename="rag_output.txt"):
    """Performs RAG query and writes all output to a file."""
    
    print(f"\nQUERY: '{query}'", flush=True)
    
    # Run the chain
    result = qa_chain.invoke({"query": query})
    
    # Write output to file instead of console
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"GENERATED ANSWER (Powered by Gemini):\n{result['result']}\n")
        f.write("="*80 + "\n")
        
        f.write("\n--- SOURCE DOCUMENTS (Context Used from ChromaDB) ---\n")
        for i, doc in enumerate(result["source_documents"]):
            source_path = doc.metadata.get('source', 'N/A')
            source_file = os.path.basename(source_path)
            
            f.write(f"\n[Source {i+1}] File: {source_file}\n")
            f.write(f"  Content Snippet: \"{doc.page_content.strip()[:200]}...\"\n")
            f.write(f"  Full Content:\n{doc.page_content}\n") 
            
    print(f"\nSUCCESS: Output written to {filename}. Check the file in your project folder.")
    return result

# --- 8. Execute the Suggested Query (Using the new file output function) ---
suggested_query = "What is the strong recommendation regarding partial nephrectomy?"
run_rag_query_to_file(suggested_query)