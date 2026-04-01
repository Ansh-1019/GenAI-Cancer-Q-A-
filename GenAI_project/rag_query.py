import os
from dotenv import load_dotenv
import time
# --- 1. Load Environment Variables ---
# This loads GOOGLE_API_KEY from your .env file
load_dotenv()

# --- 2. Use Modern LangChain Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# RetrievalQA is now found in the 'langchain_classic' package
from langchain_classic.chains import RetrievalQA 

# Import the Gemini LLM integration
from langchain_google_genai import ChatGoogleGenerativeAI


# --- 3. Configuration (Must match your ingestion script) ---

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
vector_db_path = "./chroma_knowledge_base" # Path confirmed from your directory structure


# --- 4. Load the Components ---

print("--- 1. Loading Vector Store and Embeddings ---")
try:
    # Initialize the embeddings again
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Load the persisted Chroma database
    vectorstore = Chroma(
    persist_directory=vector_db_path,
    embedding_function=embeddings,
    collection_name="cancer_rag_collection"  # MUST MATCH
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
        temperature=0.0 # Set low temperature for factual, consistent answers
    )
    
    print(f"SUCCESS: Gemini LLM ('{LLM_MODEL_NAME}') initialized.")
    
except Exception as e:
    print(f"ERROR: Could not initialize Gemini LLM. Ensure GOOGLE_API_KEY is set in your .env file. Error: {e}")
    exit()


# --- 6. Setup the Retrieval-Augmentation Chain ---

print("--- 3. Setting up Retrieval Chain ---")
# INCREASED k to 5 to retrieve more context for better chance of finding the answer
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
print("SUCCESS: RetrievalQA Chain ready.")


# # --- 7. Define the Query Function ---

# def run_rag_query(query: str):
#     """Performs the full RAG pipeline query."""
#     print(f"\nQUERY: '{query}'", flush=True) # Flush here too
    
#     # Run the chain (Use .invoke() for the latest LangChain version compatibility)
#     result = qa_chain.invoke({"query": query})
    
#     # --- Output Generation ---
#     print("\n" + "="*80, flush=True)
#     print("GENERATED ANSWER (Powered by Gemini):", flush=True)
#     print(result["result"], flush=True)
#     print("="*80, flush=True)
    
#     print("\n--- SOURCE DOCUMENTS (Context Used from ChromaDB) ---", flush=True)
#     for i, doc in enumerate(result["source_documents"]):
#         source_path = doc.metadata.get('source', 'N/A')
#         source_file = os.path.basename(source_path)
        
#         # Ensure all print statements explicitly flush
#         print(f"\n[Source {i+1}] File: {source_file}", flush=True)
#         print(f"  Content Snippet: \"{doc.page_content.strip()[:200]}...\"", flush=True)
#     time.sleep(2)
#     return result



# --- 7. Define the Query Function (Modified for File Output) ---
def run_rag_query_to_file(query: str, filename="rag_output.txt"):
    """Performs RAG query and writes all output to a file."""
    
    # Run the chain (Use .invoke() for the latest LangChain version compatibility)
    result = qa_chain.invoke({"query": query})
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"QUERY: '{query}'\n")
        f.write("="*80 + "\n")
        f.write("GENERATED ANSWER (Powered by Gemini):\n")
        f.write(result["result"] + "\n")
        f.write("="*80 + "\n")
        
        f.write("\n--- SOURCE DOCUMENTS (Context Used from ChromaDB) ---\n")
        for i, doc in enumerate(result["source_documents"]):
            source_path = doc.metadata.get('source', 'N/A')
            source_file = os.path.basename(source_path)
            
            f.write(f"\n[Source {i+1}] File: {source_file}\n")
            f.write(f"  Content Snippet: \"{doc.page_content.strip()[:200]}...\"\n")
            f.write(f"  Full Content:\n{doc.page_content}\n") # Write full content for debug
            
    print(f"\nSUCCESS: Output written to {filename}")
    return result

# --- 8. Execute the Suggested Query (Modified to use the new function) ---
# suggested_query = "What safety concerns are associated with immune checkpoint inhibitors?"

# run_rag_query_to_file(suggested_query)

# --- 8. Execute the Suggested Query ---

# Changed to a simpler query for initial testing confidence
# suggested_query = "What is the strong recommendation regarding partial nephrectomy?"
# run_rag_query(suggested_query)

def run_rag_query_ui(query: str):
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = [
        {
            "content": doc.page_content[:500],
            "metadata": doc.metadata
        }
        for doc in result["source_documents"]
    ]
    return answer, sources
