import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="./chroma_knowledge_base",
    embedding_function=embeddings,
    collection_name="cancer_rag_collection"
)

print("Total documents:", db._collection.count())

# --- Configuration (Must match your indexing script) ---
CHROMA_DB_PATH = "./chroma_knowledge_base"
COLLECTION_NAME = "cancer_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

print(f"--- Accessing Local ChromaDB at {CHROMA_DB_PATH} ---")

if not os.path.exists(CHROMA_DB_PATH):
    print("[ERROR] The ChromaDB directory was not found. Please ensure your indexing script ran successfully.")
else:
    try:
        # Initialize Embedding Model
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Load the existing vector store from the persistent directory
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # Access the underlying collection to count the documents
        count = vectorstore._collection.count()
        
        print("\n✅ SUCCESS: Database connection established!")
        print(f"   Total Documents in Knowledge Base: {count} (ICD, PubMed, EMA)")

        # --- Simple Query Test ---
        # Query designed to pull from multiple sources (e.g., ICD-10 code and PubMed abstract)
        query = "What is the ICD-10 code for breast cancer, and what is the current drug research?"
        
        print("\n--- Running Retrieval Test ---")
        docs = vectorstore.similarity_search(query, k=3) 
        
        print(f"✅ Retrieved {len(docs)} relevant chunks. Showing sources:")
        
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source Type: {doc.metadata.get('document_type', 'N/A')}")
            print(f"Title: {doc.metadata.get('title', 'N/A')}")
            print(f"Content Snippet: {doc.page_content[:150]}...")
            
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load the database. Error: {e}")