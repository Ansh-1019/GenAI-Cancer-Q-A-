import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Any

# =================================================================
#                         CONFIGURATION
# =================================================================

# 1. Input Folder (where you save the two PDFs)
PDF_FOLDER = "./guideline_pdfs" 
# 2. Database Settings (MUST match your existing local DB folder)
CHROMA_DB_PATH = "./chroma_knowledge_base"
COLLECTION_NAME = "cancer_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# 3. Chunking Strategy (Optimized for detailed guidelines)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# =================================================================
#                         PART 1: LOAD AND SPLIT PDFs
# =================================================================

def load_and_split_pdfs(folder_path: str) -> List[Document]:
    """Loads all PDFs from a folder, chunks them, and extracts metadata."""
    
    all_documents = []
    
    # Check if the path exists (ensure the user created the folder)
    if not os.path.isdir(folder_path):
        print(f"[ERROR] PDF folder not found at '{folder_path}'. Please create it and save the PDFs.")
        return all_documents
        
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in '{folder_path}'. Skipping ingestion.")
        return all_documents

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""] # Aggressive splitting for structured docs
    )

    print(f"\n--- 1. Loading and Splitting {len(pdf_files)} Guideline PDFS ---")

    for file_path in pdf_files:
        try:
            # PyPDFLoader is reliable for structured text extraction
            loader = PyPDFLoader(file_path)
            
            # Load and split documents
            raw_docs = loader.load()
            
            # Extract clean filename for metadata
            file_name = os.path.basename(file_path)
            
            # Add file-specific metadata and split into chunks
            for doc in raw_docs:
                doc.metadata["source"] = "EAU/ESMO Clinical Guideline"
                doc.metadata["document_name"] = file_name
                
            # Split the document objects
            chunks = text_splitter.split_documents(raw_docs)
            all_documents.extend(chunks)
            
            print(f"  Processed {file_name}: created {len(chunks)} chunks.")

        except Exception as e:
            print(f"[FATAL ERROR] Could not process PDF {file_path}. Error: {e}")
            
    return all_documents

# =================================================================
#                         PART 2: MERGE INTO CHROMADB
# =================================================================

def add_documents_to_chroma(new_documents: List[Document]):
    """Loads the existing ChromaDB and adds the new documents."""
    if not new_documents:
        return

    print(f"\n--- 2. Merging {len(new_documents)} documents into ChromaDB ---")

    # 1. Initialize Embedding Model
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Load the EXISTING ChromaDB instance
    if os.path.exists(CHROMA_DB_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("Existing ChromaDB loaded successfully.")
    else:
        print(f"[FATAL] ChromaDB folder not found at {CHROMA_DB_PATH}. Run initial indexer script first.")
        return

    # 3. Add the new documents to the existing index
    print(f"Indexing new documents (Merging with ICD/PubMed/EMA data)...")
    vectorstore.add_documents(new_documents)

    # 4. Persist (Save) the updated database
    vectorstore.persist()
    print(f"\nSUCCESS: All Guideline data successfully merged and saved to ChromaDB.")

# =================================================================
#                           MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    
    # Ensure the EAU and ESMO PDFs are in a subfolder named 'guideline_pdfs'
    # EAU File: EAU-Guidelines-on-Renal-Cell-Carcinoma-2024 (1).pdf
    # ESMO File: PIIS0923753421021840.pdf

    # Step 1: Load and chunk the PDFs
    guideline_documents = load_and_split_pdfs(PDF_FOLDER)
    
    # Step 2: Merge chunks into the existing ChromaDB
    add_documents_to_chroma(guideline_documents)

    print("\n\n--- RAG PIPELINE DATA ACQUISITION COMPLETE ---")