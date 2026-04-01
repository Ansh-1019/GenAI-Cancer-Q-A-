import requests
import json
import os
import time
from typing import Dict, Any, List
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# =================================================================
#                         CONFIGURATION
# =================================================================

# 1. Input File Paths
PUBMED_FILE = "./RAG KB/pubmed_cancer_abstracts.json"
EMA_FILE = "./RAG KB/non_epar-documents_json_20251121t060507z.json"

# 2. ChromaDB Settings (MUST match your existing local DB folder)
CHROMA_DB_PATH = "./chroma_knowledge_base"
COLLECTION_NAME = "cancer_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# =================================================================
#                         PART 1: DATA LOADING AND METADATA
# =================================================================

def metadata_func(record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamically pulls relevant metadata based on the source structure."""
    
    # Check if the record looks like EMA data
    if record.get("status") or record.get("publish_date"):
        metadata["title"] = record.get("name")
        metadata["document_type"] = record.get("type", "EMA Document")
        metadata["status"] = record.get("status", "N/A")
        metadata["source"] = "European Medicines Agency"
        metadata["source_url"] = record.get("url", "N/A")
    
    # Check if the record looks like PubMed data
    elif record.get("pmid"):
        metadata["title"] = record.get("title")
        metadata["pmid"] = record.get("pmid")
        metadata["document_type"] = "PubMed Abstract"
        metadata["source"] = "NIH PubMed"
        metadata["source_url"] = record.get("source_url")
    
    return metadata

def load_data_from_json(file_path: str, content_key: str, encoding: str = 'utf-8') -> List[Document]:
    """Loads a JSON file into LangChain Document objects with specified encoding."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}. Skipping.")
        return []

    # FIX 2: Implement custom file reading with error encoding
    # We load the data first using the fallback encoding (latin-1) for EMA files
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[FATAL ERROR] Could not read/decode {file_path} with {encoding} encoding. Error: {e}")
        return []

    # FIX 1: The LangChain JSONLoader expects a JSON string, not a loaded Python object.
    # The simplest fix is to create the documents manually if the loader is being problematic.
    
    documents = []
    
    # Assume data is a list of dictionaries at the root level (most common API output)
    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict) and record.get(content_key):
                # Safely extract content and apply metadata function
                doc = Document(
                    page_content=record[content_key],
                    metadata=metadata_func(record, {})
                )
                documents.append(doc)
            # Else: skip problematic elements (which caused the original mismatch error)
            
    print(f"Loaded {len(documents)} documents from {os.path.basename(file_path)}.")
    return documents

# =================================================================
#                         PART 2: MERGE AND INDEX with ChromaDB
# =================================================================

def create_faiss_index(all_documents: List[Document]):
    """Loads the existing ChromaDB and adds all documents."""
    if not all_documents:
        print("[FATAL] No documents to index. Indexing aborted.")
        return

    print(f"\n--- Starting Data Ingestion ---")
    print(f"Total documents to index: {len(all_documents)}")
    
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DB_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("Existing ChromaDB loaded successfully.")
    else:
        print(f"[FATAL] ChromaDB folder not found at {CHROMA_DB_PATH}. Cannot merge. Run the ICD script first.")
        return

    print(f"Indexing new documents...")
    vectorstore.add_documents(all_documents)

    vectorstore.persist()
    print(f"\nSUCCESS: All data (PubMed and EMA) successfully merged and saved to ChromaDB.")

# =================================================================
#                           MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    
    # Step 1: Load PubMed Data (Use 'abstract' key)
    # The default encoding 'utf-8' is used here
    pubmed_docs = load_data_from_json(
        PUBMED_FILE, 
        content_key='abstract'
    )
    
    # Step 2: Load EMA Data (Use 'name' key and the fallback encoding for the fix)
    ema_docs = load_data_from_json(
        EMA_FILE, 
        content_key='name',
        encoding='latin-1' # <-- Applying the fix here
    )
    
    # Step 3: MERGE THE TWO DATASETS
    all_merged_docs = pubmed_docs + ema_docs
    
    print(f"\n--- MERGE COMPLETE ---")
    print(f"Total PubMed/EMA documents to add: {len(all_merged_docs)}")
    
    # Step 4: Index the merged data
    create_faiss_index(all_merged_docs)