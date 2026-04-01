import requests
import os
import json
from typing import Dict, Any, List
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb


client = chromadb.CloudClient(
  api_key='ck-BDnq7oaUW454eqM7KNY6SzVbgic2mv2bgpwYJXkpn9rz',
  tenant='2f4c8907-f54c-4764-88db-9cbb376ba023',
  database='chroma_knowledge_base'
)

# Configuration (Keep the same)
CHROMA_DB_PATH = "./chroma_knowledge_base"
COLLECTION_NAME = "cancer_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
ICD_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
MAX_LIMIT_PER_CALL = 500  # Max results allowed per request

# --- NEW: Define a specific search pattern for cancer codes ---
# We will search for all codes starting with C00 through C96 (Malignant Neoplasms)
# NOTE: The API treats the 'terms' parameter as a search string, not a code range.
# We will search by common chapter names instead.
SEARCH_CHAPTERS = [
    "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", # Malignant Neoplasms
    "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", # Malignant Neoplasms
    "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", # Malignant Neoplasms
    # ... and so on up to C96... (truncated for brevity)
    "Melanoma", # Use a keyword that is highly standardized
    "Lymphoma", # Use a keyword that is highly standardized
    "Sarcoma"
]
MAX_RECORDS_TO_FETCH = 7500 # The total limit of the API

# =================================================================
#                         PART 1: FETCH ICD-10 DATA
# =================================================================

def fetch_icd_codes() -> List[Dict[str, Any]]:
    all_structured_data = []
    
    print("\n--- 1. Fetching Cancer-Related ICD-10-CM Codes (Paginated) ---")
    
    # Iterate through the specific chapter codes/keywords
    for chapter_term in SEARCH_CHAPTERS:
        current_offset = 0
        total_records_found = 0
        
        while current_offset < MAX_RECORDS_TO_FETCH:
            
            params = {
                "terms": chapter_term, 
                "count": MAX_LIMIT_PER_CALL,
                "offset": current_offset,
                "df": "code,name"
            }
            
            try:
                response = requests.get(ICD_URL, params=params, timeout=15)
                response.raise_for_status() 
                data_array = response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"[FATAL ERROR] API failed for term {chapter_term}. Error: {e}")
                break

            if len(data_array) < 4 or not data_array[1]:
                # If codes list (data_array[1]) is empty, we reached the end for this term
                break
                
            codes_list = data_array[1]
            display_names_list = data_array[3]

            # Structure the data
            for i in range(len(codes_list)):
                code = codes_list[i]
                description = display_names_list[i][0] if display_names_list[i] and display_names_list[i][0] else "N/A"
                
                all_structured_data.append({
                    "code": code,
                    "description": description,
                    "data_type": "ICD-10-CM Code",
                    "source": "NIH Clinical Tables",
                    "search_text": f"ICD-10 Code {code}: {description}"
                })
            
            records_fetched_in_this_call = len(codes_list)
            current_offset += records_fetched_in_this_call
            total_records_found = data_array[0] # The total count is the first element
            
            print(f"  Fetched {records_fetched_in_this_call} for '{chapter_term}'. Total: {total_records_found}. Offset: {current_offset}")

            if current_offset >= total_records_found:
                break
            
            time.sleep(0.5) # Be kind to the API server

    print(f"\nSuccessfully retrieved and structured {len(all_structured_data)} total ICD-10 records.")
    return all_structured_data

# =================================================================
#                         PART 2: INGESTION INTO CHROMADB
# =================================================================

def create_icd_documents(icd_data: List[Dict[str, Any]]) -> List[Document]:
    """Converts the structured ICD data into LangChain Document objects."""
    documents = []
    for record in icd_data:
        # Create a document for each code/description pair
        doc = Document(
            page_content=record["search_text"], # Use the custom search_text for content
            metadata={
                "title": record["description"],
                "code": record["code"],
                "document_type": record["data_type"],
                "source": record["source"]
            }
        )
        documents.append(doc)
    return documents

def add_documents_to_chroma(new_documents: List[Document]):
    """Loads the existing ChromaDB and adds the new ICD documents."""
    if not new_documents:
        return

    print(f"\n--- 2. Adding {len(new_documents)} documents to ChromaDB ---")

    # 1. Initialize Embedding Model
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Load the EXISTING ChromaDB instance
    # This ensures we don't overwrite the previous PubMed/EMA data
    if os.path.exists(CHROMA_DB_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("Existing ChromaDB loaded successfully.")
    else:
        print(f"[WARNING] ChromaDB folder not found at {CHROMA_DB_PATH}. Creating a new one.")
        vectorstore = Chroma.from_documents(
            documents=new_documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME
        )
        vectorstore.persist()
        print(f"\nSUCCESS: New ChromaDB created and indexed.")
        return

    # 3. Add the new documents to the existing index
    print(f"Indexing new documents...")
    vectorstore.add_documents(new_documents)

    # 4. Persist (Save) the updated database
    vectorstore.persist()
    print(f"\nSUCCESS: ICD-10 data successfully merged and saved to ChromaDB.")

# =================================================================
#                           MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    # Step 1: Fetch the ICD-10 data
    icd_data = fetch_icd_codes()
    
    # Step 2: Convert data to LangChain documents
    icd_documents = create_icd_documents(icd_data)

    # Step 3: Add documents to the existing ChromaDB
    add_documents_to_chroma(icd_documents)

    print("\n\n--- RAG PIPELINE DATA ACQUISITION COMPLETE ---")