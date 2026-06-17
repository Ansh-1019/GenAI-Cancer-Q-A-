import os
os.environ["HF_HUB_OFFLINE"] = "1"

import requests
import json
from typing import Dict, Any, List
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_knowledge_base"
COLLECTION_NAME = "cancer_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
OPENFDA_URL = "https://api.fda.gov/drug/event.json"
MAX_RECORDS = 1 

# Use a single, reliable search term for this attempt
DRUG_NAME_SEARCH = "NIVOLUMAB" 

# =================================================================
#                         PART 1: FETCH OPENFDA DATA
# =================================================================

def fetch_fda_adverse_events() -> List[Dict[str, Any]]:
    """Fetches aggregated adverse event reports from openFDA API using a simplified query."""
    
    search_query = f'patient.drug.medicinalproduct:"{DRUG_NAME_SEARCH}"'
    
    params = {
        "search": search_query,
        "limit": MAX_RECORDS, 
        "count": "patient.reaction.reactionmeddrapt.exact" # Aggregate reactions
    }
    
    print(f"\n--- 1. Searching openFDA for adverse events related to {DRUG_NAME_SEARCH} ---")
    
    try:
        response = requests.get(OPENFDA_URL, params=params, timeout=15)
        response.raise_for_status() 
        data = response.json()
        
        total_reports = data.get('meta', {}).get('results', {}).get('total', 0)
        reaction_summary = data.get('results', []) 
        
        if total_reports > 0 and reaction_summary:
            top_reactions = [
                f"{item.get('term')} (Count: {item.get('count', 0)})" 
                for item in reaction_summary[:10]
            ]
            
            return [{
                "drug_searched": DRUG_NAME_SEARCH,
                "data_type": "Adverse Event Summary",
                "total_reports_found": total_reports,
                "most_reported_reactions": "Top 10 reported reactions: " + "; ".join(top_reactions),
                "source": "openFDA Adverse Event API",
                "caveat": "Adverse event reports are UNVERIFIED and do not establish a causal relationship."
            }]
        
        elif total_reports == 0:
            print(f"[Warning] Drug search successful, but found 0 reports. Skipping.")
            return []
            
        else:
             print("[Warning] API call successful, but aggregate data structure was unexpected.")
             return []

    except requests.exceptions.RequestException as e:
        print(f"[FATAL ERROR] Error fetching openFDA data. Skipping.")
        return []

# =================================================================
#                         PART 2: INGESTION INTO CHROMADB
# =================================================================

def create_fda_documents(fda_data: List[Dict[str, Any]]) -> List[Document]:
    """Converts the structured FDA data into LangChain Document objects."""
    documents = []
    for record in fda_data:
        # Create a document for the drug adverse event summary
        page_content = (
            f"Drug Searched: {record['drug_searched']}\n"
            f"Data Type: {record['data_type']}\n"
            f"Total Reports Found: {record['total_reports_found']}\n"
            f"Reactions: {record['most_reported_reactions']}\n"
            f"Caveat: {record['caveat']}"
        )
        doc = Document(
            page_content=page_content,
            metadata={
                "title": f"Adverse Event Summary for {record['drug_searched']}",
                "source": record["source"],
                "document_type": record["data_type"]
            }
        )
        documents.append(doc)
    return documents

def add_documents_to_chroma(new_documents: List[Document]):
    """Loads the existing ChromaDB and adds the new FDA documents."""
    if not new_documents:
        return

    print(f"\n--- 2. Adding {len(new_documents)} documents to ChromaDB ---")

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

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
        print(f"\n[SUCCESS] New ChromaDB created and indexed.")
        return

    print(f"Indexing new documents...")
    vectorstore.add_documents(new_documents)

    vectorstore.persist()
    print(f"\n[SUCCESS] openFDA adverse event data successfully merged and saved to ChromaDB.")

# =================================================================
#                           MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    results = fetch_fda_adverse_events()

    if not results:
        print("No adverse event data returned from openFDA.")
    else:
        print("\nFetched OpenFDA Data:")
        print(json.dumps(results, indent=2))
        
        fda_documents = create_fda_documents(results)
        add_documents_to_chroma(fda_documents)
        
    print("\n\n--- RAG PIPELINE DATA ACQUISITION COMPLETE ---")
