import requests
import json
import os
from typing import Dict, Any, List
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Configuration (Keep the same) ---
CHROMA_DB_PATH = "./chroma_knowledge_base"
COLLECTION_NAME = "cancer_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
OPENFDA_URL = "https://api.fda.gov/drug/event.json"
MAX_RECORDS = 1 

# Use a single, reliable search term for this final attempt
DRUG_NAME_SEARCH = "NIVOLUMAB" 

# =================================================================
#                         PART 1: FETCH OPENFDA DATA (FINAL FIX)
# =================================================================

def fetch_fda_adverse_events() -> List[Dict[str, Any]]:
    """Fetches aggregated adverse event reports from openFDA API using a simplified query."""
    
    # NEW SIMPLIFIED QUERY: Search the general 'medicinalproduct' field.
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
        # Print the specific URL that failed for diagnosis
        print(f"[FATAL ERROR] Error fetching openFDA data. URL: {response.url if 'response' in locals() else 'N/A'}. Skipping.")
        return []

# --- The rest of the script (create_fda_documents and add_documents_to_chroma) 
# --- should be kept the same as the logic is correct.

# (Your existing code continues below...)
if __name__ == "__main__":
    results = fetch_fda_adverse_events()

    if not results:
        print("No adverse event data returned from openFDA.")
    else:
        print("\nFetched OpenFDA Data:")
        print(json.dumps(results, indent=2))
