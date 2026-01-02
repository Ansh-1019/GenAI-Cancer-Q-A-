import json

path = "./RAG KB/non_epar-documents_json_20251121t060507z.json"

with open(path, "rb") as f:
    raw = f.read()

print(raw[:500])
