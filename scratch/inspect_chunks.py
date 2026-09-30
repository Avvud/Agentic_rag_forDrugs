"""
scratch/inspect_chunks.py — Query ChromaDB to see all stored chunks.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
import config

client = chromadb.PersistentClient(path=config.CHROMA_PATH)
col = client.get_collection("drug_interactions")
results = col.get(include=["documents", "metadatas"])

ids = results["ids"]
docs = results["documents"]
metas = results["metadatas"]

print(f"Total chunks in Chroma: {len(ids)}")
print()
print(f"{'chunk_id':<20} | {'page':>4} | {'section':<35} | {'len':>5} | first 120 chars")
print("-" * 110)
for cid, doc, meta in zip(ids, docs, metas):
    preview = (doc or "")[:120].replace("\n", " ")
    print(f"{cid:<20} | {meta.get('page',0):>4} | {meta.get('section',''):<35} | {len(doc or ''):>5} | {preview}")

print()
# Print full text for antihistamines and acid reducers
print("="*70)
print("FULL TEXT: Antihistamines")
print("="*70)
for cid, doc, meta in zip(ids, docs, metas):
    if meta.get("section","").lower() == "antihistamines":
        print(doc)
        print()

print("="*70)
print("FULL TEXT: Acid Reducers/H2 Blockers")
print("="*70)
for cid, doc, meta in zip(ids, docs, metas):
    if "cimetidine" in (doc or "").lower():
        print(f"chunk_id: {cid}")
        print(doc)
        print()
