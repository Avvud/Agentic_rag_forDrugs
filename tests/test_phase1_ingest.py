"""
tests/test_phase1_ingest.py — Phase 1 Ingestion Tests

P1-1: Running ingest twice gives the same chunk count (no duplicates).
P1-2: Every chunk has non-empty required fields; chunk_id values are unique.
P1-3: At least 12 chunks have a section matching an OTC category name.
P1-4: Cimetidine chunk contains theophylline, warfarin, phenytoin in SAME chunk.
P1-5: Antihistamines chunk contains sedatives+alcohol, NOT nicotine/laxative.
P1-6: No chunk shorter than 40 or longer than 2000 characters.
P1-7: Semantic: querying Chroma with "cimetidine interactions" returns cimetidine chunk in top 3.
P1-8: Page numbers in metadata are within 1..page_count.
"""

import os
import sys
import shutil
import tempfile
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from ingest import ingest, OTC_CATEGORIES

import chromadb
from google import genai
from google.genai.types import EmbedContentConfig


# ---------------------------------------------------------------------------
# Module-scoped fixture: run ingestion once into a temp ChromaDB path
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ingestion_result(tmp_path_factory):
    """Run ingestion once and return (chunk_count, chroma_path)."""
    chroma_path = str(tmp_path_factory.mktemp("chroma_test"))
    count = ingest(pdf_path=config.PDF_PATH, chroma_path=chroma_path)
    return count, chroma_path


@pytest.fixture(scope="module")
def collection(ingestion_result):
    """Get the ChromaDB collection from the test ingestion."""
    _, chroma_path = ingestion_result
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection("drug_interactions")


@pytest.fixture(scope="module")
def all_chunks(collection):
    """Get all chunks from the collection as a list of dicts."""
    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for i, doc_id in enumerate(result["ids"]):
        chunks.append({
            "chunk_id": doc_id,
            "text": result["documents"][i],
            "page": result["metadatas"][i]["page"],
            "section": result["metadatas"][i]["section"],
            "source": result["metadatas"][i]["source"],
        })
    return chunks


# ---------------------------------------------------------------------------
# P1-1: Idempotency — running ingest twice gives same chunk count
# ---------------------------------------------------------------------------
def test_p1_1_idempotent_ingest(ingestion_result):
    """Running ingest twice gives the same chunk count (no duplicates)."""
    first_count, chroma_path = ingestion_result

    # Run a second time to the same path
    second_count = ingest(pdf_path=config.PDF_PATH, chroma_path=chroma_path)

    assert first_count == second_count, (
        f"Idempotency fail: first={first_count}, second={second_count}"
    )

    # Also verify the collection has exactly that many items
    client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_collection("drug_interactions")
    assert coll.count() == second_count, (
        f"Collection count {coll.count()} != ingest return {second_count}"
    )

    print(f"P1-1 PASS: Ingest twice -> {first_count} chunks each time")


# ---------------------------------------------------------------------------
# P1-2: Every chunk has non-empty required fields; chunk_ids unique
# ---------------------------------------------------------------------------
def test_p1_2_chunk_fields_and_uniqueness(all_chunks):
    """Every chunk has non-empty chunk_id, page, section, source, text; IDs unique."""
    chunk_ids = []
    for c in all_chunks:
        assert c["chunk_id"], f"Empty chunk_id in chunk: {c}"
        assert c["page"], f"Empty page in chunk {c['chunk_id']}"
        assert c["section"], f"Empty section in chunk {c['chunk_id']}"
        assert c["source"], f"Empty source in chunk {c['chunk_id']}"
        assert c["text"] and len(c["text"].strip()) > 0, (
            f"Empty text in chunk {c['chunk_id']}"
        )
        chunk_ids.append(c["chunk_id"])

    # Check uniqueness
    unique_ids = set(chunk_ids)
    assert len(unique_ids) == len(chunk_ids), (
        f"Duplicate chunk_ids: {len(chunk_ids)} total, {len(unique_ids)} unique"
    )

    print(f"P1-2 PASS: {len(chunk_ids)} chunks, all have required fields, all IDs unique")


# ---------------------------------------------------------------------------
# P1-3: At least 12 chunks match an OTC category name
# ---------------------------------------------------------------------------
def test_p1_3_otc_categories_present(all_chunks):
    """At least 12 chunks have a section matching an OTC category name."""
    # Normalise category names for comparison
    category_set = {cat.lower() for cat in OTC_CATEGORIES}

    found_categories = set()
    for c in all_chunks:
        if c["section"].lower() in category_set:
            found_categories.add(c["section"])

    missing = [cat for cat in OTC_CATEGORIES if cat.lower() not in {fc.lower() for fc in found_categories}]

    print(f"  Found categories ({len(found_categories)}): {sorted(found_categories)}")
    if missing:
        print(f"  Missing categories: {missing}")

    assert len(found_categories) >= 12, (
        f"Only {len(found_categories)} OTC categories found, need >= 12. "
        f"Missing: {missing}"
    )

    print(f"P1-3 PASS: {len(found_categories)} OTC categories found (>= 12)")


# ---------------------------------------------------------------------------
# P1-4: Cimetidine chunk has theophylline, warfarin, phenytoin in SAME chunk
# ---------------------------------------------------------------------------
def test_p1_4_cimetidine_chunk_contents(all_chunks):
    """The cimetidine chunk contains theophylline, warfarin, phenytoin."""
    cimetidine_chunks = [
        c for c in all_chunks
        if "cimetidine" in c["text"].lower()
    ]

    assert len(cimetidine_chunks) >= 1, "No chunk contains 'cimetidine'"

    # Check the first cimetidine chunk has all three drugs
    chunk = cimetidine_chunks[0]
    text_lower = chunk["text"].lower()

    assert "theophylline" in text_lower, (
        f"Chunk {chunk['chunk_id']} missing 'theophylline'"
    )
    assert "warfarin" in text_lower, (
        f"Chunk {chunk['chunk_id']} missing 'warfarin'"
    )
    assert "phenytoin" in text_lower, (
        f"Chunk {chunk['chunk_id']} missing 'phenytoin'"
    )

    print(
        f"P1-4 PASS: Chunk {chunk['chunk_id']} (section={chunk['section']}) "
        f"contains cimetidine, theophylline, warfarin, phenytoin"
    )


# ---------------------------------------------------------------------------
# P1-5: Antihistamines chunk contains sedatives+alcohol, NOT nicotine/laxative
# ---------------------------------------------------------------------------
def test_p1_5_antihistamines_chunk_contents(all_chunks):
    """Antihistamines chunk has sedatives+alcohol, not nicotine/laxative."""
    anti_chunks = [
        c for c in all_chunks
        if c["section"].lower() == "antihistamines"
    ]

    assert len(anti_chunks) >= 1, "No chunk with section 'Antihistamines'"

    chunk = anti_chunks[0]
    text_lower = chunk["text"].lower()

    # Must contain
    assert "sedative" in text_lower or "sedatives" in text_lower, (
        f"Antihistamines chunk missing 'sedatives'"
    )
    assert "alcohol" in text_lower, (
        f"Antihistamines chunk missing 'alcohol'"
    )

    # Must NOT contain text from other categories
    assert "nicotine" not in text_lower, (
        f"Antihistamines chunk contains 'nicotine' (from another category)"
    )
    assert "laxative" not in text_lower, (
        f"Antihistamines chunk contains 'laxative' (from another category)"
    )

    print(
        f"P1-5 PASS: Antihistamines chunk {chunk['chunk_id']} has "
        f"sedatives+alcohol, no nicotine/laxative"
    )


# ---------------------------------------------------------------------------
# P1-6: No chunk shorter than 40 or longer than 2000 characters
# ---------------------------------------------------------------------------
def test_p1_6_chunk_length_bounds(all_chunks):
    """No chunk < 40 chars or > 2000 chars."""
    outliers = []
    for c in all_chunks:
        length = len(c["text"])
        if length < 40 or length > 2000:
            outliers.append((c["chunk_id"], length))

    if outliers:
        print(f"  Outliers: {outliers}")

    assert len(outliers) == 0, (
        f"{len(outliers)} chunks outside [40, 2000] range: {outliers}"
    )

    lengths = [len(c["text"]) for c in all_chunks]
    print(
        f"P1-6 PASS: All {len(all_chunks)} chunks in [40, 2000] chars. "
        f"min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}"
    )


# ---------------------------------------------------------------------------
# P1-7: Semantic search — "cimetidine interactions" returns cimetidine in top 3
# ---------------------------------------------------------------------------
def test_p1_7_semantic_cimetidine_search(collection):
    """Querying 'cimetidine interactions' returns the cimetidine chunk in top 3."""
    # Embed the query with RETRIEVAL_QUERY task type
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents="cimetidine interactions",
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = resp.embeddings[0].values

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["metadatas", "documents"],
    )

    result_ids = results["ids"][0]
    result_sections = [m["section"] for m in results["metadatas"][0]]

    print(f"  Top 3 results: {list(zip(result_ids, result_sections))}")

    # Check if any result contains "cimetidine"
    found_cimetidine = False
    for i, doc in enumerate(results["documents"][0]):
        if "cimetidine" in doc.lower():
            found_cimetidine = True
            print(f"  Cimetidine found at rank {i+1}: {result_ids[i]}")
            break

    assert found_cimetidine, (
        f"Cimetidine chunk not in top 3. Got: {result_ids}"
    )

    print(f"P1-7 PASS: Cimetidine chunk found in top 3 semantic results")


# ---------------------------------------------------------------------------
# P1-8: Page numbers are within 1..page_count
# ---------------------------------------------------------------------------
def test_p1_8_page_numbers_in_range(all_chunks):
    """All page numbers in metadata are within 1..page_count."""
    import pymupdf

    doc = pymupdf.open(config.PDF_PATH)
    page_count = len(doc)
    doc.close()

    out_of_range = []
    for c in all_chunks:
        page = c["page"]
        if page < 1 or page > page_count:
            out_of_range.append((c["chunk_id"], page))

    assert len(out_of_range) == 0, (
        f"Chunks with page outside [1, {page_count}]: {out_of_range}"
    )

    pages_used = sorted(set(c["page"] for c in all_chunks))
    print(
        f"P1-8 PASS: All pages in [1, {page_count}]. "
        f"Pages used: {pages_used}"
    )
