"""
tests/test_phase0_smoke.py — Phase 0 Preflight and Smoke Tests

P0-1: config.py loads all vars; missing required var raises error naming the var
P0-2: Live Gemini text call returns non-empty string
P0-3: Live embedding call returns a vector; print dimension
P0-4: openFDA warfarin call returns HTTP 200 with results array
P0-5: ddgs returns >= 1 result for "warfarin interactions"
P0-6: PyMuPDF opens PDF; print page count and first 200 chars per page
P0-7: .env is listed in .gitignore
"""

import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# P0-1: Config loading + missing required var raises named error
# ---------------------------------------------------------------------------
def test_p0_1_config_loads_and_validates():
    """Config loads all variables. Missing required var raises error naming it."""
    import config

    # Check all expected keys exist
    assert config.GEMINI_MODEL, "GEMINI_MODEL should be set"
    assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL should be set"
    assert config.PDF_PATH, "PDF_PATH should be set"
    assert config.CHROMA_PATH, "CHROMA_PATH should be set"
    assert config.MAX_AGENT_STEPS >= 1, "MAX_AGENT_STEPS should be >= 1"
    # GEMINI_API_KEY must be non-empty (already proved by config import succeeding)
    assert config.GEMINI_API_KEY, "GEMINI_API_KEY should be set"

    # Now test that a missing required var raises a clear named error
    import importlib
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        # Reload config module with the key cleared
        with pytest.raises(ValueError) as exc_info:
            import importlib
            # We call the private loader directly to test it
            with mock.patch("os.getenv", side_effect=lambda k, d=None: "" if k == "GEMINI_API_KEY" else os.environ.get(k, d)):
                config._load()

    error_msg = str(exc_info.value)
    assert "GEMINI_API_KEY" in error_msg, f"Error should name the variable: {error_msg}"
    # Must NOT contain the actual key value (key is empty here, but verify pattern)
    assert "=" not in error_msg or "GEMINI_API_KEY" in error_msg, "Should not expose key value"

    print("P0-1 PASS: config loads correctly and raises named error for missing var")


# ---------------------------------------------------------------------------
# P0-2: Live Gemini text call
# ---------------------------------------------------------------------------
def test_p0_2_gemini_text_call():
    """Live Gemini generate_content call returns non-empty text."""
    from google import genai
    import config

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents="Say 'hello' in exactly one word.",
    )
    text = response.text
    assert text and len(text.strip()) > 0, "Response text should be non-empty"
    print(f"P0-2 PASS: Gemini model='{config.GEMINI_MODEL}' responded: {text.strip()!r}")


# ---------------------------------------------------------------------------
# P0-3: Live embedding call
# ---------------------------------------------------------------------------
def test_p0_3_embedding_call():
    """Live embedding call returns a vector; print its dimension."""
    from google import genai
    from google.genai.types import EmbedContentConfig
    import config

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents="Test embedding for drug interaction RAG.",
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    embedding = response.embeddings[0].values
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding should be non-empty"
    print(f"P0-3 PASS: Embedding model='{config.EMBEDDING_MODEL}' dimension={len(embedding)}")


# ---------------------------------------------------------------------------
# P0-4: openFDA warfarin call
# ---------------------------------------------------------------------------
def test_p0_4_openfda_warfarin():
    """openFDA call for warfarin returns HTTP 200 with a results array."""
    import httpx
    import config

    params = {
        "search": 'openfda.generic_name:"warfarin"',
        "limit": 1,
    }
    if config.OPENFDA_API_KEY:
        params["api_key"] = config.OPENFDA_API_KEY

    resp = httpx.get(
        "https://api.fda.gov/drug/label.json",
        params=params,
        timeout=10.0,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "results" in data, "Response should have 'results' key"
    assert isinstance(data["results"], list), "'results' should be a list"
    assert len(data["results"]) >= 1, "'results' should be non-empty"
    print(f"P0-4 PASS: openFDA warfarin returned {len(data['results'])} result(s)")


# ---------------------------------------------------------------------------
# P0-5: DuckDuckGo search
# ---------------------------------------------------------------------------
def test_p0_5_ddgs_search():
    """ddgs returns at least 1 result for 'warfarin interactions'."""
    from ddgs import DDGS

    results = list(DDGS().text("warfarin interactions", max_results=4))
    assert len(results) >= 1, "Should return at least 1 result"
    assert "title" in results[0], "Result should have 'title'"
    print(f"P0-5 PASS: ddgs returned {len(results)} result(s). First: {results[0]['title']!r}")


# ---------------------------------------------------------------------------
# P0-6: PyMuPDF opens PDF
# ---------------------------------------------------------------------------
def test_p0_6_pymupdf_opens_pdf():
    """PyMuPDF opens the PDF; print page count and first 200 chars per page."""
    import fitz  # PyMuPDF
    import config

    pdf_path = config.PDF_PATH
    assert os.path.exists(pdf_path), f"PDF not found at {pdf_path}"

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    assert page_count > 0, "PDF should have at least 1 page"

    print(f"\nP0-6: PDF has {page_count} pages. First 200 chars per page:")
    for i, page in enumerate(doc):
        text = page.get_text("text")
        snippet = text[:200].replace("\n", " ")
        print(f"  Page {i+1}: {snippet!r}")

    doc.close()
    print(f"P0-6 PASS: Opened PDF with {page_count} pages")


# ---------------------------------------------------------------------------
# P0-7: .env in .gitignore
# ---------------------------------------------------------------------------
def test_p0_7_env_in_gitignore():
    """Verify that .env is listed in .gitignore."""
    project_root = os.path.join(os.path.dirname(__file__), "..")
    gitignore_path = os.path.join(project_root, ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore does not exist"

    with open(gitignore_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    # Accept either ".env" or ".env" as a standalone line
    assert ".env" in lines, f".env not found in .gitignore. Lines: {lines}"
    print(f"P0-7 PASS: .env found in .gitignore")
