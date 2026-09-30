"""
tests/test_phase2_tools.py — Unit and integration tests for Phase 2 agent tools.

Tests:
P2-1:  search_pdf("antihistamine drowsiness") -> 1-4 results, all 5 keys, top section Antihistamines
P2-2:  get_category_warnings("nighttime sleep aids") -> found: True + chunk_id
P2-3:  get_category_warnings("antihistamines") and "Antihistamines" -> same chunk
P2-4:  get_category_warnings("nonsense xyz") -> found: False + non-empty available_categories
P2-5:  openfda_label_lookup("warfarin") -> found: True, non-empty text, set_id, dailymed_url
P2-6:  openfda_label_lookup("warfarin", "bad_section") -> error, NO HTTP request made
P2-7:  openfda_label_lookup("zzzznotadrug") -> found: False
P2-8:  Very long label -> text <= 2000 chars + truncated: True (mocked)
P2-9:  Empty vs set OPENFDA_API_KEY -> api_key param logic (mocked)
P2-10: web_search("ibuprofen warfarin interaction") -> results with title, url, snippet
P2-11: Force exception in each tool -> returns {"error": ...}
"""

import pytest
from unittest.mock import patch, MagicMock
import httpx
import chromadb

import config
import ingest
import tools


@pytest.fixture(scope="module", autouse=True)
def ensure_chroma_db():
    """Ensure ChromaDB is populated before running tools tests."""
    chroma_path = config.CHROMA_PATH
    client = chromadb.PersistentClient(path=chroma_path)
    collections = [c.name for c in client.list_collections()]
    if "drug_interactions" not in collections:
        print("\nChromaDB collection 'drug_interactions' not found. Running ingest()...")
        ingest.ingest()
    else:
        coll = client.get_collection("drug_interactions")
        if coll.count() == 0:
            print("\nChromaDB collection is empty. Running ingest()...")
            ingest.ingest()


# ---------------------------------------------------------------------------
# P2-1: search_pdf live search
# ---------------------------------------------------------------------------
def test_p2_1_search_pdf():
    results = tools.search_pdf("antihistamine drowsiness", top_k=4)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert 1 <= len(results) <= 4, f"Expected 1..4 results, got {len(results)}"

    required_keys = {"chunk_id", "page", "section", "text", "score"}
    for r in results:
        assert isinstance(r, dict), f"Result item is not dict: {r}"
        assert required_keys.issubset(r.keys()), f"Missing keys in result: {r.keys()}"
        assert r["chunk_id"], "chunk_id is empty"
        assert r["page"] > 0, "page must be positive int"
        assert r["text"], "text is empty"
        assert isinstance(r["score"], float), "score must be float"

    top_result = results[0]
    assert "antihistamine" in top_result["section"].lower() or "antihistamine" in top_result["text"].lower(), (
        f"Top result section/text does not match expected query: {top_result['section']}"
    )


# ---------------------------------------------------------------------------
# P2-2: get_category_warnings case-insensitive lookup
# ---------------------------------------------------------------------------
def test_p2_2_get_category_warnings_sleep_aids():
    result = tools.get_category_warnings("nighttime sleep aids")
    assert isinstance(result, dict)
    assert result.get("found") is True, f"Category not found: {result}"
    assert "chunk_id" in result and result["chunk_id"], "chunk_id missing in category warning"
    assert "section" in result and "Nighttime Sleep Aids" in result["section"]
    assert "text" in result and len(result["text"]) > 0


# ---------------------------------------------------------------------------
# P2-3: get_category_warnings exact vs lowercase -> same chunk
# ---------------------------------------------------------------------------
def test_p2_3_get_category_warnings_case_insensitive():
    res_lower = tools.get_category_warnings("antihistamines")
    res_exact = tools.get_category_warnings("Antihistamines")

    assert res_lower.get("found") is True, f"Failed lowercase lookup: {res_lower}"
    assert res_exact.get("found") is True, f"Failed exact lookup: {res_exact}"
    assert res_lower["chunk_id"] == res_exact["chunk_id"], (
        f"Chunk IDs differ: {res_lower['chunk_id']} vs {res_exact['chunk_id']}"
    )


# ---------------------------------------------------------------------------
# P2-4: get_category_warnings non-existent category
# ---------------------------------------------------------------------------
def test_p2_4_get_category_warnings_not_found():
    result = tools.get_category_warnings("nonsense xyz drug category")
    assert isinstance(result, dict)
    assert result.get("found") is False
    assert "available_categories" in result
    assert isinstance(result["available_categories"], list)
    assert len(result["available_categories"]) > 0, "available_categories should be non-empty list"


# ---------------------------------------------------------------------------
# P2-5: openfda_label_lookup live warfarin lookup
# ---------------------------------------------------------------------------
def test_p2_5_openfda_label_lookup_warfarin():
    result = tools.openfda_label_lookup("warfarin")
    assert isinstance(result, dict)
    assert result.get("found") is True, f"FDA lookup failed: {result}"
    assert result.get("drug_name") == "warfarin"
    assert result.get("text"), "Label text is empty"
    assert result.get("set_id"), "spl_set_id is missing"
    assert "dailymed.nlm.nih.gov" in result.get("dailymed_url", ""), (
        f"Invalid DailyMed URL: {result.get('dailymed_url')}"
    )
    assert result.get("set_id") in result.get("dailymed_url", "")


# ---------------------------------------------------------------------------
# P2-6: openfda_label_lookup invalid section -> error, NO HTTP request
# ---------------------------------------------------------------------------
@patch("httpx.get")
def test_p2_6_openfda_label_lookup_invalid_section(mock_httpx_get):
    result = tools.openfda_label_lookup("warfarin", section="bad_section_name")
    assert isinstance(result, dict)
    assert "error" in result, f"Expected error dict, got: {result}"
    assert "Invalid section" in result["error"]
    mock_httpx_get.assert_not_called()


# ---------------------------------------------------------------------------
# P2-7: openfda_label_lookup non-existent drug
# ---------------------------------------------------------------------------
def test_p2_7_openfda_label_lookup_nonexistent_drug():
    result = tools.openfda_label_lookup("zzzznotadrug12345")
    assert isinstance(result, dict)
    assert result.get("found") is False, f"Expected found: False, got: {result}"
    assert "reason" in result or "error" in result


# ---------------------------------------------------------------------------
# P2-8: openfda_label_lookup long text truncation (>2000 chars)
# ---------------------------------------------------------------------------
@patch("httpx.get")
def test_p2_8_openfda_label_lookup_truncation(mock_httpx_get):
    long_text = "A" * 5000
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "drug_interactions": [long_text],
                "openfda": {"spl_set_id": ["test-set-id"]},
                "effective_time": "20250101",
            }
        ]
    }
    mock_httpx_get.return_value = mock_response

    result = tools.openfda_label_lookup("testdrug")
    assert result.get("found") is True
    assert result.get("truncated") is True, "Expected truncated: True"
    assert len(result["text"]) == 2000, f"Expected text length 2000, got {len(result['text'])}"


# ---------------------------------------------------------------------------
# P2-9: openfda_label_lookup api_key parameter handling & privacy
# ---------------------------------------------------------------------------
@patch("httpx.get")
def test_p2_9_openfda_label_lookup_api_key(mock_httpx_get, caplog):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "drug_interactions": ["Interaction text"],
                "openfda": {"spl_set_id": ["set123"]},
                "effective_time": "20250101",
            }
        ]
    }
    mock_httpx_get.return_value = mock_resp

    # Case A: empty OPENFDA_API_KEY
    with patch.object(config, "OPENFDA_API_KEY", ""):
        tools.openfda_label_lookup("aspirin")
        call_params = mock_httpx_get.call_args[1]["params"]
        assert "api_key" not in call_params, "api_key should not be sent when empty"

    # Case B: non-empty OPENFDA_API_KEY
    mock_key = "SECRET_KEY_12345"
    with patch.object(config, "OPENFDA_API_KEY", mock_key):
        tools.openfda_label_lookup("aspirin")
        call_params = mock_httpx_get.call_args[1]["params"]
        assert call_params.get("api_key") == mock_key, "api_key should be sent when configured"

    # Verify key is never logged
    for record in caplog.records:
        assert mock_key not in record.getMessage(), "API key found in log output!"


# ---------------------------------------------------------------------------
# P2-10: web_search live DuckDuckGo query
# ---------------------------------------------------------------------------
def test_p2_10_web_search():
    results = tools.web_search("ibuprofen warfarin interaction", max_results=4)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, "Expected at least 1 search result"

    required_keys = {"title", "url", "snippet"}
    for r in results:
        assert isinstance(r, dict)
        assert required_keys.issubset(r.keys()), f"Missing keys in result: {r.keys()}"
        assert r["title"], "title is empty"
        assert r["url"].startswith("http"), f"Invalid URL: {r['url']}"


# ---------------------------------------------------------------------------
# P2-11: Exception handling in all 4 tools
# ---------------------------------------------------------------------------
def test_p2_11_tool_exceptions_return_error():
    # 1. search_pdf exception
    with patch("google.genai.Client", side_effect=RuntimeError("Gemini error")):
        res1 = tools.search_pdf("test")
        assert isinstance(res1, dict)
        assert "error" in res1
        assert "Gemini error" in res1["error"]

    # 2. get_category_warnings exception
    with patch("chromadb.PersistentClient", side_effect=RuntimeError("Chroma error")):
        res2 = tools.get_category_warnings("Antihistamines")
        assert isinstance(res2, dict)
        assert "error" in res2
        assert "Chroma error" in res2["error"]

    # 3. openfda_label_lookup exception
    with patch("httpx.get", side_effect=httpx.RequestError("Network error")):
        res3 = tools.openfda_label_lookup("warfarin")
        assert isinstance(res3, dict)
        assert "error" in res3
        assert "Network error" in res3["error"]

    # 4. web_search exception
    with patch("tools.DDGS", side_effect=RuntimeError("DDGS error")):
        res4 = tools.web_search("test")
        assert isinstance(res4, dict)
        assert "error" in res4
        assert "DDGS error" in res4["error"]
