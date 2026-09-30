"""
tests/test_phase3_verification.py — Unit tests for Phase 3 citation verification & agent loop control.

ALL tests use hand-built fake data / mocks. NO live LLM calls.

Tests:
P3-1:  Valid pdf citation (real chunk_id, real substring quote) passes
P3-2:  PDF citation with real chunk_id but invented quote -> dropped
P3-3:  PDF citation with non-existent chunk_id -> dropped
P3-4:  Quote differing only by whitespace/line breaks -> passes
P3-5:  Quote differing by one changed word -> dropped
P3-6:  FDA citation: valid quote passes; wrong set_id -> dropped; wrong quote -> dropped
P3-7:  Web citation with URL not in run's search results -> dropped
P3-8:  Dropped citation -> warning set, marker [n] removed/flagged, source_summary recomputed
P3-9:  Mocked model keeps requesting tools -> stops at MAX_AGENT_STEPS + safe message
P3-10: Tool returns {"error": ...} -> loop continues, no crash
P3-11: Malformed JSON from model -> retry once, then safe message
"""

import pytest
from unittest.mock import patch, MagicMock

import agent
import tools
from models import (
    AgentAnswer,
    AgentResult,
    FdaLabelCitation,
    PdfCitation,
    WebCitation,
    ToolCall,
)

# Sample document text for PDF testing
SAMPLE_CHUNK_ID = "PDF-p6-c2"
SAMPLE_CHUNK_TEXT = (
    "Antihistamines are used to treat allergy symptoms. "
    "Do not drink alcoholic beverages while taking this product. "
    "Be careful when driving a motor vehicle or operating machinery."
)


# ---------------------------------------------------------------------------
# P3-1: Valid PDF citation passes
# ---------------------------------------------------------------------------
def test_p3_1_valid_pdf_citation():
    answer = AgentAnswer(
        answer="Avoid alcohol when taking antihistamines [1].",
        citations=[
            PdfCitation(
                id=1,
                chunk_id=SAMPLE_CHUNK_ID,
                page=6,
                section="Antihistamines",
                quote="Do not drink alcoholic beverages while taking this product.",
            )
        ],
        source_summary="pdf_only",
    )
    run_context = {"pdf_chunks": {SAMPLE_CHUNK_ID: SAMPLE_CHUNK_TEXT}}

    verified, warning = agent.verify_citations(answer, run_context)

    assert len(verified.citations) == 1, "Valid citation was dropped"
    assert verified.citations[0].id == 1
    assert warning is None, f"Unexpected warning: {warning}"
    assert verified.source_summary == "pdf_only"


# ---------------------------------------------------------------------------
# P3-2: PDF citation with invented quote is dropped
# ---------------------------------------------------------------------------
def test_p3_2_pdf_invented_quote_dropped():
    answer = AgentAnswer(
        answer="Antihistamines cause severe permanent kidney damage [1].",
        citations=[
            PdfCitation(
                id=1,
                chunk_id=SAMPLE_CHUNK_ID,
                page=6,
                section="Antihistamines",
                quote="Antihistamines cause severe permanent kidney damage.",
            )
        ],
        source_summary="pdf_only",
    )
    run_context = {"pdf_chunks": {SAMPLE_CHUNK_ID: SAMPLE_CHUNK_TEXT}}

    verified, warning = agent.verify_citations(answer, run_context)

    assert len(verified.citations) == 0, "Invented quote citation was not dropped"
    assert warning is not None and "dropped" in warning
    assert verified.source_summary == "none"


# ---------------------------------------------------------------------------
# P3-3: PDF citation with non-existent chunk_id is dropped
# ---------------------------------------------------------------------------
def test_p3_3_pdf_nonexistent_chunk_id_dropped():
    answer = AgentAnswer(
        answer="Avoid alcohol [1].",
        citations=[
            PdfCitation(
                id=1,
                chunk_id="PDF-p99-c99",
                page=99,
                section="Unknown",
                quote="Do not drink alcoholic beverages while taking this product.",
            )
        ],
        source_summary="pdf_only",
    )
    run_context = {"pdf_chunks": {SAMPLE_CHUNK_ID: SAMPLE_CHUNK_TEXT}}

    verified, warning = agent.verify_citations(answer, run_context)

    assert len(verified.citations) == 0, "Non-existent chunk_id citation was not dropped"
    assert warning is not None and "not found" in warning
    assert verified.source_summary == "none"


# ---------------------------------------------------------------------------
# P3-4: Quote differing only by whitespace/line breaks passes
# ---------------------------------------------------------------------------
def test_p3_4_quote_whitespace_formatting():
    quote_with_newlines_spaces = (
        "Do  not   drink \n alco-\n holic  beverages \n while taking  this product."
    )
    answer = AgentAnswer(
        answer="Avoid alcohol [1].",
        citations=[
            PdfCitation(
                id=1,
                chunk_id=SAMPLE_CHUNK_ID,
                page=6,
                section="Antihistamines",
                quote=quote_with_newlines_spaces,
            )
        ],
        source_summary="pdf_only",
    )
    run_context = {"pdf_chunks": {SAMPLE_CHUNK_ID: SAMPLE_CHUNK_TEXT}}

    verified, warning = agent.verify_citations(answer, run_context)

    assert len(verified.citations) == 1, f"Whitespace-normalized quote was dropped: {warning}"
    assert warning is None


# ---------------------------------------------------------------------------
# P3-5: Quote with one changed word is dropped
# ---------------------------------------------------------------------------
def test_p3_5_quote_one_word_changed_dropped():
    quote_altered = "Do not drink sugary beverages while taking this product."
    answer = AgentAnswer(
        answer="Avoid sugary drinks [1].",
        citations=[
            PdfCitation(
                id=1,
                chunk_id=SAMPLE_CHUNK_ID,
                page=6,
                section="Antihistamines",
                quote=quote_altered,
            )
        ],
        source_summary="pdf_only",
    )
    run_context = {"pdf_chunks": {SAMPLE_CHUNK_ID: SAMPLE_CHUNK_TEXT}}

    verified, warning = agent.verify_citations(answer, run_context)

    assert len(verified.citations) == 0, "Altered word quote was not dropped"
    assert warning is not None


# ---------------------------------------------------------------------------
# P3-6: FDA label citation verification (valid vs wrong set_id vs wrong quote)
# ---------------------------------------------------------------------------
def test_p3_6_fda_label_verification():
    fda_run_context = {
        "fda_results": [
            {
                "drug_name": "warfarin",
                "section": "drug_interactions",
                "set_id": "fda-set-12345",
                "text": "Concomitant use of NSAIDs such as ibuprofen with warfarin increases risk of severe bleeding.",
                "dailymed_url": "https://dailymed.nlm.nih.gov/lookup.cfm?setid=fda-set-12345",
            }
        ]
    }

    # Case A: Valid FDA citation
    ans_valid = AgentAnswer(
        answer="NSAIDs increase bleeding risk with warfarin [1].",
        citations=[
            FdaLabelCitation(
                id=1,
                drug="warfarin",
                section="drug_interactions",
                set_id="fda-set-12345",
                effective_time="20240101",
                url="https://dailymed.nlm.nih.gov/lookup.cfm?setid=fda-set-12345",
                quote="ibuprofen with warfarin increases risk of severe bleeding.",
            )
        ],
        source_summary="fda_label_only",
    )
    ver_valid, warn_valid = agent.verify_citations(ans_valid, fda_run_context)
    assert len(ver_valid.citations) == 1, f"Valid FDA citation dropped: {warn_valid}"
    assert warn_valid is None

    # Case B: Mismatched set_id -> dropped
    ans_wrong_set_id = AgentAnswer(
        answer="NSAIDs increase bleeding risk [1].",
        citations=[
            FdaLabelCitation(
                id=1,
                drug="warfarin",
                section="drug_interactions",
                set_id="WRONG-SET-ID",
                effective_time="20240101",
                url="https://dailymed.nlm.nih.gov/lookup.cfm?setid=WRONG-SET-ID",
                quote="ibuprofen with warfarin increases risk of severe bleeding.",
            )
        ],
        source_summary="fda_label_only",
    )
    ver_wrong_set, warn_wrong_set = agent.verify_citations(ans_wrong_set_id, fda_run_context)
    assert len(ver_wrong_set.citations) == 0, "FDA citation with wrong set_id was not dropped"
    assert warn_wrong_set is not None

    # Case C: Invented quote -> dropped
    ans_wrong_quote = AgentAnswer(
        answer="NSAIDs are completely safe [1].",
        citations=[
            FdaLabelCitation(
                id=1,
                drug="warfarin",
                section="drug_interactions",
                set_id="fda-set-12345",
                effective_time="20240101",
                url="https://dailymed.nlm.nih.gov/lookup.cfm?setid=fda-set-12345",
                quote="NSAIDs are completely safe to take with warfarin without precautions.",
            )
        ],
        source_summary="fda_label_only",
    )
    ver_wrong_q, warn_wrong_q = agent.verify_citations(ans_wrong_quote, fda_run_context)
    assert len(ver_wrong_q.citations) == 0, "FDA citation with wrong quote was not dropped"
    assert warn_wrong_q is not None


# ---------------------------------------------------------------------------
# P3-7: Web citation verification
# ---------------------------------------------------------------------------
def test_p3_7_web_citation_verification():
    web_run_context = {
        "web_urls": {"https://www.fda.gov/drugs/safety-info", "https://medlineplus.gov/druginfo"}
    }

    # Case A: URL present -> passes
    ans_valid = AgentAnswer(
        answer="Check FDA safety information [1].",
        citations=[
            WebCitation(
                id=1,
                title="FDA Drug Safety Information",
                url="https://www.fda.gov/drugs/safety-info",
            )
        ],
        source_summary="with_web",
    )
    ver_valid, warn_valid = agent.verify_citations(ans_valid, web_run_context)
    assert len(ver_valid.citations) == 1, "Valid web citation dropped"

    # Case B: URL not in search results -> dropped
    ans_invalid = AgentAnswer(
        answer="Check random blog post [1].",
        citations=[
            WebCitation(
                id=1,
                title="Random Unverified Blog",
                url="https://randomblog.com/post123",
            )
        ],
        source_summary="with_web",
    )
    ver_invalid, warn_invalid = agent.verify_citations(ans_invalid, web_run_context)
    assert len(ver_invalid.citations) == 0, "Unverified web citation was not dropped"
    assert warn_invalid is not None


# ---------------------------------------------------------------------------
# P3-8: Marker removal, warning set, and source_summary recomputed
# ---------------------------------------------------------------------------
def test_p3_8_marker_removal_and_summary_recomputation():
    answer = AgentAnswer(
        answer="First fact from PDF [1]. Second fake fact [2].",
        citations=[
            PdfCitation(
                id=1,
                chunk_id=SAMPLE_CHUNK_ID,
                page=6,
                section="Antihistamines",
                quote="Do not drink alcoholic beverages while taking this product.",
            ),
            PdfCitation(
                id=2,
                chunk_id=SAMPLE_CHUNK_ID,
                page=6,
                section="Antihistamines",
                quote="This is an invented fake quote not in the document.",
            ),
        ],
        source_summary="pdf_only",
    )
    run_context = {"pdf_chunks": {SAMPLE_CHUNK_ID: SAMPLE_CHUNK_TEXT}}

    verified, warning = agent.verify_citations(answer, run_context)

    assert len(verified.citations) == 1
    assert verified.citations[0].id == 1
    assert "[2]" not in verified.answer, "Orphaned marker [2] was not removed from text"
    assert "[1]" in verified.answer, "Valid marker [1] was removed"
    assert warning is not None
    assert verified.source_summary == "pdf_only"


# ---------------------------------------------------------------------------
# P3-9: Infinite tool calls loop stops at MAX_AGENT_STEPS
# ---------------------------------------------------------------------------
@patch("agent.genai.Client")
def test_p3_9_infinite_tool_loop_stops_at_max_steps(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_func_call = MagicMock()
    mock_func_call.name = "search_pdf"
    mock_func_call.args = {"query": "antihistamine"}

    mock_resp = MagicMock()
    mock_resp.function_calls = [mock_func_call]
    mock_resp.candidates = [MagicMock()]

    mock_client.models.generate_content.return_value = mock_resp

    with patch("tools.search_pdf", return_value=[]):
        result = agent.run_agent("test query", max_steps=3)

    assert isinstance(result, AgentResult)
    assert "step limit" in result.answer.answer or "allowed step limit" in result.answer.answer
    assert result.verification_warning is not None
    assert "Exceeded MAX_AGENT_STEPS" in result.verification_warning
    assert len(result.tool_calls) == 3, f"Expected 3 tool calls, got {len(result.tool_calls)}"


# ---------------------------------------------------------------------------
# P3-10: Tool returning error dict does not crash agent loop
# ---------------------------------------------------------------------------
@patch("agent.genai.Client")
def test_p3_10_tool_error_dict_handled_gracefully(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_call = MagicMock()
    mock_call.name = "search_pdf"
    mock_call.args = {"query": "bad query"}
    resp_step1 = MagicMock()
    resp_step1.function_calls = [mock_call]
    resp_step1.candidates = [MagicMock()]

    resp_step2 = MagicMock()
    resp_step2.function_calls = []
    final_json = '{"answer": "I could not find information.", "citations": [], "source_summary": "none"}'
    resp_step2.text = final_json

    mock_client.models.generate_content.side_effect = [resp_step1, resp_step2]

    with patch("tools.search_pdf", return_value={"error": "Database locked"}):
        result = agent.run_agent("test query", max_steps=3)

    assert isinstance(result, AgentResult)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].error == "Database locked"
    assert result.answer.source_summary == "none"


# ---------------------------------------------------------------------------
# P3-11: Malformed JSON from model triggers retry then safe message
# ---------------------------------------------------------------------------
@patch("agent.genai.Client")
def test_p3_11_malformed_json_retry_and_safe_fallback(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    resp_bad1 = MagicMock()
    resp_bad1.function_calls = []
    resp_bad1.text = "NOT JSON TEXT AT ALL"

    resp_bad_retry = MagicMock()
    resp_bad_retry.text = "STILL NOT VALID JSON"

    mock_client.models.generate_content.side_effect = [resp_bad1, resp_bad_retry]

    result = agent.run_agent("test query", max_steps=2)

    assert isinstance(result, AgentResult)
    assert "safely" in result.answer.answer or "doctor or pharmacist" in result.answer.answer
    assert result.verification_warning is not None
    assert "JSON parsing error" in result.verification_warning
