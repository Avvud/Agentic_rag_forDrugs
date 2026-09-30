"""
agent.py — Agent function calling loop and citation verification logic.

Main entry points:
- verify_citations(answer, run_context): Verifies citation quotes & sources strictly.
- run_agent(query, max_steps): Executes agent function calling loop.
"""

import json
import logging
import re
import time
from typing import Any

import chromadb
from google import genai
from google.genai import types

import config
import prompts
import tools
from models import (
    AgentAnswer,
    AgentResult,
    Citation,
    FdaLabelCitation,
    PdfCitation,
    ToolCall,
    WebCitation,
)

log = logging.getLogger(__name__)


def _norm_text(text: str) -> str:
    """Normalize text by removing hyphenated line breaks, collapsing whitespace, and lowercasing."""
    if not text:
        return ""
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _clean_json_str(text: str) -> str:
    """Strip markdown code fences and extraneous whitespace from JSON string."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _get_pdf_chunk_text(chunk_id: str, run_context: dict) -> str | None:
    """Get full document text for chunk_id from run_context or ChromaDB."""
    # 1. Check run_context
    if "pdf_chunks" in run_context and chunk_id in run_context["pdf_chunks"]:
        return run_context["pdf_chunks"][chunk_id]

    # 2. Check ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = chroma_client.get_collection("drug_interactions")
        res = collection.get(ids=[chunk_id], include=["documents"])
        if res and res.get("documents") and len(res["documents"]) > 0:
            return res["documents"][0]
    except Exception as e:
        log.debug("Chroma lookup for %s failed: %s", chunk_id, e)

    return None


def recompute_source_summary(citations: list[Citation]) -> str:
    """Recompute source_summary strictly based on surviving citations."""
    types_present = {c.type for c in citations}
    if "web" in types_present:
        return "with_web"
    has_pdf = "pdf" in types_present
    has_fda = "fda_label" in types_present

    if has_pdf and has_fda:
        return "pdf_and_fda"
    elif has_pdf:
        return "pdf_only"
    elif has_fda:
        return "fda_label_only"
    else:
        return "none"


def verify_citations(
    answer: AgentAnswer, run_context: dict
) -> tuple[AgentAnswer, str | None]:
    """Verify citations against run_context tool outputs and ChromaDB.

    Rules:
    - PDF: chunk_id must exist, quote must be non-empty substring of chunk text (after _norm_text).
    - FDA: set_id must match an FDA result in run_context, quote must be non-empty substring of FDA text.
    - Web: URL must be present in run_context["web_urls"].

    Drops invalid citations, removes orphaned [n] markers from text, recomputes source_summary.
    """
    surviving_citations: list[Citation] = []
    dropped_info: list[str] = []

    pdf_chunks_context = run_context.get("pdf_chunks", {})
    fda_results_context = run_context.get("fda_results", [])
    web_urls_context = set(run_context.get("web_urls", []))

    for cit in answer.citations:
        # Quote truncation (max 200 chars)
        if hasattr(cit, "quote") and cit.quote and len(cit.quote) > 200:
            cit.quote = cit.quote[:200]

        if cit.type == "pdf":
            chunk_text = _get_pdf_chunk_text(cit.chunk_id, run_context)
            if chunk_text is None:
                dropped_info.append(f"Citation [{cit.id}] dropped: chunk_id '{cit.chunk_id}' not found.")
                continue

            norm_quote = _norm_text(cit.quote)
            norm_doc = _norm_text(chunk_text)

            if not norm_quote:
                dropped_info.append(f"Citation [{cit.id}] dropped: empty quote.")
                continue

            if norm_quote not in norm_doc:
                dropped_info.append(f"Citation [{cit.id}] dropped: quote not found in chunk '{cit.chunk_id}'.")
                continue

            surviving_citations.append(cit)

        elif cit.type == "fda_label":
            norm_quote = _norm_text(cit.quote)
            if not norm_quote:
                dropped_info.append(f"Citation [{cit.id}] dropped: empty quote.")
                continue

            matched = False
            for fda_res in fda_results_context:
                res_set_id = fda_res.get("set_id", "")
                if cit.set_id and res_set_id and cit.set_id != res_set_id:
                    continue  # set_id mismatch

                res_text = fda_res.get("text", "")
                if norm_quote in _norm_text(res_text):
                    matched = True
                    break

            if not matched:
                dropped_info.append(f"Citation [{cit.id}] dropped: quote/set_id not verified against FDA results.")
                continue

            surviving_citations.append(cit)

        elif cit.type == "web":
            if cit.url not in web_urls_context:
                dropped_info.append(f"Citation [{cit.id}] dropped: URL '{cit.url}' not in web search results.")
                continue

            surviving_citations.append(cit)

        else:
            dropped_info.append(f"Citation [{cit.id}] dropped: unknown citation type '{cit.type}'.")

    # Remove orphaned markers [n] for dropped citations
    surviving_ids = {c.id for c in surviving_citations}
    all_original_ids = {c.id for c in answer.citations}
    dropped_ids = all_original_ids - surviving_ids

    answer_text = answer.answer
    for d_id in dropped_ids:
        answer_text = re.sub(rf"\[{d_id}\]", "", answer_text)

    # Clean up excess spaces created by marker deletion
    answer_text = re.sub(r"\s+", " ", answer_text).strip()

    # Recompute source summary
    new_source_summary = recompute_source_summary(surviving_citations)

    # Build warning string if any citations dropped
    warning_msg = "; ".join(dropped_info) if dropped_info else None

    verified_answer = AgentAnswer(
        answer=answer_text,
        citations=surviving_citations,
        source_summary=new_source_summary,
    )

    return verified_answer, warning_msg


def _safe_fallback_result(
    message: str, tool_calls: list[ToolCall], warning: str | None = None
) -> AgentResult:
    """Helper to return a safe fallback AgentResult on failure or step limit."""
    return AgentResult(
        answer=AgentAnswer(
            answer=message,
            citations=[],
            source_summary="none",
        ),
        tool_calls=tool_calls,
        verification_warning=warning,
    )


def run_agent(query: str, max_steps: int | None = None) -> AgentResult:
    """Run the Agent function calling loop.

    Args:
        query: User input question.
        max_steps: Maximum number of tool iterations allowed (default config.MAX_AGENT_STEPS).

    Returns:
        AgentResult with answer, tool_calls, and verification_warning.
    """
    max_steps = max_steps or config.MAX_AGENT_STEPS
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    run_context: dict[str, Any] = {
        "pdf_chunks": {},
        "fda_results": [],
        "web_urls": set(),
    }
    recorded_tool_calls: list[ToolCall] = []

    # Configure Gemini content generation
    config_gen = types.GenerateContentConfig(
        system_instruction=prompts.SYSTEM_PROMPT,
        tools=tools.TOOL_DECLARATIONS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )

    messages = [types.Content(role="user", parts=[types.Part.from_text(text=query)])]

    step_count = 0

    while step_count < max_steps:
        step_count += 1
        log.info("Agent step %d/%d", step_count, max_steps)

        # Call Gemini with 429 auto-retry
        response = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=messages,
                    config=config_gen,
                )
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_retries - 1:
                    log.warning("Rate limit 429 hit on step %d. Retrying in 12s (attempt %d/%d)...", step_count, attempt + 1, max_retries)
                    time.sleep(12)
                    continue
                log.error("Gemini API call failed on step %d: %s", step_count, e)
                return _safe_fallback_result(
                    "I encountered an error communicating with the model. Please consult a pharmacist or doctor.",
                    recorded_tool_calls,
                    f"Model error: {e}",
                )

        # Check if model invoked tool(s)
        if response.function_calls:
            # Append model's response to message history
            if response.candidates and response.candidates[0].content:
                messages.append(response.candidates[0].content)

            for call in response.function_calls:
                tool_name = call.name
                tool_args = dict(call.args) if call.args else {}
                log.info("Executing tool '%s' with args: %s", tool_name, tool_args)

                func = getattr(tools, tool_name, None)
                result_count = None
                error_msg = None

                if func:
                    try:
                        tool_result = func(**tool_args)

                        if isinstance(tool_result, dict) and "error" in tool_result:
                            error_msg = tool_result["error"]
                        elif isinstance(tool_result, list):
                            result_count = len(tool_result)
                        elif isinstance(tool_result, dict):
                            result_count = 1 if tool_result.get("found", True) else 0

                    except Exception as te:
                        tool_result = {"error": str(te)}
                        error_msg = str(te)
                else:
                    tool_result = {"error": f"Unknown tool '{tool_name}'"}
                    error_msg = f"Unknown tool '{tool_name}'"

                # Record tool call
                recorded_tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result_count=result_count,
                        error=error_msg,
                    )
                )

                # Store raw output in run_context
                if tool_name in ("search_pdf", "get_category_warnings"):
                    if isinstance(tool_result, list):
                        for item in tool_result:
                            if isinstance(item, dict) and "chunk_id" in item:
                                run_context["pdf_chunks"][item["chunk_id"]] = item.get("text", "")
                    elif isinstance(tool_result, dict) and tool_result.get("found"):
                        run_context["pdf_chunks"][tool_result["chunk_id"]] = tool_result.get("text", "")

                elif tool_name == "openfda_label_lookup":
                    if isinstance(tool_result, dict) and tool_result.get("found"):
                        run_context["fda_results"].append(tool_result)

                elif tool_name == "web_search":
                    if isinstance(tool_result, list):
                        for item in tool_result:
                            if isinstance(item, dict) and "url" in item:
                                run_context["web_urls"].add(item["url"])

                # Send function response back to Gemini
                messages.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"result": tool_result},
                            )
                        ],
                    )
                )
        else:
            # Final text output from model
            final_text = response.text or ""
            log.info("Agent received final text response.")

            # Parse as AgentAnswer
            parsed_answer = None
            clean_json = _clean_json_str(final_text)
            try:
                data = json.loads(clean_json)
                parsed_answer = AgentAnswer.model_validate(data)
            except Exception as pe:
                log.warning("Failed to parse response as JSON: %s. Attempting formatting retry...", pe)
                retry_prompt = f"Extract and return ONLY a valid JSON object matching this response:\n{final_text}\nDo not use markdown formatting or code blocks."
                try:
                    retry_resp = client.models.generate_content(
                        model=config.GEMINI_MODEL,
                        contents=retry_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                        ),
                    )
                    retry_clean = _clean_json_str(retry_resp.text or "")
                    data = json.loads(retry_clean)
                    parsed_answer = AgentAnswer.model_validate(data)
                except Exception as rpe:
                    log.error("Retry parsing failed: %s", rpe)
                    return _safe_fallback_result(
                        "I could not complete the response safely. Please consult a doctor or pharmacist.",
                        recorded_tool_calls,
                        f"JSON parsing error: {rpe}",
                    )

            # Verify citations
            verified_answer, warning = verify_citations(parsed_answer, run_context)
            return AgentResult(
                answer=verified_answer,
                tool_calls=recorded_tool_calls,
                verification_warning=warning,
            )

    # Exceeded max steps
    log.warning("Agent exceeded maximum step limit (%d)", max_steps)
    return _safe_fallback_result(
        "I could not complete the request within the allowed step limit. Please consult a doctor or pharmacist.",
        recorded_tool_calls,
        f"Exceeded MAX_AGENT_STEPS ({max_steps})",
    )
