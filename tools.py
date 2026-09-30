"""
tools.py — The 4 agent tools for medical drug-interaction Q&A:
1. search_pdf(query, top_k)
2. get_category_warnings(category)
3. openfda_label_lookup(drug_name, section)
4. web_search(query, max_results)

Also exports:
- TOOL_DECLARATIONS: List of functions for Gemini function calling
- TOOL_DISPATCH: Dict mapping function name -> function object
"""

import logging
import httpx
import chromadb
from google import genai
from google.genai.types import EmbedContentConfig

import config

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore

log = logging.getLogger(__name__)

# Valid openFDA label sections
ALLOWED_OPENFDA_SECTIONS: set[str] = {
    "drug_interactions",
    "warnings",
    "boxed_warning",
    "contraindications",
}


def search_pdf(query: str, top_k: int = 4) -> list[dict] | dict:
    """Search the drug-interaction PDF via ChromaDB semantic search.

    PRIMARY tool — use this first for any drug interaction question.

    Args:
        query: Search query string.
        top_k: Number of top matching chunks to return (default 4).

    Returns:
        List of dicts with {chunk_id, page, section, text, score}
        or {"error": "<message>"} on failure.
    """
    try:
        # Embed query with Gemini (task_type="RETRIEVAL_QUERY")
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        resp = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=query,
            config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_embedding = resp.embeddings[0].values

        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = chroma_client.get_collection("drug_interactions")

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
        )

        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i in range(len(ids)):
                meta = metadatas[i] if i < len(metadatas) else {}
                dist = distances[i] if i < len(distances) else 0.0
                doc = documents[i] if i < len(documents) else ""

                formatted_results.append({
                    "chunk_id": meta.get("chunk_id", ids[i]),
                    "page": meta.get("page", 0),
                    "section": meta.get("section", ""),
                    "text": doc,
                    "score": float(dist),
                })

        return formatted_results

    except Exception as e:
        log.error("Error in search_pdf: %s", e)
        return {"error": str(e)}


def get_category_warnings(category: str) -> dict:
    """Get the full interaction warning text for a specific OTC drug category.

    Use when user names a specific category (e.g., 'antihistamines', 'pain relievers').

    Args:
        category: OTC category name (e.g., 'Antihistamines', 'Pain Relievers').

    Returns:
        Dict with {found: True, chunk_id, page, section, text} if found,
        or {found: False, available_categories: [...]} if not found,
        or {"error": "<message>"} on failure.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = chroma_client.get_collection("drug_interactions")

        # Retrieve all items from collection to inspect section metadata
        all_data = collection.get(include=["metadatas", "documents"])
        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        documents = all_data.get("documents", [])

        query_cat = category.strip().lower()

        # Collect all available category section names
        available_sections: list[str] = sorted(
            list({meta.get("section", "") for meta in metadatas if meta and meta.get("section")})
        )

        # 1. Exact match (case-insensitive)
        matched_idx = None
        for i, meta in enumerate(metadatas):
            sec = meta.get("section", "").strip()
            if sec.lower() == query_cat:
                matched_idx = i
                break

        # 2. Fuzzy / substring match
        if matched_idx is None:
            for i, meta in enumerate(metadatas):
                sec = meta.get("section", "").strip()
                if query_cat in sec.lower() or sec.lower() in query_cat:
                    matched_idx = i
                    break

        if matched_idx is not None:
            meta = metadatas[matched_idx]
            doc = documents[matched_idx]
            return {
                "found": True,
                "chunk_id": meta.get("chunk_id", ids[matched_idx]),
                "page": meta.get("page", 0),
                "section": meta.get("section", ""),
                "text": doc,
            }
        else:
            return {
                "found": False,
                "available_categories": available_sections,
            }

    except Exception as e:
        log.error("Error in get_category_warnings: %s", e)
        return {"error": str(e)}


def openfda_label_lookup(drug_name: str, section: str = "drug_interactions") -> dict:
    """Look up official FDA drug label information.

    Use AFTER search_pdf if the PDF doesn't cover the drug.

    Args:
        drug_name: Generic or brand name of the drug (e.g., 'warfarin', 'ibuprofen').
        section: Label section to retrieve. Allowed: 'drug_interactions', 'warnings',
                 'boxed_warning', 'contraindications'. Default 'drug_interactions'.

    Returns:
        Dict with {found: True, drug_name, section, text, truncated, set_id, effective_time, dailymed_url}
        or {found: False, reason: "..."} or {"error": "..."}.
    """
    # Validate section FIRST before any HTTP request
    if section not in ALLOWED_OPENFDA_SECTIONS:
        allowed_str = ", ".join(sorted(ALLOWED_OPENFDA_SECTIONS))
        return {
            "error": f"Invalid section '{section}'. Allowed sections: {allowed_str}"
        }

    try:
        base_url = "https://api.fda.gov/drug/label.json"

        params = {
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": 1,
        }
        if config.OPENFDA_API_KEY:
            params["api_key"] = config.OPENFDA_API_KEY

        headers = {"User-Agent": "AgenticRAG-MedicalQA/1.0"}

        # Search generic_name first
        response = httpx.get(base_url, params=params, headers=headers, timeout=10.0)

        # Fallback to brand_name if 404 or no results
        if response.status_code == 404 or not response.json().get("results"):
            params["search"] = f'openfda.brand_name:"{drug_name}"'
            response = httpx.get(base_url, params=params, headers=headers, timeout=10.0)

        if response.status_code == 404:
            return {
                "found": False,
                "reason": f"No FDA label found for drug '{drug_name}'",
            }

        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        if not results:
            return {
                "found": False,
                "reason": f"No FDA label found for drug '{drug_name}'",
            }

        first_result = results[0]

        if section not in first_result:
            return {
                "found": False,
                "reason": f"Section '{section}' not found in FDA label for '{drug_name}'",
            }

        raw_section = first_result[section]
        if isinstance(raw_section, list):
            sec_text = " ".join(raw_section)
        else:
            sec_text = str(raw_section)

        sec_text = sec_text.strip()

        # Truncate if >2000 chars
        truncated = False
        if len(sec_text) > 2000:
            sec_text = sec_text[:2000]
            truncated = True

        openfda_meta = first_result.get("openfda", {})
        spl_set_id = openfda_meta.get("spl_set_id", [""])[0] if openfda_meta.get("spl_set_id") else first_result.get("set_id", "")
        effective_time = first_result.get("effective_time", "")

        dailymed_url = (
            f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={spl_set_id}"
            if spl_set_id
            else ""
        )

        return {
            "found": True,
            "drug_name": drug_name,
            "section": section,
            "text": sec_text,
            "truncated": truncated,
            "set_id": spl_set_id,
            "effective_time": effective_time,
            "dailymed_url": dailymed_url,
        }

    except httpx.HTTPStatusError as e:
        log.error("HTTP error during openFDA lookup: %s", e)
        return {"error": f"openFDA HTTP error: {e.response.status_code}"}
    except Exception as e:
        log.error("Error in openfda_label_lookup: %s", e)
        return {"error": str(e)}


def web_search(query: str, max_results: int = 4) -> list[dict] | dict:
    """Search the web for drug interaction information. LAST RESORT.

    Use only after search_pdf and openfda_label_lookup fail.
    Prefer results from official medical sites (fda.gov, nih.gov, medlineplus.gov, nhs.uk, who.int, cdc.gov).

    Args:
        query: Web search query string.
        max_results: Maximum number of search results to return (default 4).

    Returns:
        List of dicts with {title, url, snippet} or {"error": "<message>"} on failure.
    """
    try:
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=max_results))

        formatted = []
        for r in raw_results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
        return formatted

    except Exception as e:
        log.error("Error in web_search: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Exported declarations and dispatch dict
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS = [search_pdf, get_category_warnings, openfda_label_lookup, web_search]

TOOL_DISPATCH = {
    "search_pdf": search_pdf,
    "get_category_warnings": get_category_warnings,
    "openfda_label_lookup": openfda_label_lookup,
    "web_search": web_search,
}
