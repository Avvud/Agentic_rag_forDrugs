"""
ingest.py — PDF ingestion and ChromaDB indexing for the drug-interaction RAG.

Strategy (based on inspected PDF layout):
- Pages 2-4:  Prose content. Chunk by paragraph/heading, max ~800 chars, 100-char overlap.
- Pages 5-9:  OTC category table (two-column: left=category name, right=interaction text).
              ONE chunk per named OTC category.
- Page 1:     Cover page — skip (no useful content).
- Page 10:    Credits/footer — skip.

Idempotent: drops and rebuilds the ChromaDB collection on every run.
"""

import os
import re
import sys
import logging
from pathlib import Path

import pymupdf  # type: ignore
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai.types import EmbedContentConfig

import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTC category definitions — derived from inspecting the PDF layout.
# Maps normalised category name -> section label used in metadata.
# ---------------------------------------------------------------------------
OTC_CATEGORIES: list[str] = [
    "Acid Reducers/H2 Blockers",
    "Antacids",
    "Antiemetics",
    "Antihistamines",
    "Antitussives",
    "Bronchodilators",
    "Laxatives",
    "Nasal Decongestants",
    "Nicotine Replacement",
    "Nighttime Sleep Aids",
    "Pain Relievers",
    "Stimulants",
    "Topical Acne Products",
]

# Page numbers (1-based) that contain category table rows.
CATEGORY_TABLE_PAGES: set[int] = {5, 6, 7, 8, 9}

# ---------------------------------------------------------------------------
# Helper: normalise text (collapse whitespace, fix hyphenated line breaks)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Collapse multiple whitespace and fix line-break hyphenation."""
    # Rejoin words broken across lines with a hyphen
    text = re.sub(r"-\s*\n\s*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Block-level extractor with sort by (x-column, y-position)
# Returns list of (x0, y0, text) tuples for a page, sorted reading order.
# ---------------------------------------------------------------------------

def _extract_blocks_sorted(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """
    Extract text blocks sorted by reading order.
    For a two-column layout, blocks with x0 < page_mid go to column 1,
    the rest to column 2.  Within each column, sort by y0.
    """
    page_width = page.rect.width
    mid = page_width / 2
    blocks = page.get_text("blocks")  # returns (x0,y0,x1,y1,text,block_no,block_type)

    left_col: list[tuple[float, float, str]] = []
    right_col: list[tuple[float, float, str]] = []

    for b in blocks:
        x0, y0, _x1, _y1, text, *_ = b
        text = text.strip()
        if not text:
            continue
        if x0 < mid:
            left_col.append((x0, y0, text))
        else:
            right_col.append((x0, y0, text))

    left_col.sort(key=lambda t: t[1])
    right_col.sort(key=lambda t: t[1])
    return left_col + right_col


# ---------------------------------------------------------------------------
# Category table parser
# ---------------------------------------------------------------------------

# Mapping from text patterns in the left column to canonical category names.
# Used to identify category boundaries.
_CATEGORY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Acid\s+Reducers|H2\s+Receptor", re.I), "Acid Reducers/H2 Blockers"),
    (re.compile(r"^Antacids", re.I), "Antacids"),
    (re.compile(r"^Antiemetics", re.I), "Antiemetics"),
    (re.compile(r"^Antihistamines", re.I), "Antihistamines"),
    (re.compile(r"^Antitussives|Cough\s+Medicine", re.I), "Antitussives"),
    (re.compile(r"^Bronchodilators", re.I), "Bronchodilators"),
    (re.compile(r"^Laxatives", re.I), "Laxatives"),
    (re.compile(r"Nasal\s+Decongestants?", re.I), "Nasal Decongestants"),
    (re.compile(r"Nicotine\s+Replacement", re.I), "Nicotine Replacement"),
    (re.compile(r"Nighttime\s+Sleep\s+Aids?", re.I), "Nighttime Sleep Aids"),
    (re.compile(r"^Pain\s+Relievers?", re.I), "Pain Relievers"),
    (re.compile(r"^Stimulants", re.I), "Stimulants"),
    (re.compile(r"Topical\s+Acne", re.I), "Topical Acne Products"),
]


def _match_category(text: str) -> str | None:
    """Return canonical category name if text starts a category, else None."""
    t = text.strip()
    for pattern, name in _CATEGORY_PATTERNS:
        if pattern.search(t):
            return name
    return None


def _parse_category_pages(doc: pymupdf.Document) -> dict[str, dict]:
    """
    Parse pages 5-9 and return one dict per OTC category with:
    {category_name: {text, page}}

    Strategy: iterate each page, collect left-column text blocks to detect
    category boundaries, collect right-column text blocks as interaction info.
    Then pair them by y-position proximity.
    """
    # We build a list of (page, y0, column, text_block) entries
    all_entries: list[tuple[int, float, str, str]] = []  # (page, y0, col, text)

    for page_num in sorted(CATEGORY_TABLE_PAGES):
        page = doc[page_num - 1]  # 0-indexed
        page_width = page.rect.width
        mid = page_width / 2

        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, _x1, _y1, text, *_ = b
            text = text.strip()
            if not text:
                continue
            col = "left" if x0 < mid else "right"
            all_entries.append((page_num, y0, col, text))

    # Sort by page, then y0
    all_entries.sort(key=lambda e: (e[0], e[1]))

    # Identify category start positions in the left column
    # Build list of (page, y0, category_name)
    cat_starts: list[tuple[int, float, str]] = []
    for page_num, y0, col, text in all_entries:
        if col == "left":
            cat = _match_category(text)
            if cat:
                cat_starts.append((page_num, y0, cat))

    # For each category, collect all right-column blocks between this category's
    # y0 and the next category's y0 (on the same page) or end of page.
    category_texts: dict[str, dict] = {}
    category_descriptions: dict[str, list[str]] = {}  # left-col description blocks

    # Also collect left-column description text (the parenthetical description)
    for i, (cat_page, cat_y0, cat_name) in enumerate(cat_starts):
        # Next category boundary
        if i + 1 < len(cat_starts):
            next_page, next_y0, _next_cat = cat_starts[i + 1]
        else:
            next_page, next_y0 = 99, 9999.0  # beyond all pages

        right_blocks: list[str] = []
        left_desc_blocks: list[str] = []

        for page_num, y0, col, text in all_entries:
            # Filter: within this category's range
            in_range = (
                (page_num == cat_page and y0 >= cat_y0) or
                (page_num > cat_page and page_num < next_page) or
                (page_num == next_page and y0 < next_y0)
            )
            if not in_range:
                continue

            if col == "right":
                # Skip "Drug Interaction Information" header
                if "Drug Interaction Information" in text:
                    continue
                right_blocks.append(text)
            else:
                # Left col: skip the category name header itself, keep description
                cat_match = _match_category(text)
                if cat_match == cat_name:
                    continue  # it's the name, skip
                if cat_match and cat_match != cat_name:
                    continue  # it's another category name, skip
                if "Category" in text and len(text) < 20:
                    continue  # skip "Category" header row
                left_desc_blocks.append(text)

        # Build combined text: category name + description + interaction info
        desc = _norm(" ".join(left_desc_blocks))
        interaction = _norm(" ".join(right_blocks))

        full_text = f"{cat_name}\n"
        if desc:
            full_text += f"{desc}\n"
        full_text += f"Drug Interaction Information:\n{interaction}"

        category_texts[cat_name] = {
            "text": full_text.strip(),
            "page": cat_page,
        }

    return category_texts


# ---------------------------------------------------------------------------
# Prose page parser → list of chunks (text, page, section)
# ---------------------------------------------------------------------------

_PROSE_HEADINGS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Drug\s+Interactions?\s*$", re.I), "Drug Interactions Overview"),
    (re.compile(r"Drug\s+Interactions?\s+and\s+Over", re.I), "OTC Drug Labels"),
    (re.compile(r"Learning\s+More\s+About", re.I), "Questions to Ask"),
    (re.compile(r"The following are examples", re.I), "OTC Drug Interaction Examples Intro"),
]

_MAX_CHUNK = 800
_OVERLAP = 100


def _detect_section(text: str, current_section: str) -> str:
    """Detect if a text block is a new section heading."""
    for pattern, name in _PROSE_HEADINGS:
        if pattern.search(text):
            return name
    return current_section


def _split_into_prose_chunks(
    text: str, page: int, section: str
) -> list[dict]:
    """
    Split a long prose text into overlapping chunks of max _MAX_CHUNK chars.
    Returns list of {text, page, section}.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + _MAX_CHUNK
        chunk = text[start:end].strip()
        if chunk and len(chunk) >= 40:
            chunks.append({"text": chunk, "page": page, "section": section})
        if end >= len(text):
            break
        start = end - _OVERLAP  # overlap
    return chunks


def _parse_prose_pages(doc: pymupdf.Document) -> list[dict]:
    """
    Parse prose pages (2, 3, 4, 5-intro) and return chunks with
    {text, page, section}.
    """
    chunks: list[dict] = []

    # Pages to process as prose (1-based). Page 5 has intro prose then table.
    prose_pages = [2, 3, 4, 5]

    for page_num in prose_pages:
        page = doc[page_num - 1]
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1])  # sort by y0

        current_section = {
            2: "Drug Interactions Overview",
            3: "OTC Drug Labels",
            4: "Questions to Ask",
            5: "OTC Drug Interaction Examples Intro",
        }.get(page_num, "General")

        # For page 5, only take prose before the category table (y0 < ~280)
        # The table header "Category" appears around y=279 on page 5
        page_height_cutoff = 280.0 if page_num == 5 else 9999.0

        accumulated: list[str] = []
        accumulated_len = 0

        for b in blocks:
            x0, y0, _x1, _y1, text, *_ = b
            text = text.strip()
            if not text:
                continue
            if y0 > page_height_cutoff and page_num == 5:
                break  # stop before category table

            # Detect section changes
            new_section = _detect_section(text, current_section)
            if new_section != current_section:
                # Flush accumulated text
                if accumulated:
                    full = _norm(" ".join(accumulated))
                    chunks.extend(_split_into_prose_chunks(full, page_num, current_section))
                    accumulated = []
                    accumulated_len = 0
                current_section = new_section

            # Skip pure heading blocks (short blocks that ARE the heading)
            is_heading = any(p.search(text) for p, _ in _PROSE_HEADINGS)
            if is_heading and len(text) < 60:
                continue

            # Skip the decorative drop-cap letter blocks (single letter blocks)
            if len(text) <= 2 and text.isupper():
                continue

            accumulated.append(text)
            accumulated_len += len(text)

            # Flush if accumulated is large enough
            if accumulated_len >= _MAX_CHUNK:
                full = _norm(" ".join(accumulated))
                chunks.extend(_split_into_prose_chunks(full, page_num, current_section))
                # Keep overlap
                overlap_text = full[-_OVERLAP:] if len(full) > _OVERLAP else full
                accumulated = [overlap_text]
                accumulated_len = len(overlap_text)

        # Flush remaining
        if accumulated:
            full = _norm(" ".join(accumulated))
            if full and len(full) >= 40:
                chunks.extend(_split_into_prose_chunks(full, page_num, current_section))

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest(pdf_path: str | None = None, chroma_path: str | None = None) -> int:
    """
    Parse PDF, chunk, embed and store in ChromaDB.
    Returns total number of chunks ingested.
    """
    pdf_path = pdf_path or config.PDF_PATH
    chroma_path = chroma_path or config.CHROMA_PATH

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_filename = Path(pdf_path).name

    # ---- Open PDF ----
    doc = pymupdf.open(pdf_path)
    page_count = len(doc)
    log.info("Opened PDF '%s' with %d pages", pdf_filename, page_count)

    # Warn on empty pages
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            log.warning("Page %d is empty (no text extracted)", i)

    # ---- Parse content ----
    all_chunks: list[dict] = []  # {text, page, section}

    # 1. Prose sections (pages 2-5 intro)
    prose_chunks = _parse_prose_pages(doc)
    all_chunks.extend(prose_chunks)
    log.info("Prose chunks: %d", len(prose_chunks))

    # 2. OTC category table (pages 5-9)
    category_data = _parse_category_pages(doc)
    for cat_name, data in category_data.items():
        all_chunks.append({
            "text": data["text"],
            "page": data["page"],
            "section": cat_name,
        })
    log.info("Category chunks: %d", len(category_data))

    doc.close()

    # ---- Assign chunk IDs and build metadata ----
    # Group by page, then assign sequential chunk number per page
    page_counters: dict[int, int] = {}
    final_chunks: list[dict] = []

    # Sort all chunks by page then section for deterministic ordering
    all_chunks.sort(key=lambda c: (c["page"], c["section"]))

    for chunk in all_chunks:
        page = chunk["page"]
        page_counters[page] = page_counters.get(page, 0) + 1
        n = page_counters[page]
        chunk_id = f"PDF-p{page}-c{n}"

        text = chunk["text"].strip()
        if len(text) < 40:
            log.warning("Skipping short chunk %s (len=%d)", chunk_id, len(text))
            continue
        if len(text) > 2000:
            log.warning("Chunk %s is long (len=%d), truncating to 2000", chunk_id, len(text))
            text = text[:2000]

        final_chunks.append({
            "chunk_id": chunk_id,
            "page": page,
            "section": chunk["section"],
            "source": pdf_filename,
            "text": text,
        })

    log.info("Total chunks after filtering: %d", len(final_chunks))

    # ---- Embed with Gemini ----
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    def _embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Embed a list of texts, returns list of float vectors."""
        embeddings = []
        # ChromaDB/Gemini: embed one at a time to avoid quota issues
        for text in texts:
            resp = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=text,
                config=EmbedContentConfig(task_type=task_type),
            )
            embeddings.append(resp.embeddings[0].values)
        return embeddings

    texts = [c["text"] for c in final_chunks]
    log.info("Embedding %d chunks with model '%s'...", len(texts), config.EMBEDDING_MODEL)
    embeddings = _embed_batch(texts)
    log.info("Embedding complete.")

    # ---- Store in ChromaDB (idempotent: drop + rebuild) ----
    os.makedirs(chroma_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    collection_name = "drug_interactions"

    # Drop existing collection if present
    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        chroma_client.delete_collection(collection_name)
        log.info("Dropped existing collection '%s'", collection_name)

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Add all chunks
    collection.add(
        ids=[c["chunk_id"] for c in final_chunks],
        embeddings=[list(e) for e in embeddings],
        documents=[c["text"] for c in final_chunks],
        metadatas=[
            {
                "chunk_id": c["chunk_id"],
                "page": c["page"],
                "section": c["section"],
                "source": c["source"],
            }
            for c in final_chunks
        ],
    )

    log.info(
        "Stored %d chunks in ChromaDB collection '%s' at '%s'",
        len(final_chunks), collection_name, chroma_path,
    )

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"INGESTION SUMMARY")
    print(f"{'='*70}")
    print(f"  PDF: {pdf_filename}")
    print(f"  Pages parsed: {page_count}")
    print(f"  Chunks created: {len(final_chunks)}")
    print(f"  Embedding model: {config.EMBEDDING_MODEL}")
    print(f"  ChromaDB path: {chroma_path}")
    print(f"{'='*70}\n")

    # ---- Full chunk list ----
    # Sanitize for Windows console (cp1252 can't handle some Unicode chars from PDF)
    def _safe(text: str) -> str:
        return text.encode("ascii", errors="replace").decode("ascii")

    print("FULL CHUNK LIST:")
    print(f"{'chunk_id':<20} | {'page':>4} | {'section':<35} | first 150 chars")
    print("-" * 120)
    for c in final_chunks:
        preview = _safe(c["text"][:150].replace("\n", " "))
        print(f"{c['chunk_id']:<20} | {c['page']:>4} | {c['section']:<35} | {preview}")

    # ---- Print special chunks ----
    print(f"\n{'='*70}")
    print("FULL TEXT: Antihistamines chunk")
    print(f"{'='*70}")
    anti_chunks = [c for c in final_chunks if c["section"].lower() == "antihistamines"]
    if anti_chunks:
        print(_safe(anti_chunks[0]["text"]))
    else:
        print("NOT FOUND")

    print(f"\n{'='*70}")
    print("FULL TEXT: Acid Reducers/H2 Blockers (cimetidine) chunk")
    print(f"{'='*70}")
    acid_chunks = [c for c in final_chunks if "cimetidine" in c["text"].lower()]
    if acid_chunks:
        print(_safe(acid_chunks[0]["text"]))
    else:
        print("NOT FOUND")

    return len(final_chunks)


if __name__ == "__main__":
    total = ingest()
    sys.exit(0)
