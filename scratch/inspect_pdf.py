"""
scratch/inspect_pdf.py — Inspect raw PDF layout to understand column ordering.
Prints full text of every page, plus block-level details.
Run with: .venv\Scripts\python.exe scratch/inspect_pdf.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows UTF-8 fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pymupdf  # noqa (suppresses fitz deprecation)

PDF_PATH = "./data/Drug-Interactions--What-You-Should-Know-low-res.pdf"

doc = pymupdf.open(PDF_PATH)
print(f"Total pages: {len(doc)}\n{'='*80}\n")

for page_num, page in enumerate(doc, start=1):
    print(f"\n{'='*80}")
    print(f"PAGE {page_num}")
    print(f"{'='*80}")

    # Raw text (may be column-scrambled)
    raw = page.get_text("text")
    print(f"--- RAW TEXT ---")
    print(raw)

    # Block-level info with bounding boxes
    print(f"\n--- BLOCKS (x0,y0,x1,y1,text) ---")
    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        text = b[4].replace("\n", " ").strip()
        if text:
            print(f"  [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}] {text[:100]!r}")

doc.close()
