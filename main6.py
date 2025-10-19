import os
import json
import fitz
from collections import Counter, defaultdict
import numpy as np
import spacy
from statistics import median
from sentence_transformers import SentenceTransformer, util

INPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"
OUTPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"

# Load NLP models
try: 
    nlp = spacy.load("xx_ent_wiki_sm")
except Exception:
    nlp = spacy.load("en_core_web_sm")
sb_model = SentenceTransformer('all-MiniLM-L6-v2')

# Utility
def normalize(text):
    return " ".join(text.strip().split())

def is_body_line(y0, page_height):
    # Exclude header/footer region
    return 0.06 * page_height < y0 < 0.93 * page_height

def get_line_features(doc):
    """Extract and yield all (page, line) blocks with font/pos/meta for full-PDF analysis."""
    all_lines = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        page_width = page.rect.width
        font_sizes = []
        # Find median for heuristics
        for block in blocks:
            if block["type"] != 0: continue
            for line in block["lines"]:
                font_sizes.extend([span["size"] for span in line["spans"]])
        median_font_size = median(font_sizes) if font_sizes else 12
        for block in blocks:
            if block["type"] != 0: continue
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"])
                if not text.strip(): continue
                spans = line["spans"]
                is_bold = any("Bold" in s['font'] or s.get('flags', 0)&2==2 for s in spans)
                font_size = max(s["size"] for s in spans)
                x = min(s["bbox"][0] for s in spans)
                y = min(s["bbox"][1] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)
                bbox = [
                    min(s["bbox"][0] for s in spans), y,
                    max(s["bbox"][2] for s in spans), y1
                ]
                all_lines.append({
                    "text": text, "font_size": font_size, "is_bold": is_bold,
                    "x0": x, "y0": y, "y1": y1, "bbox": bbox,
                    "page": page_num, "page_height": page_height,
                    "page_width": page_width, "median_font_size": median_font_size
                })
    return all_lines

def get_common_lines(all_lines, num_pages, freq_thresh=0.7, y_tol=22):
    """Detect lines repeated on most pages at similar Y; treat as header/footer/logo."""
    norm_y = defaultdict(list)
    freq = Counter()
    for l in all_lines:
        t = normalize(l["text"])
        freq[t] += 1
        norm_y[t].append(round(l["y0"]))

    min_occurs = max(2, int(num_pages * freq_thresh))
    common_lines = set()
    for t, count in freq.items():
        if count >= min_occurs:
            # Repeated line.
            # Ensure it truly repeats at close to same Y (to avoid headings with same text)
            y_spread = max(norm_y[t]) - min(norm_y[t]) if norm_y[t] else 0
            if y_spread < y_tol:
                common_lines.add(t)
    return common_lines

def group_multilines(lines, line_exclude):
    """Group adjacent lines on same page within vertical proximity, same font/bold, not excluded, as one block."""
    lines = [l for l in lines if normalize(l["text"]) not in line_exclude]
    if not lines: return []
    # Sort by page then Y
    lines = sorted(lines, key=lambda l: (l["page"], l["y0"]))
    groups = []
    curr = [lines[0]]
    for prev, nxt in zip(lines, lines[1:]):
        if nxt["page"] != prev["page"]:
            groups.append(curr)
            curr = [nxt]
            continue
        same_font = abs(nxt["font_size"] - prev["font_size"]) < 1.1
        same_bold = nxt["is_bold"] == prev["is_bold"]
        y_gap = nxt["y0"] - prev["y1"]
        if same_font and same_bold and y_gap < 1.3 * prev["font_size"]:
            curr.append(nxt)
        else:
            groups.append(curr)
            curr = [nxt]
    if curr: groups.append(curr)
    blocks = []
    for group in groups:
        block = dict(group[0])
        block["text"] = " ".join([l["text"] for l in group])
        block["y1"] = group[-1]["y1"]
        block["lines_in_block"] = len(group)
        blocks.append(block)
    return blocks

def detect_toc_pages(blocks):
    # Returns set of page numbers that are ToC
    toc_pages = set()
    for b in blocks:
        if "table of contents" in b["text"].lower():
            toc_pages.add(b["page"])
    return toc_pages

def detect_ack_pages(blocks):
    ack_pages = set()
    for b in blocks:
        if "acknowledgements" in b["text"].lower() or "copyright notice" in b["text"].lower():
            ack_pages.add(b["page"])
    return ack_pages

def is_table_row(block, lines_in_page, idx):
    """Rudimentary: A block in a set of adjacent blocks with similar structure, containing short cells, typical of tables."""
    if not block["text"]: return False
    # If previous/following blocks are short and similar Y, likely a table
    start, end = max(0, idx-2), min(len(lines_in_page), idx+3)
    siblings = lines_in_page[start:end]
    short = lambda t: len(t.strip())<15
    count_short = sum(short(l["text"]) for l in siblings)
    # Don't treat as table row if it's a typical heading
    pattern = block["text"].strip()
    is_numbered = pattern[:2].replace(".", "").isdigit()
    return (count_short > 2) and (not is_numbered)

def is_meta_page(page_num, toc_pages, ack_pages):
    # First 1-2 pages, or toc/ack/copyright pages
    return (page_num<=1) or (page_num in toc_pages) or (page_num in ack_pages)

def get_title_candidate(blocks, line_exclude, meta_pages):
    # Find the first large, bold, multi-line block on first page(s) not repeated, not a meta/cover line
    large_blocks = [b for b in blocks if b["page"] <= 1 and b["font_size"] >= b["median_font_size"] + 2 and
                    normalize(b["text"]) not in line_exclude and b["page"] not in meta_pages and
                    len(b["text"].strip()) >= 5]
    if not large_blocks:
        return "No title found"
    # Favor block with most lines or largest font
    best = sorted(large_blocks, key=lambda b: (b["lines_in_block"], b["font_size"]), reverse=True)[0]
    return best["text"]

def find_numbered_headings(blocks):
    """
    Returns (H1, H2) candidates from blocks where text starts with numbering (e.g. 1., 2.3)
    """
    h1s, h2s = [], []
    for b in blocks:
        txt = b["text"].strip()
        if txt[:2].replace(".", "").isdigit() and txt[1] in ". ":
            # 1. Heading, 2. Heading
            h1s.append(b)
        elif txt[:4].replace(".","").replace(" ","").isdigit() and "." in txt and txt[1]=='.':
            # 1.2 Subheading
            h2s.append(b)
    return h1s, h2s

def detect_headings_blocks(blocks, page2blocks, toc_pages, ack_pages, line_exclude):
    """Robust heading finder: only accept blocks not on meta pages, not repeated, and with heading-like NLP."""
    headings = []
    # Filter out lines on meta pages, repeated, or in tables
    for b in blocks:
        # No meta pages except for the section marker itself
        if b["page"] in toc_pages.union(ack_pages):
            if not any(word in b["text"].lower() for word in ("revision history","table of contents", "acknowledgements")):
                continue
        # Exclude repeated lines/logos/headers/footers
        if normalize(b["text"]) in line_exclude:
            continue
        # Exclude table rows after meta H1s (like Revision History)
        # If block is immediately after H1 and looks like a table, skip
        cur_idx = page2blocks[b["page"]].index(b)
        if cur_idx>0:
            prev_block = page2blocks[b["page"]][cur_idx-1]
            if any(word in prev_block["text"].lower() for word in ("revision history", "version history")):
                if is_table_row(b, page2blocks[b["page"]], cur_idx):
                    continue
        # Heuristic: Consider lines with strong heading cues
        txt = b["text"].strip()
        if not txt or len(txt)<3:
            continue
        # True headings: numbered or strong noun phrases
        doc = nlp(txt)
        if (
            txt[:2].replace(".", "").isdigit() and txt[1] in ". " # 1. Heading
            or (b["is_bold"] and b["font_size"]>=b["median_font_size"]+1 and len(txt)<80)
        ):
            # Not a single short word (avoid table cells), unless "section x" pattern
            if len(txt.split())>1 or "section" in txt.lower():
                headings.append(b)
        # Explicitly accept certain classic marker headings
        elif any(txt.lower().startswith(marker) for marker in [
            "revision history", "table of contents", "acknowledgements", "references"
        ]):
            headings.append(b)
    return headings

def assign_levels(headings):
    """Assign H1, H2, H3 by font clustering or numbering pattern."""
    # If heading text like "2.4 xyz", assign H2
    for h in headings:
        txt = h["text"].strip()
        if txt[:4].replace(".", "").replace(" ", "").isdigit() and "." in txt and txt[1]==".":
            h["level"] = "H2"
        elif txt[:2].replace(".", "").isdigit() and txt[1] in ". ":
            h["level"] = "H1"
        else:
            h["level"] = None # Assign by font
    # Fallback: Largest: H1, next: H2, smallest: H3
    font_sizes = sorted({h["font_size"] for h in headings}, reverse=True)
    if len(font_sizes)>1:
        for h in headings:
            if h["level"]: continue
            idx = font_sizes.index(h["font_size"])
            h["level"] = f"H{min(idx+1,3)}"
    else:
        for h in headings:
            if not h["level"]:
                h["level"] = "H1"
    headings[:] = [h for h in headings if h["level"] in {"H1","H2","H3"}]

def process_pdf(input_path, output_path):
    doc = fitz.open(input_path)
    all_lines = get_line_features(doc)
    # Step 1: Find repeated lines/logos/header/footer to globally exclude
    common_lines = get_common_lines(all_lines, doc.page_count)
    # Step 2: Group stacked lines for blocks (multi-line heading, logo, etc.)
    blocks = group_multilines(all_lines, common_lines)
    # Step 3: Detect ToC, Acknowledgement, copyright meta pages
    toc_pages = detect_toc_pages(blocks)
    ack_pages = detect_ack_pages(blocks)
    meta_pages = toc_pages.union(ack_pages)
    # For grouping, build page-index map
    page2blocks = defaultdict(list)
    for b in blocks:
        page2blocks[b["page"]].append(b)
    # Step 4: Pick real title from first page blocks (not repeated, not meta page).
    title = get_title_candidate(blocks, common_lines, meta_pages)
    # Step 5: Find actual headings, skipping known repeated/cover/meta/table content
    headings = detect_headings_blocks(
        blocks, page2blocks, toc_pages, ack_pages, common_lines
    )
    # Step 6: Assign levels H1/H2/H3 by number pattern or font size grouping
    assign_levels(headings)
    # Remove redundancy/duplicates and preserve page order
    result_headings = []
    seen = set()
    for h in sorted(headings, key=lambda x: (x["page"], x["y0"])):
        k = (h.get("level",""), normalize(h["text"]))
        if k in seen: continue
        if normalize(h["text"]) == normalize(title): continue
        seen.add(k)
        # Append trailing space for JSON output if present in PDF
        result_headings.append({
            "level": h["level"],
            "text": h["text"] + (" " if h["text"].endswith(" ") else ""),
            "page": h["page"]
        })
    # Output strict schema
    output = {"title": title, "outline": result_headings}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0]+".json")
        try:
            process_pdf(input_path, output_path)
        except Exception as e:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"title":"No title found", "outline":[]}, f, ensure_ascii=False)
            print(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    main()
