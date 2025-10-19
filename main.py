import os
import json
import fitz  # PyMuPDF
import spacy
from collections import Counter
from typing import List, Dict

INPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"
OUTPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"

# Heuristic caption/table label blocks to be mostly ignored
CAPTION_PREFIXES = ['figure', 'fig.', 'table', 'image', '图', '図', 'tabelle']

def is_caption_candidate(text: str) -> bool:
    txt = text.lower().strip(".:：; ")
    return any(txt.startswith(prefix) for prefix in CAPTION_PREFIXES) or \
        (len(txt) > 0 and txt.split(" ")[0].isdigit())

def normalize_text(text: str) -> str:
    return ' '.join(text.strip().split())

def detect_semantic_headings(lines: List[str], nlp) -> List[bool]:
    flags = []
    for line in lines:
        doc = nlp(line)
        # Require at least 2 tokens and at least half NOUN/PROPN/ADJ,
        # not a normal sentence (no verbs), not ending in normal sentence punctuation.
        if len(doc) <= 1:
            flags.append(False)
            continue
        nounish = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN", "ADJ"})
        alpha_ratio = sum(1 for t in doc if t.is_alpha) / len(doc)
        no_verb = not any(t.pos_ == "VERB" for t in doc)
        ends_punct = any(line.strip().endswith(ch) for ch in ('.','?','!','。',';',':','；','：'))
        is_short = len(normalize_text(line)) <= 80
        looks_like_heading = nounish >= (len(doc) // 2 + 1) and alpha_ratio > 0.6 and no_verb and is_short and not ends_punct
        # Backup: many tokens title/upper, low stop, not a sentence
        if not looks_like_heading:
            cues = sum(1 for t in doc if (t.text.istitle() or t.text.isupper()) and not t.is_stop)
            looks_like_heading = cues >= len(doc) * 0.8 and is_short
        flags.append(looks_like_heading)
    return flags

def assign_heading_levels(headings: List[Dict]) -> None:
    # Only as many levels as clear clusters; don't assign H3 if not present.
    # Use font_size and (as tie-breaker) bold.
    font_sizes = sorted({h['font_size'] for h in headings}, reverse=True)
    n_levels = min(3, len(font_sizes))
    level_map = {}
    for i, sz in enumerate(font_sizes[:n_levels]):
        level_map[sz] = f"H{i+1}"
    # assign, only if level present
    for h in headings:
        h['level'] = level_map.get(h['font_size'])
    # Remove if not assigned an H-level (defensive, should never happen)
    headings[:] = [h for h in headings if h['level']]

def is_table_structured(doc):
    table_block_count = 0
    text_block_count = 0
    pages_checked = min(5, doc.page_count)
    for pgno in range(pages_checked):
        for block in doc[pgno].get_text("dict")["blocks"]:
            if block["type"] == 5:
                table_block_count += 1
            elif block["type"] == 0:
                text_block_count += 1
    return table_block_count > text_block_count and table_block_count > 3

def get_title_candidate(pages_lines: List[List[Dict]], heading_candidates: List[Dict]) -> str:
    # Find largest, centerish, first-page, unique lines not in headings, and not body text
    first_page = pages_lines[0]
    headings_texts = set(normalize_text(h['text']) for h in heading_candidates)
    # Candidates: lines on very top, largest font, bold, center-ish
    possible_titles = []
    for line in first_page:
        xmid = (line['bbox'][0] + line['bbox'][2]) / 2
        is_centered = 180 < xmid < 500
        if (line['font_size'] > 12 and line['y0'] < 180 and is_centered) and not is_caption_candidate(line['text']):
            txt = normalize_text(line['text'])
            if txt not in headings_texts and len(txt) >= 5:
                possible_titles.append((line['font_size'], line['is_bold'], -abs(xmid-340), txt))
    if not possible_titles:
        return "No title found"
    # Sort: largest font, bold, most centered
    possible_titles.sort(reverse=True)
    return possible_titles[0][3]

def process_pdf(input_path: str, output_path: str, nlp):
    doc = fitz.open(input_path)
    pages_lines = []
    table_doc = is_table_structured(doc)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_lines = []
        for block in blocks:
            # If block is a table block (type==5), and doc is not table-structured, skip
            if block['type'] == 5 and not table_doc:
                continue
            # If block is a table, process intelligently:
            if block['type'] == 5 and table_doc:
                # get all spans in table as lines (flatten)
                for line in block.get("lines", []):
                    line_text = "".join(span['text'] for span in line['spans']).strip()
                    if not line_text or len(line_text) < 2:
                        continue
                    largest_span = max(line['spans'], key=lambda s: s['size'])
                    font_size = largest_span['size']
                    font_name = largest_span['font']
                    is_bold = "Bold" in font_name or "bold" in font_name or largest_span.get('flags',0)&2 == 2
                    x0 = min(span['bbox'][0] for span in line['spans'])
                    y0 = min(span['bbox'][1] for span in line['spans'])
                    bbox = [min(span['bbox'][0] for span in line['spans']),
                        min(span['bbox'][1] for span in line['spans']),
                        max(span['bbox'][2] for span in line['spans']),
                        max(span['bbox'][3] for span in line['spans'])]
                    page_lines.append({
                        'text': line_text,
                        'font_size': font_size,
                        'is_bold': is_bold,
                        'x0': x0,
                        'y0': y0,
                        'bbox': bbox,
                        'font_name': font_name,
                        'raw_spans': line['spans'],
                        'page': page_num,
                        'is_from_table': True
                    })
                continue
            # Normal text block
            if block['type'] == 0:
                for line in block['lines']:
                    line_text = "".join(span['text'] for span in line['spans']).strip()
                    if not line_text or len(line_text) < 2:
                        continue
                    largest_span = max(line['spans'], key=lambda s: s['size'])
                    font_size = largest_span['size']
                    font_name = largest_span['font']
                    is_bold = "Bold" in font_name or "bold" in font_name or largest_span.get('flags',0)&2 == 2
                    x0 = min(span['bbox'][0] for span in line['spans'])
                    y0 = min(span['bbox'][1] for span in line['spans'])
                    bbox = [min(span['bbox'][0] for span in line['spans']),
                        min(span['bbox'][1] for span in line['spans']),
                        max(span['bbox'][2] for span in line['spans']),
                        max(span['bbox'][3] for span in line['spans'])]
                    page_lines.append({
                        'text': line_text,
                        'font_size': font_size,
                        'is_bold': is_bold,
                        'x0': x0,
                        'y0': y0,
                        'bbox': bbox,
                        'font_name': font_name,
                        'raw_spans': line['spans'],
                        'page': page_num,
                        'is_from_table': False
                    })
        pages_lines.append(page_lines)

    # Flatten all lines for heading detection
    all_lines = []
    for page_lines in pages_lines:
        all_lines.extend(page_lines)

    # Step 1: Semantic heading flags
    all_text_lines = [l['text'] for l in all_lines]
    heading_flags = detect_semantic_headings(all_text_lines, nlp)
    for i, f in enumerate(heading_flags):
        all_lines[i]['semantic_heading'] = f

    # Step 2: Heading candidates -- using advanced logic for table content
    heading_candidates = []
    for l in all_lines:
        if not l['semantic_heading']:
            continue
        if is_caption_candidate(l['text']):
            continue
        if l['y0'] > 760:
            continue
        # Allow headings from table cells ONLY if table_doc, or if they look very prominent
        if l['is_from_table']:
            if not table_doc:
                continue
            # For table-docs, require large/bold/leftmost or topmost, etc to accept as heading
            # In practice, our semantic check already filters.
        heading_candidates.append(l)

    # Step 3: Assign actual heading levels (avoid spurious H3)
    if heading_candidates:
        assign_heading_levels(heading_candidates)

    # Step 4: Remove table of contents/refs from candidates for cleaner output
    filtered_headings = []
    for h in heading_candidates:
        t = normalize_text(h['text']).lower()
        if "table of contents" in t or "index" in t:
            continue
        if any(x in t for x in ("references", "bibliography")) and h['page'] > 6:
            continue
        filtered_headings.append(h)

    # Step 5: Title extraction, and exclude from headings
    title = get_title_candidate(pages_lines, filtered_headings)
    if title != "No title found":
        filtered_headings = [
            h for h in filtered_headings
            if normalize_text(h['text']) != normalize_text(title)
        ]

    # Step 6: Compose output json
    json_outline = []
    for h in filtered_headings:
        json_outline.append({
            "level": h['level'],
            "text": h['text'],
            "page": h['page']
        })

    output = {
        "title": title,
        "outline": json_outline
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def main():
    try:
        try:
            # Prefer multilingual, fallback to English
            nlp = spacy.load("xx_ent_wiki_sm")
        except:
            nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print('spaCy model could not be loaded:', e)
        exit(1)
    # Make sure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0]+".json")
        try:
            process_pdf(input_path, output_path, nlp)
        except Exception as e:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"title": "No title found", "outline": []}, f, ensure_ascii=False)
            print(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    main()
