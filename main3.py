import os
import json
import fitz
from typing import List, Dict
import spacy
from statistics import median

INPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"
OUTPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"

try:
    # Prefer multilingual, fallback to English
    NLP = spacy.load("xx_ent_wiki_sm")
except Exception:
    NLP = spacy.load("en_core_web_sm")

CAPTION_PREFIXES = [
    "figure", "fig.", "table", "image", "図", "ターブル", "tabelle"
]

def is_caption_candidate(text: str) -> bool:
    txt = text.lower().strip(".:；: ")
    for prefix in CAPTION_PREFIXES:
        if txt.startswith(prefix):
            return True
    if len(txt) > 0 and txt.split(" ")[0].isdigit():
        return True
    return False

def normalize_text(text: str) -> str:
    return ' '.join(text.strip().split())

def is_body_line(y0, page_height):
    # Exclude top 6% and bottom 7% for header/footer (tunable)
    return 0.06 * page_height < y0 < 0.93 * page_height

def detect_headings_semantic(lines: List[str], lines_meta: List[dict]) -> List[bool]:
    flags = []
    for i, line in enumerate(lines):
        doc = NLP(line)
        meta = lines_meta[i]
        # One-word heading allowed if visually salient
        if len(doc) == 1:
            if meta['is_bold'] and meta['font_size'] >= meta['median_font_size'] + 1:
                flags.append(True)
            else:
                flags.append(False)
            continue
        noun_count = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN", "ADJ"})
        alpha_ratio = sum(1 for t in doc if t.is_alpha) / max(len(doc), 1)
        non_sentence = not any(t.pos_ == "VERB" for t in doc)
        flag = (noun_count >= (len(doc) // 2 + 1)) and alpha_ratio > 0.6 and non_sentence
        if not flag:
            headings_cues = sum(1 for t in doc if (t.text.istitle() or t.text.isupper()) and not t.is_stop)
            if headings_cues >= len(doc) * 0.8 and len(doc) <= 12:
                flag = True
        # End with punctuation: nearly never heading
        if any(line.strip().endswith(x) for x in ('.', ';', ':', '!', '?', '。')):
            flag = False
        flags.append(flag)
    return flags

def cluster_heading_levels(headings: List[Dict]) -> None:
    sizes = [h['font_size'] for h in headings]
    unique_sizes = sorted(set(sizes), reverse=True)
    if len(unique_sizes) == 1:
        for h in headings:
            h['level'] = "H1"
        return
    size_counts = {}
    for sz in sizes:
        size_counts[sz] = size_counts.get(sz, 0) + 1
    # Use up to top 3 font sizes for levels
    sorted_sizes = sorted(size_counts.items(), key=lambda x: -x[0])
    size_to_level = {}
    for idx, (size, _) in enumerate(sorted_sizes[:3]):
        size_to_level[size] = f"H{idx+1}"
    for h in headings:
        level = size_to_level.get(h['font_size'], None)
        h['level'] = level
    # H3 restriction
    headings[:] = [h for h in headings if h['level'] in {"H1", "H2", "H3"}]

def get_title_candidate(pages_lines: List[List[Dict]]) -> str:
    first_page = pages_lines[0]
    candidates = [l for l in first_page if l['y0'] < 180 and l['font_size'] >= 12]
    if not candidates:
        return "No title found"
    all_texts = set()
    for page_lines in pages_lines[1:]:
        all_texts.update(normalize_text(l['text']) for l in page_lines)
    # Group closely stacked candidate lines at top, as multi-line title
    title_lines = []
    for l in sorted(candidates, key=lambda x: x['y0']):
        txt = normalize_text(l['text'])
        if txt and len(txt) >= 5 and txt not in all_texts:
            title_lines.append(txt)
        if len(title_lines) > 3:  # unlikely to have >3-line titles
            break
    if title_lines:
        return " ".join(title_lines)
    return "No title found"

def is_table_structured(doc):
    table_blocks, text_blocks = 0, 0
    pages_checked = min(5, doc.page_count)
    for pgno in range(pages_checked):
        for block in doc[pgno].get_text("dict")["blocks"]:
            if block["type"] == 5:
                table_blocks += 1
            elif block["type"] == 0:
                text_blocks += 1
    return table_blocks > text_blocks and table_blocks > 3

def process_pdf(input_path: str, output_path: str):
    doc = fitz.open(input_path)
    pages_lines = []
    table_doc = is_table_structured(doc)
    all_lines = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_lines = []
        font_sizes = []
        page_height = page.rect.height
        page_width = page.rect.width
        # Gather median font size for the page (for bold/one-word heuristic)
        for block in blocks:
            if block['type'] != 0: continue
            for line in block['lines']:
                font_sizes.extend([span['size'] for span in line['spans']])
        median_font_size = median(font_sizes) if font_sizes else 12
        # Collect body lines with meta for heading detection
        for block in blocks:
            if block['type'] != 0:
                continue
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
                # Filter out headers/footers (issue 1 fix)
                if not is_body_line(y0, page_height):
                    continue
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
                    'page_height': page_height,
                    'page_width': page_width,
                    'median_font_size': median_font_size,
                    'is_from_table': False
                })
        pages_lines.append(page_lines)
        for l in page_lines:
            all_lines.append(l)

    all_text_lines = [l['text'] for l in all_lines]
    # Pass in meta for each line (for one-word bold detection)
    semantic_flags = detect_headings_semantic(all_text_lines, all_lines)
    for i, f in enumerate(semantic_flags):
        all_lines[i]['semantic_heading'] = f

    heading_candidates = []
    for l in all_lines:
        if not l['semantic_heading']:
            continue
        if is_caption_candidate(l['text']):
            continue
        if len(l['text']) > 120:
            continue
        # Table content: suppress unless doc is table-structured or heading is unique/large (issue 4)
        if l.get('is_from_table', False) and not table_doc:
            continue
        heading_candidates.append(l)

    # --- Issue 2: Also add large, centered, bold lines as H1 if not already picked ---
    for l in all_lines:
        # Only consider if not already in headings and in body
        xmid = (l['bbox'][0]+l['bbox'][2])/2
        page_center = l['page_width']/2
        if (l['font_size'] >= l['median_font_size'] + 3 and
                abs(xmid - page_center) < l['page_width']*0.15 and
                l['is_bold'] and
                len(l['text'].strip()) > 3 and
                not is_caption_candidate(l['text']) and
                l['semantic_heading'] is False):
            if not any(h['text'] == l['text'] and h['page'] == l['page'] for h in heading_candidates):
                # Strongly-likely H1
                temp = dict(l) # Copy line dict
                temp['level'] = 'H1'
                heading_candidates.append(temp)

    # Detect heading levels
    cluster_heading_levels([h for h in heading_candidates if 'level' not in h])
    # All force H1s already marked

    # Remove repeated headings, prefer first occurrence
    seen = set()
    dedup = []
    for h in heading_candidates:
        t = normalize_text(h['text'])
        key = (h['level'], t, h['page'])
        if key in seen:
            continue
        dedup.append(h)
        seen.add(key)
    heading_candidates = dedup

    # Handle title: exclude from headings
    title = get_title_candidate(pages_lines)
    norm_title = normalize_text(title)
    for h in heading_candidates:
        if normalize_text(h['text']) == norm_title:
            h['exclude'] = True
    heading_candidates = [h for h in heading_candidates if not h.get('exclude')]

    # --- Remove Table of Contents/References if repeated (optional cleanup as before) ---
    json_outline = []
    for h in heading_candidates:
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".json")
        try:
            process_pdf(input_path, output_path)
        except Exception as e:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"title": "No title found", "outline": []}, f, ensure_ascii=False)
            print(f"Failed to process {fname}: {str(e)}")

if __name__ == "__main__":
    main()
