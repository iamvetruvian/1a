import os
import json
import fitz
import spacy
import numpy as np
from typing import List, Dict
from statistics import median
from sentence_transformers import SentenceTransformer, util

# Path config for submission
INPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"
OUTPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"

# Load spaCy small model and a compact sentence transformer (multilingual if you prefer!)
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except Exception:
    nlp = spacy.load("en_core_web_sm")

# Load SBERT model for multi-line heading grouping
sb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

CAPTION_PREFIXES = ["figure", "fig.", "table", "image", "図", "ターブル", "tabelle"]

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
    return 0.06 * page_height < y0 < 0.93 * page_height

def group_multiline_blocks(lines: List[Dict]) -> List[Dict]:
    """
    Group visually adjacent lines into heading-candidate blocks based on layout and semantic similarity.
    """
    if not lines:
        return []
    grouped = []
    curr_group = [lines[0]]
    for prev, nxt in zip(lines, lines[1:]):
        same_font = abs(nxt['font_size'] - prev['font_size']) < 1.1
        same_bold = nxt['is_bold'] == prev['is_bold']
        y_gap = nxt['y0'] - prev['bbox'][3]
        vertically_close = y_gap < 1.5 * prev['font_size']
        # Use semantic cosine similarity
        emb_prev = sb_model.encode([prev["text"]])[0]
        emb_nxt = sb_model.encode([nxt["text"]])[0]
        sim = util.cos_sim([emb_prev], [emb_nxt]).item()  # [-1,1]
        semigroup = sim > 0.63  # threshold: tune as you wish!
        if same_font and same_bold and vertically_close and semigroup:
            curr_group.append(nxt)
        else:
            grouped.append(curr_group)
            curr_group = [nxt]
    grouped.append(curr_group)
    # Combine lines in each group to one candidate
    blocks = []
    for group in grouped:
        lines_text = " ".join([l['text'] for l in group])
        lines_text_n = normalize_text(lines_text)
        l0 = group[0]
        block = dict(l0)
        block['text'] = lines_text
        block['group_lines'] = len(group)
        block['y1'] = group[-1]['bbox'][3]
        blocks.append(block)
    return blocks

def detect_headings_semantic(blocks: List[Dict]) -> List[bool]:
    """
    Use NLP, POS, casing, and block context to mark whether a candidate block is likely a heading.
    """
    flags = []
    for b in blocks:
        line = b["text"]
        doc = nlp(line)
        if not line or len(line)<2:
            flags.append(False)
            continue
        # One-word block allowed if salient
        if len(doc) == 1:
            if b['is_bold'] and b['font_size'] >= b['median_font_size'] + 1:
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
        if any(line.strip().endswith(x) for x in ('.', ';', ':', '!', '?', '。')):
            flag = False
        flags.append(flag)
    return flags

def cluster_heading_levels(blocks: List[Dict]) -> None:
    sizes = [b['font_size'] for b in blocks]
    unique_sizes = sorted(set(sizes), reverse=True)
    if len(unique_sizes) == 1:
        for b in blocks:
            b['level'] = "H1"
        return
    size_counts = {}
    for sz in sizes:
        size_counts[sz] = size_counts.get(sz, 0) + 1
    sorted_sizes = sorted(size_counts.items(), key=lambda x: -x[0])
    size_to_level = {}
    for idx, (size, _) in enumerate(sorted_sizes[:3]):
        size_to_level[size] = f"H{idx+1}"
    for b in blocks:
        level = size_to_level.get(b['font_size'], None)
        b['level'] = level
    blocks[:] = [b for b in blocks if b['level'] in {"H1", "H2", "H3"}]

def get_title_candidate(page_blocks: List[Dict], all_blocks: List[Dict]) -> str:
    # Use top-most, largest, multi-line (or single) blocks that aren't repeated as headings
    candidates = [b for b in page_blocks if b['y0'] < 180 and b['font_size'] >= 12]
    if not candidates:
        return "No title found"
    all_texts = set(normalize_text(b['text']) for b in all_blocks)
    for c in sorted(candidates, key=lambda b: (-b['font_size'], b['y0'])):
        txt = normalize_text(c['text'])
        if len(txt)>=5 and all( txt != normalize_text(b['text']) for b in all_blocks[1:]):
            return c['text']
    return candidates[0]['text']

def is_table_of_contents_page(blocks: List[Dict]) -> bool:
    # Identify ToC if one block contains or nearly equals "Table of Contents" (case insensitive)
    for b in blocks:
        if "table of contents" in b['text'].lower():
            return True
    return False

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
    all_blocks = []
    table_doc = is_table_structured(doc)
    toc_pages = set()
    pages_blocks = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_lines = []
        font_sizes = []
        page_height = page.rect.height
        for block in blocks:
            if block['type'] != 0: continue
            for line in block['lines']:
                font_sizes.extend([span['size'] for span in line['spans']])
        median_font_size = median(font_sizes) if font_sizes else 12
        for block in blocks:
            if block['type'] != 0: continue
            for line in block['lines']:
                line_text = "".join(span['text'] for span in line['spans']).strip()
                if not line_text: continue
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
                # Header/footer exclusion
                if not is_body_line(y0, page_height): continue
                page_lines.append({
                    'text': line_text,
                    'font_size': font_size,
                    'is_bold': is_bold,
                    'x0': x0,
                    'y0': y0,
                    'bbox': bbox,
                    'font_name': font_name,
                    'page': page_num,
                    'median_font_size': median_font_size,
                })
        # Group visually and semantically adjacent lines: multi-line headings
        blocks_page = group_multiline_blocks(page_lines)
        # Identify ToC page and mark it (for later skip)
        if is_table_of_contents_page(blocks_page):
            toc_pages.add(page_num)
        for b in blocks_page:
            all_blocks.append(b)
        pages_blocks.append(blocks_page)
    # Heading detection — only outside ToC
    main_blocks = [b for b in all_blocks if b['page'] not in toc_pages]
    heading_flags = detect_headings_semantic(main_blocks)
    for i, f in enumerate(heading_flags):
        main_blocks[i]['is_heading'] = f
    # Level clustering
    heading_candidates = [b for b in main_blocks if b.get('is_heading')]
    cluster_heading_levels(heading_candidates)
    # Title detection
    title = get_title_candidate(pages_blocks[0], all_blocks)
    heading_candidates = [b for b in heading_candidates if normalize_text(b['text'])!=normalize_text(title)]
    # Output
    json_outline = []
    for h in heading_candidates:
        json_outline.append({
            'level': h['level'],
            'text': h['text'],
            'page': h['page']
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
