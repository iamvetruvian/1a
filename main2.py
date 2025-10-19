import os
import json
import fitz  # PyMuPDF
from typing import List, Dict
import spacy
from spacy.tokens import DocBin
from collections import Counter, defaultdict

INPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"
OUTPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"

# Load spaCy model once at startup
try:
    # Multilingual for robustness (size: ~48MB)
    NLP = spacy.load("xx_ent_wiki_sm")
except Exception:
    # Fallback to English if xx failed
    NLP = spacy.load("en_core_web_sm")

# Heuristic block list for false positives (table/image captions, etc)
CAPTION_PREFIXES = [
    "figure", "fig.", "table", "image", "図", "ターブル", "tabelle" # multi-lingual
]

def is_caption_candidate(text: str) -> bool:
    # Check if the line starts with a likely caption pattern
    txt = text.lower().strip(".:；: ")
    for prefix in CAPTION_PREFIXES:
        if txt.startswith(prefix):
            return True
    # Also numerics + punctuation (e.g., Table 2.1: abc)
    if len(txt) > 0 and txt.split(" ")[0].isdigit():
        return True
    return False

def normalize_text(text: str) -> str:
    # Remove extra whitespace, normalize for comparison
    return ' '.join(text.strip().split())

def detect_headings_semantic(lines: List[str]) -> List[bool]:
    """
    Use spaCy's tagger to mark which lines most likely represent a heading.
    Used as a supplementary check: lines with few tokens, mostly nouns, and title case.
    """
    flags = []
    for line in lines:
        doc = NLP(line)
        # Heading-likeness features
        if len(doc) <= 1:
            flags.append(False)
            continue

        # Main heuristic: Most tokens are either PROPN, NOUN, ADJ, and the line is not a typical sentence (e.g., no verb)
        noun_count = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN", "ADJ"})
        alpha_ratio = sum(1 for t in doc if t.is_alpha) / max(len(doc),1)
        title_case = normalize_text(line).istitle()
        non_sentence = not any(t.pos_ == "VERB" for t in doc)
        # Structure: Lines with mostly alphabetic tokens, title-case, no verbs, mostly nouns/props/adjs
        flag = (noun_count >= (len(doc) // 2 + 1)) and alpha_ratio > 0.6 and non_sentence
        # Assistance: if >80% of tokens are upper or title-case with few stopwords, flag as heading
        if not flag:
            headings_cues = sum(1 for t in doc if (t.text.istitle() or t.text.isupper()) and not t.is_stop)
            if headings_cues >= len(doc) * 0.8 and len(doc) <= 12:
                flag = True
        # If line ends in punctuation, likely not a heading
        if any(line.strip().endswith(x) for x in ('.',';',':','!','?','。')):
            flag = False
        flags.append(flag)
    return flags

def cluster_heading_levels(headings: List[Dict]) -> None:
    """
    Assign a heading level (H1/H2/H3) based on block structure and semantic distance.
    - Usually, heading levels correspond to a 'cluster' of text block features, so we cluster them.
    """
    # Collect features for clustering: font size, bold/italic flags, x0 (indent/dedent), length, etc.
    # Here, we'll use font size, bold and indentation, but with a semantic fallback

    # Step 1: Get candidate features
    sizes = [h['font_size'] for h in headings]
    indents = [h['x0'] for h in headings]
    is_bold = [h['is_bold'] for h in headings]

    # Step 2: Cluster font sizes
    unique_sizes = sorted(set(sizes), reverse=True)
    if len(unique_sizes) == 1:
        # only one font size in all headings, fallback: indents or sequence
        # All are likely H1, unless indented or part of frequent heading pattern
        for h in headings:
            h['level'] = "H1"
        return

    # Use font size + bold + indent to assign levels
    size_counts = Counter(sizes)
    # Largest font size → H1, second largest → H2, etc
    sorted_sizes = sorted(size_counts.items(), key=lambda x:-x[0])
    size_to_level = {}
    for idx, (size, _) in enumerate(sorted_sizes[:3]):
        size_to_level[size] = f"H{idx+1}"

    # If > 3 sizes, group smaller font sizes as H3
    for h in headings:
        level = size_to_level.get(h['font_size'], "H3")
        h['level'] = level

    # Resolve: If two consecutive headings are both H1 (and short apart), 
    # and there's a clear indentation difference, assign one to H2
    for i in range(1, len(headings)):
        if headings[i]['level'] == "H1" and headings[i-1]['level'] == "H1":
            if abs(headings[i]['x0'] - headings[i-1]['x0']) > 20:
                headings[i]['level'] = "H2"

def get_title_candidate(pages_lines: List[List[Dict]]) -> str:
    """
    Analyze all pages' lines to find best title:
    - Title candidate: appears at top of first page, unique in doc, largest font, not a heading repeated on later pages.
    - If not found, returns "No title found"
    """
    first_page = pages_lines[0]
    # Heuristic: Find the largest font, bold, centered lines at top of pg0
    candidates = [l for l in first_page if l['y0'] < 180 and l['font_size'] >= 12]
    if not candidates:
        return "No title found"
    # Exclude likely heading by checking whether it's repeated in other pages as heading
    all_texts = set()
    for page_lines in pages_lines[1:]:
        all_texts.update(normalize_text(l['text']) for l in page_lines)
    for cand in sorted(candidates, key=lambda l: (-l['font_size'], abs(l['x0']-210))): # closer to page center & bigger
        txt = normalize_text(cand['text'])
        if len(txt) < 5:
            continue
        if txt not in all_texts:
            return txt
    return "No title found"

def process_pdf(input_path: str, output_path: str):
    doc = fitz.open(input_path)
    pages_lines = []  # List[List[Dict]]
    outline = []

    # --- Step 1: Extract line candidates from all pages ---
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        page_lines = []
        for block in blocks:
            # Only process text blocks, not images/tables separated
            if block['type'] != 0:
                continue
            for line in block['lines']:
                line_text = "".join(span['text'] for span in line['spans']).strip()
                if not line_text or len(line_text) < 2:
                    continue
                # Extract font info: largest font size in line, bold, etc
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
                    'page': page_num
                })
        pages_lines.append(page_lines)

    # --- Step 2: Gather all line candidates across pages ---
    all_lines = []
    for page_lines in pages_lines:
        all_lines.extend(page_lines)

    # --- Step 3: Find heading candidates semantically ---
    all_text_lines = [l['text'] for l in all_lines]
    semantic_flags = detect_headings_semantic(all_text_lines)
    for i, f in enumerate(semantic_flags):
        all_lines[i]['semantic_heading'] = f

    heading_candidates = []
    for l in all_lines:
        # Keep only lines with heading-like semantics and not in captions/likely tables
        if not l['semantic_heading']:
            continue
        if is_caption_candidate(l['text']):
            continue
        if len(l['text']) > 120: # Too long, not a heading
            continue
        # Optionally, skip lines that are identical to table/image captions (from their positions on page)
        if l['y0'] > 760:    # assuming A4 ~842px; likely footer/caption
            continue
        heading_candidates.append(l)

    # --- Step 4: Infer heading levels ----
    if heading_candidates:
        cluster_heading_levels(heading_candidates)
        # Remove headings below H3 level (no more than H3)
        heading_candidates = [h for h in heading_candidates if h['level'] in {"H1","H2","H3"}]

    # --- Step 5: Remove false positive headings inside Table of Contents or Reference sections ---
    # (e.g. line contains "Table of Contents" or is inside a References section)
    final_headings = []
    in_toc = False
    in_refs = False
    for h in heading_candidates:
        t = normalize_text(h['text']).lower()
        if "table of contents" in t or "index" in t:
            in_toc = True
        if any(x in t for x in ("references", "bibliography")):
            in_refs = True
        if in_toc and h['page'] < 5:
            continue
        if in_refs:
            continue
        final_headings.append(h)

    # --- Step 6: Get Title ---
    title = get_title_candidate(pages_lines)
    if title == "" or title is None:
        title = "No title found"

    # --- Step 7: Compose Output ---
    json_outline = []
    for h in final_headings:
        json_outline.append({
            "level": h['level'],
            "text": h['text'],
            "page": h['page']
        })

    output = {
        "title": title,
        "outline": json_outline
    }

    # --- Step 8: Write Output ---
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def main():
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".json")
        try:
            process_pdf(input_path, output_path)
        except Exception as e:
            # In case of failure, output a minimal JSON with proper error
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "title": "No title found",
                    "outline": []
                }, f, ensure_ascii=False)
            print(f"Failed to process {fname}: {str(e)}")
if __name__ == "__main__":
    main()
