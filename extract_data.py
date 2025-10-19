import fitz  # PyMuPDF
import os
import json
import pandas as pd
from tqdm import tqdm
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

INPUT_DIR = './inputs'
EXPECTED_DIR = './expected'
OUTPUT_CSV = 'train_data.csv'

def normalize(s):
    # Lowercase, remove all whitespace, for fuzzy matching
    return ''.join(s.lower().split()) if s else ''

def extract_labels(json_path):
    """
    Load labels from the expected JSON and build lookup by (normalized text, page)
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    label_lookup = {}  # (normalized text, page) -> label
    # Title (page=None means match on any page)
    title = data.get('title', '').strip()
    if title and title.lower() != 'no title found':
        normt = normalize(title)
        label_lookup[(normt, None)] = 'title'
    # Outline (heading levels)
    for item in data.get('outline', []):
        level = item.get('level')
        txt = item.get('text', '').strip()
        page = item.get('page', None)
        if level and txt and page is not None:
            label_lookup[(normalize(txt), int(page))] = level
    return label_lookup

def find_label(line, label_lookup):
    """
    Try to find the correct label for a line given the label lookup
    """
    normtxt = normalize(line['text'])
    p = line['page']
    # Try full match with page-specific, else title (page=None)
    if (normtxt, p) in label_lookup:
        return label_lookup[(normtxt, p)]
    if (normtxt, None) in label_lookup:
        return label_lookup[(normtxt, None)]
    return "body"

def detect_casing(text):
    words = [w for w in re.findall(r'\b\w+\b', text)]
    if not words:
        return "mixed"
    if all(w.isupper() for w in words):
        return "all_caps"
    if all(w.istitle() for w in words):
        return "capitalized"
    if words[0][0].isupper() and all(w[0].islower() or not w[0].isalpha() for w in words[1:]):
        return "sentence"
    return "mixed"

def stopword_ratio(text):
    words = [w for w in re.findall(r'\b\w+\b', text.lower())]
    if not words:
        return 0
    stop_count = sum(1 for w in words if w in STOPWORDS)
    return stop_count / len(words)

def extract_pdf_features(pdf_path):
    """
    Extract features for every text line in the PDF.
    Returns a list of dicts.
    NEW: Adds color, is_italic, casing, text_length, is_underline, font_family, stopword_ratio
    """
    doc = fitz.open(pdf_path)
    features = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:  # Only text blocks
                continue
            for line in block["lines"]:
                spans = line["spans"]
                line_text = "".join([span["text"] for span in spans]).strip()
                if not line_text:
                    continue

                font_sizes = [span["size"] for span in spans]
                font_size = max(font_sizes) if font_sizes else 0

                font_names = [span.get("font", "") for span in spans]
                font_family = Counter(font_names).most_common(1)[0][0] if font_names else ""
                is_bold = any("bold" in fn.lower() for fn in font_names)
                is_italic = any("italic" in fn.lower() or "oblique" in fn.lower() for fn in font_names)
                x0 = min(span["bbox"][0] for span in spans)
                y0 = min(span["bbox"][1] for span in spans)

                # Color: Use the most common color value in line's spans
                colors = [span.get("color", 0) for span in spans]
                color_val = Counter(colors).most_common(1)[0][0] if colors else 0
                color_hex = "#{:06X}".format(color_val) if isinstance(color_val, int) else str(color_val)

                # Underline: span flags (see PyMuPDF docs: flags & 4 == underline)
                is_underline = any(int(span.get("flags", 0)) & 4 for span in spans)

                # Casing
                casing = detect_casing(line_text)

                # Text length
                text_length = len(line_text)

                # Stopword ratio
                stop_ratio = stopword_ratio(line_text)

                features.append({
                    "file": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": line_text,
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "x0": x0,
                    "y0": y0,
                    "color": color_hex,
                    "is_italic": is_italic,
                    "casing": casing,
                    "text_length": text_length,
                    "is_underline": is_underline,
                    "font_family": font_family,
                    "stopword_ratio": stop_ratio,
                })
    return features

def main():
    rows = []
    pdfs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    for pdf_file in tqdm(pdfs, desc="Processing PDFs"):
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        json_name = os.path.splitext(pdf_file)[0] + ".json"
        json_path = os.path.join(EXPECTED_DIR, json_name)
        # Skip PDFs w/o expected output (you may want to warn here)
        if not os.path.isfile(json_path):
            print(f"Warning: Expected JSON for {pdf_file} not found, skipping...")
            continue
        label_lookup = extract_labels(json_path)
        features = extract_pdf_features(pdf_path)
        for feat in features:
            feat['label'] = find_label(feat, label_lookup)
        rows.extend(features)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Done! Data saved to {OUTPUT_CSV}")
    else:
        print("No data found.")

if __name__ == "__main__":
    main()