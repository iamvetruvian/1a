import os
import json
import fitz
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from statistics import median
import joblib
from sentence_transformers import SentenceTransformer

# === CONFIG ===
INPUT_DIR = "./tests"
OUTPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"
MODEL_PKL = "heading_classifier.pkl"
LABEL_ENCODER_PKL = "label_encoder.pkl"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L12-v2'

# === LOAD MODELS ===
clf = joblib.load(MODEL_PKL)
label_enc = joblib.load(LABEL_ENCODER_PKL)
embedder = SentenceTransformer(EMBEDDING_MODEL)

# === UTILS ===
def normalize(text):
    return " ".join(text.strip().split()) if text else ""

def is_body_line(y0, page_height):
    # Exclude PDF header/footer margin (tunable)
    return 0.06 * page_height < y0 < 0.93 * page_height

def get_line_features(doc):
    """Extract lines for the PDF as feature dicts."""
    all_lines = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        page_width = page.rect.width
        font_sizes = []
        for block in blocks:
            if block["type"] != 0: continue
            for line in block["lines"]:
                font_sizes.extend([span["size"] for span in line["spans"]])
        median_font_size = median(font_sizes) if font_sizes else 12
        for block in blocks:
            if block["type"] != 0: continue
            for line in block["lines"]:
                text = "".join([span["text"] for span in line["spans"]]).strip()
                if not text: continue
                spans = line["spans"]
                is_bold = any("bold" in s["font"].lower() or s.get("flags",0)&2==2 for s in spans)
                font_size = max(s["size"] for s in spans)
                x0 = min(s["bbox"][0] for s in spans)
                y0 = min(s["bbox"][1] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)
                all_lines.append({
                    "text": text,
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "x0": x0,
                    "y0": y0,
                    "y1": y1,
                    "page": page_num,
                    "page_height": page_height,
                    "page_width": page_width,
                    "median_font_size": median_font_size
                })
    return all_lines

def get_common_lines(all_lines, num_pages, freq_thresh=0.7, y_tol=22):
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
            y_spread = max(norm_y[t]) - min(norm_y[t]) if norm_y[t] else 0
            if y_spread < y_tol:
                common_lines.add(t)
    return common_lines

def group_multilines(lines, line_exclude):
    lines = [l for l in lines if normalize(l["text"]) not in line_exclude]
    if not lines: return []
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

def extract_features_for_model(blocks, emb_dim=384):
    # Returns array X for prediction, same feature order as used in training
    # Each block: font_size, is_bold, x0, y0, emb_0, ..., emb_383
    texts = [b["text"] for b in blocks]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    X = []
    for i, b in enumerate(blocks):
        row = [
            b["font_size"],
            int(b["is_bold"]),
            b["x0"],
            b["y0"]
        ] + list(embeddings[i])
        X.append(row)
    return np.array(X)

def process_pdf(input_path, output_path):
    doc = fitz.open(input_path)
    all_lines = get_line_features(doc)
    # Remove repeated lines (header/footer/logo)
    common_lines = get_common_lines(all_lines, doc.page_count)
    blocks = group_multilines(all_lines, common_lines)
    if not blocks:
        # fallback if all skipped
        output = {"title":"No title found", "outline":[]}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        return

    # === CLASSIFY ===
    # Features for all blocks
    X_test = extract_features_for_model(blocks)
    y_pred = clf.predict(X_test)
    pred_labels = label_enc.inverse_transform(y_pred)

    # === BUILD OUTPUT ===
    # 1. Find the block with label 'title' (if any). If multiple, largest font or lines_in_block
    title_blocks = [b for b, label in zip(blocks, pred_labels) if label=='title']
    if title_blocks:
        # Prefer largest font
        title_block = max(title_blocks, key=lambda b: (b['font_size'], b.get('lines_in_block',1)))
        title = title_block['text']
    else:
        title = "No title found"

    # 2. Headings: only those labeled H1, H2, H3. Page is from block.
    outline = []
    for b, label in zip(blocks, pred_labels):
        if label in ("H1", "H2", "H3"):
            outline.append({
                "level": label,
                "text": b["text"] + (" " if b["text"].endswith(" ") else ""),
                "page": b["page"]
            })

    result = {"title": title, "outline": outline}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    for fname in files:
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
