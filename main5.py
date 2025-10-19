import os
import json
import fitz
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util

INPUT_DIR = "./tests"
OUTPUT_DIR = "C:/Users/itsas/OneDrive/Desktop/1a"

# NLP Models
try: nlp = spacy.load("xx_ent_wiki_sm")
except Exception: nlp = spacy.load("en_core_web_sm")
sb_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize(text):
    return " ".join(text.strip().split())

def get_doc_blocks(doc):
    "Collect all lines and structure info (returns list of dicts)"
    all_blocks = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        for block in blocks:
            if block['type'] != 0: continue
            for line in block['lines']:
                text = "".join(span['text'] for span in line['spans'])
                if not text.strip(): continue
                spans = line['spans']
                font_size = max(s['size'] for s in spans)
                is_bold = any("Bold" in s['font'] or s.get('flags', 0)&2==2 for s in spans)
                x, y = min(s['bbox'][0] for s in spans), min(s['bbox'][1] for s in spans)
                y1 = max(s['bbox'][3] for s in spans)
                bbox = [min(s['bbox'][0] for s in spans), y, max(s['bbox'][2] for s in spans), y1]
                all_blocks.append({
                    'raw':line, 'text':text, 'font_size':font_size, 'is_bold':is_bold, 'x0':x, 'y0':y, 'y1':y1, 'bbox':bbox,
                    'page_height':page_height, 'page':page_num
                })
    return all_blocks

def role_classifier(block, all_blocks, page_blocks, pagelines, prevlines, nextlines):
    "Returns best-guess role (title, header, toc, heading, body, cover, etc.)"
    text = block['text'].strip()
    text_n = normalize(text)

    # Context-based semantic scoring (title often looks like summary/main subject)
    # Could expand this with a small trained classifier for even more robustness
    page_num = block['page']
    # Header: very top, small, repeated, or short: likely page header
    if block['y0'] < 0.06 * block['page_height']:
        # Is this text repeated on many pages?
        rep = sum(normalize(blk['text'])==text_n for blk in all_blocks) > 2
        if rep: return "header"
    # Footer/page-num: very bottom
    if block['y1'] > 0.93 * block['page_height']:
        if text_n.isdigit() or "page" in text_n.lower(): return "footer"
    # TOC: has Table of Contents, or a sequence of lines that look like section names plus index numbers
    if "table of contents" in text_n.lower(): return "toc"
    if any(text_n.lower().startswith(x) for x in ["appendix", "references", "index"]) and page_num > 0:
        return "toc"
    # Title candidates: summary/main noun phrase, often only on first page, not repeated elsewhere, not heading-ish
    if page_num <= 1 and len(text_n) > 5:
        main_phrases = ["road map", "business plan", "proposal", "overview", "challenge", "connecting the dots", "syllabus", "topic", "summary", "mission", "introduction"]
        nouns = sum(t.pos_ in ("NOUN","PROPN") for t in nlp(text_n))
        verbs = sum(t.pos_=="VERB" for t in nlp(text_n))
        if any(p in text_n.lower() for p in main_phrases) and nouns > verbs:
            return "title"
    # Heading: strong noun phrase, either numbered pattern or semantically a section header
    if (
        text_n[:3].replace('.','').replace(' ','').isdigit() # 1./2./3. or 1.1 etc.
        or text_n.istitle()
        or (block['is_bold'] and block['font_size'] > 10 and len(text_n)<80)
    ):
        return "heading"
    # Table/cover: if lots of short lines on page zero, likely cover/structured non-content
    if page_num==0 and len(pagelines)<6 and all(len(normalize(l['text']))<7 for l in pagelines):
        return "cover"
    # Otherwise, assume body
    return "body"

def group_heading_blocks(blocks):
    "Group multi-line candidates into single headings if they belong together (using semantic+layout similarity)."
    grouped = []
    n = len(blocks)
    i = 0
    while i < n:
        block = blocks[i]
        group = [block]
        # Keep grouping next lines if they are semantic-role-equal, close, and highly semantically similar
        while i+1 < n:
            nxt = blocks[i+1]
            if nxt['page']!=block['page']: break
            if abs(nxt['y0']-block['y1'])>1.3*block['font_size']: break # loose vertical connection
            emb1 = sb_model.encode([normalize(block['text'])])[0]
            emb2 = sb_model.encode([normalize(nxt['text'])])[0]
            if util.cos_sim([emb1],[emb2]).item()<0.63: break
            # Require same semantic role
            if nxt.get("role","")!=block.get("role",""): break
            group.append(nxt)
            block = nxt
            i += 1
        group_text = " ".join(b['text'].rstrip() for b in group)
        g = dict(group[0])
        g["text"] = group_text
        # Carry page etc.
        grouped.append(g)
        i += 1
    return grouped

def process_pdf(input_path, output_path):
    doc = fitz.open(input_path)
    all_blocks = get_doc_blocks(doc)
    page_blocks = {}
    # Prepare page_blocks for faster access
    for b in all_blocks:
        page_blocks.setdefault(b['page'],[]).append(b)
    # Assign semantic roles
    for b in all_blocks:
        prev_lines = []
        next_lines = []
        pagelines = page_blocks[b['page']]
        b["role"] = role_classifier(b, all_blocks, pagelines, pagelines, prev_lines, next_lines)
    # Group multi-line titles and headings
    title_blocks = [b for b in all_blocks if b['role']=="title"]
    heading_blocks = [b for b in all_blocks if b['role']=="heading"]
    heading_blocks = group_heading_blocks(heading_blocks)
    # Prune ToC/cover/footer/header blocks; don't include cover page for title unless it's classed as title
    # Prefer single "title" on first page unless multiple valid ones
    title = "No title found"
    if title_blocks:
        title_block = min(title_blocks, key=lambda b: (b['page'], b['y0']))
        title = title_block['text'].strip()
        # Remove if it appears as heading
        heading_blocks = [h for h in heading_blocks if normalize(h['text'])!=normalize(title)]
    # Assign heading levels: very crude, but you can comb font size clusters if you wish, or use classic patterns.
    sizes = sorted({h['font_size'] for h in heading_blocks}, reverse=True)
    level_map = {sz: f"H{i+1}" for i, sz in enumerate(sizes[:3])}
    for h in heading_blocks:
        h['level'] = level_map.get(h['font_size'],"H3")
    # Build outline
    outline = []
    for h in heading_blocks:
        outline.append({
            "level": h["level"],
            "text": h["text"],
            "page": h["page"]
        })
    output = {"title": title, "outline": outline}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"): continue
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".json")
        try:
            process_pdf(input_path, output_path)
        except Exception as e:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"title":"No title found", "outline":[]}, f, ensure_ascii=False)
            print(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    main()
