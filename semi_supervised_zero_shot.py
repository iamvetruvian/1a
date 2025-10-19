import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

LABELED_CSV = "train_data.csv"
UNLABELED_CSV = "train_data_unlabelled.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
OUTPUT_AUGMENTED = "semi_supervised_labels_zero_shot.csv"
OUTLINE_JSON_DIR = "./semi_supervised_json_outlines_zero_shot"

os.makedirs(OUTLINE_JSON_DIR, exist_ok=True)

print("Loading datasets...")
labeled_df = pd.read_csv(LABELED_CSV)
unlabeled_df = pd.read_csv(UNLABELED_CSV)

# Zero-shot label candidates
label_descs = [
    "This line is the document title.",
    "This line is a main heading.",
    "This line is a section heading.",
    "This line is a sub-section heading.",
    "This line is body text."
]
label_simple = ["title", "H1", "H2", "H3", "body"]

print("Loading sentence transformer and embedding label descriptions...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
label_embs = embedder.encode(label_descs, convert_to_tensor=True)

def zero_shot_pseudo_label(texts):
    text_embs = embedder.encode(texts, convert_to_tensor=True, batch_size=128)
    cos_scores = util.cos_sim(text_embs, label_embs)
    max_indices = cos_scores.argmax(dim=1).cpu().numpy()
    return [label_simple[i] for i in max_indices]

print("Zero-shot pseudo-labeling for unlabeled data ...")
unlabeled_texts = unlabeled_df["text"].astype(str).tolist()
unlabeled_df['pseudo_label'] = zero_shot_pseudo_label(unlabeled_texts)
print("Pseudo-label counts (unlabeled):\n", unlabeled_df['pseudo_label'].value_counts())

labeled_df['pseudo_label'] = labeled_df['label']
combined_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)

def assign_relative_hierarchy(doc_df):
    # Gather candidates by pseudo-label
    candidates = doc_df[doc_df['pseudo_label'].isin(['title', 'H1', 'H2', 'H3'])].copy()
    font_sizes = sorted(candidates['font_size'].unique(), reverse=True)
    mapping = {}

    # Assign document-relative levels robustly
    if len(font_sizes) >= 1:
        mapping[font_sizes[0]] = 'H1'
    if len(font_sizes) >= 2:
        mapping[font_sizes[1]] = 'H2'
    if len(font_sizes) >= 3:
        mapping[font_sizes[2]] = 'H3'

    def get_level(row):
        if row['pseudo_label'] == 'title':
            return 'title'
        if row['pseudo_label'] in ('H1', 'H2', 'H3') and row['font_size'] in mapping:
            return mapping[row['font_size']]
        return 'body'

    doc_df['final_level'] = doc_df.apply(get_level, axis=1)
    doc_df.loc[~doc_df['final_level'].isin(['title', 'H1', 'H2', 'H3']), 'final_level'] = 'body'
    return doc_df

print("Applying robust document-level heading normalization ...")
final_docdfs = []
for pdf_file, group in tqdm(combined_df.groupby('file'), desc="Normalizing PDFs"):
    final_docdfs.append(assign_relative_hierarchy(group.copy()))
final_df = pd.concat(final_docdfs, ignore_index=True)
final_df.to_csv(OUTPUT_AUGMENTED, index=False)
print(f"Saved: {OUTPUT_AUGMENTED}")

def save_outline_json(df, outdir):
    for pdf_file, group in df.groupby("file"):
        title_row = group[group['final_level'] == "title"]
        title = title_row['text'].iloc[0] if len(title_row) else "No title found"
        outline = [
            {
                "level": row['final_level'],
                "text": row['text'],
                "page": int(row['page']) + 1
            }
            for _, row in group[group['final_level'].isin(['H1', 'H2', 'H3'])].iterrows()
        ]
        outline_json = {"title": title, "outline": outline}
        out_path = os.path.join(outdir, f"{os.path.splitext(pdf_file)[0]}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(outline_json, f, indent=2, ensure_ascii=False)
        print(f"Saved outline JSON for {pdf_file} ({len(outline)} headings)")

save_outline_json(final_df, OUTLINE_JSON_DIR)
print(f"All outlines saved in {OUTLINE_JSON_DIR}")
