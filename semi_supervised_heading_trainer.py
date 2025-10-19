import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter

LABELED_CSV = "train_data.csv"
UNLABELED_CSV = "train_data_unlabelled.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
OUTPUT_AUGMENTED = "semi_supervised_labels.csv"
OUTLINE_JSON_DIR = "./semi_supervised_json_outlines"

import os
os.makedirs(OUTLINE_JSON_DIR, exist_ok=True)

# 1. Load datasets
labeled_df = pd.read_csv(LABELED_CSV)
unlabeled_df = pd.read_csv(UNLABELED_CSV)

# 2. Compute sentence embeddings for both
embedder = SentenceTransformer(EMBEDDING_MODEL)
def compute_embeddings(texts, batch_size=128):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs.append(embedder.encode(batch, show_progress_bar=False))
    return np.vstack(embs)

print("Embedding labeled lines ...")
labeled_embs = compute_embeddings(labeled_df["text"].astype(str).tolist())
print("Embedding unlabeled lines ...")
unlabeled_embs = compute_embeddings(unlabeled_df["text"].astype(str).tolist())

for idx, df in enumerate([labeled_df, unlabeled_df]):
    embs = labeled_embs if idx == 0 else unlabeled_embs
    for i in range(embs.shape[1]):
        df[f'emb_{i}'] = embs[:, i]

# 3. Concatenate for clustering
combined_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
emb_cols = [c for c in combined_df.columns if c.startswith("emb_")]

# 4. Cluster all to form groups (pseudo-classes)
n_clusters = 4  # title, H1, H2, body (adjust if you expect more)
print("Clustering with KMeans ...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
combined_df['cluster'] = kmeans.fit_predict(combined_df[emb_cols])

# 5. Pseudo-label: assign most common true label in each cluster
label_map = {}
labeled_clusters = combined_df.loc[combined_df['label'].notnull(), ['cluster','label']]
for cluster in labeled_clusters['cluster'].unique():
    votes = labeled_clusters.loc[labeled_clusters['cluster'] == cluster, 'label']
    if not votes.empty:
        label_map[cluster] = Counter(votes).most_common(1)[0][0]
    else:
        label_map[cluster] = "body"
combined_df['pseudo_label'] = combined_df['cluster'].map(label_map)

# 6. (Core) Normalize heading hierarchy PER DOCUMENT/FILE
def assign_relative_hierarchy(doc_df):
    # Only consider heading candidates (ignore 'body')
    heading_mask = doc_df['pseudo_label'].isin(['title', 'H1', 'H2', 'H3'])
    heading_df = doc_df[heading_mask]
    sizes = heading_df['font_size'].unique()
    font_groups = sorted(sizes, reverse=True)
    # Map
    if len(font_groups) == 0:
        return doc_df.assign(final_level="body")
    elif len(font_groups) == 1:
        for idx in heading_df.index:
            doc_df.at[idx, 'final_level'] = 'H1'
    elif len(font_groups) == 2:
        for idx in heading_df.index:
            fsize = doc_df.at[idx, 'font_size']
            doc_df.at[idx, 'final_level'] = 'H1' if fsize == font_groups[0] else 'H2'
    elif len(font_groups) >=3:
        for idx in heading_df.index:
            fsize = doc_df.at[idx, 'font_size']
            if fsize == font_groups[0]:
                doc_df.at[idx, 'final_level'] = 'H1'
            elif fsize == font_groups[1]:
                doc_df.at[idx, 'final_level'] = 'H2'
            else:
                doc_df.at[idx, 'final_level'] = 'H3'
    doc_df['final_level'] = doc_df['final_level'].fillna('body')
    return doc_df

final_docdfs = []
for pdf_file, group in combined_df.groupby('file'):
    final_docdfs.append(assign_relative_hierarchy(group.copy()))
final_df = pd.concat(final_docdfs, ignore_index=True)
final_df.to_csv(OUTPUT_AUGMENTED, index=False)
print(f"Data with pseudo-labels and normalized hierarchy saved to {OUTPUT_AUGMENTED}")

# 7. Export heading outline as JSON per PDF
def save_outline_json(df, outdir):
    for pdf_file, group in df.groupby("file"):
        # Find title, else "No title found"
        title = None
        # Page numbers are 0-based in extract_data.py, so +1
        for idx, row in group.iterrows():
            if row['final_level'] == 'title':
                title = row['text']
                break
        if not title:
            title = "No title found"
        outline = []
        for idx, row in group[group['final_level'].isin(['H1','H2','H3'])].iterrows():
            outline.append({
                "level": row['final_level'],
                "text": row['text'],
                "page": int(row['page'])+1
            })
        result = {
            "title": title,
            "outline": outline
        }
        outpath = os.path.join(outdir, f"{os.path.splitext(pdf_file)[0]}.json")
        import json
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved {outpath}")

save_outline_json(final_df, OUTLINE_JSON_DIR)
print(f"All outlines exported to {OUTLINE_JSON_DIR}")

