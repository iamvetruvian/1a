import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import joblib
from tqdm import tqdm

TRAIN_CSV = 'train_data.csv'
MODEL_PKL = 'heading_classifier.pkl'
LABEL_ENCODER_PKL = 'label_encoder.pkl'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L12-v2'

print("Loading data...")

df = pd.read_csv(TRAIN_CSV)

# Ensure these columns are present, set default if missing
# (you may want to verify your extract_data.py generates them)
for col in ['is_bold', 'is_italic', 'is_underline']:
    if col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = df[col].apply(lambda x: 1 if x in [True, 'True', 'true', '1'] else 0)
    else:
        df[col] = 0

print("Encoding labels...")
label_enc = LabelEncoder()
df['label_enc'] = label_enc.fit_transform(df['label'])

print("Computing semantic text embeddings ...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
batch_size = 128

def batched_embeddings(texts, batch_size=128):
    embeddings = []
    n = len(texts)
    for i in tqdm(range(0, n, batch_size), desc="Embedding batches"):
        emb = embedder.encode(texts[i:i+batch_size], show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)

embs = batched_embeddings(df['text'].astype(str).tolist(), batch_size=batch_size)
emb_cols = [f'emb_{i}' for i in range(embs.shape[1])]
emb_df = pd.DataFrame(embs, columns=emb_cols, index=df.index)
df = pd.concat([df, emb_df], axis=1, copy=False)

# Feature selection: feel free to expand further with new columns!
feature_cols = [
    'font_size', 'is_bold', 'x0', 'y0',
    'is_italic', 'is_underline',  # NEW
    'text_length', 'stopword_ratio'  # NEW
] + emb_cols
# You might also try:
#  'color', 'casing', 'font_family'

X = df[feature_cols]
y = df['label_enc']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print("Training LightGBM classifier ...")
lgbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.07,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    reg_alpha=1.0,
    reg_lambda=1.0,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    class_weight='balanced',
    early_stopping_rounds=30,
    verbose=50
)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
)

print("\nValidation results:")
y_pred_val = lgbm.predict(X_val)
present_label_indexes = np.unique(y_val)
present_label_names = label_enc.inverse_transform(present_label_indexes)
print(classification_report(
    y_val, y_pred_val,
    labels=present_label_indexes, target_names=present_label_names
))

# Cross-validation (optional, for deeper assessment)
print("\nCross-validating with 5 folds...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, test_idx in skf.split(X, y):
    x_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    x_val, y_val_ = X.iloc[test_idx], y.iloc[test_idx]
    lgbm_cv = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.07,
        max_depth=8,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        class_weight='balanced',
        early_stopping_rounds=30,
        verbose=-1
    )
    lgbm_cv.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val_)],
        eval_metric='multi_logloss',
    )
    preds = lgbm_cv.predict(x_val)
    acc = np.mean(preds == y_val_)
    cv_scores.append(acc)
print(f'Mean 5-fold CV accuracy: {np.mean(cv_scores):.4f} | Per-fold: {cv_scores}')

# ---- Save model and label encoder ----
joblib.dump(lgbm, MODEL_PKL)
joblib.dump(label_enc, LABEL_ENCODER_PKL)
print(f"Model saved to {MODEL_PKL}")
print(f"Label encoder saved to {LABEL_ENCODER_PKL}")
print("Training complete and ready to use on unseen PDFs!")