import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from datasets import load_from_disk

# Config
WM_TXT = "dpo/generated_text/booksum_dpo_mixed_1000/generated_clean.txt"
HUMAN_DS = "semstamp-data/original-booksum-texts"

MAX_HUMAN = 1000   
SEED = 42

# Load texts
def load_txt_lines(path):

    with open(path, "r") as f:

        return [l.strip() for l in f if len(l.strip()) > 0]

wm_texts = load_txt_lines(WM_TXT)

print(f"WM samples: {len(wm_texts)}")

human_ds = load_from_disk(HUMAN_DS)

human_texts = []

for ex in human_ds:

    if "text" in ex and isinstance(ex["text"], str):

        human_texts.append(ex["text"])

    if len(human_texts) >= MAX_HUMAN:

        break

print(f"Human samples: {len(human_texts)}")

# Labels
X_texts = wm_texts + human_texts

y = np.array([1] * len(wm_texts) + [0] * len(human_texts))

# Embedder
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

embedder = SentenceTransformer(

    "sentence-transformers/all-mpnet-base-v2",

    device=device

)


# Encode
X = embedder.encode(
    X_texts,
    batch_size=32,
    convert_to_numpy=True,
    show_progress_bar=True

)

# Train detector
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED

)

clf.fit(X, y)


# Evaluation
scores = clf.predict_proba(X)[:, 1]

auc = roc_auc_score(y, scores)
fpr, tpr, _ = roc_curve(y, scores)


print("\n===== DPO-AWARE DETECTION RESULTS =====")
print(f"AUC: {auc:.4f}")
print("TPR @ 1% FPR:", tpr[fpr <= 0.01][-1])
print("TPR @ 5% FPR:", tpr[fpr <= 0.05][-1])
print("======================================\n")