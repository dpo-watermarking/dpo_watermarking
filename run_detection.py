import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sentence_transformers import SentenceTransformer
from sampling_kmeans_utils import kmeans_reject_overlap

# CONFIG
DELTA = 0.035
BATCH_SIZE = 64

WM_PATH = "dpo/generated_text/booksum_dpo_mixed_1000/generated_paraphrased.txt"
HUMAN_PATH = "semstamp-data/original-booksum-texts"
CENTROIDS_PATH = "centroids/booksum-cluster_8_centers.pt"

# Load text helpers
def load_txt(path):
    with open(path) as f:
        return [l.strip() for l in f if len(l.strip()) > 0]


# Load texts
wm_texts = load_txt(WM_PATH)

from datasets import load_from_disk
human_ds = load_from_disk(HUMAN_PATH)
human_texts = [ex["text"] for ex in human_ds if isinstance(ex["text"], str)]

print(f"WM samples: {len(wm_texts)}")
print(f"Human samples: {len(human_texts)}")


# Embedder
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=device
)

# Centroids
cluster_centers = torch.load(
    CENTROIDS_PATH,
    map_location=device
)


# Batched scoring
def compute_scores(texts):
    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embs = embedder.encode(
            batch,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
        for j, emb in enumerate(embs):
            _, score = kmeans_reject_overlap(
                batch[j],
                embedder,
                cluster_centers,
                DELTA
            )
            scores.append(score)
        if i % (5 * BATCH_SIZE) == 0:
            print(f"Processed {i}/{len(texts)}")
    return np.array(scores)

wm_scores = compute_scores(wm_texts)
human_scores = compute_scores(human_texts)

# Metrics
y_true = np.array([1] * len(wm_scores) + [0] * len(human_scores))
y_scores = np.concatenate([wm_scores, human_scores])

auc = roc_auc_score(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)

print("\n===== DETECTION RESULTS =====")
print(f"AUC: {auc:.4f}")
print(f"TPR @ 1% FPR: {tpr[fpr <= 0.01][-1]:.4f}")
print(f"TPR @ 5% FPR: {tpr[fpr <= 0.05][-1]:.4f}")
print("============================\n")