from sklearn.metrics import roc_curve, auc
import sampling_utils
from sampling_lsh_utils import get_mask_from_seed
from sampling_kmeans_utils import get_cluster_mask, get_cluster_id
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)


# BERTScore DISABLED 
def run_bert_score(gen_sents, para_sents):
    """
    Stub for BERTScore.
    We intentionally disable it to avoid torch security issues.
    """
    return None



# Utility helpers
def flatten_gens_and_paras(gens, paras):
    new_gens = []
    new_paras = []
    for gen, para in zip(gens, paras):
        min_len = min(len(gen), len(para))
        new_gens.extend(gen[:min_len])
        new_paras.extend(para[:min_len])
    return new_gens, new_paras


def truncate_to_max_length(texts, max_length):
    new_texts = []
    for t in texts:
        t = " ".join(t.split(" ")[:max_length])
        if t[-1] not in sampling_utils.PUNCTS:
            t = t + "."
        new_texts.append(t)
    return new_texts


# k-SemStamp detection (k-means)
def detect_kmeans(sents, embedder, lmbd, k_dim, cluster_centers):
    n_sent = len(sents)
    n_watermark = 0

    curr_cluster_id = get_cluster_id(
        sents[0], embedder=embedder, cluster_centers=cluster_centers
    )
    cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)

    for i in range(1, n_sent):
        curr_cluster_id = get_cluster_id(
            sents[i], embedder=embedder, cluster_centers=cluster_centers
        )
        if curr_cluster_id in cluster_mask:
            n_watermark += 1
        cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)

    n_test_sent = n_sent - 1  # exclude prompt
    num = n_watermark - lmbd * n_test_sent
    denom = np.sqrt(n_test_sent * lmbd * (1 - lmbd))

    return num / denom



# LSH detection (kept for completeness)
def detect_lsh(sents, lsh_model, lmbd, lsh_dim, cutoff=None):
    if cutoff is None:
        cutoff = lsh_dim

    n_sent = len(sents)
    n_watermark = 0

    lsh_seed = lsh_model.get_hash([sents[0]])[0]
    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)

    for i in range(1, n_sent):
        lsh_candidate = lsh_model.get_hash([sents[i]])[0]
        if lsh_candidate in accept_mask:
            n_watermark += 1
        lsh_seed = lsh_candidate
        accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)

    n_test_sent = n_sent - 1
    num = n_watermark - lmbd * n_test_sent
    denom = np.sqrt(n_test_sent * lmbd * (1 - lmbd))

    return num / denom



# ROC helpers
def get_roc_metrics(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_roc_metrics_from_zscores(m, mp, h, dataset_path):
    mp = np.nan_to_num(mp)
    h = np.nan_to_num(h)
    n = len(mp)

    labels = [1] * n + [0] * n
    preds = np.concatenate((mp, h[:n]))

    fpr, tpr, roc_auc = get_roc_metrics(labels, preds)

    print("\n====== ROC METRICS ======")
    print(f"AUC: {roc_auc:.4f}")
    print("=========================\n")

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(os.path.join(dataset_path, "roc_curve.png"))

    np.save(os.path.join(dataset_path, "fpr.npy"), fpr)
    np.save(os.path.join(dataset_path, "tpr.npy"), tpr)

    return roc_auc, fpr



# FINAL evaluation
def evaluate_z_scores(mz, mpz, hz, dataset_path):
    mz = np.array(mz, dtype=float)
    hz = np.array(hz, dtype=float)

    if mpz is None or len(mpz) == 0:
        mpz = mz
    else:
        mpz = np.array(mpz, dtype=float)

    mz = mz[~np.isnan(mz)]
    mpz = mpz[~np.isnan(mpz)]
    hz = hz[~np.isnan(hz)]

    assert len(mz) > 0 and len(hz) > 0, "Empty z-score arrays"

    # --------- Threshold calibration from HUMAN texts ---------
    fpr_1_thresh = 2.33
    fpr_5_thresh = 1.64

    for z in np.arange(0, 6, 0.005):
        fp = np.mean(hz > z)
        if 0.0095 <= fp <= 0.0105:
            fpr_1_thresh = z
        if 0.045 <= fp <= 0.055:
            fpr_5_thresh = z

    tpr_1 = np.mean(mpz > fpr_1_thresh)
    tpr_5 = np.mean(mpz > fpr_5_thresh)

    labels = np.concatenate([np.ones(len(mz)), np.zeros(len(hz))])
    scores = np.concatenate([mz, hz])

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(os.path.join(dataset_path, "roc_curve.png"))
    plt.close()

    np.save(os.path.join(dataset_path, "fpr.npy"), fpr)
    np.save(os.path.join(dataset_path, "tpr.npy"), tpr)

    print("\n===== K-SEMSTAMP DETECTION RESULTS =====")
    print(f"AUC          : {auc_score:.4f}")
    print(f"TPR @ 1% FPR : {tpr_1:.4f}")
    print(f"TPR @ 5% FPR : {tpr_5:.4f}")
    print("=======================================\n")

    return auc_score, tpr_1, tpr_5