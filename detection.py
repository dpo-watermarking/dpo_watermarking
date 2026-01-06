import argparse
import os
import torch
import numpy as np
import pandas as pd

from tqdm import trange
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from sbert_lsh_model import SBERTLSHModel
from detection_utils import (
    detect_kmeans,
    detect_lsh,
    run_bert_score,
    evaluate_z_scores,
    flatten_gens_and_paras
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="HF dataset with generated text")
    parser.add_argument("--human_text", required=True, help="HF dataset with human/plain text")
    parser.add_argument("--detection_mode", choices=["kmeans", "lsh"], required=True)
    parser.add_argument("--cc_path", type=str, help="cluster centers (kmeans only)")
    parser.add_argument("--embedder", type=str, required=True)
    parser.add_argument("--sp_dim", type=int, default=8)
    parser.add_argument("--lmbd", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -----------------------
    # Load datasets
    # -----------------------
    dataset = load_from_disk(args.dataset_path)
    gens = dataset["text"]

    paras = dataset["para_text"] if "para_text" in dataset.column_names else None
    human_texts = load_from_disk(args.human_text)["text"][: len(gens)]

    z_scores, para_scores, human_scores = [], [], []

    # k-SEMSTAMP (kmeans)
    if args.detection_mode == "kmeans":
        assert args.cc_path is not None, "cc_path is required for kmeans detection"

        cluster_centers = torch.load(args.cc_path)
        embedder = SentenceTransformer(args.embedder)

        # --- Generated text ---
        for i in trange(len(gens), desc="kmeans_detection"):
            gen_sents = gens[i] if isinstance(gens[i], list) else sent_tokenize(gens[i])

            z = detect_kmeans(
                sents=gen_sents,
                embedder=embedder,
                lmbd=args.lmbd,
                k_dim=args.sp_dim,
                cluster_centers=cluster_centers,
            )
            z_scores.append(z)

            if paras is not None:
                para_sents = paras[i] if isinstance(paras[i], list) else sent_tokenize(paras[i])
                pz = detect_kmeans(
                    sents=para_sents,
                    embedder=embedder,
                    lmbd=args.lmbd,
                    k_dim=args.sp_dim,
                    cluster_centers=cluster_centers,
                )
                para_scores.append(pz)

        # --- Human text ---
        for i in trange(len(human_texts), desc="kmeans_human"):
            sents = sent_tokenize(human_texts[i])
            z = detect_kmeans(
                sents=sents,
                embedder=embedder,
                lmbd=args.lmbd,
                k_dim=args.sp_dim,
                cluster_centers=cluster_centers,
            )
            human_scores.append(z)

    # SEMSTAMP (LSH)
    elif args.detection_mode == "lsh":
        lsh_model = SBERTLSHModel(
            lsh_model_path=args.embedder,
            device="cuda",
            batch_size=1,
            lsh_dim=args.sp_dim,
            sbert_type="base",
        )

        for i in trange(len(gens), desc="lsh_detection"):
            gen_sents = gens[i] if isinstance(gens[i], list) else sent_tokenize(gens[i])
            z = detect_lsh(
                sents=gen_sents,
                lsh_model=lsh_model,
                lmbd=args.lmbd,
                lsh_dim=args.sp_dim,
            )
            z_scores.append(z)

            if paras is not None:
                para_sents = paras[i] if isinstance(paras[i], list) else sent_tokenize(paras[i])
                pz = detect_lsh(
                    sents=para_sents,
                    lsh_model=lsh_model,
                    lmbd=args.lmbd,
                    lsh_dim=args.sp_dim,
                )
                para_scores.append(pz)

        for i in trange(len(human_texts), desc="lsh_human"):
            sents = sent_tokenize(human_texts[i])
            z = detect_lsh(
                sents=sents,
                lsh_model=lsh_model,
                lmbd=args.lmbd,
                lsh_dim=args.sp_dim,
            )
            human_scores.append(z)

    # Save raw scores
    np.save(os.path.join(args.dataset_path, "z_scores.npy"), z_scores)
    np.save(os.path.join(args.dataset_path, "para_z_scores.npy"), para_scores)
    np.save(os.path.join(args.dataset_path, "human_z_scores.npy"), human_scores)


    # Metrics (paper-style)
    print("Evaluating z-scores...")

    if len(para_scores) == 0:
         para_scores = None

    auroc, fpr1, fpr5 = evaluate_z_scores(
       z_scores, para_scores, human_scores, args.dataset_path
    )


    print("Evaluating BERTScore...")
    gen_sents, para_sents = flatten_gens_and_paras(gens, paras)
    bert = run_bert_score(gen_sents, para_sents)

    df = pd.DataFrame(
        [[f"{auroc:.3f}", f"{fpr1:.3f}", f"{fpr5:.3f}", f"{bert:.3f}"]],
        columns=["auroc", "fpr1", "fpr5", "bert_score"],
    )

    out_path = os.path.join(args.dataset_path, "results.csv")
    df.to_csv(out_path, sep="\t", index=False)
    print("Saved:", out_path)
