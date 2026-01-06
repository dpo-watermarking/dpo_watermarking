#!/usr/bin/env python3

"""

Filter DPO pairs using the k-SemStamp k-means margin constraint.
Input JSON must be a list of dicts with keys:

  - prompt

  - chosen

  - rejected

We compute margin(text) = d2 - d1 using sentence embedding and k-means centroids.

We keep only strong pairs:

  chosen_margin >= POS_THRESH

  rejected_margin <= NEG_THRESH

This makes DPO learn the geometry you care about.

Usage:

  python dpo/scripts/filter_pairs_by_margin.py \

    --in_json dpo/data_watermark_1000/train_mixed.json \

    --out_json dpo/data_watermark_1000/train_mixed.json \

    --centroids centroids/booksum-cluster_8_centers.pt \

    --delta 0.035 \

    --pos_thresh 0.060 \

    --neg_thresh 0.020 \

    --max_items 12000

"""


import argparse, json, os

import torch

from tqdm import tqdm

from sentence_transformers import SentenceTransformer



def kmeans_margin(text: str, embedder, centers: torch.Tensor) -> float:

    with torch.no_grad():

        emb = embedder.encode(text, convert_to_tensor=True).unsqueeze(0)  # [1, d]

        dists = torch.cdist(emb, centers)  # [1, k]

        sd, _ = torch.sort(dists, dim=1)

        return (sd[0, 1] - sd[0, 0]).item()



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--in_json", required=True)

    ap.add_argument("--out_json", required=True)

    ap.add_argument("--centroids", required=True)

    ap.add_argument("--delta", type=float, default=0.035)

    ap.add_argument("--pos_thresh", type=float, default=0.060)

    ap.add_argument("--neg_thresh", type=float, default=0.020)

    ap.add_argument("--max_items", type=int, default=12000)

    args = ap.parse_args()



    print(">>> Loading centroids:", args.centroids)

    centers = torch.load(args.centroids, map_location="cpu").float()



    print(">>> Loading embedder (CPU)")

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

    embedder.eval()



    print(">>> Loading pairs:", args.in_json)

    with open(args.in_json, "r") as f:

        data = json.load(f)

    assert isinstance(data, list), "Input JSON must be a list of pairs"



    kept = []

    stats = {"total": 0, "kept": 0}



    for ex in tqdm(data):

        stats["total"] += 1

        prompt = ex.get("prompt", "")

        chosen = ex.get("chosen", "")

        rejected = ex.get("rejected", "")

        if not (prompt and chosen and rejected):

            continue


        mc = kmeans_margin(chosen, embedder, centers)

        mr = kmeans_margin(rejected, embedder, centers)


        # Strict margin separation

        if (mc >= args.pos_thresh) and (mr <= args.neg_thresh):

            ex2 = dict(ex)

            ex2["chosen_margin"] = mc

            ex2["rejected_margin"] = mr

            kept.append(ex2)



        if len(kept) >= args.max_items:

            break



    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.out_json, "w") as f:

        json.dump(kept, f, indent=2)



    print("\n>>> DONE")

    print(">>> total:", stats["total"])

    print(">>> kept :", len(kept))

    print(">>> wrote:", args.out_json)

    print(">>> note: stricter = smaller dataset but much stronger constraint learning")



if __name__ == "__main__":

    main()


