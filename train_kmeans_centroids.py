import os
import argparse
import torch
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embedder", type=str, required=True)
    parser.add_argument("--cluster_size", type=int, default=256)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    print("Loading dataset...")
    ds = load_from_disk(args.dataset)
    texts = ds["text"]

    print("Loading embedder...")
    model = SentenceTransformer(args.embedder, device="cuda")

    print("Encoding texts...")
    embeddings = []
    for t in tqdm(texts):
        emb = model.encode(t, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")
    dim = embeddings.shape[1]

    print(f"Training k-means (k={args.cluster_size}, dim={dim})...")
    kmeans = faiss.Kmeans(
        dim,
        args.cluster_size,
        niter=50,
        verbose=True,
        gpu=True
    )
    kmeans.train(embeddings)

    centroids = torch.from_numpy(kmeans.centroids)

    save_file = os.path.join(
        args.save_path,
        f"c4-cluster_{args.cluster_size}_centers.pt"
    )
    torch.save(centroids, save_file)

    print(f"✅ Saved centroids to {save_file}")
    print(f"Centroid shape: {centroids.shape}")


if __name__ == "__main__":
    main()
