#!/usr/bin/env python3
"""
Token-level constrained decoding using trained LoRA-DPO model.

"""

import os, json, argparse, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

# Make repo importable so we can reuse their watermark check
sys.path.insert(0, ".")
from sampling_kmeans_utils import kmeans_reject_overlap

# semantic check frequency
CHECK_EVERY = 8

@torch.no_grad()
def token_level_generate(
    model,
    tokenizer,
    embedder,
    clusters,
    prompt: str,
    delta: float,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
):
    device = model.device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    generated_ids = inputs["input_ids"].clone()
    token_trials = 0

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits[:, -1, :]

        if temperature and temperature > 0:
            logits = logits / temperature

        top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(top_vals, dim=-1)[0]

        # Gumbel trick for stochastic ordering
        u = torch.rand_like(probs)
        scores = torch.log(probs + 1e-12) - torch.log(-torch.log(u + 1e-12) + 1e-12)
        try_order = torch.argsort(scores, descending=True).tolist()

        chosen = None

        for j in try_order:
            cand_id = top_idx[0, j].item()
            token_trials += 1

            cand_ids = torch.cat(
                [generated_ids, torch.tensor([[cand_id]], device=device)], dim=1
            )
            cand_text = tokenizer.decode(cand_ids[0], skip_special_tokens=True)

            # CHECK BASED ON *CANDIDATE* LENGTH
            do_check = (cand_ids.shape[1] % CHECK_EVERY == 0)

            if do_check:
                ok, _ = kmeans_reject_overlap(
                    cand_text, embedder, clusters, delta=delta
                )
            else:
                ok = True

            if ok:
                chosen = cand_id
                generated_ids = cand_ids
                break

        if chosen is None:
            final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return final_text, token_trials, False

        if tokenizer.eos_token_id is not None and chosen == tokenizer.eos_token_id:
            break

    final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return final_text, token_trials, True


def pick_prompt(example):
    if isinstance(example, dict):
        if "prompt" in example:
            return example["prompt"]
        if "text" in example:
            return example["text"][:500]
        if "document" in example:
            return str(example["document"])[:500]
    raise ValueError("Cannot find prompt field")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/llama-2-7b-hf")
    ap.add_argument("--lora_path", default="models/dpo_lora_attn_BF16_WORKING")
    ap.add_argument("--centroid_path", default="centroids/booksum-cluster_8_centers.pt")
    ap.add_argument("--data_path", default="semstamp-data/booksum-vanilla")
    ap.add_argument("--out_dir", default="evaluation/token_decode_run1")
    ap.add_argument("--num_prompts", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.035)
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)

    # FIXED DEFAULT: MiniLM
    ap.add_argument(
        "--embedder_name",
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(">>> Loading base model:", args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    base.eval()

    print(">>> Loading LoRA adapters:", args.lora_path)
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading embedder:", args.embedder_name)
    embedder = SentenceTransformer(
        args.embedder_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(">>> Loading clusters:", args.centroid_path)
    clusters = torch.load(args.centroid_path, map_location="cpu")

    print(">>> Loading dataset:", args.data_path)
    ds = load_from_disk(args.data_path)
    n = min(args.num_prompts, len(ds))
    ds = ds.select(range(n))

    results = []
    accepted = 0
    total_token_trials = 0
    failed = 0

    for i, ex in enumerate(ds):
        prompt = pick_prompt(ex)

        text, trials, ok = token_level_generate(
            model=model,
            tokenizer=tokenizer,
            embedder=embedder,
            clusters=clusters,
            prompt=prompt,
            delta=args.delta,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
        )

        results.append({
            "i": i,
            "accepted": ok,
            "token_trials": trials,
            "text_preview": text[:250],
        })

        total_token_trials += trials
        accepted += int(ok)
        failed += int(not ok)

        if (i + 1) % 5 == 0:
            print(
                f"[{i+1}/{n}] accepted={accepted}, "
                f"failed={failed}, "
                f"avg_token_trials_per_accept={total_token_trials/max(1,accepted):.2f}"
            )

    summary = {
        "num_prompts": n,
        "accepted": accepted,
        "failed": failed,
        "accept_rate": accepted / max(1, n),
        "total_token_trials": total_token_trials,
        "avg_token_trials_per_prompt": total_token_trials / max(1, n),
        "avg_token_trials_per_accept": total_token_trials / max(1, accepted),
        "equiv_sentence_attempts": (total_token_trials / max(1, accepted)) / args.max_new_tokens,
        "delta": args.delta,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "lora_path": args.lora_path,
        "base_model": args.base_model,
        "embedder_name": args.embedder_name,
        "check_every": CHECK_EVERY,
    }

    with open(os.path.join(args.out_dir, "samples.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
  

if __name__ == "__main__":
    main()
