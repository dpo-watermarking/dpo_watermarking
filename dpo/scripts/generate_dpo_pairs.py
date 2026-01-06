#!/usr/bin/env python3
"""
Generate text using a trained DPO LoRA model (evaluation phase)
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def extract_prompt(text, n_sent=3):
    sents = text.split(". ")
    prompt = ". ".join(sents[:n_sent]).strip()
    if not prompt.endswith("."):
        prompt += "."
    return prompt + " "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading dataset...")
    ds = load_from_disk(args.data_path)
    ds = ds.select(range(min(args.num_prompts, len(ds))))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    outputs = []

    print("Generating text...")
    for i, ex in enumerate(tqdm(ds)):
        prompt = extract_prompt(ex["text"])

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_k=20,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        outputs.append({
            "id": i,
            "prompt": prompt,
            "generation": gen_text
        })

    out_path = os.path.join(args.out_dir, "dpo_generations.json")
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"\nDONE. Saved {len(outputs)} generations to:")
    print(out_path)


if __name__ == "__main__":
    main()
