"""
Partially adapted from nl-command
By the courtesy of the authors of:
"On the Blind Spots of Model-Based Evaluation Metrics for Text Generation"
https://arxiv.org/abs/2212.10020
"""

import argparse
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.functional import log_softmax
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from util import load_file_by_line, path_wo_ext, break_text, chunks
import natsort


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[eval_clm] Using device: {device}")


# Text entropy
def text_entropy(sen_lis, k):
    dd, num = {}, 0
    for sen in sen_lis:
        for i in range(len(sen) - k + 1):
            num += 1
            tt = " ".join(sen[i:i + k])
            dd[tt] = dd.get(tt, 0) + 1

    entro = 0.0
    for tt in dd:
        prob = dd[tt] / num
        entro -= prob * math.log(prob)
    return entro


# MLM perplexity (optional)
def mlm_perplexity(model, tokenizer, text, batch_size):
    with torch.no_grad():
        input_ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True
        ).to(device)

        seqlen = input_ids.size(-1)
        expanded = input_ids.expand(seqlen, seqlen).clone()
        masked_idxs = torch.arange(seqlen).to(device)

        expanded[torch.arange(seqlen), masked_idxs] = tokenizer.mask_token_id

        logprobs = []
        for b_idx in chunks(torch.arange(seqlen).to(device), batch_size):
            out = model(expanded[b_idx])
            lp = log_softmax(out.logits, dim=-1)
            orig = input_ids[0][b_idx]
            logprobs.append(lp[torch.arange(len(b_idx)), b_idx, orig])

        logprobs = torch.cat(logprobs)
        nll = logprobs[1:-1].mean()
        ppl = torch.exp(-nll)

    return ppl


# Causal LM perplexity (MAIN)
def eval_perplexity(model, tokenizer, texts, generation_file=None, K=500, name_suffix=""):
    ppls, lengths = [], []

    for text in tqdm(texts, desc="perplexity"):
        text = text.replace("Paraphrase:", "").replace("paraphrase:", "")
        prefix = tokenizer.bos_token or ""

        input_ids = tokenizer.encode(
            prefix + text, return_tensors="pt", truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss
            ppl = torch.exp(nll)

        ppls.append(ppl.item())
        lengths.append(input_ids.size(1))

    ppls = torch.tensor(ppls)
    lengths = torch.tensor(lengths)
    total_len = lengths.sum()

    sent_avg = ppls.mean().item()
    weighted_avg = (ppls * lengths).sum().item() / total_len.item()

    print("\n===== Perplexity =====")
    print(f"Sent Avg PPL     : {sent_avg:.4f}")
    print(f"Weighted Avg PPL : {weighted_avg:.4f}")
    print("======================")

    if generation_file is not None:
        out_path = f"{path_wo_ext(generation_file)}_{name_suffix}.ppl"
        with open(out_path, "w") as f:
            print(f"sent_avg_ppl={sent_avg:.4f}", file=f)
            print(f"weighted_avg_ppl={weighted_avg:.4f}", file=f)

    return sent_avg


# Repetition
def rep_ngram(sen_lis, n):
    reps = []
    for sen in sen_lis:
        seen, total = set(), 0
        for i in range(len(sen) - n + 1):
            total += 1
            seen.add(" ".join(sen[i:i + n]))
        if total > 0:
            reps.append(1.0 - len(seen) / total)
    return np.mean(reps)


def eval_repetition(texts):
    print("\n--- Repetition ---")
    tokens = break_text(texts)
    rep4 = rep_ngram(tokens, 4)
    print(f"4-gram repetition: {rep4:.4f}")
    return rep4


# Args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generation", nargs="+", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-s", "--output_suffix", default="")
    parser.add_argument("-mlm", "--masked_language_model", action="store_true")
    parser.add_argument("--lim", type=int)
    return parser.parse_args()


# Main
if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    if args.masked_language_model:
        model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, pad_token_id=pad_id
        ).to(device)

    model.eval()

    for gen_path in natsort.natsorted(args.generation):
        print(f"\nEvaluating: {gen_path}")

        if os.path.isdir(gen_path):
            ds = load_from_disk(gen_path)
            texts = ds["text"]
        else:
            texts = load_file_by_line(gen_path)

        if args.lim:
            texts = texts[: args.lim]

        ppl = eval_perplexity(
            model,
            tokenizer,
            texts,
            generation_file=gen_path,
            name_suffix=args.output_suffix,
        )

        rep = eval_repetition(texts)

        print(f"\nRESULT → PPL={ppl:.4f}, REP={rep:.4f}")
