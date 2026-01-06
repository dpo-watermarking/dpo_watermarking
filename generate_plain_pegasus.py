#!/usr/bin/env python3



from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from datasets import load_from_disk

import torch

from tqdm import tqdm



# ------ CONFIG -------

MODEL_NAME = "google/pegasus-arxiv"

INPUT_DATASET = "dpo/generated_text/booksum_dpo_mixed_1000/generated_clean.txt"

OUTPUT_FILE = "dpo/generated_text/booksum_dpo_mixed_1000/generated_paraphrased.txt"

DEVICE = "cpu"   # FORCE CPU — DO NOT CHANGE
MAX_NEW_TOKENS = 60
NUM_BEAMS = 5

def load_lines(path):

    with open(path, "r") as f:

        lines = [l.strip() for l in f.readlines() if l.strip()]

    return lines

def main():

    print(f"Using device: {DEVICE}")
    print("Loading Pegasus tokenizer...")
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    print("Loading Pegasus model...")
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()

    print("Loading input text...")
    texts = load_lines(INPUT_DATASET)
    outputs = []
    for text in tqdm(texts):

        inputs = tokenizer(

            text,

            truncation=True,

            padding=True,

            return_tensors="pt"

        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():

            gen = model.generate(

                **inputs,

                num_beams=NUM_BEAMS,

                max_new_tokens=MAX_NEW_TOKENS,

                early_stopping=True

            )

        paraphrased = tokenizer.decode(

            gen[0],

            skip_special_tokens=True

        )
        
        outputs.append(paraphrased)

    print("Saving paraphrased output...")

    with open(OUTPUT_FILE, "w") as f:

        for line in outputs:

            f.write(line + "\n")

    print("DONE!")

    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":

    main()