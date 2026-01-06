#!/usr/bin/env python3

"""
DPO training: hard + random (watermark_1000)
"""

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "meta-llama/llama-2-7b-hf"
DATA_PATH  = "dpo/watermark_1000/dpo_mixed_1000.json"
OUT_DIR    = "models/dpo_mixed_1000"

def main():

    print(">>> DPO MIXED 1000 STARTED")


    # Load data
    pairs = json.load(open(DATA_PATH))
    print(f">>> Loaded {len(pairs)} pairs")

    dataset = Dataset.from_list(pairs)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


    # DPO config
    training_args = DPOConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
        beta=0.1,
        max_length=512,
        max_prompt_length=256,

    )

    # Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(">>> Training started")
    trainer.train()

    print(">>> Saving model")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(">>> DPO MIXED 1000 DONE")


if __name__ == "__main__":

    main()
