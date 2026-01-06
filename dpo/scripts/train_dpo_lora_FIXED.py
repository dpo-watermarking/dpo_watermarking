#!/usr/bin/env python3

"""
LoRA-DPO with BF16 (FIXED VERSION)
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

BASE_MODEL = "meta-llama/llama-2-7b-hf"
DATA_PATH  = "dpo/data_watermark_1000/train_mixed.json"
OUT_DIR    = "models/dpo_lora_attn_BF16_WORKING"


MAX_PROMPT_LEN = 256
MAX_LEN        = 512
BETA           = 0.1
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    print(">>> Loading dataset:", DATA_PATH)

    dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]

    required = {"prompt", "chosen", "rejected"}

    if not required.issubset(dataset.column_names):

        raise ValueError(f"Missing keys: {required}")

    
    print(">>> Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    tokenizer.pad_token = tokenizer.eos_token

    
    print(">>> Loading base model (BF16)")

    base_model = AutoModelForCausalLM.from_pretrained(

        BASE_MODEL,

        torch_dtype=torch.bfloat16,

        device_map="auto",

        low_cpu_mem_usage=True,

    )

    base_model.config.use_cache = False

    

    print(">>> Applying LoRA")

    lora_config = LoraConfig(

        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        task_type=TaskType.CAUSAL_LM,

        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    )


    model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    
    print(">>> Loading reference model (BF16)")

    ref_model = AutoModelForCausalLM.from_pretrained(

        BASE_MODEL,

        torch_dtype=torch.bfloat16,

        device_map="auto",

        low_cpu_mem_usage=True,

    )

    ref_model.eval()

    

    print(">>> Configuring DPO with BF16")

    training_args = DPOConfig(

        output_dir=OUT_DIR,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=4,

        num_train_epochs=1,

        learning_rate=1e-5,

        beta=BETA,

        logging_steps=10,

        save_steps=500,

        save_strategy="steps",

        max_grad_norm=1.0,

        bf16=True,

        fp16=False,

        dataloader_num_workers=0,

        gradient_checkpointing=False,

        optim="adamw_torch",

        max_length=MAX_LEN,

        max_prompt_length=MAX_PROMPT_LEN,

        remove_unused_columns=False,

        report_to=[],

    )

    
    print(">>> Starting DPO training")

    trainer = DPOTrainer(

        model=model,

        ref_model=ref_model,

        args=training_args,

        train_dataset=dataset,

        processing_class=tokenizer,

    )


    trainer.train()

    print(">>> Saving model")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(">>> DONE!")


if __name__ == "__main__":

    main()