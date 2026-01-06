# Distilling Semantic Watermarks: Accelerating Text Generation via Preference Optimization

**Official implementation for eliminating rejection sampling in semantic watermarking through Direct Preference Optimization**

## Overview

This repository extends k-SEMSTAMP (ACL 2024) by transferring watermark constraints directly into the language model via DPO training. Our approach eliminates costly rejection sampling while maintaining detection robustness.

**Key Result:** We eliminate k-SEMSTAMP's rejection sampling bottleneck by internalizing watermark constraints through DPO training, achieving an 11.4× efficiency improvement cost while maintaining robust detection. This transforms watermarking from costly trial-and-error sampling into efficient preference-aligned generation.

---

## Quick Start

### Environment Setup
```bash
conda create -n ksem_dpo python=3.10 -y
conda activate ksem_dpo
pip install -r requirements.txt
```

### Data Preparation
```bash
python load_c4.py
python train_val_test_split.py
```
Data stored in `semstamp-data/`

### Train Semantic Clusters
```bash
python train_kmeans_centroids.py
```
Outputs k-means centroids to `centroids/booksum-cluster_8_centers.pt`

---

## DPO Training Pipeline

### 1. Generate Candidate Sentences
```bash
python dpo/scripts/generate_dpo_pairs.py
```

### 2. Construct Preference Pairs
Each candidate is embedded, assigned to nearest cluster, and labeled as valid/rejected based on watermark constraints.

**Negative sampling strategies:**
- **Random:** uniformly sampled rejected sentences
- **Hard:** nearest rejected sentences in embedding space
- **Mixed:** combination (used in paper)

```bash
python dpo/scripts/filter_pairs_by_margin.py
```
Output: `dpo/data_watermark_1000/train_mixed.json`

### 3. DPO + LoRA Fine-tuning
```bash
python dpo/scripts/train_dpo_lora_FIXED.py
```
Trains LLaMA-2-7B with LoRA (rank=16). Best checkpoint: `models/dpo_lora_attn_BF16_WORKING/checkpoint-2434`

---

## Evaluation

### Generation (No Rejection Sampling)
```bash
python dpo/scripts/generate_with_dpo_lora.py
```
Outputs to `dpo/generated_text/booksum_dpo_mixed_1000/`

### Paraphrase Attack
```bash
python generate_plain_pegasus.py \
  --input generated_clean.txt \
  --output generated_paraphrased.txt
```

### Detection Performance
```bash
python run_detection_dpo_lr.py
```
**Metrics:** AUROC, TPR@1%FPR, TPR@5%FPR

### Efficiency Analysis
```bash
python dpo/scripts/token_decode_efficiency.py
```
Measures total tokens generated per accepted sentence. Results in `evaluation/`

### Text Quality
```bash
# Perplexity & repetition
python eval_clm.py \
  -g dpo/generated_text/booksum_dpo_mixed_1000/generated_clean.txt \
  -m meta-llama/llama-2-7b-hf \
  -s dpo_gen

# Semantic & n-gram entropy
python eval_quality.py \
  dpo/generated_text/booksum_dpo_mixed_1000 \
  50 \
  --model_path meta-llama/llama-2-7b-hf
```

```










