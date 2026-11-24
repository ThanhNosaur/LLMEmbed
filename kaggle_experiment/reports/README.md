# Kaggle Experiment Results (Core Set)

This directory retains the core experiments directly supporting the main study (*When does adaptive embedding fusion help minority classes?*). Earlier exploratory runs (`Exp 1`, `Exp 2`, `Exp 5B`, `Exp 5C`, and smoke tests) remain preserved in git history (`f83d2b7`).

## 1. Core Controlled NF4 Comparison (50 epochs, seeds `42/43/44`)

| Dataset | Exp 0 *(Baseline #15)* | Exp 3 *(Chunked CoT + #15)* | Exp 4 *(AG-CMF)* | $\Delta$ (Exp 4 vs Exp 3) |
| :--- | ---: | ---: | ---: | ---: |
| **R8** ([manifest](r8_full_nf4_v6/run_manifest.json)) | 97.40% | 97.52% | **97.62%** | +0.10 pp |
| **R52** ([manifest](r52_full_nf4_v7/run_manifest.json)) | 90.11% | 89.82% | **95.61%** | **+5.79 pp** |
| **SST-2** ([0](sst2_full_nf4_exp0_2_v8/run_manifest.json), [3–4](sst2_full_nf4_exp3_4_v9/run_manifest.json)) | 93.39% | **95.76%** | 94.80% | -0.96 pp |
| **MR (IMDb proxy 40k/10k)** ([0](mr_full_nf4_exp0_v1/run_manifest.json), [3–4](mr_full_nf4_exp3_4_v4/run_manifest.json)) | 95.88% | **96.64%** | 96.48% | -0.16 pp |
| **AGNews** ([0](agnews_full_nf4_exp0_v1/run_manifest.json), [3](agnews_full_nf4_exp3_v1/run_manifest.json), [4](agnews_full_nf4_exp4_v1/run_manifest.json)) | 94.81% | **95.08%** | 95.01% | -0.07 pp |
| **5-Dataset Average** | **94.32%** | **94.96%** | **95.90%** | **+0.94 pp** |

## 2. Strong Controls on Identical FP16 Embeddings (Imbalanced Classification, 100 epochs, seeds `42/43/44`)

All five heads below share the exact same pre-extracted FP16 embeddings, train/val/test split, seeds, and validation-accuracy checkpoint selection. Frequency buckets are pre-registered by training-split class count: `Rare (<10)`, `Medium (10–49)`, `Frequent (>=50)`.

### R52 (52 classes — [manifest](r52_full_fp16_controls_v2/run_manifest.json))

| Experiment | Params | Accuracy | Macro-F1 | Rare F1 (`<10`) | Medium F1 (`10–49`) | Frequent F1 (`>=50`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp4_ctrl_avgcat` (Table 1 #11 `Avg+Cat`) | 6.96M | 91.91% ± 0.08% | 42.54% | 0.00% | 52.75% | 87.62% |
| `exp3` (Baseline #15) | 25.89M | 93.22% ± 0.04% | 49.85% | 0.00% | 68.54% | 90.34% |
| `exp4_ctrl_weighted` (#15 + Class-Weighted Loss) | 25.89M | 91.43% ± 0.38% | 74.09% | **54.35%** | 83.41% | 86.61% |
| `exp4_ctrl_mlp` (Parameter-Matched MLP) | 29.02M | 95.56% ± 0.04% | 72.77% | 39.09% | 89.59% | **92.47%** |
| **`exp4` (AG-CMF)** | 29.02M | **95.81%** ± 0.21% | **76.94%** | 50.72% | **90.43%** | 91.53% |

### Ohsumed (23 medical classes — [manifest](ohsumed_full_fp16_controls_v1/run_manifest.json))

| Experiment | Params | Accuracy | Macro-F1 | Rare F1 (`<10`, 1 cls) | Medium F1 (`10–49`, 7 cls) | Frequent F1 (`>=50`, 15 cls) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp4_ctrl_avgcat` (Table 1 #11 `Avg+Cat`) | 6.96M | 67.66% ± 2.07% | 50.75% | 0.00% | 29.91% | 63.86% |
| `exp3` (Baseline #15) | 25.89M | 69.83% ± 0.56% | 58.08% | 0.00% | 48.37% | 66.48% |
| `exp4_ctrl_weighted` (#15 + Class-Weighted Loss) | 25.89M | 65.45% ± 0.35% | 59.71% | 11.99% | 57.77% | 63.79% |
| `exp4` (AG-CMF) | 29.02M | 72.58% ± 0.59% | 65.70% | 9.89% | 62.69% | 70.83% |
| **`exp4_ctrl_mlp`** (Parameter-Matched MLP) | 29.02M | **73.06%** ± 0.20% | **66.73%** | **20.46%** | **64.31%** | **70.95%** |

## 3. Additional FP16 Reference Runs (100 epochs)

| Run | Dataset | Config | Accuracy | Macro-F1 |
| :--- | :--- | :--- | ---: | ---: |
| [R8 FP16 Exp 4](r8_r52_full_fp16_exp4_v1/run_manifest.json) | R8 | FP16, controlled, 100 epochs, seeds 42/43/44 | 97.61% | 93.17% |
| [SST-2 FP16 Exp 3](sst2_full_fp16_exp3_4_v1/run_manifest.json) | SST-2 | FP16, controlled, 100 epochs, seeds 42/43/44 | 95.45% | 95.45% |
| [SST-2 FP16 Exp 4](sst2_full_fp16_exp3_4_v1/run_manifest.json) | SST-2 | FP16, controlled, 100 epochs, seeds 42/43/44 | 95.57% | 95.56% |
| [SST-2 author baseline](sst2_author_fp16_exp0_v11/run_manifest.json) | SST-2 | FP16, author, 100 epochs, seed 42 | 93.92% | — |
