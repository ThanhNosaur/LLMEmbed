# Kaggle Experiment Results

These files contain observed metrics and run manifests. Large feature caches and model checkpoints remain in the local `kaggle_experiment/runs/archive/` directory and in the linked Kaggle notebook versions.

## 1. Controlled NF4 Ablation Matrix (35/35 Complete — 50 epochs, seeds `42/43/44`)

| Dataset | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 (AG-CMF) | Exp 5B (LLaMA-2 + DeBERTa) | Exp 5C (Qwen2.5 + DeBERTa) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R8** ([0–4](r8_full_nf4_v6/run_manifest.json), [5B](r8_r52_full_nf4_exp5b_v12/run_manifest.json), [5C](r8_r52_full_nf4_exp5c_v3/run_manifest.json)) | 97.40% | 97.24% | 97.50% | 97.52% | 97.62% | **97.65%** | 97.50% |
| **R52** ([0–4](r52_full_nf4_v7/run_manifest.json), [5B](r8_r52_full_nf4_exp5b_v12/run_manifest.json), [5C](r8_r52_full_nf4_exp5c_v3/run_manifest.json)) | 90.11% | 87.97% | 89.23% | 89.82% | **95.61%** | 95.46% | 95.21% |
| **SST-2** ([0–2](sst2_full_nf4_exp0_2_v8/run_manifest.json), [3–4](sst2_full_nf4_exp3_4_v9/run_manifest.json), [5B](sst2_full_nf4_exp5b_v3/run_manifest.json), [5C](sst2_full_nf4_exp5c_v2/run_manifest.json)) | 93.39% | 94.42% | 95.22% | **95.76%** | 94.80% | 94.50% | 95.22% |
| **MR (IMDb proxy 40k/10k)** ([0](mr_full_nf4_exp0_v1/run_manifest.json), [1](mr_full_nf4_exp1_v1/run_manifest.json), [2](mr_full_nf4_exp2_v1/run_manifest.json), [3–4](mr_full_nf4_exp3_4_v4/run_manifest.json), [5B](mr_full_nf4_exp5b_v1/run_manifest.json), [5C](mr_full_nf4_exp5c_v1/run_manifest.json)) | 95.88% | 96.20% | **96.96%** | 96.64% | 96.48% | 96.62% | 96.61% |
| **AGNews** ([0](agnews_full_nf4_exp0_v1/run_manifest.json), [1](agnews_full_nf4_exp1_v1/run_manifest.json), [2](agnews_full_nf4_exp2_v1/run_manifest.json), [3](agnews_full_nf4_exp3_v1/run_manifest.json), [4](agnews_full_nf4_exp4_v1/run_manifest.json), [5B](agnews_full_nf4_exp5b_v1/run_manifest.json), [5C](agnews_full_nf4_exp5c_v1/run_manifest.json)) | 94.81% | 94.87% | 94.93% | **95.08%** | 95.01% | 95.01% | 94.86% |
| **5-Dataset Average** | **94.32%** | **94.14%** | **94.77%** | **94.96%** | **95.90%** | **95.85%** | **95.88%** |

## 2. FP16 + 100 Epochs Comparison Runs

| Run | Dataset | Config | Accuracy | Macro-F1 |
| :--- | :--- | :--- | ---: | ---: |
| [R8 FP16 Exp 4](r8_r52_full_fp16_exp4_v1/run_manifest.json) | R8 | FP16, controlled, 100 epochs, seeds 42/43/44 | 97.61% | 93.17% |
| [R52 FP16 Exp 4](r8_r52_full_fp16_exp4_v1/run_manifest.json) | R52 | FP16, controlled, 100 epochs, seeds 42/43/44 | **95.81%** | **76.94%** |
| [SST-2 FP16 Exp 3](sst2_full_fp16_exp3_4_v1/run_manifest.json) | SST-2 | FP16, controlled, 100 epochs, seeds 42/43/44 | 95.45% | 95.45% |
| [SST-2 FP16 Exp 4](sst2_full_fp16_exp3_4_v1/run_manifest.json) | SST-2 | FP16, controlled, 100 epochs, seeds 42/43/44 | 95.57% | 95.56% |
| [SST-2 author baseline](sst2_author_fp16_exp0_v11/run_manifest.json) | SST-2 | FP16, author, 100 epochs, seed 42 | 93.92% | — |

## 3. Strong Controls on Identical FP16 Embeddings (Imbalanced Classification, 100 epochs, seeds `42/43/44`)

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

## Notes on Provenance & Scope

- R8, R52, and Ohsumed use the TextGCN splits (`TEXTGCN_COMMIT = 96222365...`). R8/R52 preprocessing has not been verified bit-for-bit against the original LLMEmbed paper.
- MR uses an IMDb 40k/10k resplit proxy (`imdb_resplit_proxy`), not the original Pang & Lee MR split.
- On imbalanced datasets (`R52` and `Ohsumed`), both `exp4` (AG-CMF) and `exp4_ctrl_mlp` strongly outperform LLMEmbed #15 (`exp3`), `Avg+Cat`, and class-weighted #15 by removing the 49-d Gram/Dropout-0.5 bottleneck. Self-attention + gating (`AG-CMF`) provides a clear additional gain over the parameter-matched MLP on highly skewed 52-class `R52` (+4.16 pp Macro-F1, +11.63 pp Rare F1), whereas the parameter-matched MLP slightly outperforms `AG-CMF` on 23-class `Ohsumed` (+1.03 pp Macro-F1).
