# Kaggle experiment results

These files contain observed metrics and run manifests. Large feature caches and model checkpoints remain in the local `kaggle_experiment/runs/archive/` directory and in the linked Kaggle notebook versions.

| Run | Dataset | Config | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [R8 full](r8_full_nf4_v6/run_manifest.json) | R8 | NF4, controlled, 50 epochs, seeds 42/43/44 | 97.40% | 97.24% | 97.50% | 97.52% | 97.62% |
| [R52 full](r52_full_nf4_v7/run_manifest.json) | R52 | NF4, controlled, 50 epochs, seeds 42/43/44 | 90.11% | 87.97% | 89.23% | 89.82% | 95.61% |
| SST-2 full ([Exp 0–2](sst2_full_nf4_exp0_2_v8/run_manifest.json), [Exp 3–4](sst2_full_nf4_exp3_4_v9/run_manifest.json)) | SST-2 | NF4, controlled, 50 epochs, seeds 42/43/44 | 93.39% | 94.42% | 95.22% | 95.76% | 94.80% |
| [SST-2 author baseline](sst2_author_fp16_exp0_v11/run_manifest.json) | SST-2 | FP16, author, 100 epochs, seed 42 | 93.92% | — | — | — | — |
| [SST-2 smoke](sst2_smoke_llama2_v5/run_manifest.json) | SST-2 | NF4, controlled, 256 samples per split, 2 epochs, seed 42 | 52.73% | 55.47% | 91.02% | 90.23% | 50.00% |

R8 and R52 use the TextGCN train/test splits. Their preprocessing has not been verified against the original LLMEmbed paper, so these numbers are not a direct reproduction of the paper's table. SST-2 uses the GLUE train/validation split. Exp 0–2 and Exp 3–4 were run in separate Kaggle versions with the same comparison key, study key, seeds and precision. NF4 is a quantized variant, while the author's compatibility setting uses FP16. The SST-2 smoke run checks the pipeline only; its accuracy is not a benchmark result.

The SST-2 author-compatibility run uses the full 67,349/872 train/validation split and a single seed. Its 93.92% accuracy is 1.84 percentage points below the 95.76% reported for Table 1 configuration 15 in the original paper. The run uses a different GPU environment and is not a bit-for-bit reproduction. Its protocol and precision differ from the controlled NF4 ablation above, so the two Exp 0 rows are not a paired comparison.
