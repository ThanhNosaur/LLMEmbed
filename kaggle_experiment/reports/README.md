# Kaggle experiment results

These files contain observed metrics and run manifests. Large feature caches and model checkpoints remain in the local `kaggle_experiment/runs/archive/` directory and in the linked Kaggle notebook versions.

| Run | Dataset | Config | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [R8 full](r8_full_nf4_v6/run_manifest.json) | R8 | NF4, controlled, 50 epochs, seeds 42/43/44 | 97.40% | 97.24% | 97.50% | 97.52% | 97.62% |
| [R52 full](r52_full_nf4_v7/run_manifest.json) | R52 | NF4, controlled, 50 epochs, seeds 42/43/44 | 90.11% | 87.97% | 89.23% | 89.82% | 95.61% |
| [SST-2 full, part 1](sst2_full_nf4_exp0_2_v8/run_manifest.json) | SST-2 | NF4, controlled, 50 epochs, seeds 42/43/44 | 93.39% | 94.42% | 95.22% | pending | pending |
| [SST-2 smoke](sst2_smoke_llama2_v5/run_manifest.json) | SST-2 | NF4, controlled, 256 samples per split, 2 epochs, seed 42 | 52.73% | 55.47% | 91.02% | 90.23% | 50.00% |

R8 and R52 use the TextGCN train/test splits. Their preprocessing has not been verified against the original LLMEmbed paper, so these numbers are not a direct reproduction of the paper's table. SST-2 uses the GLUE train/validation split; Exp 3–4 are pending, so the full ablation is incomplete. NF4 is a quantized variant, while the author's compatibility setting uses FP16. The SST-2 smoke run checks the pipeline only; its accuracy is not a benchmark result.
