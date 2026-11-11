# Kaggle experiment results

These files contain observed metrics and run manifests. Large feature caches and model checkpoints remain in the local `kaggle_experiment/runs/archive/` directory and in the linked Kaggle notebook versions.

| Run | Dataset | Config | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [R8 full](r8_full_nf4_v6/run_manifest.json) | R8 | NF4, controlled, 50 epochs, seeds 42/43/44 | 97.40% | 97.24% | 97.50% | 97.52% | 97.62% |
| [SST-2 smoke](sst2_smoke_llama2_v5/run_manifest.json) | SST-2 | NF4, controlled, 256 samples per split, 2 epochs, seed 42 | 52.73% | 55.47% | 91.02% | 90.23% | 50.00% |

R8 uses the TextGCN train/test split. Its preprocessing has not been verified against the original LLMEmbed paper, so these numbers are not a direct reproduction of the paper's table. NF4 is also a quantized variant, while the author's compatibility setting uses FP16. The SST-2 smoke run checks the pipeline only; its accuracy is not a benchmark result.
