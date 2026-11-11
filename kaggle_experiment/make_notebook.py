import json
import os

repo_dir = r"c:\Users\FX517ZC\Documents\LLMEmbed\LLMEmbed-ACL2024"
out_dir = r"c:\Users\FX517ZC\Documents\LLMEmbed\kaggle_experiment"

# Read modified code files
files_to_pack = [
    "DownstreamModel.py",
    "MyDataset.py",
    "model_op.py",
    "model_op_multi.py",
    "main.py",
    "bert_embedding/rep_extract.py",
    "roberta_embedding/rep_extract.py",
    "llama2_embedding/rep_extract.py",
]

file_contents = {}
for rel_path in files_to_pack:
    full_path = os.path.join(repo_dir, rel_path.replace("/", os.sep))
    with open(full_path, "r", encoding="utf-8") as f:
        file_contents[rel_path] = f.read()

# Build setup cell code that creates these files
setup_code_lines = [
    "# Create directory structure and author's adapted code files\n",
    "import os\n",
    "\n",
    "os.makedirs('bert_embedding', exist_ok=True)\n",
    "os.makedirs('roberta_embedding', exist_ok=True)\n",
    "os.makedirs('llama2_embedding', exist_ok=True)\n",
    "os.makedirs('dataset', exist_ok=True)\n",
    "os.makedirs('hf_model', exist_ok=True)\n",
    "\n"
]

for filename, content in file_contents.items():
    escaped_content = content.replace("\\", "\\\\").replace("'''", "\\'\\'\\'").replace('"""', '\\"\\"\\"')
    setup_code_lines.append(f"with open('{filename}', 'w', encoding='utf-8') as f:\n")
    setup_code_lines.append(f'    f.write("""{escaped_content}""")\n\n')

setup_code = "".join(setup_code_lines)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 LLMEmbed: Rethinking Lightweight LLM's Genuine Function in Text Classification (ACL 2024)\n",
            "\n",
            "Official Paper: *https://aclanthology.org/2024.acl-long.433*\n",
            "\n",
            "This notebook runs the exact experimental pipeline of the author:\n",
            "1. Feature extraction from **BERT-large** (`bert-large-uncased`, 1024-d pooler output)\n",
            "2. Feature extraction from **RoBERTa-large** (`roberta-large`, 1024-d [CLS] output)\n",
            "3. Feature extraction from **LLaMA-2-7B** (`daryl149/llama-2-7b-chat-hf`, top-5 layers mean pooled, 5x4096-d)\n",
            "4. Multi-modal fusion with **Co-occurrence Pooling** + **Downstream Classifier** (`DownstreamModel.py`)\n",
            "5. Evaluation on **SST-2** benchmark (Accuracy & F1 Score)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Check GPU environment\n",
            "!nvidia-smi\n",
            "import torch\n",
            "print(f'PyTorch Version: {torch.__version__}')\n",
            "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'Device Name: {torch.cuda.get_device_name(0)}')\n",
            "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Install dependencies\n",
            "!pip install -q transformers datasets accelerate scikit-learn tqdm\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            setup_code
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Extract BERT-large Embeddings for SST-2\n",
            "import os\n",
            "os.environ['STEP'] = '128'\n",
            "!python bert_embedding/rep_extract.py 0 sst2\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Extract RoBERTa-large Embeddings for SST-2\n",
            "import os\n",
            "os.environ['STEP'] = '128'\n",
            "!python roberta_embedding/rep_extract.py 0 sst2\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Extract LLaMA-2-7B Embeddings for SST-2\n",
            "import os\n",
            "# Safe batch size to ensure no VRAM OOM on 16GB GPU\n",
            "os.environ['LLAMA_STEP'] = '32'\n",
            "!python llama2_embedding/rep_extract.py 0 sst2\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Verify extracted tensor shapes\n",
            "import torch, os\n",
            "for folder in ['llama2_embedding/sst2/dataset_tensor', 'bert_embedding/sst2/dataset_tensor', 'roberta_embedding/sst2/dataset_tensor']:\n",
            "    print(f'=== {folder} ===')\n",
            "    if os.path.exists(folder):\n",
            "        for f in sorted(os.listdir(folder)):\n",
            "            t = torch.load(os.path.join(folder, f), weights_only=False)\n",
            "            print(f'  {f}: shape={t.shape}, dtype={t.dtype}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Train and evaluate DownstreamModel (LLMEmbed)\n",
            "# Usage: python main.py <cuda_no> <task> <epoches> <SIGMA> <batch_size> <lr>\n",
            "!python main.py 0 sst2 10 0.1 1024 0.0001\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Summary & Comparison with ACL 2024 Paper (Table 2)\n",
            "print('='*60)\n",
            "print('ACL 2024 Paper (Table 2) SST-2 Benchmarks:')\n",
            "print('  BERT-large:      0.8761')\n",
            "print('  RoBERTa-large:   0.9025')\n",
            "print('  LLaMA2 7B:       0.8922')\n",
            "print('  LLMEmbed (Ours): 0.9553')\n",
            "print('='*60)\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = os.path.join(out_dir, "llmembed_experiment.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("Wrote notebook to:", nb_path)

meta = {
    "id": "ngtunthanh/llmembed-acl2024-experiment",
    "title": "LLMEmbed ACL2024 Experiment",
    "code_file": "llmembed_experiment.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

meta_path = os.path.join(out_dir, "kernel-metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Wrote metadata to:", meta_path)
