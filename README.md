# LLMEmbed & LLMEmbed++

Dự án nghiên cứu, tái lập và mở rộng bài báo **LLMEmbed: Rethinking Lightweight LLM’s Genuine Function in Text Classification** (ACL 2024).

Bên cạnh việc lưu trữ mã nguồn gốc của nhóm tác giả, repository này triển khai kiến trúc mở rộng **LLMEmbed++** nhằm giải quyết các hạn chế về trích xuất tầng sâu, ngữ cảnh văn bản dài và cơ chế trộn đa mô hình (Cross-Model Fusion) thông qua hệ thống thực nghiệm có kiểm soát trên Kaggle GPU.

---

## Cấu trúc Thư mục

```text
LLMEmbed/
├── README.md                                  # Tài liệu tổng quan dự án
├── BanThao_LLMEmbed.md                        # Bản thảo thiết kế nghiên cứu & thực nghiệm LLMEmbed++
├── LLMEmbed - Rethinking Lightweight ...pdf   # Bài báo gốc LLMEmbed (ACL 2024)
├── LLMEmbed-ACL2024/                          # Mã nguồn gốc của nhóm tác giả (ACL 2024)
│   ├── bert_embedding/                        # Trích xuất đặc trưng BERT-large
│   ├── llama2_embedding/                      # Trích xuất đặc trưng LLaMA-2-7B
│   ├── roberta_embedding/                     # Trích xuất đặc trưng RoBERTa-large
│   ├── DownstreamModel.py                     # Kiến trúc Classifier Head gốc (cấu hình #15)
│   ├── MyDataset.py                           # Bộ nạp dữ liệu đặc trưng
│   ├── main.py                                # Điểm vào huấn luyện/đánh giá gốc
│   ├── model_op.py                            # Huấn luyện phân loại nhị phân (SST-2, MR)
│   └── model_op_multi.py                      # Huấn luyện phân loại đa lớp (AGNews, R8, R52)
└── kaggle_experiment/                         # Triển khai LLMEmbed++ & hệ thống thực nghiệm Kaggle
    ├── README.md                              # Hướng dẫn chi tiết thực nghiệm, CLI & cấu hình Kaggle
    ├── requirements.txt                       # Các thư viện phụ thuộc chuẩn hóa
    ├── llmembed_plusplus.ipynb                # Notebook tự chứa (standalone) chạy trực tiếp trên Kaggle
    ├── make_plusplus_notebook.py              # Script đóng gói mã nguồn vào notebook độc lập
    ├── llmembed/                              # Package Python cốt lõi của LLMEmbed++
    │   ├── data.py                            # Tải dữ liệu, kiểm chứng SHA256 & chia train/val/test
    │   ├── features.py                        # Trích xuất đặc trưng đa tầng, CoT Prompt, chunking & memmap cache
    │   ├── models.py                          # 15 phép fusion Bảng 1, ChunkPool & Adaptive Gated Fusion (AG-CMF)
    │   ├── train.py                           # Huấn luyện Classifier Head có kiểm soát & khôi phục checkpoint
    │   ├── report.py                          # Tổng hợp kết quả CSV, tính độ lệch chuẩn & vẽ biểu đồ giải thích
    │   └── run.py                             # Giao diện dòng lệnh (CLI) điều phối toàn bộ pipeline
    ├── reports/                               # Báo cáo kết quả thực nghiệm & run manifests trên Kaggle
    │   ├── README.md                          # Bảng tổng hợp kết quả thực nghiệm đã ghi nhận
    │   ├── r8_full_nf4_v6/                    # Kết quả R8 (Exp 0–4, NF4, 50 epochs, seeds 42/43/44)
    │   ├── r52_full_nf4_v7/                   # Kết quả R52 (Exp 0–4, NF4, 50 epochs, seeds 42/43/44)
    │   ├── r8_r52_full_nf4_exp5b_v12/         # Kết quả R8 & R52 Exp 5B (LLaMA-2 + DeBERTa, NF4, 50 epochs)
    │   ├── r8_r52_full_nf4_exp5c_v3/          # Kết quả R8 & R52 Exp 5C (Qwen2.5 + DeBERTa, NF4, 50 epochs)
    │   ├── sst2_full_nf4_exp0_2_v8/           # Kết quả SST-2 Exp 0–2 (NF4, 50 epochs, seeds 42/43/44)
    │   ├── sst2_full_nf4_exp3_4_v9/           # Kết quả SST-2 Exp 3–4 (NF4, 50 epochs, seeds 42/43/44)
    │   ├── sst2_full_nf4_exp5b_v3/            # Kết quả SST-2 Exp 5B (LLaMA-2 + DeBERTa, NF4, 50 epochs)
    │   ├── sst2_full_nf4_exp5c_v2/            # Kết quả SST-2 Exp 5C (Qwen2.5 + DeBERTa, NF4, 50 epochs)
    │   ├── sst2_author_fp16_exp0_v11/         # Kết quả SST-2 Baseline theo protocol tác giả (FP16, 100 epochs)
    │   └── sst2_smoke_llama2_v5/              # Kết quả chạy kiểm thử vận hành (Smoke test)
    └── tests/                                 # Bộ kiểm thử tự động (Unit Tests)
        └── test_experiments.py
```

---

## Tóm tắt Lộ trình Thực nghiệm (LLMEmbed++)

| Mã thực nghiệm | Tầng LLM (sâu $\rightarrow$ nông) | Phương thức trích xuất | Xử lý văn bản dài | Kiến trúc trộn (Fusion) |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 0** | `32, 31, 30, 29, 28` | Văn bản thô, Masked Mean Pooling | 1 đoạn (giới hạn context) | Bảng 1 #15 (`Cat+Co+Avg+Cat`) |
| **Exp 1** | `32, 28, 24, 20, 16` | Văn bản thô, Strided Multi-Depth ($s=4$) | Như Exp 0 | Bảng 1 #15 |
| **Exp 2** | `32, 28, 24, 20, 16` | Single-Pass CoT Prompt + Anchor/Cosine Pooling | 1 đoạn (giữ nguyên hậu tố prompt) | Bảng 1 #15 |
| **Exp 3** | `32, 28, 24, 20, 16` | Như Exp 2 | Cửa sổ trượt (Sliding Windows) + Learned Chunk Attention | Bảng 1 #15 |
| **Exp 4 (5A)** | `32, 28, 24, 20, 16` | Như Exp 3 | Như Exp 3 | **AG-CMF** (Adaptive Gated Cross-Model Fusion) |
| **Exp 5B** | LLaMA-2 như Exp 4 | Thay BERT bằng DeBERTa-v3-large | Như Exp 3 | **AG-CMF** |
| **Exp 5C** | Qwen2.5-7B: `28, 24, 20, 16, 12` | Qwen2.5-7B + DeBERTa-v3-large + RoBERTa-large | Như Exp 3 | **AG-CMF** |

---

## Kết quả Thực nghiệm Đã Ghi Nhận

Chi tiết đầy đủ (bao gồm đối chứng `FP16 + 100 epochs` và đối chứng mất cân bằng trên `R52` & `Ohsumed`) nằm tại [`kaggle_experiment/reports/README.md`](kaggle_experiment/reports/README.md):

| Tập dữ liệu | Cấu hình chạy | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 (AG-CMF) | Exp 5B (LLaMA-2 + DeBERTa) | Exp 5C (Qwen2.5 + DeBERTa) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R8** | NF4, controlled, 50 epochs, seeds `42/43/44` | 97.40% | 97.24% | 97.50% | 97.52% | 97.62% | **97.65%** | 97.50% |
| **R52** | NF4, controlled, 50 epochs, seeds `42/43/44` | 90.11% | 87.97% | 89.23% | 89.82% | **95.61%** | 95.46% | 95.21% |
| **SST-2** | NF4, controlled, 50 epochs, seeds `42/43/44` | 93.39% | 94.42% | 95.22% | **95.76%** | 94.80% | 94.50% | 95.22% |
| **MR (IMDb proxy)** | NF4, controlled, 50 epochs, seeds `42/43/44` | 95.88% | 96.20% | **96.96%** | 96.64% | 96.48% | 96.62% | 96.61% |
| **AGNews** | NF4, controlled, 50 epochs, seeds `42/43/44` | 94.81% | 94.87% | 94.93% | **95.08%** | 95.01% | 95.01% | 94.86% |
| **Trung bình 5 tập** | NF4, controlled, 50 epochs, seeds `42/43/44` | **94.32%** | **94.14%** | **94.77%** | **94.96%** | **95.90%** | **95.85%** | **95.88%** |

**Đối chứng mạnh trên cùng embedding FP16 (100 epochs, 3 seeds):**
- **R52 (52 lớp):** `Exp 4 (AG-CMF)` đạt **95.81% Acc / 76.94% Macro-F1 / 50.72% Rare F1**, vượt `Matched MLP` cùng 29.02M tham số (`95.56% / 72.77% / 39.09%`), `Baseline #15` (`93.22% / 49.85% / 0.00%`), `Avg+Cat` (`91.91% / 42.54% / 0.00%`) và `#15 + Class-Weighted Loss` (`91.43% / 74.09% / 54.35%`).
- **Ohsumed (23 lớp y khoa):** `Matched MLP` (`73.06% Acc / 66.73% Macro-F1`) và `Exp 4 (AG-CMF)` (`72.58% / 65.70%`) đều vượt xa `Baseline #15` (`69.83% / 58.08%`), `Avg+Cat` (`67.66% / 50.75%`) và `#15 + Class-Weighted Loss` (`65.45% / 59.71%`).

---

## Hướng dẫn Sử dụng Nhanh

### 1. Chạy trên Kaggle Notebook
1. Tải file [`kaggle_experiment/llmembed_plusplus.ipynb`](kaggle_experiment/llmembed_plusplus.ipynb) lên Kaggle Notebook.
2. Bật **GPU T4 ×2**, bật **Internet**, và cấu hình biến `HF_TOKEN` trong **Kaggle Secrets** (để tải `meta-llama/Llama-2-7b-hf`).
3. Chỉnh `RUN_MODE = 'smoke'` để kiểm tra nhanh hoặc `RUN_MODE = 'full'` để chạy thực nghiệm đầy đủ.

### 2. Chạy qua dòng lệnh (CLI) tại máy cục bộ
```bash
cd kaggle_experiment
pip install -r requirements.txt

# Xem kế hoạch thực nghiệm (không tải model/dữ liệu)
python -m llmembed.run --plan

# Chạy bộ kiểm thử tự động (Unit Tests)
python -m unittest discover -s tests -v

# Tái đóng gói notebook độc lập sau khi chỉnh sửa mã nguồn
python make_plusplus_notebook.py
```

Xem thêm hướng dẫn chi tiết về tham số dòng lệnh, cơ chế lưu cache `float16` memmap và quy chuẩn đánh giá tại [`kaggle_experiment/README.md`](kaggle_experiment/README.md).
