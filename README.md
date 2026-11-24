# LLMEmbed & LLMEmbed++

Dự án nghiên cứu, tái lập và mở rộng bài báo **LLMEmbed: Rethinking Lightweight LLM’s Genuine Function in Text Classification** (ACL 2024).

Bên cạnh việc lưu trữ mã nguồn gốc của nhóm tác giả, repository này triển khai kiến trúc mở rộng **LLMEmbed++** nhằm giải quyết các hạn chế về trích xuất tầng sâu, ngữ cảnh văn bản dài và cơ chế trộn đa mô hình (Cross-Model Fusion) thông qua hệ thống thực nghiệm có kiểm soát trên Kaggle GPU.

---

## Cấu trúc Thư mục

```text
LLMEmbed/
├── README.md                                  # Tài liệu tổng quan dự án
├── BaoCao_ThucNghiem_LLMEmbed.md              # Báo cáo thực nghiệm ngắn gọn & kết luận chính
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
    │   └── sst2_author_fp16_exp0_v11/         # Kết quả SST-2 Baseline theo protocol tác giả (FP16, 100 epochs)
    └── tests/                                 # Bộ kiểm thử tự động (Unit Tests)
        └── test_experiments.py
```

---

## Tóm tắt Thực nghiệm Cốt lõi

| Mã thực nghiệm | Mô tả ngắn |
| :--- | :--- |
| **Exp 0** | Baseline nội bộ theo LLMEmbed (`#15 Cat+Co+Avg+Cat`, 5 tầng cuối `32..28`) |
| **Exp 3** | Trích xuất cách quãng + CoT prompt + Chunking + đầu trộn `#15` |
| **Exp 4 (`AG-CMF`)** | Giữ nguyên embedding của Exp 3, thay đầu trộn bằng **Adaptive Gated Cross-Model Fusion** (~29.02M tham số) |
| **Đối chứng mạnh (FP16)** | So sánh `Exp 4` trên cùng 100% embedding với `Avg+Cat` (`#11`), `Matched MLP` (~29.02M), và `#15 + Class-Weighted Loss` trên `R52` & `Ohsumed` |

---

## Kết quả Thực nghiệm Cốt lõi

Chi tiết đầy đủ nằm tại [`BaoCao_ThucNghiem_LLMEmbed.md`](BaoCao_ThucNghiem_LLMEmbed.md) và [`kaggle_experiment/reports/README.md`](kaggle_experiment/reports/README.md):

| Tập dữ liệu | Cấu hình | Exp 0 *(Baseline)* | Exp 3 | Exp 4 *(AG-CMF)* | $\Delta$ (Exp 4 vs Exp 3) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **R8** | NF4, controlled, 50 ep, 3 seeds | 97.40% | 97.52% | **97.62%** | +0.10 pp |
| **R52** | NF4, controlled, 50 ep, 3 seeds | 90.11% | 89.82% | **95.61%** | **+5.79 pp** |
| **SST-2** | NF4, controlled, 50 ep, 3 seeds | 93.39% | **95.76%** | 94.80% | -0.96 pp |
| **MR (IMDb proxy)** | NF4, controlled, 50 ep, 3 seeds | 95.88% | **96.64%** | 96.48% | -0.16 pp |
| **AGNews** | NF4, controlled, 50 ep, 3 seeds | 94.81% | **95.08%** | 95.01% | -0.07 pp |
| **Trung bình 5 tập** | NF4, controlled, 50 ep, 3 seeds | **94.32%** | **94.96%** | **95.90%** | **+0.94 pp** |

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
