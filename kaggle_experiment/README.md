# LLMEmbed++: thực nghiệm trên Kaggle

Triển khai bản thảo `BanThao_LLMEmbed.md`, đối chiếu Bảng 1–3 và mục Limitations của bài LLMEmbed ACL 2024. Mã gốc trong `LLMEmbed-ACL2024/`, notebook SST-2 cũ và log cũ được giữ nguyên. Notebook mới: **`llmembed_plusplus.ipynb`**.

## Bắt đầu trên Kaggle

1. Import `llmembed_plusplus.ipynb` vào một Kaggle Notebook, bật **Internet** và **GPU T4 ×2**.
2. Xin quyền dùng [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), tạo token Hugging Face có quyền đọc, thêm vào Kaggle Secrets tên **`HF_TOKEN`** và cấp quyền cho notebook. Không dán token vào cell.
3. Chạy mặc định `RUN_MODE='smoke'`: SST-2, tối đa 256 mẫu mỗi split, 2 epochs, seed 42. Đây chỉ là kiểm tra vận hành.
4. Khi chạy được, đổi `RUN_MODE='full'`: toàn bộ dữ liệu, 50 epochs, seeds 42/43/44. Chọn **một dataset mỗi phiên**, và chọn các experiment phù hợp thời gian/dung lượng còn lại.
5. `PRECISION='nf4'` tiết kiệm VRAM; `PRECISION='fp16'` dùng model không lượng tử hóa. Giữ cùng precision cho toàn bộ chuỗi ablation; NF4 không phải tái lập số liệu FP16 của bài báo.
6. Save Version để lưu output trước khi phiên kết thúc. Các tensor 7B không nằm trong notebook/ZIP; được tải khi chạy.

Notebook tự giải nén mã nguồn đã nhúng, không cần upload repo hay dữ liệu trước. Các lệnh `subprocess.run(..., check=True)` dừng khi có lỗi. Chưa có token LLaMA thì có thể chạy riêng `EXPERIMENTS=['exp5c']` với Qwen công khai; kết quả này thuộc Exp 5C.

## Các thực nghiệm

| ID | 5 tầng LLM (thứ tự sâu → nông) | Trích xuất | Gộp đoạn | Fusion |
|---|---|---|---|---|
| exp0 | 32,31,30,29,28 | Raw, masked mean | Một đoạn bị giới hạn context | Table 1 #15 |
| exp1 | 32,28,24,20,16 | Raw, masked mean | Như exp0 | #15 |
| exp2 | 32,28,24,20,16 | Prompt + anchor/cosine | Một đoạn; giữ nguyên suffix | #15 |
| exp3 | 32,28,24,20,16 | Như exp2 | Sliding windows + learned attention | #15 |
| exp4 / Exp 5A | 32,28,24,20,16 | Như exp3 | Như exp3 | AG-CMF |
| exp5b | LLaMA-2 như exp4 | BERT → DeBERTa-v3-large | Như exp3 | AG-CMF |
| exp5c | Qwen2.5-7B: 28,24,20,16,12 | Qwen + DeBERTa + RoBERTa | Như exp3 | AG-CMF |

Chiều embedding và số tầng lấy từ config model, không cố định 4096/32 cho Qwen. Có thể chọn `--next-model meta-llama/Meta-Llama-3-8B` khi có quyền truy cập; bộ chọn tầng vẫn lấy 5 tầng từ cuối với stride chỉ định.

- BERT dùng `pooler_output`, RoBERTa dùng token đầu như code gốc. DeBERTa dùng token đầu, vì không có BERT pooler.
- `gamma=0.5`, `tau=0.1`, chunk 256 **content tokens**, overlap 32. Mỗi tokenizer chia đoạn riêng; mỗi nguồn gộp đoạn riêng rồi mới fusion.
- Không giới hạn số đoạn mặc định (`--max-chunks 0`). Nếu chủ động đặt cap, lấy các cửa sổ trải đều và báo tỷ lệ token thực sự được giữ.
- Learned chunk attention thêm vị trí tương đối, cờ đoạn đầu/cuối. AG-CMF gồm 7 phép chiếu riêng, MHA, cổng softmax và tương tác Hadamard; mặc định K=1024, 8 heads.
- Exp 3 và 4 dùng chung cache. BERT/RoBERTa không chia đoạn được tái sử dụng giữa Exp 0/1/2. Các nguồn/chế độ khác có cache riêng.

## Dữ liệu và so sánh công bằng

| Dataset | Nguồn tự tải | Train/test gốc | Ý nghĩa |
|---|---|---|---|
| SST-2 | `stanfordnlp/sst2` | 67.349 / 872 | Dùng split validation có nhãn làm test, theo bài LLMEmbed |
| AGNews | `fancyzhx/ag_news` | 120.000 / 7.600 | Split train/test công khai |
| R8 | TextGCN, commit cố định | 5.485 / 2.189 | Đúng số mẫu; chưa xác nhận tiền xử lý trùng tác giả |
| R52 | TextGCN, commit cố định | 6.532 / 2.568 | Đúng số mẫu; chưa xác nhận tiền xử lý trùng tác giả |
| MR proxy | `stanfordnlp/imdb`, chia lại seed 42 | 40.000 / 10.000 | **Không phải bộ chia MR đã được tác giả xác nhận** |

Repo không chứa dữ liệu MR và không mô tả cách tạo split 40k/10k. IMDb gốc là 25k/25k. `--mr-source imdb_resplit` ghép hai split **có nhãn**, chia stratified 80/20, không dùng 50k mẫu unsupervised. Cờ này phải được chọn tường minh khi dùng CLI; notebook đã hiển thị rõ lựa chọn proxy. Không thay bằng `rotten_tomatoes` hay MR 10.662 câu và gọi là cùng benchmark.

Có dữ liệu tác giả thì dùng `--data-dir /kaggle/input/my-data --mr-source local`, cấu trúc:

```text
my-data/
  mr/train.json
  mr/test.json
  r8/train.json
  r8/test.json
  r52/train.json
  r52/test.json
```

Chấp nhận `[{'text': '...', 'label': 0}, ...]` hoặc `{'text': [...], 'label': [...]}` trong JSON chuẩn (dấu ngoặc kép), hoặc JSONL mỗi dòng một record. Nhãn chuỗi được ánh xạ chung theo tập train. Từ chối nhãn test lạ, nhãn -1, văn bản trống, và độ dài mảng không khớp. Dữ liệu local luôn được ưu tiên nếu thư mục dataset tồn tại. `benchmark_status=local_split_unverified` là ghi nhận nguồn, không tự chứng nhận nó khớp paper.

**Protocol mặc định `controlled`:** tách stratified 10% từ train làm validation bằng seed 42, giữ các lớp singleton trong train. Tất cả experiment/training seed dùng cùng split. Huấn luyện AdamW, LR 1e-4, weight decay 0.01, cosine scheduler, effective batch 1024. Gradient accumulation với micro batch 64 giảm VRAM; dropout khiến nó không đồng nhất từng bit với một forward batch 1024. Chọn checkpoint có validation accuracy tốt nhất (giữ epoch đầu khi hòa), đánh giá test một lần sau huấn luyện. Accuracy và macro/micro F1 tính trên toàn bộ mẫu, không lấy trung bình metric từng batch. Không dùng SST-2 validation/test để chọn epoch.

**Protocol `author`:** chạy riêng Exp 0, chat checkpoint như mã gốc trong repo, mean gồm padding cố định, softmax trước CrossEntropy như mã gốc, Adam, toàn bộ train, epoch cuối, không validation/scheduler. Đây là chế độ tương thích đường tính và mục tiêu huấn luyện của code tác giả; precision, môi trường và dữ liệu vẫn có thể khác, không đảm bảo khớp từng bit hoặc đạt đúng 0.9620. Code sửa cho `controlled` xuất logits trực tiếp.

```bash
python -m llmembed.run --datasets sst2 --experiments exp0 --protocol author --precision fp16 --epochs 100 --seeds 42 --output runs/author
```

`controlled` giữ lại validation nên số mẫu thực sự dùng để cập nhật head ít hơn bảng paper; so sánh đóng góp bằng **baseline tự chạy cùng protocol**. Báo cáo số liệu paper riêng, không chép 0.9620 vào hàng kết quả thực nghiệm.

## Lệnh CLI

Chạy từ thư mục `kaggle_experiment` sau khi cài requirements. Dùng torch có CUDA trên Kaggle; kiểm thử tại máy chỉ cần torch >=2.6 CPU.

```bash
# Xem kế hoạch, không tải model/dữ liệu
python -m llmembed.run --plan

# Chạy thử
python -m llmembed.run --datasets sst2 --limit 256 --epochs 2 --seeds 42 --output runs/smoke

# Ablation đầy đủ một dataset
python -m llmembed.run --datasets r52 --experiments exp0 exp1 exp2 exp3 exp4 --precision fp16 --output runs/full

# MR proxy, được ghi nhãn riêng trong mọi kết quả
python -m llmembed.run --datasets mr --mr-source imdb_resplit --output runs/full

# Khảo sát backbone
python -m llmembed.run --datasets sst2 --experiments exp4 exp5b exp5c --output runs/full

# 15 phép fusion trong Bảng 1 và 4 stride; tái sử dụng cache
python -m llmembed.run --datasets r52 --experiments exp0 --fusion-sweep --stride-sweep --output runs/sweeps

# Tách extraction và training; các tham số còn lại phải giống nhau
python -m llmembed.run --datasets r52 --stage extract --output runs/full
python -m llmembed.run --datasets r52 --stage train --output runs/full

# Tổng hợp lại các result.json đã có
python -m llmembed.report runs/full
```

`--sigma`, `--gamma`, `--tau`, `--stride`, `--chunk-size`, `--hidden` được lưu trong cấu hình. Muốn chọn hyperparameter thì dùng **validation**, cố định lựa chọn trước khi công bố test; việc chạy sweep không tự chọn “best test”.

## Lưu trữ, chạy tiếp, bộ nhớ

- Nạp lần lượt từng backbone, tắt gradient và KV cache, dùng `AutoModel` để không tạo logits từ vựng. Dynamic padding trong controlled. LLM FP16 có thể được Accelerate chia trên hai GPU; NF4 có thể vừa một GPU.
- Batch extraction bắt đầu 2 cho LLM/16 cho encoder, giảm một nửa khi CUDA OOM, dừng với lỗi nếu batch 1 vẫn không đủ. Model-load OOM không tự thay backbone hoặc precision. Giảm batch, dùng NF4 hoặc chọn GPU phù hợp.
- Feature float16 lưu phẳng `values.bin` và offsets, đọc bằng memory map; RAM không chứa toàn bộ feature. Sau mỗi 32 documents ghi state atomically. Khởi động lại sẽ bỏ phần đuôi chưa commit và tiếp tục từ document đã lưu.
- Cache key gồm SHA256 dữ liệu/split, model/revision/commit, tokenizer theo model đó, precision, pooling/layers/prompt/chunk, version extractor. Không trộn cache smoke/full hay các cấu hình khác nhau.
- Checkpoint `last.pt` lưu optimizer, scheduler, RNG và epoch; `best.pt` dùng đánh giá. Chạy lại cùng lệnh tiếp tục từ epoch hoàn tất. Thay số epochs tạo run mới vì cosine schedule thay đổi.
- `--reuse-cache /kaggle/input/old-output/features` đọc cache hoàn tất tại input chỉ đọc. Để tiếp tục cache dở hoặc training, sao chép output cũ sang thư mục working bằng `RESUME_DIR` trong notebook rồi chạy cùng cấu hình. Không chạy đồng thời hai tiến trình vào cùng output.
- Một cache LLaMA không chia đoạn tốn khoảng **40 KiB/document** (5×4096×2 bytes): AGNews train+test khoảng **4,87 GiB**; chunking nhân thêm số đoạn. Nhiều experiment + model weights + checkpoints có thể vượt dung lượng phiên. Chạy theo dataset/experiment, Save Version rồi gắn output làm input khi cần.

## Kết quả xuất ra

- `environment.json`: phiên bản Python/thư viện/CUDA/GPU và lệnh chạy; không có HF token.
- `data/*/manifest.json`: nguồn, SHA256, ánh xạ nhãn, counts, split seed, cảnh báo văn bản trùng train/test.
- `features/*/state.json`: model commit, progress, thời gian extraction, tỷ lệ token gốc giữ lại theo tokenizer, token weights.
- `results/*/`: config, best/last checkpoint, history, `predictions.csv`, metric test, mean gate, trọng số đoạn của 4 mẫu đầu test.
- `metrics.csv`: từng seed; `ablation.csv`: mean/sample std và delta theo **điểm phần trăm**, chỉ ghép baseline cùng split/cấu hình/seed. Một seed thì std để trống.
- `average_five_datasets.csv`: chỉ tạo khi đủ 5 datasets và cùng seeds. Cột protocol vẫn nêu MR proxy; số trung bình này không tự trở thành đối chiếu chính xác với paper.
- `results/*/source_weights.png`, `explanations/*/tokens_*.png`: trọng số cổng trên test và heatmap cosine token. Vẽ đoạn đầu của tối đa 4 document; toàn bộ trọng số các đoạn của 4 document đó vẫn lưu trong state JSON. Trọng số token không phải giải thích nhân quả hay chuỗi suy luận được sinh.
- Thời gian extraction là compute ban đầu của các cache, không tính tải/nạp model. Training time gồm train + validation, không gồm ghi checkpoint; inference đo riêng. Khi dùng lại cache, không diễn giải thời gian trích xuất đã lưu thành thời gian phiên hiện tại.
- Điện năng mặc định **không đo**. `--power-watts 140` chỉ tạo **ước lượng** với giả định tổng công suất 140 W; không ghi số đó là phép đo, không bao gồm CPU/toàn hệ thống.

## Các điểm cần sửa khi diễn giải bản thảo

1. Co-occurrence của code tác giả là `2*sigmoid(sigma*X)-1 = tanh(sigma*X/2)`. Biểu thức `tanh(2*sigma*X)` trong paper/bản thảo không tương đương. Baseline ở đây dùng đúng code tác giả; nhánh Hadamard AG-CMF giữ công thức riêng trong bản thảo.
2. Cosine token pooling với gamma/tau cố định **không được học**. Chỉ chunk attention, projections, MHA, gate và classifier được học. Single-pass prompt không chứng minh có CoT reasoning.
3. Mask chỉ cho token văn bản gốc tham gia cosine pool, loại prompt/special/padding/anchor; suffix luôn được giữ. Văn bản được token hóa riêng rồi ghép token IDs với prefix/suffix để kiểm soát budget, không decode/re-tokenize các đoạn.
4. Tám đoạn 256, overlap 32 chỉ phủ tối đa 1.824 content tokens. Muốn tránh cắt, dùng số đoạn không giới hạn; tỷ lệ giữ token không đồng nghĩa giữ mọi quan hệ ngữ cảnh dài giữa các đoạn.
5. R52=0.4217 trong Table 2 thuộc RoBERTa và một số cấu hình Cat; cấu hình Cat ba nguồn đạt **0.9326**. Bản thảo đã đảo MR/AGNews ở vài hàng (RoBERTa, IO, Cat, Cat+Co). Cần chép lại từ PDF khi lập bảng đối thủ.
6. Không có bảo đảm trước về accuracy >0.968/0.973, tăng tốc 8–10× hoặc <0.25 kWh. Các giá trị đó là mục tiêu, không phải kết quả đã được kiểm chứng.
7. Table 1 #1–14 được triển khai theo mô tả fusion và dùng cùng MLP 1024→256; repo chỉ cung cấp head #15 nên không khẳng định kiến trúc head của 14 cấu hình còn lại được tác giả công bố đầy đủ.

## Kiểm tra và tái tạo notebook

```bash
python -m unittest discover -s tests -v
python make_plusplus_notebook.py
```

Tests dùng mô hình tiny ngẫu nhiên lưu local: đủ Exp 0–5, kích thước backbone khác nhau, forward/backward 15 fusion, che padding/prompt, chunk coverage, recovery cache, checkpoint resume, cache reuse và CSV/PNG. Không tải model 7B và không xác nhận chất lượng phân loại. Build notebook nhúng chính các file Python đã kiểm tra; notebook gốc không bị ghi đè.

## Nguồn đối chiếu

- [LLMEmbed, ACL 2024](https://aclanthology.org/2024.acl-long.433/) và [mã tác giả](https://github.com/ChunLiu-cs/LLMEmbed-ACL2024).
- [TextGCN: corpus và split Reuters](https://github.com/yao8839836/text_gcn/tree/962223652e9bb164ac2d83cd09fc7b8845ce860b/data).
- [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), [AGNews](https://huggingface.co/datasets/fancyzhx/ag_news), [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb).
- [Transformers 4.51.3: bitsandbytes](https://huggingface.co/docs/transformers/v4.51.3/en/quantization/bitsandbytes), [DeBERTa](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/deberta-v2), [Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json).
