# Khi nào hợp nhất embedding thích nghi có ích?

**Bản nháp báo cáo thực nghiệm LLMEmbed++ · 19/11/2026 · Lưu hành nội bộ nhóm**

Tên tiếng Anh dự kiến: *When Does Adaptive Embedding Fusion Help? An Empirical Study of Frozen Models for Text Classification*

## Tóm tắt

Nhóm khảo sát cách chọn tầng ẩn, trích xuất embedding bằng prompt, chia đoạn và hợp nhất đặc trưng từ các mô hình đóng băng. Báo cáo tổng hợp **105 lượt chạy** theo giao thức có kiểm soát trên Kaggle, gồm 5 bộ dữ liệu, 7 cấu hình và 3 seed. Phát hiện nổi bật nằm ở **R52**. Khi dùng cùng đặc trưng trích xuất, thay bộ hợp nhất cơ sở bằng bộ hợp nhất thích nghi làm accuracy tăng từ **89,82% lên 95,61%** và macro-F1 tăng từ **31,74% lên 74,41%**. Hiệu quả nhỏ hơn trên R8 và accuracy giảm trên SST-2, IMDb và AGNews. Kết quả R52 là cơ sở để kiểm tra tiếp khả năng nhận diện các lớp ít mẫu.

## 1. Thiết kế thực nghiệm

### Dữ liệu và giao thức

| Bộ dữ liệu | Train nguồn | Tập đánh giá | Nguồn và cách chia |
|---|---:|---:|---|
| SST-2 | 67.349 | 872 | GLUE train/validation. Dùng validation của GLUE làm tập đánh giá cuối. |
| IMDb, proxy cho MR | 40.000 | 10.000 | Gộp hai split có nhãn của IMDb, sau đó chia lại 80/20 theo nhãn với seed 42. |
| AGNews | 120.000 | 7.600 | Train/test từ bộ dữ liệu Hugging Face. |
| R8 | 5.485 | 2.189 | Train/test theo TextGCN, gồm 8 lớp. |
| R52 | 6.532 | 2.568 | Train/test theo TextGCN, gồm 52 lớp. |

Các kích thước ở cột train là số mẫu trước khi tách validation nội bộ. Giao thức `controlled` tách khoảng 10% dữ liệu train theo nhãn để chọn checkpoint theo accuracy. Mỗi cấu hình chính huấn luyện tối đa 50 epoch, với seed huấn luyện **42, 43, 44** trên cùng một split. Các backbone giữ nguyên trọng số. NF4 áp dụng cho nhánh LLM, còn các encoder BERT/RoBERTa hoặc DeBERTa chạy FP16. Các cấu hình trong cùng một tập có cùng dấu vân tay dữ liệu và khóa so sánh trong CSV.

### Các cấu hình khảo sát

| Cấu hình | Thay đổi được đánh giá |
|---|---|
| Exp 0 | Baseline nội bộ theo LLMEmbed, dùng Llama-2-7B, BERT-large và RoBERTa-large. Gộp trung bình token ở 5 tầng cuối của Llama, sau đó hợp nhất theo `Cat + Co + Avg + Cat`. |
| Exp 1 | Chọn 5 tầng cách quãng của Llama: 16, 20, 24, 28 và 32. |
| Exp 2 | Thêm prompt theo tác vụ. Gộp vector token cuối với trung bình có trọng số cosine của các token văn bản. |
| Exp 3 | Thêm chia đoạn chồng lấp và gộp đoạn có trọng số vị trí học được. |
| Exp 4 | Giữ đặc trưng trích xuất của Exp 3. Thay bộ hợp nhất cơ sở bằng bộ hợp nhất có phép chiếu, self-attention giữa các nguồn, gate và tương tác Hadamard. |
| Exp 5B | Dùng cấu hình Exp 4 và thay BERT-large bằng DeBERTa-v3-large. |
| Exp 5C | Dùng cấu hình Exp 5B và thay Llama-2-7B bằng Qwen2.5-7B. |

Exp 0–4 là chuỗi thí nghiệm bổ sung thành phần. Mỗi bước mô tả hiệu ứng của toàn bộ thay đổi ở bước đó. Ví dụ, Exp 2 đổi cả prompt và cách gộp token.

## 2. Kết quả chính

### Accuracy trên năm bộ dữ liệu

Các ô theo từng bộ dữ liệu là **trung bình ± độ lệch chuẩn mẫu** qua 3 seed, tính theo %. Cột cuối lấy trung bình của năm bộ dữ liệu với trọng số bằng nhau.

| Cấu hình | SST-2 | IMDb proxy | AGNews | R8 | R52 | TB 5 tập |
|---|---:|---:|---:|---:|---:|---:|
| Exp 0 | 93,39 ± 0,24 | 95,88 ± 0,08 | 94,81 ± 0,09 | 97,40 ± 0,12 | 90,11 ± 0,12 | 94,32 |
| Exp 1 | 94,42 ± 0,24 | 96,20 ± 0,11 | 94,87 ± 0,09 | 97,24 ± 0,07 | 87,97 ± 0,18 | 94,14 |
| Exp 2 | 95,22 ± 0,52 | **96,96 ± 0,05** | 94,93 ± 0,02 | 97,50 ± 0,13 | 89,23 ± 0,06 | 94,77 |
| Exp 3 | **95,76 ± 0,34** | 96,64 ± 0,02 | **95,08 ± 0,08** | 97,52 ± 0,11 | 89,82 ± 0,22 | 94,96 |
| Exp 4 | 94,80 ± 0,65 | 96,48 ± 0,23 | 95,01 ± 0,19 | 97,62 ± 0,12 | **95,61 ± 0,16** | **95,91** |
| Exp 5B | 94,50 ± 0,41 | 96,62 ± 0,07 | 95,01 ± 0,25 | **97,65 ± 0,11** | 95,46 ± 0,29 | 95,85 |
| Exp 5C | 95,22 ± 0,35 | 96,61 ± 0,07 | 94,86 ± 0,05 | 97,50 ± 0,34 | 95,21 ± 0,04 | 95,88 |

- **Exp 4 so với Exp 3.** Accuracy tăng 5,79 điểm % trên R52 và 0,11 điểm % trên R8. Accuracy giảm 0,96 điểm % trên SST-2, 0,16 trên IMDb và 0,07 trên AGNews. Delta được tính trước khi làm tròn.
- **Exp 1 so với Exp 0.** Tầng cách quãng cải thiện SST-2 khoảng 1,03 điểm % nhưng làm R52 giảm 2,14 điểm %.
- **Exp 5B và 5C.** Thay backbone cho kết quả gần Exp 4 khi tính trung bình năm tập. Trên R52, cả hai cấu hình đều đạt macro-F1 thấp hơn Exp 4.

### Phân tích R52

Macro-F1 cho mỗi lớp trọng số bằng nhau, nên làm rõ sự khác biệt giữa các mô hình khi số mẫu mỗi lớp chênh lệch lớn. Bảng dưới tính lại từ dự đoán của từng seed. Nhóm lớp ít mẫu dùng ngưỡng khám phá **dưới 10 mẫu test**, gồm 28/52 lớp.

| Cấu hình | Accuracy (%) | Macro-F1 toàn tập (%) | F1 trung bình của 28 lớp ít mẫu (%) | Số lớp có ít nhất một dự đoán đúng qua từng seed |
|---|---:|---:|---:|---:|
| Exp 0 | 90,11 | 34,98 | 6,0 | 25–26 |
| Exp 3 | 89,82 | 31,74 | 3,0 | 22–23 |
| Exp 4 | 95,61 | 74,41 | 59,8 | 44–45 |

Exp 4 cải thiện accuracy từ **5,69 đến 6,00 điểm %** so với Exp 3 ở cả ba seed. Mức tăng macro-F1 trung bình là **42,67 điểm %**. Hai cấu hình dùng cùng dữ liệu và đặc trưng trích xuất, nhưng Exp 4 thay nhiều phần của đầu phân loại cùng lúc và tăng số tham số được huấn luyện từ **25,89 lên 29,02 triệu**. Phân tích theo lớp hiện có giá trị khám phá trên R52 và cần được xác nhận độc lập.

## 3. Kết quả FP16 và đối chiếu bài gốc

| Giao thức | Tập | Cấu hình | Epoch | Seed | Accuracy (%) | Macro-F1 (%) |
|---|---|---|---:|---|---:|---:|
| Tương thích tác giả | SST-2 | Exp 0 | 100 | 42 | 93,92 | 93,92 |
| Controlled | SST-2 | Exp 3 | 100 | 42/43/44 | 95,45 ± 0,46 | 95,45 ± 0,46 |
| Controlled | SST-2 | Exp 4 | 100 | 42/43/44 | 95,57 ± 0,24 | 95,56 ± 0,24 |
| Controlled | R8 | Exp 4 | 100 | 42/43/44 | 97,61 ± 0,16 | 93,17 ± 0,18 |
| Controlled | R52 | Exp 4 | 100 | 42/43/44 | 95,81 ± 0,21 | 76,94 ± 0,80 |

Lượt tương thích tác giả dùng Llama-2-7B-Chat. Accuracy SST-2 của lượt này kém **1,84 điểm %** so với mức **95,76%** được báo cáo cho cấu hình `Cat + Co + Avg + Cat` trong bài gốc. Giao thức controlled dùng Llama-2-7B bản base trong Exp 0–4. Các lượt FP16 và NF4 ở đây còn khác về số epoch hoặc giao thức. Cần phép đối chứng cùng cấu hình và ngân sách huấn luyện để đánh giá riêng tác động của độ chính xác số.

Mức **96,20%** trong bài gốc là trung bình trên năm bộ dữ liệu theo giao thức của tác giả. Mức **95,91%** của Exp 4 là trung bình nội bộ, với IMDb chia lại và tiền xử lý R8/R52 chưa đối chiếu đầy đủ với bài gốc. Hai con số nên đứng ở hai bảng riêng trong bản thảo bài báo. [LLMEmbed, ACL 2024, Bảng 2](https://aclanthology.org/2024.acl-long.433.pdf)

## 4. Giới hạn khi diễn giải

- **Mẫu trùng giữa các tập.** Manifest ghi nhận văn bản trùng khớp tuyệt đối với tập train nguồn ở 132 mẫu đánh giá IMDb, 4 mẫu R8 và 6 mẫu R52. Các điểm số trên dùng toàn bộ split đã lưu. Cần tính lại metric sau khi loại mẫu trùng để đo độ nhạy.
- **CoT và khả năng giải thích.** Exp 2 dùng prompt theo tác vụ và gộp cosine với hệ số cố định. Heatmap mô tả trọng số gộp token. Hiện chưa có phép kiểm định chuỗi suy luận hay độ tin cậy của heatmap như một lời giải thích.
- **Văn bản dài và chi phí.** Số liệu cache IMDb cho thấy Exp 3 lưu giữ 100% token theo bộ đếm của pipeline, so với khoảng 92,54% ở Exp 0. Tổng thời gian trích xuất đã ghi giảm từ khoảng 8,9 xuống 7,1 giờ. Phép đo giữa các lượt chạy còn cần chuẩn hóa phần cứng, batch và điều kiện chạy để kết luận về tốc độ. Điện năng chưa được đo.
- **Phạm vi kết luận.** Ba seed thay đổi khởi tạo và quá trình huấn luyện trên cùng một split. Phân tích R52 cho thấy kết quả lặp lại giữa các seed trên split đó. Cần một tập mất cân bằng độc lập để kiểm tra khả năng khái quát.

## 5. Hướng viết bài và việc cần chốt trong nhóm

Câu hỏi nghiên cứu đề xuất là **khi nào hợp nhất embedding thích nghi cải thiện khả năng phân loại các lớp ít mẫu?** R52 là kết quả trung tâm. SST-2, IMDb, AGNews và R8 cho thấy hiệu quả thay đổi theo bộ dữ liệu. Các nghiên cứu trước đã khảo sát hợp nhất embedding đa mô hình và cách chọn tầng, nên phần khác biệt cần làm rõ là kết quả trên lớp ít mẫu dưới các đối chứng đủ mạnh. [Gwak & Jung, ACL 2026](https://aclanthology.org/2026.acl-long.1786/)

1. **Chạy đối chứng trên cache Exp 3 của R52.** So sánh Exp 4 với `Avg + Cat`, một MLP có số tham số gần tương đương và một baseline dùng weighted cross-entropy. Dùng cùng split và seed. Nếu Exp 4 vẫn dẫn, thử bỏ gate và self-attention lần lượt để xác định thành phần có đóng góp.
2. **Kiểm tra độ bền của kết quả.** Tính lại accuracy và macro-F1 sau khi lọc mẫu trùng. Báo cáo kết quả theo nhóm tần suất lớp dựa trên số mẫu train và xác nhận trên một bộ dữ liệu mất cân bằng độc lập.
3. **Chốt phạm vi bản thảo.** Đưa phân tích R52 lên phần chính, giữ ma trận năm tập để mô tả phạm vi kết quả và đưa FP16 vào phần đối chiếu bổ sung. Với đối chứng và xác nhận độc lập, có thể phát triển thành một bài báo thực nghiệm ngắn hoặc bài workshop.

## Nguồn số liệu

- Các bảng SST-2, R8, R52 và IMDb Exp 3–4 lấy từ [kho báo cáo Kaggle](kaggle_experiment/reports/README.md) và các CSV trong những thư mục chạy tương ứng.
- AGNews, các cấu hình IMDb còn lại và các lượt FP16 controlled được tổng hợp từ `tmp/results_dl/`, bằng các tệp `llmembed_runs/full/metrics.csv` và `environment.json`.
- Phân tích R52 tính từ `predictions.csv` trong `kaggle_experiment/runs/archive/r52/full_nf4_v7/`. Bảng FP16 tương thích tác giả lấy từ [manifest của lượt chạy SST-2](kaggle_experiment/reports/sst2_author_fp16_exp0_v11/run_manifest.json).
- Mô tả cấu hình đối chiếu với [mã điều phối thí nghiệm](kaggle_experiment/llmembed/run.py), [mã fusion và chunk pooling](kaggle_experiment/llmembed/models.py) và [bài LLMEmbed gốc](https://aclanthology.org/2024.acl-long.433/).
