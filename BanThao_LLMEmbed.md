# Bản Thảo Thiết Kế & Thực Hiện Thực Nghiệm: Dự Án LLMEmbed++

**Tên đề tài gợi ý:** *LLMEmbed++: Enhancing Lightweight LLM Text Classification via CoT-Guided Extraction, Hierarchical Long-Text Chunking, and Adaptive Cross-Model Fusion*  
**Bài báo nền tảng (Baseline):** *LLMEmbed: Rethinking Lightweight LLM’s Genuine Function in Text Classification (ACL 2024)*  
**Mã nguồn gốc tham chiếu:** `https://github.com/ChunLiu-cs/LLMEmbed-ACL2024`

---

## 1. Mục tiêu Nghiên cứu, Phạm vi Đóng góp & Giả thuyết Khoa học

Dựa trên kết quả thực nghiệm có kiểm soát (35/35 cấu hình trên 5 tập chuẩn + đối chứng mạnh trên `R52` và `Ohsumed`), tuyên bố đóng góp chính của nghiên cứu được **thu hẹp chặt chẽ** vào:

1. **Phân tích và khắc phục nút thắt trộn đặc trưng (Fusion Bottleneck) trên phân loại văn bản mất cân bằng / lớp ít mẫu:** Trên các tập nhiều lớp lệch phân bố như `R52` (52 nhãn) và `Ohsumed` (23 nhãn y khoa), đầu trộn `Cat + Co + Avg + Cat` (#15) của LLMEmbed gốc (nén tương tác qua ma trận Gram 49 chiều + Dropout 0.5) làm sụp hoàn toàn nhóm lớp hiếm (`Rare F1 = 0.00%`). Thay bằng đầu hợp nhất giàu dung lượng có `LayerNorm + GELU + Dropout 0.2` (`Matched MLP` hoặc `AG-CMF`) nâng `Macro-F1` tăng mạnh trên cả `R52` (`+22.9` đến `+27.1 pp`) lẫn `Ohsumed` (`+7.6` đến `+8.7 pp`) mà không gây đánh đổi Accuracy như `class-weighted loss`.
2. **Tách bạch lợi ích riêng của Fusion Thích nghi (`AG-CMF`) trước các đối chứng mạnh:** Khi giữ cố định 100% cùng bộ embedding FP16 và cân bằng số tham số (~29.02M), cơ chế self-attention + gating của `AG-CMF` cho lợi ích bổ sung rõ rệt trên bài toán nhiều lớp cực lệch như `R52` (`+4.16 pp` Macro-F1 và `+11.63 pp` Rare F1 so với `Matched MLP`), trong khi trên `Ohsumed` (23 lớp, chỉ 1 lớp `<10` mẫu) `Matched MLP` nhỉnh hơn nhẹ (`+1.03 pp` Macro-F1).
3. **Ngoài phạm vi tuyên bố chính (Explicitly Out of Scope):** Các thực nghiệm hiện tại **không** khẳng định suy luận CoT thực thụ, khả năng giải thích nhân quả (trọng số attention/gate chỉ là công cụ chẩn đoán), vượt SOTA toàn diện trên A100 FP16, hay tiết kiệm điện năng đo trực tiếp. Tập `MR` trong thực nghiệm dùng proxy IMDb 40k/10k nên không so trực tiếp 1-1 với split MR gốc của bài báo.

---

## 2. Chuẩn bị Môi trường, Dữ liệu và Cấu hình Cơ sở

### 2.1. Thống kê 5 Bộ dữ liệu Thực nghiệm
Giữ nguyên cách chia tập dữ liệu chuẩn của bài báo gốc và CARP (Sun et al., 2023) để đảm bảo so sánh công bằng tuyệt đối (Fair Comparison):

| Bộ dữ liệu | Tác vụ | Số nhãn ($C$) | Tập Train ($N_{\text{train}}$) | Tập Test ($N_{\text{test}}$) | Độ dài tối đa (từ) | Đặc điểm thử thách chính |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **SST-2** | Phân tích cảm xúc câu | $2$ | $67,349$ | $872$ | $56$ | Câu ngắn, nhiều cấu trúc đảo ngữ/ẩn dụ |
| **MR** | Đánh giá phim (Review) | $2$ | $40,000$ | $10,000$ | $2,470$ | **Văn bản rất dài**, tốn bộ nhớ VRAM và thời gian trích xuất |
| **AGNews** | Phân loại tin tức | $4$ | $120,000$ | $7,600$ | $177$ | Tập dữ liệu lớn, kiểm tra tốc độ huấn luyện batch lớn |
| **R8** | Phân loại tài liệu Reuters | $8$ | $5,485$ | $2,189$ | $964$ | Văn bản dài trung bình, phân phối nhãn lệch |
| **R52** | Phân loại tài liệu Reuters | $52$ | $6,532$ | $2,568$ | $1,039$ | **Số lượng nhãn lớn ($52$)**, dễ bị quá khớp khi nối vector (`Cat`) |

### 2.2. Cấu hình Phần cứng & Siêu tham số (Hyperparameters)
* **Phần cứng khuyến nghị:** $1 \times$ GPU NVIDIA A100 (40GB) hoặc RTX 3090/4090 (24GB) với độ chính xác bán phần `bfloat16` / `float16` khi rút trích đặc trưng.
* **Mô hình nền tiêu chuẩn (Standard Backbones):**
  * `meta-llama/Llama-2-7b-hf` ($d_{\text{llama2}} = 4096$, $32$ khối Transformer).
  * `bert-large-uncased` ($d_{\text{bert}} = 1024$, $24$ khối Transformer).
  * `roberta-large` ($d_{\text{roberta}} = 1024$, $24$ khối Transformer).
* **Siêu tham số huấn luyện Classifier Head:**
  * Kích thước lô (Batch size): $N_b = 1024$ (giống bài báo gốc).
  * Số kỷ nguyên (Epochs): $50 \text{--} 100$ epochs (hội tụ quanh epoch $50$).
  * Tối ưu hóa (Optimizer): AdamW với tốc độ học khởi tạo $\eta = 1 \times 10^{-4}$, Weight Decay $\lambda = 1 \times 10^{-2}$, Cosine Annealing LR Scheduler.
  * Chiều không gian ẩn đồng bộ: $K = 1024$.

---

## 3. Chi Tiết 6 Thực Nghiệm Cốt Lõi (Step-by-Step Implementation)

Dưới đây là lộ trình thực hiện tuần tự từ **Thực nghiệm 0 (Baseline)** đến **Thực nghiệm 5 (Mở rộng)** để bạn dễ dàng lập bảng Ablation Study.

---

### Thực nghiệm 0 (Exp 0): Tái hiện Baseline LLMEmbed Gốc
**Mục tiêu:** Chạy lại mã nguồn gốc của tác giả với cấu hình số 15 (`Cat + Co + Avg + Cat`) để xác nhận kết quả cơ sở (Avg. Accuracy $\approx 0.9620$).

* **Cách trích xuất:**
  * Đưa văn bản thô $x_i$ vào LLaMA2-7B, lấy trung bình tất cả các token tại $5$ khối cuối cùng ($d_m \in \{1, 2, 3, 4, 5\}$, tương ứng với các khối $\{28, 29, 30, 31, 32\}$):
    $$\phi_i^{(\text{llama2}, d_m)} = \frac{1}{L_i} \sum_{t=1}^{L_i} h_{i, t}^{(33 - d_m)} \in \mathbb{R}^{4096}$$
  * Trích xuất vector câu ở tầng cuối cùng của BERT-large ($\phi_i^{(\text{bert}, 1)} \in \mathbb{R}^{1024}$) và RoBERTa-large ($\phi_i^{(\text{roberta}, 1)} \in \mathbb{R}^{1024}$).
* **Cách trộn (`Cat + Co + Avg + Cat`):**
  * Chiếu tuyến tính $5$ vector của LLaMA2 từ $4096 \rightarrow 1024$, ghép với BERT và RoBERTa thành ma trận $\hat{\phi}_i \in \mathbb{R}^{7 \times 1024}$.
  * Tính tích ngoài chuẩn hóa:
    $$\psi_{i, \text{co}} = \tanh\left(2\sigma \left(\hat{\phi}_i \hat{\phi}_i^T\right)\right)$$
  * Nối $\psi_{i, \text{co}}$ với vector trung bình $\text{Avg}\left(\{\phi_i^{(\text{llama2}, d_m)}\}_{d_m=1}^5\right)$ rồi đưa vào tầng phân loại tuyến tính.

---

### Thực nghiệm 1 (Exp 1): Trích xuất Đa tầng Cách quãng (Strided Multi-Depth Extraction)
**Cơ sở lý thuyết:** Các khối liền kề nhau ở cuối mạng ($\{28, 29, 30, 31, 32\}$) có ma trận biểu diễn tương đồng cao (dư thừa thông tin). Theo các nghiên cứu phân tích tầng ẩn của LLM (*Layer-Aware Embedding Fusion, 2025*), các tầng trung gian (từ tầng $16$ đến tầng $28$) nắm giữ cấu trúc cú pháp và ngữ nghĩa phân loại tốt hơn tầng cuối cùng (vốn bị thiên lệch cho nhiệm vụ dự đoán từ tiếp theo).

* **Thiết kế thay đổi:**
  * Thay tập chỉ số độ sâu liền kề $d_m \in \{1, 2, 3, 4, 5\}$ (tức các khối $\{32, 31, 30, 29, 28\}$) bằng **tập khối cách quãng với bước nhảy $s = 4$**:
    $$\mathcal{B}_{\text{strided}} = \{16, 20, 24, 28, 32\}$$
  * Thử nghiệm phụ (Hyperparameter Search cho bước nhảy $s$): So sánh $s \in \{1, 2, 4, 6\}$:
    * $s = 1$: $\{28, 29, 30, 31, 32\}$ (Bài báo gốc)
    * $s = 2$: $\{24, 26, 28, 30, 32\}$
    * $s = 4$: $\{16, 20, 24, 28, 32\}$ (Đề xuất chính)
    * $s = 6$: $\{8, 14, 20, 26, 32\}$
* **Chi phí tính toán phát sinh:** Bằng $0$ (vẫn lấy đúng $5$ vector từ $1$ lần chạy qua LLaMA2-7B).

---

### Thực nghiệm 2 (Exp 2): CoT-Guided PromptEOL & Explainable Token Attention Pooling
**Cơ sở lý thuyết:** Giải quyết **Limitation 2** của bài báo gốc. Do LLaMA2 dùng mặt nạ nhân quả một chiều (Causal Attention Mask), token ở vị trí $t$ chỉ nhìn thấy các token $\le t$. Nếu lấy trung bình cộng mọi token như bài báo gốc, các token ở đầu câu mang rất ít ngữ cảnh toàn cục. Ngược lại, nếu bọc văn bản bằng một **Pretended CoT + PromptEOL template**, token cuối cùng của chuỗi sẽ đóng vai trò "điểm hội tụ" (anchor token) gom toàn bộ ngữ nghĩa suy luận của câu.

#### 2.1. Thiết kế Template Đầu vào (Single-Pass CoT Prompt)
Thay vì đưa văn bản thô $x_i$ vào LLaMA2, ta bọc $x_i$ bằng khung mẫu ngắn gọn tùy theo tác vụ (không yêu cầu mô hình sinh chữ):
* **Với bài toán Cảm xúc (SST-2, MR):**
  > `Text: "{x_i}". After analyzing the key sentiment clues step by step, the overall sentiment of this text in one word is:`
* **Với bài toán Chủ đề Tin tức (AGNews, R8, R52):**
  > `Article: "{x_i}". After identifying the core domain clues and reasoning step by step, the topic category of this article in one word is:`

#### 2.2. Kết hợp Anchor Token & Learnable Token-Level Attention
Tại mỗi khối $b \in \mathcal{B}_{\text{strided}}$, gọi chuỗi vector ẩn của các token là $H^{(b)} = [h_1^{(b)}, h_2^{(b)}, \dots, h_L^{(b)}] \in \mathbb{R}^{L \times 4096}$.
1. **Vector hội tụ cuối câu (CoT-Anchor Vector):** Lấy trực tiếp vector của token cuối cùng $h_L^{(b)}$.
2. **Vector trung bình có trọng số chú ý (Explainable Attention Pooling):** Trong quá trình huấn luyện Classifier Head (hoặc qua một mô-đun chọn lọc token nhẹ), ta tính trọng số chú ý $\alpha_t^{(b)}$ cho từng token thuộc văn bản gốc $x_i$:
   $$\alpha_t^{(b)} = \frac{\exp\left(\text{sim}\left(h_t^{(b)}, h_L^{(b)}\right) / \tau\right)}{\sum_{k=1}^{L-1} \exp\left(\text{sim}\left(h_k^{(b)}, h_L^{(b)}\right) / \tau\right)}$$
   Trong đó $\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ là độ tương đồng cosine giữa từng từ $h_t^{(b)}$ trong câu với vector hội tụ suy luận $h_L^{(b)}$ ở cuối câu, và $\tau = 0.1$ là tham số nhiệt độ (temperature).
3. **Biểu diễn tổng hợp tại khối $b$:**
   $$\phi_i^{(\text{llama2}, b)} = \gamma \cdot h_L^{(b)} + (1 - \gamma) \sum_{t=1}^{L-1} \alpha_t^{(b)} h_t^{(b)}$$
   Với $\gamma \in [0, 1]$ (mặc định $\gamma = 0.5$). Đặc biệt, cách tính $\alpha_t^{(b)}$ dựa trên tương đồng cosine với $h_L^{(b)}$ **không cần lưu toàn bộ tensor kích thước $[N, L, 4096]$ vào ổ cứng**, mà được tính trực tiếp ngay trong lúc rút trích đặc trưng (on-the-fly extraction)! Bạn vừa giữ được kích thước lưu trữ siêu nhẹ của LLMEmbed, vừa xuất ra được danh sách các từ có trọng số $\alpha_t^{(b)}$ cao nhất để vẽ **Heatmap Giải thích (Explainability)**!

#### 2.3. Mã nguồn PyTorch cho bước Rút trích Exp 1 + Exp 2
```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def extract_cot_strided_llama_embeddings(
    model, tokenizer, texts, task_type="sentiment", 
    strided_blocks=[16, 20, 24, 28, 32], gamma=0.5, tau=0.1
):
    """
    Trích xuất đặc trưng LLaMA2 tại các khối cách quãng với CoT-Guided Prompt
    và Cosine-Guided Token Attention Pooling (không tốn thêm bộ nhớ lưu trữ).
    """
    prompts = []
    for x in texts:
        if task_type == "sentiment":
            p = f'Text: "{x}". After analyzing the key sentiment clues step by step, the overall sentiment of this text in one word is:'
        else:
            p = f'Article: "{x}". After identifying the core domain clues and reasoning step by step, the topic category of this article in one word is:'
        prompts.append(p)

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(model.device)
    
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple gồm 33 tensors (1 embedding + 32 blocks)
    attention_mask = inputs["attention_mask"]  # [B, L]

    # Tìm vị trí của token thực cuối cùng (trước padding) cho mỗi câu trong batch
    seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
    batch_size = len(texts)
    batch_indices = torch.arange(batch_size, device=model.device)

    extracted_blocks = []
    for b in strided_blocks:
        H_b = hidden_states[b]  # [B, L, 4096]
        
        # 1. Lấy CoT-Anchor Vector (token cuối cùng của mỗi mẫu)
        h_last = H_b[batch_indices, seq_lengths]  # [B, 4096]
        
        # 2. Tính Cosine Similarity giữa từng token và h_last để làm trọng số chú ý
        H_b_norm = F.normalize(H_b, p=2, dim=-1)          # [B, L, 4096]
        h_last_norm = F.normalize(h_last, p=2, dim=-1).unsqueeze(1)  # [B, 1, 4096]
        sim_scores = torch.bmm(H_b_norm, h_last_norm.transpose(1, 2)).squeeze(-1) / tau  # [B, L]
        
        # Che (mask) các vị trí padding
        sim_scores = sim_scores.masked_fill(attention_mask == 0, -1e9)
        alpha = F.softmax(sim_scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        
        # 3. Tổ hợp có trọng số và kết hợp với h_last
        h_attn = torch.sum(alpha * H_b, dim=1)  # [B, 4096]
        phi_b = gamma * h_last + (1.0 - gamma) * h_attn  # [B, 4096]
        extracted_blocks.append(phi_b.cpu())

    # Trả về tensor kích thước [B, 5, 4096] và trọng số alpha ở khối cuối để làm Explainability
    return torch.stack(extracted_blocks, dim=1), alpha.squeeze(-1).cpu()
```

---

### Thực nghiệm 3 (Exp 3): Xử lý Văn bản Dài bằng Hierarchical Chunk-and-Fuse
**Cơ sở lý thuyết:** Giải quyết **Limitation 1** của bài báo gốc trên các bộ dữ liệu văn bản dài (MR, R8, R52).
* Trong bài báo gốc, văn bản dài tới $2,470$ từ (trên tập MR) khiến BERT và RoBERTa bị cắt bỏ toàn bộ phần văn bản sau $512$ tokens (mất tới $75\%$ nội dung đánh giá phim ở phần kết luận!). Đồng thời, việc nạp chuỗi dài $2,470$ từ vào LLaMA2 khiến ma trận Self-Attention kích thước $L \times L$ chiếm hết VRAM, làm giảm khả năng xử lý song song và kéo dài thời gian trích xuất tập MR lên tới $5\text{h}:43\text{m}:45\text{s}$.

#### 3.1. Quy trình Chia đoạn có Chồng lấp (Sliding Window Chunking)
Với một văn bản dài $x_i$ có độ dài $L_i > L_{\max}$ (đặt $L_{\max} = 256$ hoặc $384$ tokens):
1. Chia $x_i$ thành $M_i$ đoạn văn bản con (chunks) $\{c_{i, 1}, c_{i, 2}, \dots, c_{i, M_i}\}$, mỗi đoạn dài tối đa $L_{\max} = 256$ tokens với phần chồng lấp (overlap) là $32$ tokens (giới hạn tối đa $M_{\max} = 8$ đoạn để bao phủ hơn $2,000$ tokens).
2. Đưa tất cả các đoạn $c_{i, m}$ qua LLaMA2-7B, BERT-large và RoBERTa-large theo lô lớn (nhờ độ dài cố định ngắn $256$ tokens, GPU không bao giờ bị tràn bộ nhớ và tốc độ tính toán tăng vọt).
3. **Tổng hợp Đa cấp có Trọng số vị trí (Position-Aware Chunk Aggregation):**
   Trong các bài viết đánh giá (MR) hoặc tin tức (R8, R52), câu chủ đề và câu kết luận thường nằm ở **đoạn đầu tiên ($m=1$)** và **đoạn cuối cùng ($m=M_i$)**. Khi huấn luyện Classifier Head, ta gộp $M_i$ đoạn của văn bản $x_i$ thông qua cơ chế **Chunk-Level Self-Attention**:
   $$\beta_{i, m} = \frac{\exp\left(w_c^T \tanh\left(W_c z_{i, m} + b_c\right)\right)}{\sum_{k=1}^{M_i} \exp\left(w_c^T \tanh\left(W_c z_{i, k} + b_c\right)\right)}, \quad \phi_i = \sum_{m=1}^{M_i} \beta_{i, m} z_{i, m}$$
   Trong đó $z_{i, m}$ là vector đặc trưng của đoạn thứ $m$, và $\{W_c, b_c, w_c\}$ là các tham số siêu nhẹ được học cùng với Classifier Head.

---

### Thực nghiệm 4 (Exp 4): Bộ Trộn Đặc Trưng Thích Nghi (Adaptive Cross-Model & Gated MoE Fusion)
**Cơ sở lý thuyết:** Đây là đóng góp cốt lõi về mặt kiến trúc mạng để thay thế hoàn toàn Bảng 1 của bài báo gốc.
Trong LLMEmbed gốc:
* Phép nối trực tiếp (`Cat`) làm số chiều tăng vọt lên $5 \times 4096 + 2 \times 1024 = 22,528$ chiều, dẫn đến hiện tượng quá khớp nghiêm trọng trên tập R52 (độ chính xác tụt xuống $0.4217$).
* Phép `Co-occurrence Pooling` (`Cat + Co`) nhân ma trận $\hat{\phi}_i \hat{\phi}_i^T$ tạo ra ma trận tương quan $7 \times 7 = 49$ phần tử (hoặc tương tác tuyến tính cố định) khiến độ chính xác trên R52 chỉ đạt $0.8895$, buộc tác giả phải nối bù thêm vector `Avg` của LLaMA2 (`Cat + Co + Avg + Cat`) mới lên được $0.9568$.

Chúng ta đề xuất kiến trúc **Adaptive Gated Cross-Model Fusion (AG-CMF)** gồm 3 thành phần nối tiếp nhau:

#### 4.1. Căn chỉnh Không gian & Chuẩn hóa Lớp (Modality-Specific Projection)
Với $5$ vector từ LLaMA2 ($\{\phi_i^{(\text{llama2}, b)}\}_{b=1}^5 \in \mathbb{R}^{4096}$), $1$ vector từ BERT ($\phi_i^{(\text{bert})} \in \mathbb{R}^{1024}$) và $1$ vector từ RoBERTa ($\phi_i^{(\text{roberta})} \in \mathbb{R}^{1024}$), ta chiếu tất cả về cùng không gian chuẩn hóa $K = 1024$ chiều:
$$\hat{\phi}_{i, j} = \text{LayerNorm}\left(\text{GELU}\left(W_j \phi_{i, j} + b_j\right)\right) \in \mathbb{R}^{1024}, \quad j \in \{1, \dots, 7\}$$
Xếp chồng $7$ vector này thành ma trận đặc trưng đa nguồn $\hat{\Phi}_i = [\hat{\phi}_{i, 1}; \dots; \hat{\phi}_{i, 7}] \in \mathbb{R}^{7 \times 1024}$.

#### 4.2. Tương tác Chéo Đa đầu (Multi-Head Cross-Source Self-Attention)
Thay vì nhân ma trận cố định $\hat{\Phi}_i \hat{\Phi}_i^T$ như `Co-occurrence`, ta cho $7$ nguồn đặc trưng trao đổi thông tin qua một lớp **Multi-Head Self-Attention (MHA)** (coi $7$ nguồn như một chuỗi gồm $7$ tokens, mỗi token $1024$ chiều):
$$\tilde{\Phi}_i = \text{LayerNorm}\left(\hat{\Phi}_i + \text{MHA}\left(\hat{\Phi}_i, \hat{\Phi}_i, \hat{\Phi}_i\right)\right) \in \mathbb{R}^{7 \times 1024}$$

#### 4.3. Cổng Chuyên gia Thích nghi & Tương tác Bậc hai (MoE Gating + Hadamard Bilinear Interaction)
1. **Cổng trọng số động (Dynamic Router / MoE Gate):** Mỗi mẫu văn bản $x_i$ tự quyết định xem tầng nào của LLaMA2 hay BERT/RoBERTa là đáng tin cậy nhất thông qua mạng cổng:
   $$g_i = \text{softmax}\left(W_{\text{gate}} \tilde{\Phi}_i\right) \in \mathbb{R}^{7 \times 1}, \quad \psi_{i, \text{gated}} = \sum_{j=1}^7 g_{i, j} \tilde{\phi}_{i, j} \in \mathbb{R}^{1024}$$
2. **Tương tác bậc hai không nổ số chiều (Hadamard Generative-Discriminative Fusion):** Tách biểu diễn tổng hợp của nhánh Sinh (Generative: trung bình có trọng số của $5$ tầng LLaMA2, ký hiệu $u_{\text{gen}} \in \mathbb{R}^{1024}$) và nhánh Phân biệt (Discriminative: trung bình có trọng số của BERT và RoBERTa, ký hiệu $u_{\text{disc}} \in \mathbb{R}^{1024}$). Tính tích Hadamard (nhân từng phần tử $\odot$) kết hợp chuẩn hóa hàm mũ (Power Normalization):
   $$\psi_{i, \text{inter}} = \tanh\left(2\sigma \left(u_{\text{gen}} \odot u_{\text{disc}}\right)\right) \in \mathbb{R}^{1024}$$
3. **Vector biểu diễn cuối cùng:**
   $$\psi_i = \text{Concat}\left(\psi_{i, \text{gated}}, \psi_{i, \text{inter}}\right) \in \mathbb{R}^{2048}$$
   Đưa $\psi_i$ (chỉ $2048$ chiều, rất gọn nhẹ và giàu thông tin bậc cao) qua tầng phân loại có Dropout ($p = 0.2$) để dự đoán nhãn $\tilde{y}_i$.

#### 4.4. Mã nguồn PyTorch đầy đủ cho Mô-đun `AG-CMF` (Thay thế Classifier Head trong repo gốc)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGatedCrossModelFusion(nn.Module):
    def __init__(self, num_classes, llama_dim=4096, plm_dim=1024, hidden_dim=1024, num_heads=8, dropout=0.2, sigma=0.2):
        super().__init__()
        self.sigma = sigma
        
        # 1. Các tầng chiếu (Projection layers) cho 5 tầng LLaMA2, 1 BERT, 1 RoBERTa
        self.llama_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(llama_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(5)
        ])
        self.bert_proj = nn.Sequential(
            nn.Linear(plm_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.roberta_proj = nn.Sequential(
            nn.Linear(plm_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. Multi-Head Cross-Source Self-Attention trên 7 nguồn đặc trưng
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # 3. Cổng MoE (Dynamic Gating Router) học trọng số đóng góp của từng nguồn
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 4. Đầu phân loại (Classifier Head) nhận đầu vào 2048 chiều (Gated + Hadamard Interaction)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, llama_feats, bert_feat, roberta_feat):
        """
        llama_feats: Tensor [B, 5, 4096]
        bert_feat: Tensor [B, 1024]
        roberta_feat: Tensor [B, 1024]
        """
        # Bước 1: Chiếu cả 7 nguồn về cùng chiều K = 1024
        proj_list = [self.llama_projs[idx](llama_feats[:, idx, :]) for idx in range(5)]
        proj_list.append(self.bert_proj(bert_feat))
        proj_list.append(self.roberta_proj(roberta_feat))
        
        # Xếp chồng thành [B, 7, 1024]
        Phi_hat = torch.stack(proj_list, dim=1)
        
        # Bước 2: Tương tác chéo qua Multi-Head Attention
        attn_out, _ = self.cross_attn(Phi_hat, Phi_hat, Phi_hat)
        Phi_tilde = self.attn_norm(Phi_hat + attn_out)  # [B, 7, 1024]
        
        # Bước 3: Tính trọng số MoE Gate cho 7 nguồn
        gate_logits = self.gate_net(Phi_tilde)          # [B, 7, 1]
        gate_weights = F.softmax(gate_logits, dim=1)    # [B, 7, 1]
        psi_gated = torch.sum(gate_weights * Phi_tilde, dim=1)  # [B, 1024]
        
        # Bước 4: Tương tác bậc hai Hadamard giữa nhánh Generative (0..4) và Discriminative (5..6)
        u_gen = Phi_tilde[:, :5, :].mean(dim=1)         # [B, 1024]
        u_disc = Phi_tilde[:, 5:, :].mean(dim=1)        # [B, 1024]
        hadamard_prod = u_gen * u_disc
        psi_inter = torch.tanh(2.0 * self.sigma * hadamard_prod)  # Power Normalization [B, 1024]
        
        # Bước 5: Nối và phân loại
        psi_final = torch.cat([psi_gated, psi_inter], dim=-1)     # [B, 2048]
        logits = self.classifier(psi_final)
        
        return logits, gate_weights.squeeze(-1)  # Trả kèm gate_weights [B, 7] để vẽ biểu đồ phân tích đóng góp
```

---

### Thực nghiệm 5 (Exp 5): Khảo sát Tính Tổng quát trên Thế hệ Backbone Mới (Scalability & Generalization)
**Mục tiêu:** Chứng minh phương pháp của bạn không chỉ hoạt động tốt trên bộ ba (LLaMA2-7B + BERT + RoBERTa) mà còn tạo ra bước nhảy vọt khi thay thế bằng các mô hình mã nguồn mở hiện đại hơn có cùng quy mô tham số ($\sim 7\text{B}\text{--}8\text{B}$):
* **Cấu hình A (Standard - So sánh trực tiếp với bài báo gốc):** `LLaMA2-7B` + `BERT-large` + `RoBERTa-large`.
* **Cấu hình B (Upgraded Discriminative):** `LLaMA2-7B` + `DeBERTa-v3-large` + `RoBERTa-large`.
* **Cấu hình C (Next-Gen Lightweight LLMs):** `Llama-3-8B` (hoặc `Qwen2.5-7B`) + `DeBERTa-v3-large` + `RoBERTa-large`.

---

## 4. Kết Quả Thực Nghiệm Đã Ghi Nhận (Observed Results)

Toàn bộ số liệu dưới đây là **kết quả đo thực tế** trên Kaggle GPU (`Tesla T4 × 2`), trung bình qua **3 seeds (`42, 43, 44`)** (chi tiết tại `kaggle_experiment/reports/README.md`).

### Bảng 1: Ablation Study Đầy Đủ 35/35 (`NF4`, `controlled`, 50 epochs, 3 seeds)

| ID | Cấu hình Thực nghiệm | SST-2 | MR *(IMDb proxy)* | AGNews | R8 | R52 | Avg. 5 tập | $\Delta$ vs Exp 0 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 0** | Baseline NF4 ($\{28..32\}$, Mean Pool, `#15`) | 93.39% | 95.88% | 94.81% | 97.40% | 90.11% | 94.32% | — |
| **Exp 1** | $+$ Strided Multi-Depth ($\{16, 20, 24, 28, 32\}$) | 94.42% | 96.20% | 94.87% | 97.24% | 87.97% | 94.14% | -0.18 pp |
| **Exp 2** | $+$ CoT Prompt & Cosine Anchor Pooling | 95.22% | **96.96%** | 94.93% | 97.50% | 89.23% | 94.77% | +0.45 pp |
| **Exp 3** | $+$ Hierarchical Chunking + Learned ChunkPool | **95.76%** | 96.64% | **95.08%** | 97.52% | 89.82% | 94.96% | +0.64 pp |
| **Exp 4** | $+$ Adaptive Gated Cross-Model Fusion (`AG-CMF`) | 94.80% | 96.48% | 95.01% | 97.62% | **95.61%** | **95.90%** | **+1.58 pp** |
| **Exp 5B** | `AG-CMF` + DeBERTa-v3-large (thay BERT) | 94.50% | 96.62% | 95.01% | **97.65%** | 95.46% | 95.85% | +1.53 pp |
| **Exp 5C** | `AG-CMF` + Qwen2.5-7B + DeBERTa-v3-large | 95.22% | 96.61% | 94.86% | 97.50% | 95.21% | 95.88% | +1.56 pp |

*(Tham chiếu: bài báo LLMEmbed ACL 2024 báo cáo cấu hình #15 trên A100 FP16, 100 epochs đạt SST-2 `95.76%`, MR gốc `95.49%`, AGNews `95.83%`, R8 `98.22%`, R52 `95.68%`, trung bình `96.20%`. Chạy lại bản `FP16 + 100 epochs` của `Exp 4` trên T4 đạt R52 **`95.81%`** (+0.13 pp vs bài báo), R8 `97.61%`, SST-2 `95.57%`.)*

---

### Bảng 2: Đối Chứng Mạnh trên Dữ Liệu Mất Cân Bằng (Cùng Bộ Embedding FP16, 100 epochs, 3 seeds)

Cả 5 đầu phân loại dùng chung 100% cùng bộ embedding FP16, cùng split train/val/test, cùng 3 seeds và cùng tiêu chí chọn checkpoint theo `val_accuracy`. Nhóm tần suất lớp được chốt trước theo số mẫu train: `Rare (<10)`, `Medium (10–49)`, `Frequent (>=50)`.

| Tập dữ liệu | Cấu hình | Tham số | Accuracy | Macro-F1 | Rare F1 (`<10`) | Medium F1 (`10–49`) | Frequent F1 (`>=50`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **R52** *(52 lớp)* | `exp4_ctrl_avgcat` (`Avg+Cat` #11) | 6.96M | 91.91% ± 0.08% | 42.54% | 0.00% | 52.75% | 87.62% |
| **R52** *(52 lớp)* | `exp3` (Baseline #15) | 25.89M | 93.22% ± 0.04% | 49.85% | 0.00% | 68.54% | 90.34% |
| **R52** *(52 lớp)* | `exp4_ctrl_weighted` (#15 + Class-Weighted Loss) | 25.89M | 91.43% ± 0.38% | 74.09% | **54.35%** | 83.41% | 86.61% |
| **R52** *(52 lớp)* | `exp4_ctrl_mlp` (MLP cân bằng tham số) | 29.02M | 95.56% ± 0.04% | 72.77% | 39.09% | 89.59% | **92.47%** |
| **R52** *(52 lớp)* | **`exp4` (AG-CMF)** | 29.02M | **95.81%** ± 0.21% | **76.94%** | 50.72% | **90.43%** | 91.53% |
| **Ohsumed** *(23 lớp)* | `exp4_ctrl_avgcat` (`Avg+Cat` #11) | 6.96M | 67.66% ± 2.07% | 50.75% | 0.00% | 29.91% | 63.86% |
| **Ohsumed** *(23 lớp)* | `exp3` (Baseline #15) | 25.89M | 69.83% ± 0.56% | 58.08% | 0.00% | 48.37% | 66.48% |
| **Ohsumed** *(23 lớp)* | `exp4_ctrl_weighted` (#15 + Class-Weighted Loss) | 25.89M | 65.45% ± 0.35% | 59.71% | 11.99% | 57.77% | 63.79% |
| **Ohsumed** *(23 lớp)* | `exp4` (AG-CMF) | 29.02M | 72.58% ± 0.59% | 65.70% | 9.89% | 62.69% | 70.83% |
| **Ohsumed** *(23 lớp)* | **`exp4_ctrl_mlp`** (MLP cân bằng tham số) | 29.02M | **73.06%** ± 0.20% | **66.73%** | **20.46%** | **64.31%** | **70.95%** |

**Phân tích rút ra từ Bảng 2:**
- Trên cả hai miền độc lập (`R52` tin tức tài chính và `Ohsumed` y khoa), `Baseline #15` và `Avg+Cat` đều sụp hoàn toàn ở lớp hiếm (`Rare F1 = 0.00%`), còn `class-weighted loss` đánh đổi quá nặng `Accuracy` (`-4.38 pp` trên R52, `-7.13 pp` trên Ohsumed so với `Exp 4`).
- Phần cải thiện chính trước `Baseline #15` đến từ việc bỏ nút thắt Gram 49 chiều / Dropout 0.5 bằng đầu phân loại giàu dung lượng có `LayerNorm + GELU + Dropout 0.2`. Riêng cơ chế **self-attention + gating (`AG-CMF`)** vượt rõ `Matched MLP` cùng 29.02M tham số trên bài toán 52 lớp cực lệch (`R52`: `+4.16 pp` Macro-F1, `+11.63 pp` Rare F1), nhưng không vượt `Matched MLP` trên bài toán 23 lớp ít cực đoan hơn (`Ohsumed`: `-1.03 pp` Macro-F1).

---

## 5. Quy Trình Tổ Chức Mã Nguồn & Chạy Lệnh Thực Tế

Bạn có thể tải mã nguồn gốc của tác giả về và nâng cấp theo cấu trúc thư mục sau:

```text
LLMEmbed-PlusPlus/
├── data/                        # Chứa 5 bộ dữ liệu: SST-2, MR, AGNews, R8, R52
├── extract_features.py          # Nâng cấp từ code gốc: Tích hợp Exp 1 (Strided), Exp 2 (CoT-Prompt), Exp 3 (Chunking)
├── models/
│   ├── baseline_fusion.py       # Giữ nguyên 15 phép trộn của Bảng 1 bài báo gốc (Exp 0)
│   └── ag_cmf_head.py           # Chứa class AdaptiveGatedCrossModelFusion (Exp 4)
├── train_eval.py                # Kịch bản huấn luyện Classifier Head (Batch size = 1024, 50-100 epochs)
├── visualize_explainability.py  # Xuất Heatmap từ khóa (alpha_t) và biểu đồ trọng số cổng (gate_weights)
└── run_all_ablations.sh         # Script tự động chạy toàn bộ Exp 0 -> Exp 5 và xuất kết quả ra CSV
```

### Kịch bản Bash (`run_all_ablations.sh`) để chạy tự động toàn bộ thực nghiệm:
```bash
#!/bin/bash
set -e

DATASETS=("sst2" "mr" "agnews" "r8" "r52")

for DS in "${DATASETS[@]}"; do
    echo "===================================================="
    echo "Đang xử lý bộ dữ liệu: $DS"
    echo "===================================================="

    # 1. Chạy Exp 0: Baseline gốc (Blocks 28-32, Raw text, Mean pooling)
    python extract_features.py --dataset $DS --blocks 28 29 30 31 32 --mode raw_mean --save_dir ./feats/exp0/$DS
    python train_eval.py --dataset $DS --feat_dir ./feats/exp0/$DS --fusion_type cat_co_avg_cat --seed 42

    # 2. Chạy Exp 1: Strided Blocks (Blocks 16, 20, 24, 28, 32)
    python extract_features.py --dataset $DS --blocks 16 20 24 28 32 --mode raw_mean --save_dir ./feats/exp1/$DS
    python train_eval.py --dataset $DS --feat_dir ./feats/exp1/$DS --fusion_type cat_co_avg_cat --seed 42

    # 3. Chạy Exp 2 & Exp 3: Strided Blocks + CoT-Guided Token Attention + Hierarchical Chunking
    python extract_features.py --dataset $DS --blocks 16 20 24 28 32 --mode cot_attn_chunk --chunk_size 256 --save_dir ./feats/exp2_3/$DS
    python train_eval.py --dataset $DS --feat_dir ./feats/exp2_3/$DS --fusion_type cat_co_avg_cat --seed 42

    # 4. Chạy Exp 4 (Full LLMEmbed++): Kết hợp đặc trưng Exp 2_3 với bộ trộn AG-CMF
    python train_eval.py --dataset $DS --feat_dir ./feats/exp2_3/$DS --fusion_type ag_cmf --lr 1e-4 --epochs 50 --seed 42
done
```