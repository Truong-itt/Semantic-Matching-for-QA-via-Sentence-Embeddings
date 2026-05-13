# BÁO CÁO ĐỀ TÀI

## Semantic Matching for QA via Sentence Embeddings
### Unsupervised vs Supervised Sentence Selection

> Bài toán sentence-selection question answering trên SQuAD v1.1, được triển khai trong project `sentence_selection_qa`.

**Học phần:** Xử lý ngôn ngữ tự nhiên (420300138501)  
**Người thực hiện:** Hồ Duy Trường  
**Giáo viên hướng dẫn:** TS. Bùi Thanh Hùng  
**Thời gian hoàn thành:** Tháng 5 năm 2026

---

## 1. Tóm tắt

Project này chuyển bài toán QA từ span extraction sang sentence selection: với một câu hỏi và một đoạn văn đã tách câu, hệ thống chọn ra câu có khả năng chứa đáp án cao nhất.

Hệ thống được xây dựng theo 2 hướng:

- **Unsupervised:** so khớp question-sentence bằng cosine hoặc euclidean trên embedding.
- **Supervised:** biến mỗi cặp (question, sentence) thành vector đặc trưng và huấn luyện bộ phân loại nhị phân.

Các thành phần chính trong project:

- Tiền xử lý SQuAD v1.1 sang dạng sentence-selection.
- Encoder: TF-IDF + SVD, SBERT, BM25.
- Mô hình: unsupervised selector, supervised selector, BiLSTM encoder (đã implement nhưng chưa nối vào runner chính).
- Đánh giá: Accuracy@k, MRR, precision/recall/F1, biểu đồ và error analysis.
- Demo web Flask để nhập question/context và nhận câu được chọn.

### 1.1 Tính cấp thiết của đề tài

Trong QA thực tế, một câu hỏi thường không cần một câu trả lời có độ chính xác ký tự tuyệt đối, mà cần một đơn vị thông tin đủ gần và đủ đúng về mặt ngữ nghĩa. Với nhiều hệ thống retrieval, sentence-level selection là một cách cân bằng giữa hiệu năng, độ phức tạp và khả năng giải thích. Nó nhẹ hơn span extraction, nhưng vẫn giữ được khả năng lọc nhiễu rất tốt trong context dài.

Trong bối cảnh project này, sentence selection còn có giá trị như một baseline học thuật: nó cho phép so sánh trực quan giữa lexical matching, dense semantic matching và supervised reranking. Nhờ đó, người đọc có thể thấy rõ mức đóng góp của từng tầng xử lý thay vì chỉ nhìn vào một model cuối cùng.

### 1.2 Mục tiêu học thuật và kỹ thuật

Về học thuật, đề tài nhằm làm rõ ba câu hỏi chính:

- Khi nào lexical overlap là đủ tốt để chọn câu?
- Khi nào cần embedding ngữ nghĩa như SBERT để vượt qua giới hạn lexical?
- Khi nào supervised reranking thực sự tạo thêm giá trị so với similarity thuần túy?

Về kỹ thuật, project hướng đến một pipeline có thể chạy ổn định trên CPU, có thể mở rộng về sau, và đủ rõ ràng để báo cáo lại bằng số liệu, biểu đồ và error analysis.

### 1.3 Phạm vi và giả định của bài toán

Project làm việc với giả định rằng câu trả lời thường nằm gọn trong một câu. Đây là giả định quan trọng vì nó cho phép biến QA thành bài toán xếp hạng câu. Khi giả định này không còn đúng, ví dụ answer span kéo dài qua nhiều câu hoặc context chứa answer phân tán, sentence selection sẽ bắt đầu bộc lộ giới hạn.

Giả định này không phải hạn chế riêng của project mà là một thiết kế có chủ đích. Mục tiêu ở đây là xây dựng một hệ QA đơn giản, dễ kiểm soát và dễ so sánh trước khi nghĩ đến các mô hình lớn hơn.

---

## 1. Tổng quan nghiên cứu

### 1.1 Sentence selection trong QA cổ điển

Trước khi transformer trở nên phổ biến, nhiều hệ QA đi theo hướng pipeline: truy xuất tài liệu liên quan, chọn đoạn văn, rồi chọn câu hoặc span trong đoạn đó. Sentence selection xuất hiện như một lớp trung gian có tác dụng giảm không gian tìm kiếm. Trong nhiều hệ thống, bước này có thể được xem như document reranking ở mức vi mô.

Khi chỉ cần chọn đúng câu chứa thông tin trả lời, độ phức tạp của vấn đề giảm đáng kể. Điều này đặc biệt hữu ích trong các tình huống context dài, nơi việc xử lý tất cả token bằng một mô hình span extractor là tốn kém và dễ bị nhiễu.

### 1.2 Lexical matching và sparse retrieval

Các phương pháp lexical matching như TF-IDF, cosine similarity hoặc BM25 dựa trên giả định rằng những từ trùng nhau là tín hiệu mạnh cho sự liên quan. Trong QA, giả định này thường đúng ở mức nào đó vì câu hỏi và câu chứa đáp án thường có ít nhất một số từ khóa tương đồng.

TF-IDF cung cấp một baseline rất tốt vì nó đơn giản, dễ triển khai và có thể hoạt động ổn ngay cả khi không có GPU. BM25 lại cải tiến TF-IDF bằng cách tính đến độ dài câu và độ bão hòa tf, nên thường mạnh hơn trong retrieval thực tế. Tuy vậy, cả hai đều phụ thuộc nhiều vào lexical overlap và dễ bị hạn chế khi câu đúng diễn đạt khác câu hỏi.

### 1.3 Dense embeddings và semantic matching

Sự phát triển của sentence embedding, đặc biệt là SBERT, đã thay đổi cách so khớp văn bản. Thay vì chỉ đếm từ xuất hiện, dense embedding cố gắng mã hóa ngữ nghĩa của cả câu vào một vector có cấu trúc tốt hơn cho similarity. Điều này giúp mô hình xử lý tốt các trường hợp paraphrase và diễn đạt lại.

Trong bài toán sentence selection, dense embedding có lợi rõ rệt khi câu hỏi và câu trả lời không chia sẻ nhiều token nhưng vẫn mang quan hệ ngữ nghĩa mạnh. Đây chính là lý do SBERT thường vượt TF-IDF trong các benchmark ngắn và vừa.

### 1.4 Supervised reranking

Nếu dense embedding là cách nâng cấp biểu diễn, thì supervised reranking là cách nâng cấp quyết định. Thay vì chỉ dựa vào một metric cố định, mô hình được học trực tiếp từ nhãn câu đúng. Điều này cho phép classifier học ra boundary phù hợp hơn với dữ liệu.

Trong nhiều hệ retrieval hiện đại, reranking là một chiến lược rất quan trọng: bước đầu dùng retrieval nhanh để lấy top-k, sau đó một model mạnh hơn xếp hạng lại. Project này không triển khai full pipeline hai tầng, nhưng tinh thần supervised selector vẫn gần với reranking, vì nó học cách phân biệt positive và negative pairs trong cùng context.

### 1.5 Vị trí của project trong dòng nghiên cứu

Project này không nhằm vượt qua các mô hình SOTA, mà nhằm xây dựng một baseline có cấu trúc tốt và giải thích rõ ràng. Nó nằm ở giao điểm của ba hướng:

- retrieval cổ điển.
- sentence embedding.
- supervised pair classification.

Nhờ vậy, project có thể được dùng như một bài thực hành học thuật đầy đủ, đồng thời là nền tảng cho các nghiên cứu sâu hơn sau này.

---

## 2. Bài toán và dữ liệu

### 2.1 Định nghĩa bài toán

Cho câu hỏi `Q` và đoạn văn `P` được tách thành `N` câu `{S1, S2, ..., SN}`, mục tiêu là chọn chỉ số câu `Si` có xác suất chứa câu trả lời cao nhất.

**Input**

- `question`: câu hỏi tự nhiên.
- `sentences`: danh sách câu trong context.
- `label`: chỉ số câu chứa answer span trong dữ liệu huấn luyện/đánh giá.

**Output**

- `predicted_idx`: chỉ số câu được dự đoán.

### 2.2 Bộ dữ liệu

Project dùng **SQuAD v1.1** từ HuggingFace `datasets`:

```python
from datasets import load_dataset
dataset = load_dataset("squad")
```

Quá trình chuyển đổi sang dạng sentence-selection được thực hiện trong `preprocessing/convert_dataset.py`:

1. Tách passage thành câu bằng tokenizer.
2. Tìm câu đầu tiên chứa `answer_text` theo so khớp không phân biệt hoa/thường.
3. Gán `label` là index của câu đó.
4. Loại bỏ mẫu không tìm thấy câu chứa đáp án.

**Kích thước processed data hiện có trong repo:**

- `data/processed/train.json`: 4,972 mẫu
- `data/processed/val.json`: 998 mẫu

Mỗi sample có cấu trúc:

```json
{
  "id": "...",
  "question": "...",
  "sentences": ["...", "..."],
  "label": 1
}
```

### 2.3 Sentence tokenizer

Tokenizer nằm trong `preprocessing/sentence_tokenizer.py`:

- Ưu tiên `nltk.sent_tokenize` với `punkt`.
- Có fallback regex khi NLTK không khả dụng.
- Hàm `find_answer_sentence_idx()` trả về sentence index đầu tiên chứa đáp án.

---

## 3. Kiến trúc project

### 3.1 Luồng chạy chính

`main.py` là entry point thống nhất cho toàn bộ pipeline. Các mode hiện có:

- `full`: preprocess -> unsupervised -> supervised -> compare.
- `preprocess`: tạo `data/processed/train.json` và `val.json`.
- `unsupervised`: chạy các baseline không giám sát.
- `supervised`: huấn luyện các classifier có giám sát.
- `compare`: gộp kết quả từ JSON đã lưu và xuất bảng/biểu đồ tổng hợp.
- `demo`: chạy demo CLI cho một cặp question/context.

### 3.2 Các thư mục chính

| Thành phần | Vai trò |
|---|---|
| `preprocessing/` | Tách câu và chuyển SQuAD sang sentence-selection |
| `embeddings/` | TF-IDF encoder, SBERT encoder, BM25 encoder, BiLSTM encoder |
| `models/` | Unsupervised selector và supervised selector |
| `evaluation/` | Metrics, bảng kết quả, plotting, error analysis |
| `experiments/` | Runner cho unsupervised và supervised experiments |
| `webapp/` | Flask demo |
| `results/` | CSV, JSON, plots, error reports |
| `saved_models/` | Model pickle của supervised experiments |

---

## 4. Phương pháp

### 4.1 Encoder

#### TF-IDF + Truncated SVD

Encoder này là baseline nhẹ, dùng cho cả unsupervised và supervised:

- `TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)`
- `TruncatedSVD(n_components=256)`
- `Normalizer(copy=False)`

Ưu điểm: nhanh, dễ chạy trên CPU, ít phụ thuộc.

Nhược điểm: phụ thuộc lexical overlap, khó bắt paraphrase sâu.

#### Sentence-BERT

- Model: `all-MiniLM-L6-v2`
- Output dimension: 384
- Encoding đã được chuẩn hoá L2

Ưu điểm: biểu diễn ngữ nghĩa tốt hơn, đặc biệt với paraphrase.

#### So sánh TF-IDF và SBERT

Hai encoder này đại diện cho hai cách tiếp cận rất khác nhau.

TF-IDF xem văn bản như một tập từ có trọng số. Nó rất hiệu quả khi vấn đề chính là nhận diện từ khóa. Ngược lại, SBERT xem câu như một đơn vị ngữ nghĩa tương đối hoàn chỉnh và cố gắng đưa các câu gần nghĩa vào gần nhau trong không gian vector.

Trong sentence selection, TF-IDF thường tốt khi câu gold chứa từ khóa trùng với question. SBERT mạnh hơn khi câu gold được paraphrase hoặc khi những từ khóa quan trọng không xuất hiện nguyên dạng trong câu hỏi.

#### Vai trò của SVD trong TF-IDF

SVD giúp biến vector sparse chiều lớn thành dense vector kích thước cố định. Điều này có hai tác dụng:

1. Giảm chiều để tăng tốc tính similarity.
2. Làm cho biểu diễn dễ dùng hơn với classifier có giám sát.

Tuy nhiên, SVD không biến TF-IDF thành semantic encoder theo đúng nghĩa. Nó chỉ nén thông tin lexical theo không gian thấp chiều hơn. Vì vậy, dù tiện lợi, TF-IDF + SVD vẫn là một baseline thống kê chứ không phải model hiểu ngữ nghĩa sâu.

#### BM25

- Dùng `rank_bm25.BM25Okapi`.
- Là baseline retrieval sparse, không neural.

#### BiLSTM Encoder

`embeddings/bilstm_encoder.py` đã được implement như một encoder tái sử dụng, nhưng hiện chưa được nối vào các experiment runner chính.

### 4.2 Unsupervised selector

`models/unsupervised.py` sử dụng vector similarity để chọn câu.

- `cosine`: điểm càng cao càng tốt.
- `euclidean`: khoảng cách được negate để API đồng nhất là “cao hơn = tốt hơn”.

#### Công thức trực giác

Với cosine similarity, hai vector càng cùng hướng thì score càng cao:

$$
cos(q, s) = (q \cdot s) / (||q|| ||s||)
$$

Với euclidean distance, hai vector càng gần nhau thì distance càng nhỏ. Trong project, distance được đổi dấu để thành score:

$$
score(q, s) = -||q - s||_2
$$

Điều này giúp selector luôn dùng cùng một quy ước: score lớn hơn nghĩa là câu tốt hơn.

Quy trình:

1. Encode `question` và toàn bộ `sentences`.
2. Tính score cho từng câu.
3. `argmax(score)` trả về index câu được chọn.

### 4.3 Supervised selector

`models/supervised.py` biến bài toán thành phân loại nhị phân trên cặp `(question, sentence)`.

#### Feature vector

Với mỗi cặp `(q, s)`:

- cosine similarity: 1 chiều
- element-wise product: `D` chiều
- absolute difference: `D` chiều
- concatenation `[q; s]`: `2D` chiều

Tổng kích thước: `1 + 3D`.

#### Vì sao chọn các feature này

Các feature này được chọn vì chúng bổ sung lẫn nhau:

- cosine similarity nắm bắt mức độ gần nhau tổng quát.
- element-wise product làm nổi bật các chiều mà question và sentence cùng kích hoạt.
- absolute difference cho biết mức lệch theo từng chiều.
- concatenation giữ nguyên thông tin riêng của từng vector để classifier tự học tương tác.

Nếu chỉ dùng cosine, mô hình supervised sẽ không khác nhiều so với unsupervised. Việc thêm product, diff và concat làm không gian đặc trưng giàu hơn, cho phép classifier học các pattern tinh vi hơn.

#### Cách xây dựng negative samples

Negative samples được lấy ngẫu nhiên từ các câu còn lại trong cùng context. Cách này có hai ưu điểm:

1. Negative đủ khó vì chúng đến từ cùng passage nên thường cùng chủ đề.
2. Mô hình buộc phải phân biệt câu đúng với câu “gần đúng”, thay vì chỉ học tín hiệu quá dễ.

Điểm cần lưu ý là negative sampling ngẫu nhiên có thể làm kết quả dao động nhẹ giữa các lần chạy. Tuy nhiên với cấu hình hiện tại và random seed cố định, kết quả vẫn đủ ổn định để so sánh.

#### Classifiers

| Model | Cấu hình mặc định |
|---|---|
| Logistic Regression | `max_iter=1000`, `C=1.0`, `solver='lbfgs'` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `random_state=42` |
| XGBoost | `n_estimators=300`, `max_depth=6`, `learning_rate=0.1` |


### 4.4 Evaluation

`evaluation/metrics.py` đóng vai trò tính toán và chuẩn hóa toàn bộ thước đo. Điểm quan trọng nhất là cách đánh giá được thiết kế thống nhất cho cả unsupervised và supervised, nên các runner khác nhau có thể dùng chung một pipeline kiểm tra mà không cần viết lại logic đo lường.

Các metric chính gồm:

- Accuracy@k: đo tỷ lệ mẫu có câu gold nằm trong top-k dự đoán.
- Recall@k: trong bài toán 1-of-N này, bằng với Accuracy@k.
- MRR: mean reciprocal rank, phản ánh vị trí xếp hạng của câu đúng.
- Precision / Recall / F1: trong thiết lập top-1, các giá trị này trùng với accuracy.

Ngoài các thước đo số học, project còn sinh các biểu đồ trực quan để hỗ trợ phân tích:

- Accuracy@k comparison.
- MRR comparison.
- Confusion/rank distribution.

Các biểu đồ này giúp người đọc không chỉ biết model nào tốt hơn mà còn thấy được mức độ cải thiện ở các ngưỡng top-k.

### 4.5 Ý nghĩa của MRR trong bài toán này

MRR đặc biệt hữu ích khi ta muốn biết câu đúng thường đứng ở vị trí nào. Hai hệ thống có thể cùng Accuracy@1 nhưng MRR khác nhau nếu một hệ thường xếp câu đúng ở vị trí 2-3 trong những trường hợp sai.

Trong sentence selection, điều này quan trọng vì đôi khi top-1 sai nhưng top-3 vẫn chứa câu đúng. Khi đó, một downstream reranker hoặc answer extractor vẫn có thể tận dụng top-k candidate.

### 4.6 Vì sao top-5 gần như rất cao

Khi context chỉ có vài câu, việc câu đúng có mặt trong top-5 trở nên khá dễ. Do đó, Accuracy@5 thường tiến gần 1.0. Điều này không có nghĩa mô hình đã giải quyết hoàn toàn bài toán, mà chỉ cho thấy candidate generation ở mức câu đã khá tốt.

Nói cách khác, thách thức thực sự nằm ở top-1, nơi mô hình phải sắp xếp chính xác nhất giữa những câu rất gần nhau về mặt ngữ nghĩa.

---

## 5. Kết quả thực nghiệm

### 5.1 Thiết lập thực nghiệm

Các run hiện tại được thực hiện trên môi trường Windows, Python 3.10, chạy CPU. Đây là thiết lập đủ để kiểm tra các baseline của project mà không cần hạ tầng GPU chuyên dụng.

Tham số dữ liệu hiện dùng trong run:

- Train: 4.972 mẫu.
- Validation: 998 mẫu.

Mặc dù kích thước dữ liệu này nhỏ hơn bộ SQuAD gốc, nó đã đủ để làm các so sánh tương đối giữa encoder và classifier, đồng thời giúp quá trình chạy nhanh hơn và thuận tiện hơn khi demo.

### 5.2 Kết quả unsupervised

Kết quả được lưu trong `results/tables/unsupervised_results.json`.

| System | Acc@1 | Acc@3 | Acc@5 | MRR | F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Cosine | 0.7244 | 0.9439 | 0.9850 | 0.8338 | 0.7244 |
| TF-IDF + Euclidean | 0.7244 | 0.9439 | 0.9850 | 0.8338 | 0.7244 |
| BM25 | 0.7214 | 0.9539 | 0.9850 | 0.8393 | 0.7214 |
| SBERT + Cosine | 0.7615 | 0.9599 | 0.9870 | 0.8594 | 0.7615 |
| SBERT + Euclidean | 0.7615 | 0.9599 | 0.9870 | 0.8594 | 0.7615 |

Phân tích kết quả:

- SBERT + Cosine đạt Accuracy@1 cao nhất trong nhóm unsupervised.
- TF-IDF + Cosine là baseline mạnh, đặc biệt xét theo chi phí tính toán.
- BM25 có MRR tốt, phản ánh năng lực retrieval sparse đáng kể.
- Cosine và Euclidean gần như tương đương vì vector đã được normalize tốt.

### 5.3 Kết quả supervised

Kết quả được lưu trong `results/tables/supervised_results.json`.

| System | Acc@1 | Acc@3 | Acc@5 | MRR | F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | 0.6964 | 0.9259 | 0.9760 | 0.8116 | 0.6964 |
| TF-IDF + RF | 0.7335 | 0.9369 | 0.9850 | 0.8356 | 0.7335 |
| TF-IDF + XGB | 0.7194 | 0.9489 | 0.9870 | 0.8337 | 0.7194 |
| SBERT + LR | 0.7555 | 0.9599 | 0.9830 | 0.8550 | 0.7555 |
| SBERT + RF | 0.7565 | 0.9539 | 0.9890 | 0.8568 | 0.7565 |
| TF-IDF + Cosine (Unsup) | 0.7244 | 0.9439 | 0.9850 | 0.8338 | 0.7244 |

### 5.4 Ablation theo neg_per_pos

| neg_per_pos | Acc@1 | Acc@3 | Acc@5 | MRR |
|---|---:|---:|---:|---:|
| 1 | 0.7034 | 0.9319 | 0.9770 | 0.8154 |
| 3 | 0.6964 | 0.9259 | 0.9760 | 0.8116 |
| 5 | 0.6954 | 0.9309 | 0.9780 | 0.8105 |

### 5.5 Bàn luận thực nghiệm

Các kết quả trên đưa ra vài nhận xét quan trọng:

1. SBERT là encoder mạnh nhất trong toàn bộ các mô hình hiện tại. Khi embedding ngữ nghĩa tốt hơn, cả unsupervised lẫn supervised đều hưởng lợi.
2. Random Forest tỏ ra phù hợp với feature pairwise do project xây dựng. Nó bắt được quan hệ phi tuyến tốt hơn logistic regression trong nhiều trường hợp.
3. XGBoost có hiệu năng cạnh tranh nhưng chưa vượt RF ở Acc@1 trong run hiện tại.
4. Việc tăng số negative mỗi positive không cải thiện mạnh kết quả, điều đó cho thấy bài toán không chỉ phụ thuộc vào class balance mà còn phụ thuộc lớn vào chất lượng biểu diễn.

### 5.6 Kết luận ngắn từ kết quả

Nếu xét chung toàn bộ hệ thống, mô hình có Accuracy@1 cao nhất là SBERT + Cosine. Nếu xét riêng nhóm supervised, SBERT + RF là tốt nhất. Nếu xét yếu tố chi phí, TF-IDF + Cosine vẫn là lựa chọn rất thực dụng.

---

## 6. Demo web

### 6.1 Triển khai

Demo web của project được chạy qua `run_webapp.py`.

```bash
python run_webapp.py
```

Sau khi khởi động, ứng dụng mở tại `http://127.0.0.1:5000`.

### 6.2 Chức năng hiện có

Web app hiện cung cấp các chức năng sau:

- Giao diện nhập question và context.
- Dự đoán sentence được chọn bằng metric cosine hoặc euclidean.
- So sánh cosine và euclidean trong cùng một request.
- Hiển thị danh sách ví dụ mẫu để kiểm thử nhanh.

### 6.3 Cách hoạt động của backend

Trong `webapp/app.py`, encoder TF-IDF được fit theo kiểu lazy load. Nghĩa là encoder chỉ được khởi tạo khi có request đầu tiên hoặc khi server start, thay vì phải fit trong quá trình build ứng dụng.

Nguồn dữ liệu fit encoder gồm:

- Các ví dụ hard-coded trong ứng dụng.
- Một phần dữ liệu validation nếu `data/processed/val.json` tồn tại.

Sau khi encoder sẵn sàng, app dùng `UnsupervisedSelector` để chấm điểm từng câu trong context. Kết quả trả về có cả raw score lẫn normalized score để frontend dễ hiển thị.

### 6.4 Các API endpoint

Ba endpoint chính của app là:

- GET /: trả về trang HTML.
- POST /api/predict: dự đoán sentence tốt nhất cho question/context.
- POST /api/compare: so sánh cosine và euclidean song song.
- GET /api/examples: trả ví dụ mẫu.

### 6.5 Vai trò của web demo trong project

Web demo không chỉ là phần trình diễn giao diện. Về mặt kỹ thuật, nó xác nhận rằng pipeline sentence-selection có thể được đóng gói thành một dịch vụ nhỏ, dễ dùng, dễ kiểm tra. Đây là bước rất hữu ích nếu sau này muốn đưa project lên server hoặc tích hợp vào một hệ thống QA lớn hơn.

---

## 7. Hạn chế và hướng phát triển

### 7.1 Hạn chế hiện tại

1. BiLSTM chưa được nối vào runner chính. Mặc dù encoder đã implement, hiện nó chưa tham gia vào thí nghiệm chính thức.
2. Web demo mới dừng ở TF-IDF unsupervised scoring. Chưa expose SBERT hay supervised model qua API.
3. Bài toán vẫn ở mức chọn câu. Chưa có span extraction hoặc answer generation cuối cùng.
4. Feature engineering supervised còn tương đối đơn giản. Chưa có cross-encoder hoặc ranking loss.
5. Pipeline hiện được tối ưu cho tiếng Anh. Nếu chuyển sang tiếng Việt, tokenizer và dữ liệu cần thay đổi đáng kể.

### 7.2 Hướng phát triển ngắn hạn

1. Tích hợp BiLSTM vào experiment runner.
2. Expose các model mạnh hơn trong web app.
3. Thử thêm các biến thể metric và hyperparameter tuning.
4. Chạy thêm error analysis theo nhóm question type.
5. Tăng mức độ chi tiết của logging và lưu trữ experiment.

### 7.3 Hướng phát triển dài hạn

1. Fine-tune SBERT bằng contrastive hoặc ranking loss.
2. Chuyển supervised sang learning-to-rank thay vì binary classification.
3. Mở rộng sang multilingual sentence selection.
4. Kết hợp sentence selection với span extraction để thành full QA pipeline.
5. Đóng gói thành API service có monitoring và khả năng scale.

### 7.4 Hướng phát triển nghiên cứu

Về mặt nghiên cứu, project có thể mở rộng theo ba trục:

- Trục biểu diễn: thay encoder bằng model ngữ nghĩa mạnh hơn.
- Trục học máy: chuyển từ classification sang ranking.
- Trục hệ thống: tối ưu speed, cache và deployment.

---

## 8. Kết luận

Project đã xây dựng thành công một pipeline sentence-selection QA hoàn chỉnh từ dữ liệu, encoder, selector, evaluation cho đến demo web. Điểm mạnh lớn nhất của project là tính mô-đun: mỗi thành phần được tách rõ, dễ thay thế và dễ mở rộng.

Từ thực nghiệm hiện tại có thể rút ra ba kết luận chính:

- SBERT cho chất lượng tốt hơn TF-IDF trong hầu hết các cấu hình.
- Supervised RF trên SBERT là mô hình mạnh nhất trong nhóm supervised.
- TF-IDF baseline vẫn hữu ích vì tốc độ nhanh và chi phí thấp.

Ở góc độ học thuật, đề tài cho thấy sự khác biệt giữa lexical matching và semantic matching, đồng thời minh họa rõ vai trò của supervised reranking trong sentence selection.

---

## 9. Tài liệu tham khảo

[1] Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text.

[2] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

[3] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.

[4] Scikit-learn Documentation. https://scikit-learn.org/

[5] Sentence-Transformers Documentation. https://www.sbert.net/

[6] HuggingFace Datasets. https://huggingface.co/datasets/squad

[7] NLTK Documentation. https://www.nltk.org/

[8] XGBoost Documentation. https://xgboost.readthedocs.io/

---

## 10. Phụ lục

### A. Lệnh chạy nhanh

```bash
cd d:\2026\natural_language_processing\sentence_selection_qa

# Cài dependencies chính
pip install datasets nltk scikit-learn numpy matplotlib sentence-transformers xgboost rank_bm25 flask torch

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Tiền xử lý dữ liệu
python main.py --mode preprocess --max_train 5000 --max_val 1000

# Chạy unsupervised experiments
python main.py --mode unsupervised --max_train 5000 --max_val 1000

# Chạy supervised experiments
python main.py --mode supervised --max_train 5000 --max_val 1000

# Gộp kết quả đã lưu
python main.py --mode compare

# Demo CLI
python main.py --mode demo --question "Who created Python?" --context "Python was created by Guido van Rossum in 1991. It is widely used in data science."

# Demo web
python run_webapp.py
```

### B. Tham số mô hình

**TF-IDF**

- n_components: 256
- max_features: 50.000
- ngram_range: (1, 2)
- sublinear_tf: true

**SBERT**

- model: all-MiniLM-L6-v2
- output dimension: 384
- batch size: 64
- device: auto

**BM25**

- k1: 1.5
- b: 0.75

**BiLSTM**

- embed_dim: 100
- hidden_dim: 128
- output_dim: 256
- num_layers: 2
- pooling: max / mean / last

**Classifier**

- LR: max_iter=1000, C=1.0, solver=lbfgs
- RF: n_estimators=200, max_depth=10
- XGB: n_estimators=300, max_depth=6, learning_rate=0.1

### C. File output chính

- results/tables/unsupervised_results.json
- results/tables/supervised_results.json
- results/tables/all_results.csv
- results/plots/unsupervised_acc_at_k.png
- results/plots/supervised_acc_at_k.png
- results/plots/all_mrr.png
- results/plots/ablation_neg_ratio.png
- results/error_analysis/tfidf_cosine_error_report.json
- results/error_analysis/tfidf_lr_error_report.json
- saved_models/tfidf_lr.pkl
- saved_models/tfidf_rf.pkl
- saved_models/tfidf_xgb.pkl
- saved_models/sbert_lr.pkl
- saved_models/sbert_rf.pkl

### D. Ghi chú về phiên bản hiện tại

- Web app hiện thiên về demo unsupervised TF-IDF.
- BiLSTM đã có code nhưng chưa được tích hợp vào runner.
- Các số liệu trong phần kết quả khớp với các file JSON hiện có trong results/tables.

---

**Trạng thái:** Hoàn tất theo code và kết quả hiện có trong project  
**Ngày cập nhật:** 13/05/2026