# Training Model — NER tiếng Việt (SKILL / JOB / TIME / LOC)

Dự án huấn luyện mô hình nhận diện thực thể (NER) cho câu tự giới thiệu của người tìm việc tiếng Việt, dựa trên
[VPhoBertTagger](https://github.com/datnvhust/VPhoBertTagger) (PhoBERT). Các thực thể cần trích xuất:

| Nhãn | Ý nghĩa | Ví dụ |
|---|---|---|
| `SKILL` | Kỹ năng | tiếng Anh, pha chế, nhập liệu |
| `JOB` | Công việc | chạy bàn, giao báo |
| `TIME` | Thời gian rảnh / ca làm | ca tối, thứ 2 đến thứ 6 |
| `LOC` | Địa điểm (quận/huyện, thành phố) | Hải Châu, Đà Nẵng |

Nhãn theo định dạng BIO (`O`, `B-SKILL`, `I-SKILL`, `B-JOB`, …) — xem `LABEL2ID_JOBNER` trong
[constant.py](phobert_ner/VPhoBertTagger/vphoberttagger/constant.py).

Ví dụ: `tôi giỏi [tiếng anh|SKILL], rảnh vào [tối|TIME] và ở [Hải Châu|LOC]`

## Bối cảnh và Problem Statement

**Bối cảnh (kịch bản giả định).** Các nền tảng tuyển dụng việc bán thời gian / thời vụ (phục vụ, gia sư, giao hàng…) thường
nhận hồ sơ dưới dạng một đoạn tự giới thiệu viết tự do bằng tiếng Việt, hay có teencode, viết tắt (`T2`, `T4`) hoặc không dấu.
Nhà tuyển dụng phải đọc tay từng đoạn để biết ứng viên *làm được gì, rảnh lúc nào, ở khu vực nào*; việc này chậm và khó lọc/tìm kiếm.
Dự án giả định bối cảnh Đà Nẵng (nhãn LOC gồm 7 quận/huyện và "Đà Nẵng").

**Problem statement.** Cho một câu tự giới thiệu tiếng Việt của người tìm việc, tự động trích xuất các thực thể
`SKILL`, `JOB`, `TIME`, `LOC` (gán nhãn BIO mức token) để có thể lọc và ghép ứng viên với tin tuyển dụng mà không cần đọc tay.
Bài toán khó ở chỗ: câu ngắn, nhiều biến thể viết (có dấu / không dấu / viết tắt), thực thể nhiều từ
(`từ 6 giờ chiều đến 10 giờ tối`), và câu phủ định (`không biết bơi nên không làm cứu hộ`).

### Research questions

- **RQ1:** Fine-tune PhoBERT có vượt baseline từ điển + regex không, và vượt ở loại thực thể / nhóm câu nào?
- **RQ2:** Thêm lớp CRF hoặc LSTM-CRF lên PhoBERT có cải thiện so với softmax thuần không (xét micro-F1 và P/R)?
- **RQ3:** Model xử lý thế nào với câu không dấu và câu phủ định so với câu có dấu thông thường?

Phần trả lời nằm ở mục [Nhận xét theo research questions](#nhận-xét-theo-research-questions).

## Cấu trúc thư mục

```
phobert_ner/
├── VPhoBertTagger/          # repo gốc (đã sửa nhẹ, task mới `jobner`)
├── data_build/              # sinh + kiểm tra dữ liệu, baseline, báo cáo
│   ├── lexicon.py           # từ điển thực thể + biến thể
│   ├── templates.py         # template sinh câu
│   ├── handwritten*.txt     # câu viết tay (markup [text|TAG]) — mỗi người 1 file
│   ├── test_hard_raw.txt    # tập test khó viết tay
│   ├── build_data.py        # pipeline sinh dữ liệu → data_dir/
│   ├── validate_data.py     # kiểm tra định dạng + thống kê
│   ├── baseline_dict.py     # baseline từ điển + regex
│   ├── predict_conll.py     # xuất dự đoán của model ra file CoNLL
│   └── eval_report.py       # F1 mức thực thể theo loại / nhóm
├── data_dir/                # train/dev/test/test_hard.txt (4 cột, tab) + metadata
├── train_softmax.sh / train_crf.sh / train_lstm_crf.sh
├── notebook_colab.ipynb         # train trên Colab (1 người)
├── notebook_colab_team.ipynb    # train trên Colab theo nhánh của từng thành viên
├── requirements_working.txt     # thư viện chạy được trên Python 3.12
├── CONTRIBUTING.md          # quy trình đóng góp dữ liệu cho team
└── SETUP_REPORT.md          # báo cáo chi tiết: thay đổi repo, quy ước gán nhãn, baseline
```

## Cài đặt

Yêu cầu: Python 3.12, Java (cho VnCoreNLP tách từ). `requirements.txt` gốc của VPhoBertTagger **không cài được**
trên Python 3.12, hãy dùng bản đã chỉnh:

```bash
cd phobert_ner
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements_working.txt
```

VnCoreNLP (`phobert_ner/vncorenlp/`) không nằm trong git — tải lại bằng notebook Colab hoặc thủ công
(`py_vncorenlp.download_model()` lỗi trên Windows vì cần `wget`; xem `SETUP_REPORT.md` §3).

## Quy trình

```bash
cd phobert_ner

# 1. Sinh dữ liệu từ handwritten*.txt + template → data_dir/
python data_build/build_data.py

# 2. Kiểm tra định dạng và thống kê
python data_build/validate_data.py

# 3. Baseline từ điển + regex (mốc để so sánh)
python data_build/baseline_dict.py

# 4. Train (nên dùng GPU / Colab; mỗi checkpoint ~515MB)
bash train_softmax.sh      # hoặc train_crf.sh, train_lstm_crf.sh
```

Sau khi train, script tự chạy `predict_conll.py` và `eval_report.py` cho `test` và `test_hard`
→ `outputs/<arch>/report_*.txt`.

Dự đoán một câu thô (chạy từ thư mục `phobert_ner/`, cần Java):

```bash
python VPhoBertTagger/main.py predict --task jobner ...
```

## Định dạng dữ liệu

Mỗi dòng là một token, 4 cột cách nhau bằng tab: `token  pos  chunk  ner`. Cột `pos` và `chunk` luôn là `_`
(model chỉ đọc `token` và `ner`). Các câu cách nhau bằng một dòng trống.

```
em	_	_	O
biet	_	_	O
nhap	_	_	B-SKILL
lieu	_	_	I-SKILL
```

## Quyết định thiết kế và lý do

| Quyết định | Lý do |
|---|---|
| Dùng `vinai/phobert-base` | Mô hình pretrained tiếng Việt; nhẹ hơn bản large, đủ để fine-tune trên Colab với ~1.5k câu train. |
| Tách từ bằng VnCoreNLP trước khi đưa vào PhoBERT | PhoBERT được pretrain trên văn bản đã tách từ, đưa câu thô sẽ lệch phân bố. |
| So sánh softmax / CRF / LSTM-CRF với cùng siêu tham số | Để chênh lệch kết quả quy về kiến trúc (RQ2), không phải do chỉnh tham số. Siêu tham số: `lr=3e-5`, batch 32, `max_seq_length=128` (câu dài nhất 55 subword), tối đa 30 epoch, early-stop 5. |
| Chọn model theo **dev**, không phải test | Code gốc chọn model trên tập test (rò rỉ); đã sửa trong `trainer.py` (xem [SETUP_REPORT.md](phobert_ner/SETUP_REPORT.md) §2). |
| Baseline từ điển + regex | Mốc tối thiểu để biết PhoBERT có đáng dùng không (RQ1). |
| Dữ liệu = câu viết tay + sinh từ template, chia 80/10/10 **theo nhóm** | Chưa có dữ liệu thật; chia theo nhóm (câu gốc và các bản biến thể cùng một tập) để tránh rò rỉ. Số nhóm nằm ở >1 tập: 0. |
| Thêm `test_hard` viết tay riêng | `test` chứa nhiều câu từ template nên dễ; `test_hard` phản ánh tốt hơn khả năng tổng quát hoá. |
| Báo cáo F1 theo nhãn và theo nhóm (có dấu / không dấu / phủ định) | Để trả lời RQ3 và thấy model yếu ở đâu thay vì chỉ một con số tổng. |
| Phủ định vẫn gán nhãn bình thường | Bài toán là trích xuất thực thể, không phải hiểu ý nghĩa; cờ `negated` lưu riêng ở file meta. |
| Chuẩn hoá dấu kiểu cũ (`hòa → hoà`) | Khớp `normalize_text` của repo gốc lúc suy luận, để train và predict cùng quy ước. |

Quy ước gán nhãn chi tiết (SKILL, TIME, LOC) ở [SETUP_REPORT.md](phobert_ner/SETUP_REPORT.md) §4.1.

## Kết quả

So sánh 4 phương pháp trên cùng `data_dir/`, cùng siêu tham số, 1 lần train / 1 seed mỗi kiến trúc
(notebook: [notebook_colab_compare.ipynb](phobert_ner/notebook_colab_compare.ipynb)). F1 mức thực thể (seqeval).
Số liệu thô: [results/compare.csv](results/compare.csv), biểu đồ: [results/compare.png](results/compare.png).

| Model | test micro-F1 | test_hard micro-F1 | test_hard P / R |
|---|---|---|---|
| Baseline (từ điển + regex) | 0.752 | 0.827 | 0.894 / 0.770 |
| PhoBERT + softmax | 0.934 | 0.868 | 0.884 / 0.852 |
| PhoBERT + CRF | 0.932 | **0.883** | 0.879 / 0.888 |
| PhoBERT + LSTM-CRF | **0.938** | 0.878 | 0.859 / 0.898 |

F1 theo nhãn trên test_hard:

| Model | JOB | LOC | SKILL | TIME |
|---|---|---|---|---|
| Baseline | 0.821 | 1.000 | 0.714 | 0.803 |
| softmax | 0.815 | 0.938 | 0.894 | 0.819 |
| CRF | 0.800 | 0.952 | 0.937 | 0.822 |
| LSTM-CRF | 0.867 | 0.954 | 0.919 | 0.794 |

F1 theo nhóm câu (test / test_hard):

| Nhóm | Baseline | softmax | CRF | LSTM-CRF |
|---|---|---|---|---|
| có dấu | 0.767 / 0.801 | 0.938 / 0.861 | 0.936 / 0.873 | 0.947 / 0.890 |
| không dấu | 0.716 / 0.923 | 0.923 / 0.895 | 0.921 / 0.923 | 0.917 / 0.829 |
| phủ định | 0.776 / 0.944 | 0.956 / 0.920 | 0.964 / 0.932 | 0.973 / 0.923 |

### Nhận xét theo research questions

- **RQ1 (PhoBERT vs baseline):** cả 3 biến thể PhoBERT vượt baseline về micro-F1 trên cả hai tập: +0.18 trên `test`
  và +0.04 đến +0.056 trên `test_hard`. Lợi thế đến từ recall (baseline bỏ sót nhiều SKILL/JOB không có trong từ điển).
  Tuy nhiên baseline không thua đều: nó tốt hơn ở LOC (1.000) và ở nhóm không dấu / phủ định trên test_hard.
- **RQ2 (thêm CRF / LSTM-CRF):** chênh lệch giữa 3 kiến trúc nhỏ. Trên `test` chỉ ~0.005 (LSTM-CRF cao nhất), trên
  `test_hard` CRF và LSTM-CRF hơn softmax khoảng 0.01–0.016. Với 1 seed và test_hard chỉ có 196 thực thể, chênh lệch cỡ
  này **chưa đủ để kết luận chắc chắn** kiến trúc nào tốt hơn; có thể nói CRF/LSTM-CRF cải thiện recall trên test_hard
  (0.888 / 0.898 so với 0.852) nhưng đổi lại precision thấp hơn.
- **RQ3 (không dấu, phủ định):** trên `test`, câu không dấu thấp hơn câu có dấu khoảng 0.015–0.03 ở cả 3 model;
  câu phủ định không bị ảnh hưởng (còn cao hơn). Trên `test_hard` nhóm không dấu chỉ có 21 câu nên số dao động mạnh
  (LSTM-CRF 0.829 so với CRF 0.923), không nên diễn giải quá mức.

### Phân tích lỗi

Số lỗi theo loại (so thực thể gold với thực thể dự đoán):

| Model | Tập | Thiếu | Thừa | Sai ranh giới | Sai nhãn | Tổng |
|---|---|---|---|---|---|---|
| Baseline | test | 535 | 8 | 109 | 135 | 787 |
| Baseline | test_hard | 32 | 4 | 8 | 5 | 49 |
| softmax | test | 21 | 21 | 127 | 5 | 174 |
| softmax | test_hard | 12 | 4 | 13 | 4 | 33 |
| CRF | test | 10 | 26 | 134 | 5 | 175 |
| CRF | test_hard | 7 | 7 | 12 | 3 | 29 |
| LSTM-CRF | test | 10 | 38 | 113 | 5 | 166 |
| LSTM-CRF | test_hard | 3 | 11 | 14 | 3 | 31 |

Nhận xét:

- **Baseline** lỗi chủ yếu là *thiếu* (535/787 trên test): từ điển không phủ hết JOB và SKILL (F1 trên test chỉ 0.525 và 0.494).
- **Sai ranh giới là lỗi chính của cả 3 model PhoBERT trên test** (113–134 trên ~170 lỗi, khoảng 70–77%), và chiếm khoảng 40–48% trên test_hard.
  Ví dụ (softmax, test_hard): gold `từ 6 giờ chiều đến 10 giờ tối` nhưng model chỉ lấy `6 giờ chiều`; gold `dạy kèm`, model lấy `dạy`;
  gold `chu nhat`, model lấy `tru chu nhat` (kéo thêm từ "trừ").
  Khoảng thời gian dài và cụm nhiều từ là chỗ khó nhất.
- **CRF và LSTM-CRF giảm lỗi *thiếu* nhưng tăng lỗi *thừa*:** trên test_hard, lỗi thiếu 12 → 7 → 3 (softmax → CRF → LSTM-CRF),
  lỗi thừa 4 → 7 → 11. Khớp với kết quả recall tăng, precision giảm ở RQ2.
- **Câu phủ định gây cả hai hướng lỗi.** Dù nhãn vẫn gán cho thực thể bị phủ định, model đôi khi bỏ sót
  (`Em chưa bao giờ phục vụ…` → bỏ `phục vụ`; `không phải người Đà Nẵng…` → bỏ `Đà Nẵng`), và đôi khi gán thừa
  (`không biết bơi nên không làm cứu hộ` → thừa `cứu hộ` ở trường hợp gold không gán; `về khuya` → TIME).
- **Một số lỗi khác:** LOC không dấu bị bỏ (`quan hai chau`, `di thanh khe`); viết tắt `T2` gán nhầm SKILL thay vì TIME, `T4` bị bỏ.

Lưu ý: các ví dụ trên là 15 lỗi đầu tiên của softmax trên test_hard, chưa phải thống kê đầy đủ. Chưa kiểm tra xem lỗi ranh giới
trên `test` có tập trung ở một số template hay không.

### Hạn chế

- Dữ liệu là câu viết tay + sinh tự động, không phải câu thật của người tìm việc; `test` cao (~0.93) một phần vì chứa câu từ template,
  `test_hard` (~0.87–0.88) đáng tin hơn nhưng chỉ có 102 câu / 196 thực thể.
- Mỗi kiến trúc chỉ train 1 seed nên chênh lệch giữa softmax / CRF / LSTM-CRF chưa có ý nghĩa thống kê.
- Nhãn do một người gán, chưa đo độ đồng thuận giữa người gán.

Model, confusion matrix và file dự đoán của softmax: [Google Drive](https://drive.google.com/drive/folders/1sB2yuWcbcShMBRWG0VeC1jklLIyb35rw).

## Nguồn và đóng góp của nhóm

**Lấy từ nguồn ngoài (đã ghi nguồn):**
- [VPhoBertTagger](https://github.com/datnvhust/VPhoBertTagger) (`phobert_ner/VPhoBertTagger/`): code train / predict PhoBERT + softmax / CRF / LSTM-CRF. Chúng tôi chỉ sửa nhẹ (6 file, +34/−12 dòng, xem `phobert_ner/repo_changes.diff`): thêm task `jobner`, chọn model theo dev thay vì test, `weights_only=False` cho torch mới, sửa khởi tạo trọng số CRF/LSTM và một lỗi lúc predict LSTM-CRF.
- Mô hình [`vinai/phobert-base`](https://huggingface.co/vinai/phobert-base) và bộ tách từ VnCoreNLP.
- Thư viện: PyTorch, Hugging Face Transformers, seqeval.

**Do nhóm tự làm:** toàn bộ `phobert_ner/data_build/` (từ điển, template, câu viết tay, pipeline sinh/kiểm tra dữ liệu, baseline, `predict_conll.py`, `eval_report.py`),
`data_dir/`, script `train_*.sh`, các notebook Colab, thiết kế thí nghiệm, `results/` và phần phân tích trong README này.

## Đóng góp dữ liệu

Mỗi người tạo nhánh riêng, viết câu vào `phobert_ner/data_build/handwritten_<ten>.txt`, push nhánh rồi train trên
Colab. **Không** push `data_dir/` hay `outputs/` (dễ conflict / quá nặng). Chi tiết xem
[CONTRIBUTING.md](phobert_ner/CONTRIBUTING.md).

## Lưu ý

- [SETUP_REPORT.md](phobert_ner/SETUP_REPORT.md) là bản ghi lại lúc cài đặt ban đầu (28/10/2026, mới smoke test, số liệu dữ liệu cũ);
  số liệu hiện hành là ở README này.
- Dữ liệu là câu viết tay + sinh tự động, không phải câu thật của người tìm việc.
- Chuẩn hoá dấu theo kiểu cũ (`hòa → hoà`) để khớp `normalize_text` của repo lúc suy luận.
