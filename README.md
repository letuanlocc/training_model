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

## Kết quả

### PhoBERT + softmax (seqeval, mức thực thể)

| Nhãn | test F1 | test_hard F1 |
|---|---|---|
| JOB | 0.924 | 0.815 |
| LOC | 0.987 | 0.938 |
| SKILL | 0.875 | 0.894 |
| TIME | 0.954 | 0.819 |
| **micro** | **0.934** (P 0.932 / R 0.935) | **0.868** (P 0.884 / R 0.852) |

Theo nhóm câu (F1): có dấu 0.938 / 0.861, không dấu 0.923 / 0.895, phủ định 0.956 / 0.920 (test / test_hard).
Câu không có thực thể: test sạch 29/30, test_hard sạch 4/4.

Model, confusion matrix, file dự đoán và báo cáo nằm trên
[Google Drive](https://drive.google.com/drive/folders/1sB2yuWcbcShMBRWG0VeC1jklLIyb35rw).
Mới có kết quả của softmax; crf và lstm_crf chưa có số liệu.

### Baseline từ điển + regex (chạy lại trên dữ liệu có JOB)

| Tập | Baseline micro F1 | PhoBERT softmax |
|---|---|---|
| test | 0.752 (P 0.856 / R 0.671) | **0.934** |
| test_hard | 0.827 (P 0.894 / R 0.770) | **0.868** |

PhoBERT softmax vượt baseline trên cả hai tập; khoảng cách trên `test_hard` (+0.04) nhỏ hơn nhiều so với `test` (+0.18).
Baseline yếu nhất ở recall (từ điển không phủ hết cách diễn đạt), đặc biệt SKILL trên test_hard (R 0.565).

## Đóng góp dữ liệu

Mỗi người tạo nhánh riêng, viết câu vào `phobert_ner/data_build/handwritten_<ten>.txt`, push nhánh rồi train trên
Colab. **Không** push `data_dir/` hay `outputs/` (dễ conflict / quá nặng). Chi tiết xem
[CONTRIBUTING.md](phobert_ner/CONTRIBUTING.md).

## Lưu ý

- Model chưa được train đầy đủ trong báo cáo cài đặt ban đầu (chỉ smoke test) — xem
  [SETUP_REPORT.md](phobert_ner/SETUP_REPORT.md) mục 6–8 để biết những gì đã/ chưa kiểm chứng.
- Dữ liệu là câu viết tay + sinh tự động, không phải câu thật của người tìm việc.
- Chuẩn hoá dấu theo kiểu cũ (`hòa → hoà`) để khớp `normalize_text` của repo lúc suy luận.
