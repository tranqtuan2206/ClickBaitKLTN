# ViClickbait Detector — Phát hiện Clickbait tiếng Việt

Công cụ phát hiện **tiêu đề/bài báo clickbait tiếng Việt** bằng học sâu, kèm giao diện web cho phép dán link bài báo để phân tích và **giải thích vì sao** tiêu đề đó bị coi là clickbait.

Kết hợp 2 hướng tiếp cận:

- **Mô hình học sâu**: PhoBERT (văn bản) + ViT (hình ảnh), ghép bằng Cross-Modal Attention.
- **Luật ngôn ngữ**: nhận diện các dấu hiệu như "khoảng trống tò mò", câu hỏi, danh sách con số, viết hoa, gọi trực tiếp người đọc...

## Tính năng

- 🔗 Dán link bài báo (VnExpress, Thanh Niên, ...) → tự lấy tiêu đề + ảnh bìa.
- 📝 **Text only**: phân tích tiêu đề, highlight các cụm từ đáng ngờ.
- 🖼️ **Image only**: phân tích ảnh bìa, xuất heatmap (Attention Rollout) cho biết mô hình đang nhìn vào đâu.
- 🧠 **Multimodal**: dùng cả tiêu đề và ảnh.
- ⚖️ **So sánh**: chạy cùng lúc cả 3 mô hình trên một bài báo.
- 🕘 Lịch sử phân tích, chỉnh ngưỡng quyết định (mặc định 0.5).

## Cấu trúc thư mục

```
ClickBaitKLTN/
├── app.py                  # Giao diện Streamlit
├── scraper.py              # Lấy tiêu đề / ảnh bìa từ link bài báo
├── signals.py              # Phát hiện dấu hiệu clickbait theo luật
├── text_utils.py           # Tách từ tiếng Việt (pyvi) cho PhoBERT
├── train.py                # Huấn luyện mô hình
├── evaluate.py             # Đánh giá trên tập validation
├── download_images.py      # Tải ảnh thumbnail của dataset
├── requirements.txt
├── data/
│   ├── text/clickbait_dataset_vietnamese.csv   # Dataset
│   └── images/                                 # Ảnh thumbnail (tự tải, không push git)
└── *.pth                   # Trọng số mô hình (không push git)
```

## Bộ dữ liệu ViClickbait-2025

- **3.414** tiêu đề tin tức tiếng Việt, gán nhãn `clickbait` / `non-clickbait`.
- Thu thập từ **8 trang báo lớn** (VnExpress, Thanh Niên, Dân Trí, ...).
- Các cột chính: `id`, `url`, `title`, `lead_paragraph`, `category`, `publish_datetime`, `source`, `thumbnail_url`, `label`.

> ⚠️ Dataset đang trong quá trình xem xét công bố, chưa có DOI. Thông tin trích dẫn sẽ được cập nhật sau khi bài báo khoa học được chấp nhận.

## Cài đặt

Yêu cầu **Python 3.11** (khuyến nghị) và pip.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Nếu có GPU NVIDIA, nên cài PyTorch bản CUDA trước cho khớp driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Cách sử dụng

**1. Tải ảnh của dataset** (chỉ cần chạy một lần, ~3000 ảnh):

```bash
python download_images.py
```

**2. Huấn luyện** (chọn 1 trong 3 chế độ):

```bash
python train.py --mode text
python train.py --mode image
python train.py --mode multimodal --epochs 5 --batch-size 4 --lr 2e-5
```

Trọng số tốt nhất được lưu thành `text_only_clickbait_model.pth`, `image_only_clickbait_model.pth`, `multimodal_clickbait_model.pth`.

**3. Đánh giá** trên tập validation (80/20, seed 42):

```bash
python evaluate.py --mode multimodal
```

In ra Precision / Recall / F1, Macro F1 và ma trận nhầm lẫn.

**4. Chạy giao diện web:**

```bash
streamlit run app.py
```

Mở `http://localhost:8501`, dán link hoặc nhập tiêu đề (kèm ảnh nếu dùng mode Image / Multimodal) rồi bấm **Phân tích**.

## Lưu ý

- Cần có file `.pth` tương ứng trong thư mục gốc, nếu không app sẽ cảnh báo và dùng **trọng số ngẫu nhiên** (kết quả vô nghĩa).
- Mode **Image / Multimodal** bắt buộc phải có ảnh.
- `data/images/` và các file `.pth` được đặt trong `.gitignore` vì quá nặng (hàng trăm MB → vài GB), không đẩy lên GitHub.
- Lịch sử phân tích được lưu tự động vào `analysis_history.json`.
- Một số trang báo chặn crawler hoặc đổi cấu trúc meta → khi lấy link lỗi, hãy nhập tiêu đề và upload ảnh thủ công.

## Giấy phép

Dự án phục vụ mục đích **học thuật và phi thương mại**. Dataset dự kiến phát hành theo giấy phép CC BY 4.0 sau khi được chấp thuận. Vui lòng ghi nguồn khi sử dụng.

Mọi thắc mắc hoặc đề nghị hợp tác nghiên cứu, vui lòng liên hệ nhóm thực hiện.
