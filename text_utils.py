"""Tiện ích xử lý văn bản dùng chung cho train / evaluate / app.

PhoBERT được pretrain trên văn bản ĐÃ TÁCH TỪ (word-segmented, các từ ghép nối
bằng dấu '_'). Vì vậy trước khi tokenize cần tách từ bằng pyvi để khớp phân phối
dữ liệu lúc pretrain.
"""
from pyvi import ViTokenizer


def segment_text(text: str) -> str:
    """Tách từ tiếng Việt: "sân bay" → "sân_bay" (định dạng PhoBERT mong đợi)."""
    return ViTokenizer.tokenize(str(text))
