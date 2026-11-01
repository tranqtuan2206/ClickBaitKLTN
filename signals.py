"""Trích xuất dấu hiệu clickbait theo quy tắc (rule-based) từ tiêu đề tiếng Việt.

Mỗi tín hiệu là một span (start, end) trong chuỗi gốc + danh mục + giải thích,
để app.py highlight trực tiếp lên tiêu đề.

- Cụm nhiều từ  → so khớp trên chuỗi đã bỏ dấu (chịu được gõ thiếu dấu)
- Từ đơn        → so khớp trên chuỗi gốc (tránh false positive: "bạn" vs "bán/bản")
"""
import re
import unicodedata
from dataclasses import dataclass

# Màu highlight + tên danh mục
CATEGORIES = {
    'curiosity':   ('#ff5252', 'Khoảng trống tò mò'),
    'sensational': ('#ff9800', 'Cảm thán / cường điệu'),
    'question':    ('#ffd600', 'Câu hỏi / dấu chấm than'),
    'listicle':    ('#448aff', 'Danh sách / con số'),
    'address':     ('#69f0ae', 'Gọi trực tiếp người đọc'),
    'emphasis':    ('#ce93d8', 'Viết hoa / nhấn mạnh'),
}


@dataclass
class Signal:
    start: int
    end: int
    text: str
    category: str
    explanation: str

    @property
    def color(self):
        return CATEGORIES[self.category][0]

    @property
    def label(self):
        return CATEGORIES[self.category][1]


def _normalize_keep_len(s: str) -> str:
    """Bỏ dấu tiếng Việt nhưng GIỮ NGUYÊN độ dài chuỗi → map index về bản gốc."""
    out = []
    for ch in s:
        decomposed = unicodedata.normalize('NFD', ch)
        base = ''.join(c for c in decomposed if not unicodedata.combining(c))
        if ch in ('đ', 'Đ'):
            base = 'd' if ch == 'đ' else 'D'
        out.append(base if len(base) == 1 else ch)
    return ''.join(out)


# ---------- Cụm nhiều từ: so khớp trên chuỗi bỏ dấu ----------
# (pattern, danh mục, giải thích)
PHRASE_PATTERNS = [
    # Khoảng trống tò mò
    (r"bạn\s+sẽ\s+không\s+tin", 'curiosity', 'Tạo tò mò: hứa điều "khó tin" nhưng không nói là gì.'),
    (r"bạn\s+sẽ\s+bất\s+ngờ", 'curiosity', 'Tạo tò mò: gợi ý kết quả bất ngờ mà không tiết lộ.'),
    (r"bạn\s+không\s+(?:thể|nên|phải)", 'curiosity', 'Gọi trực tiếp + tò mò: "bạn không thể..." kích thích tò mò.'),
    (r"không\s+ngờ", 'curiosity', 'Cụm "không ngờ" — chiêu kinh điển của clickbait.'),
    (r"sự\s+thật", 'curiosity', '"Sự thật..." ngụ ý còn thông tin giấu sau tiêu đề.'),
    (r"bí\s+mật", 'curiosity', '"Bí mật..." — chiêu tạo tò mò cổ điển.'),
    (r"phía\s+sau", 'curiosity', '"Phía sau..." — ám chỉ có điều chưa được nói ra.'),
    (r"l[ýi]\s+do", 'curiosity', '"Lý do..." — liệt kê mà chưa cho biết nội dung.'),
    (r"v[ìi]\s+sao|t[ăa]i\s+sao", 'curiosity', 'Câu hỏi "vì sao/tại sao" — tạo khoảng trống thông tin.'),
    (r"t[ăa]i\s+th[ếe]\s+nào", 'curiosity', 'Câu hỏi mở "tại sao thế nào" không kèm trả lời.'),
    (r"chuyện\s+g[ì]|điều\s+g[ì]", 'curiosity', 'Câu hỏi mở mà tiêu đề không trả lời.'),
    (r"n[óo]\s+l[àa]\s+g[ì]", 'curiosity', 'Cụm "nó là gì" — hút tò mò.'),

    # Cảm thán / cường điệu
    (r"ngỡ\s+ngàng", 'sensational', 'Từ cảm thán cường điệu.'),
    (r"chấn\s+động", 'sensational', 'Từ cảm thán cường điệu.'),
    (r"kinh\s+hoàng", 'sensational', 'Từ cảm thán cường điệu.'),
    (r"g[âa]y\s+s[ốo]c", 'sensational', '"Gây sốc" — ngôn ngữ giật gân.'),
    (r"kh[ông]\s+tưởng", 'sensational', '"Không tưởng" — phóng đại.'),
    (r"cực\s+(?:đỉnh|hot|ngầu|ngon|đã|khủng)", 'sensational', 'Từ phóng đại ("cực ...").'),
    (r"ch[ỉi]\s+có\s+(?:t[àa]i|ở|d[âa]n)", 'sensational', '"Chỉ có ở..." — đánh bóng exclusive.'),
    (r"đ[ộo]c\s+nh[áa]t", 'sensational', '"Độc nhất" — phóng đại.'),
    (r"kh[ông]\s+th[ể]\s+b[ỏ]\s+quen", 'sensational', '"Không thể bỏ qua" — ép người đọc nhấp.'),
    (r"s[ốo]c\s+nặng|s[ốo]c\s+tr[ốo]ng", 'sensational', 'Từ cảm thán cường điệu.'),
    (r"kinh\s+ngư[aạ]c|đáng\s+kinh\s+ngư[aạ]c", 'sensational', 'Từ cảm thán cường điệu.'),
    (r"b[áa]o\s+động|c[ẩa]nh\s+b[áa]o", 'sensational', 'Từ gây lo sợ / giật gân.'),
    (r"b[ỏ]\s+lỡ", 'sensational', '"Bỏ lỡ" — ép người đọc nhấp ngay.'),

    # Danh sách / con số
    (r"\btop\s*\d+", 'listicle', 'Danh sách "top N" — công thức clickbait quen thuộc.'),
    (r"\d+\s+l[ýi]\s+do", 'listicle', 'Liệt kê "N lý do" — hứa hẹn nội dung dễ đọc.'),
    (r"\d+\s+c[áa]ch\b", 'listicle', 'Liệt kê "N cách" — công thức danh sách.'),
    (r"\d+\s+b[íi]\s+quy[ếe]t", 'listicle', 'Liệt kê "N bí quyết".'),
    (r"\d+\s+d[áa]u\s+hi[êe]u", 'listicle', 'Liệt kê "N dấu hiệu".'),
    (r"\d+\s+điều\b|\d+\s+l[ỗio]i\b|\d+\s+thứ\b", 'listicle', 'Liệt kê bằng con số.'),

    # Gọi trực tiếp người đọc (nhiều từ)
    (r"\bc[áa]c\s+bạn\b|\bchúng\s+ta\b", 'address', 'Gọi nhóm người đọc — tạo cảm giác liên quan.'),
]

# ---------- Từ đơn / ký hiệu: so khớp trên chuỗi GỐC (tránh lỗi bỏ dấu) ----------
# (pattern, danh mục, giải thích, flags) — ALLCAPS KHÔNG được dùng IGNORECASE
WORD_PATTERNS = [
    (r"\bbạn\b", 'address', 'Gọi thẳng người đọc ("bạn") — tạo cảm giác liên quan cá nhân.', re.IGNORECASE),
    (r"\bđỉnh\b", 'sensational', 'Từ phóng đại ("đỉnh").', re.IGNORECASE),
    (r"\bnóng\b", 'sensational', 'Từ "nóng" — giật gân.', re.IGNORECASE),
    (r"\bkhủng\b", 'sensational', 'Từ phóng đại ("khủng").', re.IGNORECASE),
    (r"\btrend\b", 'sensational', 'Từ "trend" — giật gân.', re.IGNORECASE),
    (r"!{2,}|\?{2,}", 'emphasis', 'Dấu than/hỏi lặp lại — cường điệu.', 0),
    (r"\?+|!+", 'question', 'Dấu chấm hỏi/than tạo nhịp giật gân, gợi tò mò.', 0),
    (r"\.{3}|…", 'curiosity', 'Dấu "..." ngắt quãng — ngụ ý còn phần hấp dẫn phía sau.', 0),
    (r"\b[A-ZÀ-ỸĐ]{4,}\b", 'emphasis', 'Viết hoa toàn bộ một từ — nhấn mạnh giật gân.', 0),
]

_PHRASE_COMPILED = [
    (re.compile(_normalize_keep_len(pat), re.IGNORECASE), cat, why)
    for pat, cat, why in PHRASE_PATTERNS
]
_WORD_COMPILED = [
    (re.compile(pat, flags), cat, why)
    for pat, cat, why, flags in WORD_PATTERNS
]


def extract_signals(title: str):
    """Trả về list[Signal] sort theo vị trí, loại overlap (giữ span dài hơn)."""
    if not title:
        return []

    norm = _normalize_keep_len(title)
    found = []

    for regex, cat, why in _PHRASE_COMPILED:
        for m in regex.finditer(norm):
            found.append(Signal(m.start(), m.end(), title[m.start():m.end()], cat, why))
    for regex, cat, why in _WORD_COMPILED:
        for m in regex.finditer(title):
            found.append(Signal(m.start(), m.end(), m.group(), cat, why))

    # Sort: span dài trước, rồi theo vị trí — span ngắn không nuốt span dài
    found.sort(key=lambda s: (-(s.end - s.start), s.start))

    kept = []
    for sig in found:
        if any(not (sig.end <= k.start or sig.start >= k.end) for k in kept):
            continue
        kept.append(sig)
    kept.sort(key=lambda s: s.start)
    return kept


def _escape(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def render_highlighted(title: str, signals):
    """Sinh HTML tiêu đề với các span được highlight theo danh mục màu."""
    if not signals:
        return _escape(title)
    parts, cursor = [], 0
    for sig in signals:
        parts.append(_escape(title[cursor:sig.start]))
        tooltip = _escape(f"{sig.label}: {sig.explanation}")
        parts.append(
            f'<span style="background:{sig.color}55; border-bottom:2px solid {sig.color}; '
            f'border-radius:3px; padding:0 2px; cursor:help;" title="{tooltip}">'
            f'{_escape(sig.text)}</span>'
        )
        cursor = sig.end
    parts.append(_escape(title[cursor:]))
    return ''.join(parts)


def summarize(signals):
    """Gom nhóm theo danh mục → list dict cho panel giải thích."""
    groups = {}
    for sig in signals:
        g = groups.setdefault(sig.category, {'count': 0, 'examples': [], 'why': sig.explanation})
        g['count'] += 1
        if len(g['examples']) < 3 and sig.text not in g['examples']:
            g['examples'].append(sig.text)
    return [
        {
            'category': cat,
            'label': CATEGORIES[cat][1],
            'color': CATEGORIES[cat][0],
            **data,
        }
        for cat, data in sorted(groups.items(), key=lambda kv: -kv[1]['count'])
    ]
