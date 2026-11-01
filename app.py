"""ViClickbait Detector — giao diện Streamlit.

Tính năng:
- Dán link bài báo (VnExpress, Thanh Niên, ...) → tự lấy tiêu đề + ảnh bìa
- 3 chế độ: text only (highlight dấu hiệu), image only (heatmap attention), multimodal (cả hai)
- Highlight dấu hiệu clickbait theo quy tắc + xác suất từ mô hình
- Lịch sử phân tích (lưu file JSON)
"""
import io
import json
import os
import sys
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel, ViTModel

from scraper import fetch_article
from signals import extract_signals, render_highlighted, summarize, CATEGORIES
from text_utils import segment_text

if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')

st.set_page_config(page_title="ViClickbait Detector", page_icon="📰", layout="wide")

HISTORY_FILE = 'analysis_history.json'


# ===================== Kiến trúc mô hình (giống hệt train.py) =====================
class TextOnlyClickbaitModel(nn.Module):
    def __init__(self, phobert_name="vinai/phobert-base"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        hidden = self.text_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values=None):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(torch.mean(out.last_hidden_state, dim=1)).squeeze(1)


class ImageOnlyClickbaitModel(nn.Module):
    def __init__(self, vit_name="google/vit-base-patch16-224"):
        super().__init__()
        self.image_encoder = ViTModel.from_pretrained(vit_name)
        hidden = self.image_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        out = self.image_encoder(pixel_values=pixel_values)
        return self.classifier(torch.mean(out.last_hidden_state, dim=1)).squeeze(1)


class MultimodalClickbaitModel(nn.Module):
    def __init__(self, phobert_name="vinai/phobert-base", vit_name="google/vit-base-patch16-224", num_heads=8):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        self.image_encoder = ViTModel.from_pretrained(vit_name)
        text_h = self.text_encoder.config.hidden_size
        image_h = self.image_encoder.config.hidden_size
        self.image_projection = nn.Linear(image_h, text_h) if text_h != image_h else nn.Identity()
        self.cross_attention = nn.MultiheadAttention(text_h, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(text_h * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        image_features = self.image_projection(self.image_encoder(pixel_values=pixel_values).last_hidden_state)
        attn_output, _ = self.cross_attention(query=text_features, key=image_features, value=image_features)
        combined = torch.cat(
            (torch.mean(text_features, dim=1), torch.mean(attn_output, dim=1)), dim=1,
        )
        return self.classifier(combined).squeeze(1)


MODEL_CLASSES = {'text': TextOnlyClickbaitModel, 'image': ImageOnlyClickbaitModel, 'multimodal': MultimodalClickbaitModel}
CHECKPOINTS = {'text': 'text_only_clickbait_model.pth', 'image': 'image_only_clickbait_model.pth', 'multimodal': 'multimodal_clickbait_model.pth'}
MODE_LABELS = {'text': '📝 Text only (PhoBERT)', 'image': '🖼️ Image only (ViT)', 'multimodal': '🧠 Multimodal (PhoBERT + ViT)'}


# ===================== Load pipeline =====================
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_pipeline(mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = MODEL_CLASSES[mode]()

    ckpt = CHECKPOINTS[mode]
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        status = f"✅ {ckpt}"
    else:
        status = f"⚠️ Không tìm thấy {ckpt} — trọng số random!"
    model.to(device).eval()

    eager_vit = None
    if mode in ('image', 'multimodal'):
        # ViT bản eager để lấy attentions cho heatmap (sdpa không hỗ trợ output_attentions)
        eager_vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224", attn_implementation='eager',
        )
        eager_vit.load_state_dict(model.image_encoder.state_dict())
        eager_vit.to(device).eval()
    return model, tokenizer, processor, eager_vit, device, status


# ===================== Dự đoán + heatmap =====================
def predict(mode, title, image, pipeline):
    model, tokenizer, processor, _, device, _ = pipeline
    with torch.no_grad():
        if mode in ('text', 'multimodal'):
            enc = tokenizer(segment_text(title), max_length=256, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
        else:
            input_ids = attention_mask = None

        if mode in ('image', 'multimodal'):
            pixel_values = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
        else:
            pixel_values = None

        logit = model(input_ids, attention_mask, pixel_values)
        return torch.sigmoid(logit).item()


@torch.no_grad()
def rollout_heatmap(pipeline, image):
    """Attention rollout trên ViT → ảnh overlay vùng model tập trung."""
    _, _, processor, eager_vit, device, _ = pipeline
    if eager_vit is None:
        return None
    pixel = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
    out = eager_vit(pixel_values=pixel, output_attentions=True)
    n = out.attentions[0].shape[-1]
    roll = torch.eye(n, device=device)
    for a in out.attentions:
        a = a.mean(dim=1)                      # (B, N, N) — trung bình các head
        a = a + torch.eye(n, device=device)    # thêm identity
        a = a / a.sum(dim=-1, keepdim=True)    # chuẩn hoá hàng
        roll = roll @ a
    cls = roll[0, 0, 1:]                       # dòng CLS, bỏ token CLS
    side = int(cls.numel() ** 0.5)             # 14x14
    grid = cls.reshape(side, side)
    grid = grid / grid.max().clamp(min=1e-9)
    grid = F.interpolate(grid[None, None], size=(224, 224), mode='bilinear')[0, 0]
    mask = Image.fromarray((grid * 255).byte().cpu().numpy(), mode='L')
    mask = mask.point(lambda p: int(p * 0.55))  # độ trong suốt 55%
    base = image.convert('RGB').resize((224, 224))
    red = Image.new('RGB', base.size, (255, 40, 40))
    return Image.composite(red, base, mask)


# ===================== Lịch sử =====================
def load_history():
    try:
        with open(HISTORY_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(hist):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)


# ===================== Phân tích =====================
def run_analysis(mode, title, image):
    pipeline = load_pipeline(mode)
    prob = predict(mode, title, image, pipeline)

    signals = None
    if mode in ('text', 'multimodal'):
        signals = extract_signals(title)

    heatmap = None
    if mode in ('image', 'multimodal') and image is not None:
        heatmap = rollout_heatmap(pipeline, image)

    # Lưu lịch sử
    rec = {
        'id': uuid.uuid4().hex[:8],
        'ts': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'mode': mode,
        'title': title,
        'prob': round(prob, 4),
        'signals': [f'{s.text} ({s.label})' for s in (signals or [])][:10],
        'domain': st.session_state.get('domain', ''),
    }
    hist = load_history()
    hist.append(rec)
    save_history(hist[-50:])  # giữ 50 bản gần nhất

    return {'mode': mode, 'prob': prob, 'title': title, 'signals': signals, 'heatmap': heatmap, 'rec': rec}


def run_comparison(title, image):
    """Chạy cả 3 mô hình trên cùng một input và trả về kết quả so sánh."""
    rows = []
    for m in ('text', 'image', 'multimodal'):
        prob = predict(m, title, image, load_pipeline(m))
        rows.append({'mode': m, 'prob': prob})

    signals = extract_signals(title)
    heatmap = rollout_heatmap(load_pipeline('multimodal'), image) if image is not None else None

    rec = {
        'id': uuid.uuid4().hex[:8],
        'ts': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'mode': 'compare',
        'title': title,
        'prob': round(max(r['prob'] for r in rows), 4),
        'probs': {r['mode']: round(r['prob'], 4) for r in rows},
        'signals': [f'{s.text} ({s.label})' for s in signals][:10],
        'domain': st.session_state.get('domain', ''),
    }
    hist = load_history()
    hist.append(rec)
    save_history(hist[-50:])

    return {'mode': 'compare', 'rows': rows, 'title': title, 'signals': signals,
            'heatmap': heatmap, 'rec': rec}


def verdict_of(prob, thr, zone=0.10):
    """clickbait / non / uncertain quanh ngưỡng."""
    if prob >= thr + zone:
        return 'clickbait'
    if prob <= thr - zone:
        return 'non'
    return 'uncertain'


VERDICT_UI = {
    'clickbait': ('error', '🚨 **CLICKBAIT**'),
    'non': ('success', '✅ **NON-CLICKBAIT**'),
    'uncertain': ('warning', '⚠️ **KHÔNG CHẮC CHẮN** — nằm sát ngưỡng quyết định'),
}


def render_result(res):
    thr = st.session_state.get('threshold', 0.5)
    mode = res['mode']

    st.divider()

    # --- Mode so sánh 3 model ---
    if mode == 'compare':
        st.subheader("Kết quả — ⚖️ So sánh 3 mô hình")
        cols = st.columns(3)
        for c, row in zip(cols, res['rows']):
            with c:
                st.metric(MODE_LABELS[row['mode']], f"{row['prob']*100:.2f}%")
                st.progress(min(row['prob'], 1.0))
                fn, label = VERDICT_UI[verdict_of(row['prob'], thr)]
                getattr(st, fn)(label)
        st.caption(f'Ngưỡng quyết định: **{thr:.2f}** (±0.10 là vùng không chắc chắn) — chỉnh ở sidebar')
    else:
        prob = res['prob']
        v = verdict_of(prob, thr)
        st.subheader(f"Kết quả — {MODE_LABELS[mode]}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric('Xác suất CLICKBAIT', f'{prob*100:.2f}%',
                      delta=f'{(prob-thr)*100:+.1f} so với ngưỡng', delta_color='inverse')
            st.progress(min(prob, 1.0), text=f'P(clickbait) = {prob:.4f} | ngưỡng {thr:.2f}')
        with c2:
            fn, label = VERDICT_UI[v]
            if res['signals'] is not None:
                label += f" — {len(res['signals'])} dấu hiệu nghi ngờ"
            getattr(st, fn)(label)

    # --- Highlight tiêu đề (text / multimodal) ---
    if res['signals'] is not None:
        st.markdown("**Tiêu đề được phân tích:**")
        st.markdown(
            f'<div style="font-size:1.25rem; line-height:1.9; padding:0.6rem 0.9rem; '
            f'background:#161b22; border-radius:8px;">{render_highlighted(res["title"], res["signals"])}</div>',
            unsafe_allow_html=True,
        )
        groups = summarize(res['signals'])
        if groups:
            st.markdown("**Dấu hiệu clickbait được tìm thấy:**")
            for g in groups:
                st.markdown(
                    f'<span style="background:{g["color"]}33; color:{g["color"]}; font-weight:600; '
                    f'padding:2px 10px; border-radius:12px; margin-right:8px;">'
                    f'{g["label"]} ×{g["count"]}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f'Ví dụ: **{", ".join(g["examples"])}** — {g["why"]}')
        else:
            st.info('Không nhận diện được dấu hiệu quy tắc nào trong tiêu đề này.')

    # --- Heatmap (image / multimodal) ---
    if res['heatmap'] is not None:
        st.markdown("**Phân tích ảnh — Attention Rollout** (vùng đỏ = model tập trung):")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(res['title_img'], caption='Ảnh gốc', use_container_width=True)
        with col_b:
            st.image(res['heatmap'], caption='Attention heatmap', use_container_width=True)
    elif mode == 'image':
        st.warning('Không có ảnh để phân tích mode này.')


# ===================== Giao diện =====================
st.title("📰 ViClickbait Detector — Phát hiện Clickbait tiếng Việt")
st.caption("PhoBERT + ViT (Cross-Modal Attention) · Dataset ViClickbait-2025 · Nâng cấp: dán link + giải thích dấu hiệu")

# ---------- Sidebar: lịch sử ----------
hist = load_history()
st.sidebar.header(f"🕘 Lịch sử phân tích ({len(hist)})")
if hist and st.sidebar.button('🗑️ Xoá lịch sử'):
    save_history([])
    st.rerun()
for rec in reversed(hist[-20:]):
    mode_tag = '⚖️ so sánh' if rec['mode'] == 'compare' else rec['mode']
    with st.sidebar.expander(f"{rec['ts'][6:]} · {mode_tag} · {rec['prob']*100:.1f}%"):
        st.caption(rec.get('domain') or '(nhập tay)')
        st.write(rec['title'][:150])
        if rec.get('probs'):
            st.caption(' · '.join(f'{k} {v*100:.1f}%' for k, v in rec['probs'].items()))
        if rec.get('signals'):
            st.caption('Dấu hiệu: ' + '; '.join(rec['signals'][:5]))
        if st.button('♻️ Dùng lại tiêu đề này', key=f'reuse_{rec["id"]}'):
            st.session_state['title_val'] = rec['title']
            st.rerun()
if not hist:
    st.sidebar.caption('Chưa có lượt phân tích nào.')

# ---------- Sidebar: ngưỡng quyết định ----------
st.sidebar.divider()
st.sidebar.header('⚙️ Cấu hình')
st.sidebar.slider('Ngưỡng quyết định (P ≥ ngưỡng → clickbait):',
                  0.10, 0.90, 0.50, 0.01, key='threshold')
st.sidebar.caption('Vùng ±0.10 quanh ngưỡng → "không chắc chắn".')

# ---------- Chọn mode ----------
mode = st.radio('Chọn mô hình:', list(MODE_LABELS), format_func=lambda k: MODE_LABELS[k], horizontal=True)

# ---------- Dán link ----------
col_link, _ = st.columns([3, 1])
with col_link:
    url = st.text_input('🔗 Dán link bài báo (VnExpress, Thanh Niên, ...):',
                        placeholder='https://vnexpress.net/...')
fetch_btn = st.button('🔗 Lấy tiêu đề & ảnh từ link', type='secondary')

if fetch_btn:
    if not url.strip():
        st.warning('Hãy dán link trước.')
    else:
        with st.spinner('Đang tải bài viết...'):
            art = fetch_article(url)
        if 'error' in art:
            st.error(art['error'] + ' — Hãy dán tiêu đề thủ công bên dưới.')
        else:
            st.session_state['title_val'] = art['title']
            st.session_state['link_image'] = art['image_bytes']
            st.session_state['domain'] = art['domain']
            st.session_state['meta'] = {
                'domain': art['domain'], 'known': art['known_site'],
                'lead': art['lead'], 'published': art['published'], 'url': art['final_url'],
            }
            st.session_state.pop('result', None)
            st.rerun()

# ---------- Meta card sau khi lấy link ----------
meta = st.session_state.get('meta')
if meta:
    badge = '✅' if meta['known'] else '🌐'
    st.caption(f"{badge} **{meta['domain']}**" + (f" · Đăng lúc: {meta['published']}" if meta['published'] else ''))
    if meta['lead']:
        with st.expander('Mô tả / lead từ trang'):
            st.write(meta['lead'])

# ---------- Tiêu đề + ảnh ----------
if 'title_val' not in st.session_state:
    st.session_state['title_val'] = ''
title_input = st.text_area('Tiêu đề bài báo:', key='title_val', height=110,
                           placeholder='Ví dụ: Bạn sẽ không tin chuyện gì xảy ra tiếp theo...')

uploaded = st.file_uploader('Ảnh thumbnail (tùy chọn — nếu link không lấy được):',
                            type=['jpg', 'jpeg', 'png', 'webp'])

# Ưu tiên ảnh upload > ảnh lấy từ link
link_img_bytes = st.session_state.get('link_image')
image = None
if uploaded:
    image = Image.open(uploaded)
elif link_img_bytes:
    image = Image.open(io.BytesIO(link_img_bytes))

# Xem trước ảnh lấy từ link (ảnh upload đã có preview sẵn của Streamlit)
if image is not None and not uploaded:
    st.image(image, caption='Ảnh bìa lấy từ link — dùng cho mode Image / Multimodal', width=240)

if mode in ('image', 'multimodal') and image is None:
    st.info('Mode này cần ảnh: dán link có ảnh bìa hoặc upload ảnh.')

# ---------- Nút phân tích ----------
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    predict_btn = st.button('🔍 Phân tích', type='primary', use_container_width=True)
with col_btn2:
    compare_btn = st.button('⚖️ So sánh cả 3 mô hình', use_container_width=True)

if predict_btn:
    if not title_input.strip() and mode != 'image':
        st.warning('Hãy nhập tiêu đề (hoặc dán link).')
    elif mode in ('image', 'multimodal') and image is None:
        st.error('Mode này cần ảnh — hãy dán link có ảnh bìa hoặc upload ảnh.')
    else:
        with st.spinner(f'Đang phân tích mode {mode}...'):
            title_show = title_input.strip() or '(không có tiêu đề)'
            res = run_analysis(mode, title_show, image)
            res['title_img'] = image
        st.session_state['result'] = res
        st.rerun()

if compare_btn:
    if not title_input.strip():
        st.warning('Hãy nhập tiêu đề (hoặc dán link).')
    elif image is None:
        st.error('So sánh cần ảnh (cho model Image & Multimodal) — hãy dán link có ảnh bìa hoặc upload ảnh.')
    else:
        with st.spinner('Đang chạy cả 3 mô hình...'):
            res = run_comparison(title_input.strip(), image)
            res['title_img'] = image
        st.session_state['result'] = res
        st.rerun()

# ---------- Hiển thị kết quả (giữ qua các lần rerun) ----------
if 'result' in st.session_state and st.session_state['result']:
    render_result(st.session_state['result'])
