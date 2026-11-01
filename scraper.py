"""Lấy tiêu đề, mô tả và ảnh bìa từ link bài báo Việt Nam (og:*/twitter:* meta)."""
import io
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
    )
}

# Một số trang báo Việt Nam phổ biến (chỉ để hiển thị nhận diện, không chặn link lạ)
KNOWN_SITES = [
    'vnexpress.net', 'thanhnien.vn', 'tuoitre.vn', 'dantri.com.vn',
    'vietnamnet.vn', '24h.com.vn', 'cafebiz.vn', 'genk.vn', 'kenh14.vn',
    'soha.vn', 'yan.vn', 'vtv.vn', 'vov.vn', 'nld.com.vn', 'laodong.vn',
]


def _meta_content(soup, *attrs):
    """Tìm nội dung theo nhiều thuộc tính meta khác nhau."""
    for kw in attrs:
        tag = soup.find('meta', attrs=kw)
        if tag and tag.get('content'):
            val = tag.get('content').strip()
            if val:
                return val
    return None


def fetch_article(url, timeout=12):
    """Trả về dict article hoặc dict{'error': ...}. Luôn có ít nhất title/image nếu trang có og tags."""
    url = (url or '').strip()
    if not url:
        return {'error': 'Chưa nhập link.'}
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    domain = urlparse(url).netloc.lower().replace('www.', '')
    if not domain or '.' not in domain:
        return {'error': 'Link không hợp lệ.'}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {'error': 'Trang phản hồi quá lâu (timeout).'}
    except requests.exceptions.HTTPError as e:
        return {'error': f'Trang trả về lỗi HTTP {e.response.status_code} (có thể chặn bot).'}
    except requests.exceptions.RequestException:
        return {'error': 'Không kết nối được tới link này.'}

    # Một số trang trả JSON lỗi dù status 200
    if 'application/json' in resp.headers.get('Content-Type', ''):
        return {'error': 'Trang trả về dữ liệu không phải HTML (có thể chặn bot).'}

    soup = BeautifulSoup(resp.text, 'html.parser')

    title = (
        _meta_content(soup, {'property': 'og:title'}, {'name': 'og:title'},
                      {'property': 'twitter:title'}, {'name': 'twitter:title'})
        or (soup.title.string.strip() if soup.title and soup.title.string else '')
    )
    lead = _meta_content(
        soup,
        {'property': 'og:description'}, {'name': 'og:description'},
        {'property': 'twitter:description'}, {'name': 'twitter:description'},
        {'name': 'description'},
    ) or ''
    image_url = _meta_content(
        soup,
        {'property': 'og:image'}, {'name': 'og:image'},
        {'property': 'twitter:image'}, {'name': 'twitter:image'},
        {'name': 'twitter:image:src'},
    ) or ''
    published = _meta_content(
        soup,
        {'property': 'article:published_time'},
        {'name': 'article:published_time'},
        {'property': 'og:updated_time'},
    ) or ''

    if not title:
        return {'error': 'Không lấy được tiêu đề từ trang này (thử dán tiêu đề thủ công).'}

    # Tải ảnh bìa (nếu có)
    image_bytes = None
    if image_url:
        try:
            if image_url.startswith('//'):
                image_url = 'https:' + image_url
            img_resp = requests.get(image_url, headers=HEADERS, timeout=timeout)
            if img_resp.status_code == 200 and len(img_resp.content) > 1024:
                image_bytes = img_resp.content
        except requests.exceptions.RequestException:
            image_bytes = None

    return {
        'title': re.sub(r'\s+', ' ', title).strip(),
        'lead': re.sub(r'\s+', ' ', lead).strip(),
        'image_bytes': image_bytes,
        'image_url': image_url,
        'published': published,
        'domain': domain,
        'final_url': resp.url,
        'known_site': any(domain.endswith(s) for s in KNOWN_SITES),
    }
