import os
import sys
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel, ViTModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.amp import autocast

from text_utils import segment_text

# Đảm bảo in ký tự tiếng Việt không bị lỗi bảng mã trên Windows console
if sys.platform == 'win32':
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr is not None:
        sys.stderr.reconfigure(encoding='utf-8')

# 0. Tham số dòng lệnh
parser = argparse.ArgumentParser(description='Đánh giá mô hình clickbait')
parser.add_argument(
    '--mode', type=str, default='multimodal',
    choices=['text', 'image', 'multimodal'],
    help='Chế độ mô hình: text, image, multimodal'
)
parser.add_argument('--batch-size', type=int, default=8, help='Batch size khi đánh giá')
args = parser.parse_args()
MODE = args.mode

# 1. Cấu hình đường dẫn
repo_dir = '.'
csv_path = os.path.join(repo_dir, 'data', 'text', 'clickbait_dataset_vietnamese.csv')
images_dir = os.path.join(repo_dir, 'data', 'images')

df = pd.read_csv(csv_path)

# 2. Khởi tạo Tokenizer và Processor
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 3. Định nghĩa Dataset (giống hệt lúc train)
class ViClickbaitDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, processor, max_length=256):
        self.df = dataframe
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = segment_text(str(row['title']))
        text_encoding = self.tokenizer(
            title, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        url = str(row['thumbnail_url'])
        clean_url = url.split('?')[0]
        img_name = os.path.basename(clean_url)
        if not img_name or not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif')):
            img_name = f"thumb_{idx}.jpg"

        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        image_encoding = self.processor(images=image, return_tensors="pt")
        label = 1.0 if row['label'] == 'clickbait' else 0.0

        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': image_encoding['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

dataset = ViClickbaitDataset(df, images_dir, tokenizer, image_processor)

# Chia đúng tỷ lệ 80/20 và cố định seed 42 khớp với lúc train
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


# 4. Khai báo lại kiến trúc mô hình (giống hệt train.py)
class TextOnlyClickbaitModel(nn.Module):
    def __init__(self, phobert_name="vinai/phobert-base"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        hidden_size = self.text_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values=None):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = torch.mean(text_outputs.last_hidden_state, dim=1)
        return self.classifier(pooled).squeeze(1)


class ImageOnlyClickbaitModel(nn.Module):
    def __init__(self, vit_name="google/vit-base-patch16-224"):
        super().__init__()
        self.image_encoder = ViTModel.from_pretrained(vit_name)
        hidden_size = self.image_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        pooled = torch.mean(image_outputs.last_hidden_state, dim=1)
        return self.classifier(pooled).squeeze(1)


class MultimodalClickbaitModel(nn.Module):
    def __init__(self, phobert_name="vinai/phobert-base", vit_name="google/vit-base-patch16-224", num_heads=8):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        self.image_encoder = ViTModel.from_pretrained(vit_name)

        text_hidden_size = self.text_encoder.config.hidden_size
        image_hidden_size = self.image_encoder.config.hidden_size

        if text_hidden_size != image_hidden_size:
            self.image_projection = nn.Linear(image_hidden_size, text_hidden_size)
        else:
            self.image_projection = nn.Identity()

        self.cross_attention = nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state
        image_features = self.image_projection(image_features)

        attn_output, _ = self.cross_attention(query=text_features, key=image_features, value=image_features)
        text_pooled = torch.mean(text_features, dim=1)
        multimodal_pooled = torch.mean(attn_output, dim=1)

        combined = torch.cat((text_pooled, multimodal_pooled), dim=1)
        out = self.classifier(combined)
        return out.squeeze(1)


MODEL_CLASSES = {
    'text': TextOnlyClickbaitModel,
    'image': ImageOnlyClickbaitModel,
    'multimodal': MultimodalClickbaitModel,
}
CHECKPOINT_NAMES = {
    'text': 'text_only_clickbait_model.pth',
    'image': 'image_only_clickbait_model.pth',
    'multimodal': 'multimodal_clickbait_model.pth',
}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL_CLASSES[MODE]().to(device)

    model_weights_path = CHECKPOINT_NAMES[MODE]
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, weights_only=True))
        print(f"Đã tải trọng số từ {model_weights_path}!")
    else:
        print(f"CẢNH BÁO: Không tìm thấy {model_weights_path}! Hãy chạy train.py --mode {MODE} trước.")

    model.eval()
    all_preds = []
    all_labels = []

    print(f"Đang đánh giá mode {MODE.upper()} trên tập Validation...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            with autocast("cuda"):
                outputs = model(input_ids, attention_mask, pixel_values)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Tính toán các chỉ số đánh giá theo yêu cầu khóa luận
    print(f"\n================ KẾT QUẢ ĐÁNH GIÁ [{MODE.upper()}] ================")
    print(classification_report(all_labels, all_preds, target_names=['Non-Clickbait (0)', 'Clickbait (1)'], digits=4))

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Macro F1-score chính: {macro_f1:.4f}")

    print("\nConfusion Matrix (Ma trận nhầm lẫn):")
    print(confusion_matrix(all_labels, all_preds))
    print("================================================================")
