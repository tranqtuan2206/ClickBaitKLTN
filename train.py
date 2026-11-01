import os
import sys
import copy
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel, ViTModel
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from text_utils import segment_text

# Đảm bảo in ký tự tiếng Việt không bị lỗi bảng mã trên Windows console
if sys.platform == 'win32':
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr is not None:
        sys.stderr.reconfigure(encoding='utf-8')

# 0. Tham số dòng lệnh
parser = argparse.ArgumentParser(description='Huấn luyện mô hình clickbait')
parser.add_argument(
    '--mode', type=str, default='multimodal',
    choices=['text', 'image', 'multimodal'],
    help='Chế độ mô hình: text (chỉ PhoBERT), image (chỉ ViT), multimodal (cả hai)'
)
parser.add_argument('--epochs', type=int, default=3, help='Số epoch huấn luyện')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
args = parser.parse_args()
MODE = args.mode

# 1. Đường dẫn dữ liệu tại máy local
repo_dir = '.'
csv_path = os.path.join(repo_dir, 'data', 'text', 'clickbait_dataset_vietnamese.csv')
images_dir = os.path.join(repo_dir, 'data', 'images')

df = pd.read_csv(csv_path)
print(f"Đã tải xong! Tổng số lượng mẫu dữ liệu: {len(df)}")
print(f"Chế độ huấn luyện: {MODE.upper()}")

# 2. Khởi tạo Tokenizer và Image Processor
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 3. Định nghĩa PyTorch Dataset (Đồng bộ cách gọi tên ảnh với script download)
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

        # Xử lý văn bản (tách từ trước khi tokenize — chuẩn PhoBERT)
        title = segment_text(str(row['title']))
        text_encoding = self.tokenizer(
            title,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Xử lý hình ảnh (Khớp chuẩn logic đặt tên file với download_images.py)
        url = str(row['thumbnail_url'])
        clean_url = url.split('?')[0]
        img_name = os.path.basename(clean_url)
        if not img_name or not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif')):
            img_name = f"thumb_{idx}.jpg"

        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback nếu ảnh lỗi hoặc thiếu
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        image_encoding = self.processor(images=image, return_tensors="pt")

        # Nhãn nhị phân
        label = 1.0 if row['label'] == 'clickbait' else 0.0

        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': image_encoding['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

dataset = ViClickbaitDataset(df, images_dir, tokenizer, image_processor)

# 4. Chia tập Train / Val (cố định seed 42 để kết quả nhất quán, num_workers=0 để tránh lỗi đa tiến trình trên Windows)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print(f"Số lượng mẫu Train: {len(train_dataset)} | Số lượng mẫu Validation: {len(val_dataset)}")

# 5. Định nghĩa Kiến trúc Mô hình theo 3 chế độ
class TextOnlyClickbaitModel(nn.Module):
    """Chỉ sử dụng PhoBERT cho tiêu đề"""

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
    """Chỉ sử dụng ViT cho thumbnail"""

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
    """Kết hợp PhoBERT + ViT + Cross-Modal Attention"""

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
            nn.Linear(256, 1),
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

# Đoạn mã huấn luyện chính
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL_CLASSES[MODE]().to(device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Không tìm thấy GPU'
    print(f"Đã khởi tạo mô hình trên thiết bị: {device} | Tên GPU: {gpu_name}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")

    def evaluate_model(model, dataloader, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                with autocast("cuda"):
                    outputs = model(input_ids, attention_mask, pixel_values)
                    loss = criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    num_epochs = args.epochs
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN MODE {MODE.upper()} ({num_epochs} epochs) ---")

    # Theo dõi trọng số epoch tốt nhất (dựa trên Val Acc)
    best_val_acc = -1.0
    best_epoch = -1
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"[{MODE}] Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            with autocast("cuda"):
                outputs = model(input_ids, attention_mask, pixel_values)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * input_ids.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        val_loss, val_acc = evaluate_model(model, val_loader, device)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Hoàn tất:")
        print(f" - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f" - Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc*100:.2f}%\n")

        # Cập nhật trọng số tốt nhất nếu val_acc cải thiện
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  ★ Lưu trọng số tốt nhất mới tại epoch {best_epoch} (Val Acc: {best_val_acc*100:.2f}%)\n")

    print(f"Huấn luyện mode {MODE.upper()} hoàn thành!")
    print(f"Epoch tốt nhất: {best_epoch}/{num_epochs} với Val Acc = {best_val_acc*100:.2f}%")
    ckpt_name = CHECKPOINT_NAMES[MODE]
    torch.save(best_model_wts, ckpt_name)
    print(f"Đã lưu trọng số TỐT NHẤT (epoch {best_epoch}) thành {ckpt_name}!")
