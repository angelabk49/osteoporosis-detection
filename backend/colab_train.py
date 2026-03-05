"""
colab_train.py — Google Colab Training Script
==============================================
Upload this file + your data/ folder to Google Colab.

Instructions:
1. Open Google Colab (colab.research.google.com)
2. Set Runtime → Change runtime type → GPU (T4)
3. Upload your project as a zip or mount Google Drive
4. Run this script

This script trains ResNet-50, DenseNet-121, and EfficientNet-B0
and saves the best checkpoints.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import efficientnet_b0
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ─── Configuration ───────────────────────────────────────────────────────────
BATCH_SIZE = 16       # Can be larger with GPU
EPOCHS = 50
PATIENCE = 5
IMAGE_SIZE = 384
NUM_WORKERS = 2       # Works fine on Colab Linux

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── CLAHE Preprocessing ────────────────────────────────────────────────────
class CLAHETransform:
    """Apply CLAHE to enhance bone density contrast in X-ray images."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, pil_img):
        img_array = np.array(pil_img.convert("L"))
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        enhanced = clahe.apply(img_array)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

# ─── Dataset ─────────────────────────────────────────────────────────────────
classes = ['normal', 'osteopenia', 'osteoporosis']

class KneeDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.paths)

# ─── Transforms ──────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_tf = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Data Loading ────────────────────────────────────────────────────────────
def load_data(base_dir, batch_size=BATCH_SIZE, val_split=0.2, seed=42):
    """Load train/val/test data from the data/ directory."""

    def load_split(split):
        paths, labels = [], []
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(base_dir, split, cls)
            if not os.path.exists(cls_dir):
                print(f"  Warning: {cls_dir} not found, skipping")
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(cls_dir, f))
                    labels.append(i)
        return paths, labels

    full_train_paths, full_train_labels = load_split("train")
    test_paths, test_labels = load_split("test")

    print(f"Total train images: {len(full_train_paths)}")
    print(f"Total test images: {len(test_paths)}")
    for i, cls in enumerate(classes):
        tr_count = full_train_labels.count(i)
        te_count = test_labels.count(i)
        print(f"  {cls}: train={tr_count}, test={te_count}")

    # Stratified train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        full_train_paths, full_train_labels,
        test_size=val_split, stratify=full_train_labels, random_state=seed
    )

    train_ds = KneeDataset(train_paths, train_labels, train_tf)
    val_ds = KneeDataset(val_paths, val_labels, eval_tf)
    test_ds = KneeDataset(test_paths, test_labels, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_labels

# ─── Model Definitions ──────────────────────────────────────────────────────
def get_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model.to(device)

def get_densenet():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.to(device)

def get_effnet():
    model = efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return model.to(device)

# ─── Training ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def train_model(model, name, train_loader, val_loader, train_labels, epochs=EPOCHS):
    labels_np = np.array(train_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_np), y=labels_np)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(f"  {name} | Epoch {epoch+1:02d}/{epochs} | Val Acc: {val_acc:.2f}% | Loss: {avg_loss:.4f} | LR: {lr:.1e}")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if val_acc >= 85 and patience_counter >= PATIENCE:
            print(f"  {name} — early stopping at epoch {epoch+1} (best: {best_acc:.2f}%)")
            break

    print(f"  ✅ {name} training complete. Best val accuracy: {best_acc:.2f}%\n")
    # Reload best weights
    model.load_state_dict(torch.load(f"checkpoints/{name}_best.pth", map_location=device))
    return model

# ─── Ensemble Evaluation ────────────────────────────────────────────────────
def ensemble_evaluate(models, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.stack([torch.softmax(m(x), 1) for m in models]).mean(0)
            preds = probs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Set your data directory here ──
    # If using Google Drive:
    #   from google.colab import drive
    #   drive.mount('/content/drive')
    #   DATA_DIR = '/content/drive/MyDrive/Osteoporosis Knee X ray/data'
    # If uploaded directly:
    #   DATA_DIR = '/content/data'
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("=" * 60)
    print("  ODS — Training Pipeline")
    print(f"  Device: {device}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {EPOCHS}")
    print("=" * 60)

    # Load data
    print("\n📂 Loading dataset...")
    train_loader, val_loader, test_loader, train_labels = load_data(DATA_DIR)

    # Train models
    print("\n🏋️ Training ResNet-50...")
    resnet = train_model(get_resnet(), "resnet", train_loader, val_loader, train_labels)

    print("🏋️ Training DenseNet-121...")
    densenet = train_model(get_densenet(), "densenet", train_loader, val_loader, train_labels)

    print("🏋️ Training EfficientNet-B0...")
    effnet = train_model(get_effnet(), "effnet", train_loader, val_loader, train_labels)

    # Ensemble evaluation
    print("=" * 60)
    print("📊 Ensemble Evaluation on Test Set")
    print("=" * 60)
    all_models = [resnet, densenet, effnet]
    for m in all_models:
        m.eval()

    preds, labels = ensemble_evaluate(all_models, test_loader)
    print(classification_report(labels, preds, target_names=classes, zero_division=0))
    acc = (preds == labels).mean() * 100
    print(f"Ensemble Test Accuracy: {acc:.2f}%")

    print("\n✅ Training complete! Checkpoints saved in checkpoints/")
    print("   - checkpoints/resnet_best.pth")
    print("   - checkpoints/densenet_best.pth")
    print("   - checkpoints/effnet_best.pth")
    print("\nDownload these files and place them in your backend/checkpoints/ folder.")
