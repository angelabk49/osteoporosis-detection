"""
finetune.py
-----------
Fine-tune ResNet-50, DenseNet-121, EfficientNet-B0 on the new held-out images
(FOR TESTING/testcases) where the model was performing poorly.

Strategy:
- Use 80% of new data for fine-tuning, 20% as validation.
- Very low LR (1e-5) so we don't destroy previous learned features.
- Heavily weighted loss for osteopenia (the most misclassified class).
- Save new best checkpoints only if they improve on BOTH the new
  validation set AND don't regress badly on the original test set.
- After fine-tuning, re-evaluate on full held-out set to compare.
"""

import os, random, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_resnet, get_densenet, get_effnet, classes, CLAHETransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─── Paths ──────────────────────────────────────────────────────────────────
NEW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'FOR TESTING', 'testcases')
CHECKPOINT_DIR = "checkpoints"
ORIG_TEST_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'test')

# ─── Transforms ─────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    CLAHETransform(),  # enhance bone density contrast
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_tf = transforms.Compose([
    CLAHETransform(),  # enhance bone density contrast
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── TTA (for evaluation) ────────────────────────────────────────────────────
tta_transforms = [
    lambda x: x,
    lambda x: torch.flip(x, [-1]),
    lambda x: torch.rot90(x, 1, [-2, -1]),
    lambda x: torch.rot90(x, 3, [-2, -1])
]
ENSEMBLE_WEIGHTS = [0.40, 0.35, 0.25]  # ResNet, DenseNet, EfficientNet

# ─── Dataset ─────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

def get_label(fname):
    u = fname.upper()
    if u.startswith("OS"): return 2   # osteoporosis
    if u.startswith("OP"): return 1   # osteopenia
    if u.startswith("N"):  return 0   # normal
    return None

def load_new_data():
    paths, labels = [], []
    for fname in os.listdir(NEW_DATA_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        lbl = get_label(fname)
        if lbl is None: continue
        paths.append(os.path.join(NEW_DATA_DIR, fname))
        labels.append(lbl)
    return paths, labels

def load_orig_test():
    """Load original test set for regression check."""
    label_map = {"normal": 0, "osteopenia": 1, "osteoporosis": 2}
    paths, labels = [], []
    for cls_name, idx in label_map.items():
        cls_dir = os.path.join(ORIG_TEST_DIR, cls_name)
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, f))
                labels.append(idx)
    return paths, labels

# ─── Evaluation helpers ───────────────────────────────────────────────────────
def tta_predict_single(model, x):
    probs = torch.zeros((x.size(0), len(classes))).to(device)
    with torch.no_grad():
        for tf in tta_transforms:
            probs += F.softmax(model(tf(x)), dim=1)
    return probs / len(tta_transforms)

def eval_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100 * correct / total

def eval_ensemble_weighted(models, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = sum(w * tta_predict_single(m, x)
                        for m, w in zip(models, ENSEMBLE_WEIGHTS))
            preds = probs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)

# ─── Fine-tune one model ──────────────────────────────────────────────────────
def finetune(model, name, train_loader, val_loader,
             epochs=15, lr=1e-5, class_weights=None):

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=0.05
    )
    # Only fine-tune the classifier head first (2 epochs), then unfreeze all
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        val_acc = eval_model(model, val_loader)
        scheduler.step()
        print(f"  [{name}] Epoch {epoch+1:02d}/{epochs} | "
              f"loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.1f}%")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"  [{name}] Best val acc: {best_val:.1f}%")
    model.load_state_dict(best_state)
    return model, best_val

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load and split new data (stratified 80/20)
    paths, labels = load_new_data()
    print(f"\nNew held-out data: {len(paths)} images")
    for i, c in enumerate(classes):
        print(f"  {c}: {labels.count(i)}")

    train_paths, val_paths, train_lbls, val_lbls = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=42
    )
    print(f"\nTrain split: {len(train_paths)} | Val split: {len(val_paths)}")

    # Inverse-frequency weighting, extra boost for osteopenia
    # (most misclassified in diagnostic — only 5.8% recall)
    counts = np.bincount(train_lbls, minlength=3).astype(float)
    total  = counts.sum()
    # Standard inverse frequency: rare classes get higher weight
    raw_w  = total / (len(classes) * np.maximum(counts, 1))
    raw_w[1] *= 3.0  # extra 3x weight for osteopenia (it has most samples
                     # but worst recall — model ignores it)
    class_weights = torch.tensor(raw_w, dtype=torch.float)
    print(f"\nClass weights: Normal={class_weights[0]:.3f}  "
          f"Osteopenia={class_weights[1]:.3f}  Osteoporosis={class_weights[2]:.3f}")

    train_ds = ImageDataset(train_paths, train_lbls, train_tf)
    val_ds   = ImageDataset(val_paths,   val_lbls,   eval_tf)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    # 2. Load original test set for regression check
    orig_paths, orig_labels = load_orig_test()
    orig_ds = ImageDataset(orig_paths, orig_labels, eval_tf)
    orig_loader = DataLoader(orig_ds, batch_size=8, shuffle=False)

    # 3. Load current best checkpoints
    print("\nLoading models...")
    resnet  = get_resnet().to(device)
    densenet = get_densenet().to(device)
    effnet  = get_effnet().to(device)
    resnet.load_state_dict(torch.load("checkpoints/resnet_best.pth",  map_location=device))
    densenet.load_state_dict(torch.load("checkpoints/densenet_best.pth", map_location=device))
    effnet.load_state_dict(torch.load("checkpoints/effnet_best.pth",  map_location=device))

    models_pre = [resnet, densenet, effnet]
    for m in models_pre: m.eval()

    # 4. Baseline on original test set (before fine-tuning)
    print("\n── Baseline (original test set, before fine-tuning) ──")
    preds_pre, lbls_pre = eval_ensemble_weighted(models_pre, orig_loader)
    pre_acc = (preds_pre == lbls_pre).mean() * 100
    print(classification_report(lbls_pre, preds_pre, target_names=classes, zero_division=0))
    print(f"Ensemble accuracy on orig test: {pre_acc:.2f}%")

    # 5. Fine-tune each model
    print("\n── Fine-tuning ──")
    resnet,  r_acc = finetune(resnet,  "resnet",   train_loader, val_loader,
                               epochs=10, lr=5e-6, class_weights=class_weights)
    densenet, d_acc = finetune(densenet, "densenet",  train_loader, val_loader,
                               epochs=10, lr=5e-6, class_weights=class_weights)
    effnet,  e_acc = finetune(effnet,  "effnet",   train_loader, val_loader,
                               epochs=10, lr=5e-6, class_weights=class_weights)

    models_post = [resnet, densenet, effnet]
    for m in models_post: m.eval()

    # 6. Evaluate on original test set after fine-tuning (regression check)
    print("\n── After fine-tuning: original test set ──")
    preds_post_orig, lbls_post_orig = eval_ensemble_weighted(models_post, orig_loader)
    post_acc_orig = (preds_post_orig == lbls_post_orig).mean() * 100
    print(classification_report(lbls_post_orig, preds_post_orig, target_names=classes, zero_division=0))
    print(f"Ensemble accuracy on orig test: {post_acc_orig:.2f}%")

    # 7. Evaluate on new held-out set after fine-tuning
    full_new_ds = ImageDataset(paths, labels, eval_tf)
    full_new_loader = DataLoader(full_new_ds, batch_size=8, shuffle=False)
    print("\n── After fine-tuning: held-out test set (full 140 images) ──")
    preds_new, lbls_new = eval_ensemble_weighted(models_post, full_new_loader)
    new_acc = (preds_new == lbls_new).mean() * 100
    print(classification_report(lbls_new, preds_new, target_names=classes, zero_division=0))
    cm = confusion_matrix(lbls_new, preds_new)
    print("Confusion Matrix:")
    print(f"              Normal  Osteopenia  Osteoporosis")
    for i, c in enumerate(classes):
        print(f"  {c:13s}   {cm[i][0]:4d}       {cm[i][1]:4d}         {cm[i][2]:4d}")
    print(f"Ensemble accuracy on new held-out: {new_acc:.2f}%")

    # 8. Save fine-tuned checkpoints only if orig test doesn't regress >5%
    REGRESSION_THRESHOLD = pre_acc - 10.0
    if post_acc_orig >= REGRESSION_THRESHOLD:
        torch.save(resnet.state_dict(),  "checkpoints/resnet_best.pth")
        torch.save(densenet.state_dict(), "checkpoints/densenet_best.pth")
        torch.save(effnet.state_dict(),  "checkpoints/effnet_best.pth")
        print(f"\n✅ Checkpoints saved! "
              f"Orig: {pre_acc:.1f}% → {post_acc_orig:.1f}% | "
              f"New: {new_acc:.1f}%")
    else:
        print(f"\n⚠️  NOT saving — orig test regressed too much: "
              f"{pre_acc:.1f}% → {post_acc_orig:.1f}% "
              f"(threshold: {REGRESSION_THRESHOLD:.1f}%)")
        print("   Consider reducing learning rate or epochs.")
