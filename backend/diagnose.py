"""
diagnose.py
-----------
Run all held-out test images through the ensemble (with TTA)
and print a detailed diagnostic report.
"""

import os, sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_resnet, get_densenet, get_effnet, classes, CLAHETransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ---------- Preprocessing (same as model.py) ----------
transform = transforms.Compose([
    CLAHETransform(),  # enhance bone density contrast
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- TTA transforms (same as model.py) ----------
tta_transforms = [
    lambda x: x,
    lambda x: torch.flip(x, [-1]),
    lambda x: torch.rot90(x, 1, [-2, -1]),
    lambda x: torch.rot90(x, 3, [-2, -1])
]

def tta_predict(model, x):
    probs = torch.zeros((1, len(classes))).to(device)
    with torch.no_grad():
        for tf in tta_transforms:
            probs += F.softmax(model(tf(x)), dim=1)
    return probs / len(tta_transforms)

ENSEMBLE_WEIGHTS = [0.40, 0.35, 0.25]  # [ResNet-50, DenseNet-121, EfficientNet-B0]

def ensemble_predict(models, x):
    weighted_probs = sum(
        w * tta_predict(m, x)
        for m, w in zip(models, ENSEMBLE_WEIGHTS)
    )
    return weighted_probs

# ---------- Load models ----------
print("Loading models...")
resnet = get_resnet()
densenet = get_densenet()
effnet = get_effnet()

resnet.load_state_dict(torch.load("checkpoints/resnet_best.pth", map_location=device))
densenet.load_state_dict(torch.load("checkpoints/densenet_best.pth", map_location=device))
effnet.load_state_dict(torch.load("checkpoints/effnet_best.pth", map_location=device))

for m in [resnet, densenet, effnet]:
    m.to(device)
    m.eval()

models = [resnet, densenet, effnet]
print("Models loaded.\n")

# ---------- Held-out test images ----------
test_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'FOR TESTING', 'testcases')

# Determine true label from filename prefix
def get_true_label(filename):
    name = filename.upper()
    if name.startswith("OS"):
        return "osteoporosis"
    elif name.startswith("OP"):
        return "osteopenia"
    elif name.startswith("N"):
        return "normal"
    return None

# ---------- Run predictions ----------
all_true, all_pred, all_conf = [], [], []
misclassified = []

files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"Found {len(files)} test images.\n")

for fname in files:
    fpath = os.path.join(test_dir, fname)
    true_label = get_true_label(fname)
    if true_label is None:
        print(f"  SKIP (unknown prefix): {fname}")
        continue

    img = Image.open(fpath).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    probs = ensemble_predict(models, x)
    pred_idx = probs.argmax(1).item()
    pred_label = classes[pred_idx]
    confidence = probs[0][pred_idx].item()

    # Per-class probabilities
    class_probs = {cls: probs[0][i].item() for i, cls in enumerate(classes)}

    all_true.append(true_label)
    all_pred.append(pred_label)
    all_conf.append(confidence)

    if pred_label != true_label:
        misclassified.append({
            "file": fname,
            "true": true_label,
            "pred": pred_label,
            "conf": confidence,
            "probs": class_probs
        })

# ---------- Also run individual model predictions for comparison ----------
print("="*70)
print("INDIVIDUAL MODEL ANALYSIS (first 5 OS images)")
print("="*70)

os_files = [f for f in files if f.upper().startswith("OS")][:5]
for fname in os_files:
    fpath = os.path.join(test_dir, fname)
    img = Image.open(fpath).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    print(f"\n  {fname} (true: osteoporosis)")
    for model, name in zip(models, ["ResNet-50", "DenseNet-121", "EfficientNet"]):
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()
            print(f"    {name:15s} -> {classes[pred_idx]:14s} "
                  f"[N={probs[0][0]:.3f}  OP={probs[0][1]:.3f}  OS={probs[0][2]:.3f}]")
    
    # Ensemble with TTA
    ens_probs = ensemble_predict(models, x)
    ens_pred = classes[ens_probs.argmax(1).item()]
    print(f"    {'Ensemble+TTA':15s} -> {ens_pred:14s} "
          f"[N={ens_probs[0][0]:.3f}  OP={ens_probs[0][1]:.3f}  OS={ens_probs[0][2]:.3f}]")

# ---------- Report ----------
print("\n" + "="*70)
print("CLASSIFICATION REPORT (held-out test set)")
print("="*70)

# Convert to indices for confusion matrix
true_idx = [classes.index(t) for t in all_true]
pred_idx = [classes.index(p) for p in all_pred]

print(classification_report(true_idx, pred_idx, target_names=classes, zero_division=0))

cm = confusion_matrix(true_idx, pred_idx)
print("Confusion Matrix:")
print(f"{'':>15s}  {'Normal':>8s}  {'Osteopenia':>10s}  {'Osteoporosis':>12s}")
for i, cls in enumerate(classes):
    print(f"  {cls:>13s}  {cm[i][0]:>8d}  {cm[i][1]:>10d}  {cm[i][2]:>12d}")

# ---------- Misclassifications ----------
print(f"\n{'='*70}")
print(f"MISCLASSIFIED IMAGES ({len(misclassified)} / {len(all_true)})")
print(f"{'='*70}")

for m in misclassified:
    p = m["probs"]
    print(f"  {m['file']:20s}  true={m['true']:14s}  pred={m['pred']:14s}  "
          f"conf={m['conf']:.3f}  "
          f"[N={p['normal']:.3f} OP={p['osteopenia']:.3f} OS={p['osteoporosis']:.3f}]")

# ---------- Summary ----------
correct = sum(1 for t, p in zip(all_true, all_pred) if t == p)
total = len(all_true)
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Total images:     {total}")
print(f"  Correct:          {correct}")
print(f"  Misclassified:    {total - correct}")
print(f"  Accuracy:         {100*correct/total:.1f}%")
print(f"  Avg confidence:   {np.mean(all_conf):.3f}")

# Per-class accuracy
for cls in classes:
    cls_true = [i for i, t in enumerate(all_true) if t == cls]
    cls_correct = sum(1 for i in cls_true if all_pred[i] == cls)
    if cls_true:
        print(f"  {cls:14s}:    {cls_correct}/{len(cls_true)} = {100*cls_correct/len(cls_true):.1f}%")
