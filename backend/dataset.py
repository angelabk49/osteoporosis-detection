"""
dataset.py
----------
Purpose:
- Define PyTorch Dataset and DataLoader.
- Apply ONLINE (runtime) image transformations.
- Provide pretrained CNN models (ResNet, EfficientNet, DenseNet).

Key Points:
- Online augmentations occur EVERY epoch.
- Train/Test split is stratified.
- Models are ImageNet-pretrained and adapted for 3 classes.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision.models import efficientnet_b0


# ---------- CLAHE Preprocessing (critical for X-ray classification) ----------
class CLAHETransform:
    """Apply CLAHE to enhance bone density contrast in X-ray images.
    Converts to grayscale, applies CLAHE, converts back to 3-channel RGB."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, pil_img):
        img_array = np.array(pil_img.convert("L"))  # grayscale
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        enhanced = clahe.apply(img_array)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        from PIL import Image as _Image
        return _Image.fromarray(enhanced_rgb)
effnet_path = os.path.join(os.path.dirname(__file__), 'efficientnet_b0_local.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import platform
NUM_WORKERS = 0 if platform.system() == "Windows" else 2

base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
classes = ['normal', 'osteopenia', 'osteoporosis']
train_files = set(os.listdir(os.path.join(base_dir, "train", "osteoporosis")))
test_files  = set(os.listdir(os.path.join(base_dir, "test", "osteoporosis")))
class KneeDataset(Dataset):
    def __init__(self, paths, labels, transform=None,return_paths=False):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.return_paths=return_paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.return_paths:
            return img, self.labels[idx],self.paths[idx]
        return img, self.labels[idx]

    def __len__(self):
        return len(self.paths)


def load_data(
    batch_size=8,
    val_split=0.2,
    seed=42
):
    """
    Returns:
        train_loader : DataLoader (train subset, augmented)
        val_loader   : DataLoader (validation subset, no augmentation)
        test_loader  : DataLoader (held-out test set)
        train_labels : list[int] (for class-weight computation)
    """

    def load_split(split):
        paths, labels = [], []
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(base_dir, split, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(cls_dir, f))
                    labels.append(i)
        return paths, labels

    # -------------------- Load raw paths --------------------
    full_train_paths, full_train_labels = load_split("train")
    test_paths, test_labels = load_split("test")

    # -------------------- Train / Val split --------------------
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        full_train_paths,
        full_train_labels,
        test_size=val_split,
        stratify=full_train_labels,
        random_state=seed
    )

    # -------------------- Transforms --------------------
    train_tf = transforms.Compose([
        CLAHETransform(),  # enhance bone density contrast
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_tf = transforms.Compose([
        CLAHETransform(),  # enhance bone density contrast
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------- Datasets --------------------
    train_ds = KneeDataset(train_paths, train_labels, train_tf)
    val_ds   = KneeDataset(val_paths, val_labels, eval_tf)
    test_ds  = KneeDataset(test_paths, test_labels, eval_tf,return_paths=True)

    # -------------------- Loaders --------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    return train_loader, val_loader, test_loader, train_labels


def get_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model.to(device)

def get_effnet(num_classes=3, device=torch.device("cpu")):
    model = efficientnet_b0(weights=None)  # no online download

    if os.path.exists(effnet_path):
        state = torch.load(effnet_path, map_location=device)
        model.load_state_dict(state)
        print("[EffNet] Loaded local pretrained weights")
    else:
        print("[EffNet] Local weights not found — initialized randomly")

    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

def get_densenet():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.to(device)
