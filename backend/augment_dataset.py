"""
augment_dataset.py
------------------
Purpose:
- Perform OFFLINE data augmentation on knee X-ray images.
- Expand dataset size on disk before training.
- Save augmented images permanently into database_augmented/.

Key Points:
- Uses medically safe augmentations (rotation, flip, brightness, crop, shift).
- Oversamples the 'normal' class slightly to improve recall.
- This script is run ONCE before training.
"""

import os
import random
from PIL import Image
import torchvision.transforms as transforms

# Dataset paths
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database_original')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database_augmented')

classes = ["normal", "osteopenia", "osteoporosis"]

# Create output directories
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# Augmentation operations
augmentation_ops = [
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.1)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
]

# Augmentation loop
for cls in classes:
    src = os.path.join(input_dir, cls)
    dst = os.path.join(output_dir, cls)

    aug_per_image = 7 if cls == "normal" else 5

    for img_name in os.listdir(src):
        img = Image.open(os.path.join(src, img_name)).convert("RGB")

        # Save original
        img.save(os.path.join(dst, img_name))

        # Save augmented versions
        for i in range(aug_per_image):
            ops = random.sample(augmentation_ops, k=random.randint(2, 3))
            aug_img = transforms.Compose(ops)(img)
            aug_img.save(os.path.join(dst, f"{img_name.split('.')[0]}_aug{i+1}.jpg"))

print("Offline augmentation complete.")
