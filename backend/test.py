"""
test.py
--------
Purpose:
- Evaluate ensemble of CNN models (ResNet, DenseNet, EfficientNet) on test data.
- Apply Test Time Augmentation (TTA) to improve predictions.
- Save detailed predictions (image path, true label, predicted label) to CSV.
"""
from sklearn.preprocessing import label_binarize


import torch
import torch.nn.functional as F
import pandas as pd
from dataset import load_data, get_resnet, get_densenet, get_effnet, classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,roc_curve,auc
)

from torch.utils.data import Subset, DataLoader
os.makedirs("checkpoints/reports", exist_ok=True)
os.makedirs("checkpoints/confusion_matrices", exist_ok=True)

# ------------------------------
# Test Time Augmentation (TTA)
# ------------------------------
tta_transforms = [
    lambda x: x,  # original
    lambda x: torch.flip(x, [-1]),  # horizontal flip
    lambda x: torch.rot90(x, 1, [-2, -1]),  # rotate 90
    lambda x: torch.rot90(x, 3, [-2, -1])   # rotate 270
]


def tta_predict(model, x):
    """
    Apply TTA transforms and average predictions.
    x: single batch tensor
    returns: averaged softmax probabilities
    """
    model.eval()
    probs = torch.zeros((x.size(0), len(classes))).to(device)
    with torch.no_grad():
        for tf in tta_transforms:
            probs += F.softmax(model(tf(x)), dim=1)
    return probs / len(tta_transforms)


# Ensemble weights: [ResNet-50, DenseNet-121, EfficientNet-B0]
ENSEMBLE_WEIGHTS = [0.40, 0.35, 0.25]

def ensemble_predict(models, loader):
    """
    Weighted ensemble prediction with TTA.
    Supports datasets that return (x, y) or (x, y, path).
    """
    all_preds, all_labels, all_probs, all_paths = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, paths = batch
                all_paths.extend(paths)
            else:
                x, y = batch

            x = x.to(device)

            # Weighted ensemble with TTA
            probs = sum(
                w * tta_predict(m, x)
                for m, w in zip(models, ENSEMBLE_WEIGHTS)
            )

            preds = probs.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        all_paths
    )

def save_confusion_matrix(y_true, y_pred, fold):
    cm = confusion_matrix(y_true, y_pred)

    # Save CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(f"checkpoints/confusion_matrices/cm_fold_{fold}.csv")

    # Save image
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"checkpoints/confusion_matrices/cm.png")
    plt.close()
def save_combined_roc(labels, probs, classes, save_dir="checkpoints/roc"):
    """
    Saves a single ROC plot containing all class-wise ROC curves
    """
    os.makedirs(save_dir, exist_ok=True)

    # Binarize labels: (N,) -> (N, C)
    y_true = label_binarize(labels, classes=range(len(classes)))

    plt.figure(figsize=(7, 6))

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{cls} (AUC = {roc_auc:.3f})"
        )

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves(All classes)")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_path = os.path.join(save_dir, "roc_all_classes.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined ROC curve -> {save_path}")


if __name__ == "__main__":
    print("Using device:", device)
    _,_,test_loader,_=load_data(batch_size=8)
    # Load models once
    resnet = get_resnet().to(device)
    resnet.load_state_dict(torch.load("checkpoints/resnet_best.pth", map_location=device))

    densenet = get_densenet().to(device)
    densenet.load_state_dict(torch.load("checkpoints/densenet_best.pth", map_location=device))

    effnet = get_effnet().to(device)
    effnet.load_state_dict(torch.load("checkpoints/effnet_best.pth", map_location=device))

    models = [resnet, densenet, effnet]
    preds, labels,probs,paths = ensemble_predict(models, test_loader)

        # Save classification report
    report = classification_report(
            labels, preds,
            target_names=classes,
            output_dict=True,
            zero_division=0)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"checkpoints/reports/classification_report_test.csv")
    print(classification_report(labels,preds,target_names=classes,zero_division=0))
    cm = confusion_matrix(labels, preds)
    save_combined_roc(labels, probs, classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(
        "checkpoints/confusion_matrices/confusion_matrix_test.csv"
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        "checkpoints/confusion_matrices/confusion_matrix_test.png"
    )
    plt.close()

    # ------------------------------
    # Save per-image predictions
    # ------------------------------
    results_df = pd.DataFrame({
        "image_path": paths,
        "true_label": [classes[i] for i in labels],
        "predicted_label": [classes[i] for i in preds]
    })

    results_df.to_csv(
        "checkpoints/reports/test_predictions.csv",
        index=False
    )
    acc = accuracy_score(labels, preds)
    print(f"\nFinal Test Accuracy: {acc:.4f}")

    print("\nAll test results saved successfully.")
