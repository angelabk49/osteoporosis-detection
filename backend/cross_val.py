from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
from dataset import load_data, get_resnet, get_densenet, get_effnet, classes
from test import ensemble_predict
from train import train_model
def cross_validate_ensemble(
    models_fn,
    dataset,
    n_splits=3,
    batch_size=8
):
    """
    Performs k-fold CV on TRAINING DATA ONLY
    """
    X = np.arange(len(dataset))
    y = np.array(dataset.labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset   = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Fresh models per fold
        models = [fn() for fn in models_fn]

        # ---- TRAIN EACH MODEL ----
        for m, name in zip(models, ["resnet", "effnet", "densenet"]):
            train_model(m, name, train_loader, val_loader, y[train_idx])

        # ---- ENSEMBLE PREDICTION ----
        preds, labels, _, _ = ensemble_predict(models, val_loader)

        # ---- Classification Report ----
        report = classification_report(
            labels, preds, target_names=classes, zero_division=0
        )

        report_path = f"checkpoints/reports/fold_{fold}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # ---- Confusion Matrix ----
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=classes,
                    yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix – Fold {fold}")

        cm_path = f"checkpoints/confusion_matrices/fold_{fold}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved Fold {fold} results")

        fold_results.append({
            "fold": fold,
            "accuracy": np.mean(preds == labels)
        })

    return fold_results
train_loader, val_loader, test_loader, train_labels = load_data(batch_size=8)

train_dataset = train_loader.dataset  # only train images

results = cross_validate_ensemble(
    models_fn=[get_resnet, get_effnet, get_densenet],
    dataset=train_dataset,
    n_splits=3
)

print("\n===== 3-Fold CV Summary =====")
print(pd.DataFrame(results))
