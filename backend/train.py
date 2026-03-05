"""
train.py
--------
Purpose:
- Train ResNet, EfficientNet, and DenseNet with class-balanced loss.
- Use patience-based early stopping and checkpointing.
- Save best and latest checkpoints for resuming.
- Perform ensemble predictions with soft voting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from dataset import load_data, get_resnet, get_effnet, get_densenet, classes
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50  # maximum epochs
PATIENCE = 5  # early stopping patience
TARGET_ACC = 90  # stop early if validation reaches this consistently


# -------------------- Evaluation function --------------------
def evaluate(model, loader):
    """Compute accuracy on a DataLoader"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


# -------------------- Training function --------------------
def train_model(model, name, train_loader, val_loader, train_labels, epochs=25):
    """
    Train a CNN model with:
    - Class-balanced CrossEntropyLoss with label smoothing
    - Gradient clipping
    - Checkpointing to resume training
    - Best validation accuracy tracking
    - Early stopping if accuracy consistently exceeds threshold
    """
    # Compute class weights
    labels_np = np.array(train_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_np),
        y=labels_np
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    # Optimizer with lower LR for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Learning rate scheduler (reduce LR if val acc plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Setup checkpointing
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_acc = 0
    patience_counter = 0  # Early stopping counter

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            total_loss += loss.item()

        # Evaluate on validation
        val_acc = evaluate(model, val_loader)
        print(f"{name} | Epoch {epoch+1} | Val Acc: {val_acc:.2f}% | Loss: {total_loss/len(train_loader):.4f}")

        # Step the scheduler
        scheduler.step(val_acc)

        # Save checkpoint to resume training
        ckpt_path = os.path.join(checkpoint_dir, f"{name}_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': val_acc
        }, ckpt_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(checkpoint_dir, f"{name}_best.pth")
            torch.save(model.state_dict(), best_path)
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1

        # Early stopping if accuracy consistently above 85% for 3 epochs
        if val_acc >= 85 and patience_counter >= 3:
            print(f"{name} reached consistently high accuracy (>85%), stopping early!")
            break

    print(f"{name} training complete. Best val accuracy: {best_acc:.2f}%")
    return model

# -------------------- Ensemble prediction --------------------
def ensemble_predict(models, loader):
    """
    Predict using an ensemble of models (soft voting)
    """
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.stack([torch.softmax(m(x), 1) for m in models]).mean(0)
            preds = probs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return np.array(all_preds), np.array(all_labels)


# -------------------- Main --------------------
if __name__ == "__main__":
    # Load dataset
    train_loader, val_loader, test_loader, train_labels = load_data(batch_size=8)

    print(f"Train images: {len(train_loader.dataset)}, Val images: {len(val_loader.dataset)}")

    # Train models
    resnet = train_model(get_resnet(), "resnet", train_loader, val_loader, train_labels)
    effnet = train_model(get_effnet(), "effnet", train_loader, val_loader, train_labels)
    densenet = train_model(get_densenet(), "densenet", train_loader, val_loader, train_labels)

    # Ensemble evaluation
    models = [resnet, effnet, densenet]
    preds, labels = ensemble_predict(models, val_loader)
    print("\nEnsemble Results:")
    print(classification_report(labels, preds, target_names=classes, zero_division=0))

    # Save ensemble models state dicts for later use
    ensemble_path = os.path.join("checkpoints", "ensemble_models.pth")
    torch.save({k: m.state_dict() for k, m in zip(["resnet", "effnet", "densenet"], models)}, ensemble_path)
    print(f"Ensemble models saved to {ensemble_path}")
