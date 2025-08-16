# src/train.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd

from model import SpectrogramResNet
from data_setup import create_logo_dataloaders


class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, patience=15, verbose=False, delta=0, path='best_model.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_step(model, dataloader, loss_fn, optimizer, device):
    """Performs one epoch of training."""
    model.train()
    train_loss, train_acc = 0, 0
    all_preds, all_labels = [], []

    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits, y.float())
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_probs = torch.sigmoid(y_logits)
        y_preds = torch.round(y_pred_probs)
        all_preds.extend(y_preds.cpu().detach().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = train_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, precision, recall, f1


def val_step(model, dataloader, loss_fn, device):
    """Performs one epoch of validation."""
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validation"):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y.float())
            val_loss += loss.item()

            y_pred_probs = torch.sigmoid(y_logits)
            y_preds = torch.round(y_pred_probs)
            all_preds.extend(y_preds.cpu().detach().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = val_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, precision, recall, f1


def train(config: dict, test_fan_id: str):
    """Main training function for a single LOGO fold"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Training Fold | Test Fan: {test_fan_id} | Device: {device} ---")

    # Create DataLoaders for the current fold
    train_loader, val_loader, _, class_to_idx = create_logo_dataloaders(
        data_dir=config["data_dir"],
        test_fan_id=test_fan_id,
        image_size=config["image_size"],
        batch_size=config["batch_size"]
    )
    
    # Instantiate model
    model = SpectrogramResNet(num_classes=1, dropout_rate=config["dropout_rate"]).to(device)
    
    # Dynamically calculate class weights for the current fold
    # Get the labels from the training subset
    train_labels = [train_loader.dataset.dataset.targets[i] for i in train_loader.dataset.indices]
    num_abnormal = sum(1 for label in train_labels if label == class_to_idx['abnormal'])
    num_normal = len(train_labels) - num_abnormal
    weight_for_abnormal = num_normal / num_abnormal

    print(f"[INFO] Abnormal weight for this fold: {weight_for_abnormal:.2f}")
    pos_weight_tensor = torch.tensor([weight_for_abnormal], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Use unique model path for each fold
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"best_model_test{test_fan_id}.pth"

    early_stopper = EarlyStopping(patience=config["patience"], verbose=True, path=model_path)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Create a dictionary to store training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': []
    }

    # Training loop
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc, _, _, train_f1 = train_step(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc, val_prec, val_recall, val_f1 = val_step(
            model, val_loader, loss_fn, device
        )
        
        print(f"\nEpoch: {epoch+1} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Precision: {val_prec:.4f} | Val Recall: {val_recall:.4f}")
        
        # Append results to history dictionary
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_recall)

        # Step the scheduler based on the validation loss
        scheduler.step(val_loss)

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Return the history
    return pd.DataFrame(history)


# Block for testing
if __name__ == '__main__':
    # Configuration dictionary
    CONFIG = {
        "data_dir": "../data/processed/fan",
        "image_size": (224, 224),
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "dropout_rate": 0.4,
        "patience": 15,
        "model_path": "models/best_fan_model.pth"
    }
    
    train(CONFIG)