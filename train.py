import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from safetensors.torch import save_file

from model import get_model, ViolenceDetector
from dataset import create_data_loaders


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


def train_epoch(model, train_loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (frames, labels) in enumerate(train_loader):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return running_loss / total, correct / total


def validate(model, val_loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * frames.size(0)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_model(model: ViolenceDetector, save_path: str, config: Dict) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    weights_path = save_path.with_suffix(".safetensors")
    save_file(model.state_dict(), str(weights_path))
    print(f"Model weights saved to: {weights_path}")

    config_path = save_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to: {config_path}")


def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    num_frames: int = 16,
    batch_size: int = 8,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 7,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
) -> Dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading dataset...")
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir, num_frames=num_frames, batch_size=batch_size,
        val_split=val_split, num_workers=num_workers, seed=seed
    )

    config = {
        "num_frames": num_frames,
        "lstm_hidden_size": lstm_hidden_size,
        "lstm_num_layers": lstm_num_layers,
        "dropout": dropout,
        "frame_size": 224
    }

    print("\nCreating model...")
    model = get_model(
        num_frames=num_frames, lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers, dropout=dropout, device=device
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "learning_rate": []}
    best_val_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Target accuracy: 85%+")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs} (lr: {current_lr:.2e})")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Time: {time.time() - epoch_start:.1f}s")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, str(output_path / "best_model"), config)
            print(f"  New best model! Val Acc: {val_acc:.4f}")

        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        if val_acc >= 0.85:
            print(f"\nTarget accuracy (85%) reached at epoch {epoch}!")

    print("-" * 60)
    print(f"\nTraining completed in {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    save_model(model, str(output_path / "final_model"), config)

    history_path = output_path / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Violence Detection Model")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        data_dir=args.data_dir, output_dir=args.output_dir, num_frames=args.num_frames,
        batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers, dropout=args.dropout, patience=args.patience,
        val_split=args.val_split, num_workers=args.workers, seed=args.seed
    )


if __name__ == "__main__":
    main()
