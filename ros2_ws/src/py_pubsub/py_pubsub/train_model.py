"""
Training Script for Gesture Classification.
Loads the gesture dataset, trains the CNN, evaluates performance,
and saves the trained model weights.

Usage:
    python3 train_model.py --data_dir ./gesture_data --epochs 20 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from .gesture_model import (
    GestureCNN, GestureDataset,
    train_transform, eval_transform,
    CLASS_NAMES
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train gesture classification model')
    parser.add_argument('--data_dir', type=str, default='./gesture_data',
                        help='Path to gesture dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output_model', type=str, default='./gesture_model.pth',
                        help='Path to save trained model')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load dataset ──────────────────────────────────────────────
    full_dataset = GestureDataset(args.data_dir, transform=train_transform)
    print(f"Total samples: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("Error: No images found. Run collect_gestures.py first.")
        return

    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override transform for validation set
    val_dataset.dataset = GestureDataset(args.data_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # ── Model, loss, optimizer ────────────────────────────────────
    model = GestureCNN().to(device)

    # CrossEntropyLoss is the standard choice for multi-class classification.
    # It combines LogSoftmax + NLLLoss, making it numerically stable.
    # Other options: FocalLoss (for imbalanced data), LabelSmoothingCE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── Training loop ─────────────────────────────────────────────
    best_val_acc = 0.0
    print("\n" + "=" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>9.4f} | {val_acc:>7.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_model)
            print(f"         → Saved best model (val_acc={val_acc:.1%})")

    # ── Final evaluation ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final Evaluation on Validation Set")
    print("=" * 60)

    model.load_state_dict(torch.load(args.output_model, weights_only=True))
    _, final_acc, all_preds, all_labels = evaluate(
        model, val_loader, criterion, device)

    print(f"\nBest Validation Accuracy: {best_val_acc:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print(f"\nModel saved to: {args.output_model}")


if __name__ == '__main__':
    main()
