import argparse
import json
import sys
import time
from pathlib import Path

# training script. To use:
# python train.py --epochs 15 --batch_size 32 --lr 0.0002

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model import build_model
from training.device import print_device_summary, resolve_device
from training.transforms import get_train_transforms, get_val_transforms


DATA_DIR = Path("data/processed")
MODEL_OUTPUT_DIR = Path("models")
RUN_OUTPUT_DIR = Path("outputs")
FREEZE_BACKBONE = False


def log_line(message: str, log_lines: list[str]):
    print(message)
    log_lines.append(message)


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_accuracy_curve(history: list[dict], output_path: Path):
    epochs = [row["epoch"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    val_accuracy = [row["val_accuracy"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracy, marker="o", label="Train accuracy")
    plt.plot(epochs, val_accuracy, marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SortSnap ResNet18 classifier.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def run_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    log_lines = []
    device = resolve_device("auto", require_cuda=False)
    print_device_summary(device)
    pin_memory = device.type == "cuda"

    train_dataset = ImageFolder(DATA_DIR / "train", transform=get_train_transforms())
    val_dataset = ImageFolder(DATA_DIR / "val", transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        num_classes=len(train_dataset.classes),
        freeze_backbone=FREEZE_BACKBONE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    best_model_path = MODEL_OUTPUT_DIR / "resnet18_waste_classifier.pth"
    history = []
    epoch_durations = []

    for epoch in range(1, args.epochs + 1):
        synchronize_if_cuda(device)
        epoch_start_time = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        synchronize_if_cuda(device)
        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_duration_seconds)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_duration_seconds": epoch_duration_seconds,
            }
        )

        log_line(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"epoch_seconds={epoch_duration_seconds:.2f}",
            log_lines,
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    with open(MODEL_OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(train_dataset.classes, file, indent=2)

    average_epoch_seconds = sum(epoch_durations) / max(len(epoch_durations), 1)
    total_training_seconds = sum(epoch_durations)
    log_line(f"best_val_accuracy={best_val_accuracy:.4f}", log_lines)
    log_line(f"average_epoch_seconds={average_epoch_seconds:.2f}", log_lines)
    log_line(f"total_training_seconds={total_training_seconds:.2f}", log_lines)
    log_line(f"saved_model={best_model_path}", log_lines)

    training_results = {
        "args": {
            "data_dir": str(DATA_DIR),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "freeze_backbone": FREEZE_BACKBONE,
            "device": "auto",
            "require_cuda": False,
            "num_workers": args.num_workers,
            "augmentation": "realistic",
        },
        "classes": train_dataset.classes,
        "best_val_accuracy": best_val_accuracy,
        "average_epoch_seconds": average_epoch_seconds,
        "total_training_seconds": total_training_seconds,
        "history": history,
    }
    with open(RUN_OUTPUT_DIR / "training_results.json", "w", encoding="utf-8") as file:
        json.dump(training_results, file, indent=2)

    with open(RUN_OUTPUT_DIR / "training_output.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(log_lines))
        file.write("\n")

    save_accuracy_curve(history, RUN_OUTPUT_DIR / "training_accuracy_curve.png")


if __name__ == "__main__":
    main()
