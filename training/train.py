import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model import build_model
from training.device import print_device_summary, resolve_device
from training.transforms import get_minimal_train_transforms, get_train_transforms, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train SortSnap ResNet18 classifier.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Use auto to prefer CUDA when PyTorch can access it.",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail fast if CUDA is not available to PyTorch.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--augmentation",
        choices=["realistic", "minimal"],
        default="realistic",
        help="Use realistic augmentation for robustness experiments or minimal transforms as a baseline.",
    )
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
    device = resolve_device(args.device, require_cuda=args.require_cuda)
    print_device_summary(device)
    pin_memory = device.type == "cuda"

    train_transform = (
        get_train_transforms()
        if args.augmentation == "realistic"
        else get_minimal_train_transforms()
    )
    train_dataset = ImageFolder(args.data_dir / "train", transform=train_transform)
    val_dataset = ImageFolder(args.data_dir / "val", transform=get_val_transforms())

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
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    best_model_path = args.output_dir / "resnet18_waste_classifier.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    with open(args.output_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(train_dataset.classes, file, indent=2)

    print(f"best_val_accuracy={best_val_accuracy:.4f}")
    print(f"saved_model={best_model_path}")


if __name__ == "__main__":
    main()
