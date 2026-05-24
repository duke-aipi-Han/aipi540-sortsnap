import argparse
import json
import sys
from pathlib import Path

# Evaluation script for the SortSnap classifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model import build_model
from training.device import print_device_summary, resolve_device
from training.transforms import get_stress_test_transforms, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SortSnap classifier.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_path", type=Path, default=Path("models/resnet18_waste_classifier.pth"))
    parser.add_argument("--class_names_path", type=Path, default=Path("models/class_names.json"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
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
    return parser.parse_args()


@torch.inference_mode()
def collect_predictions(model, dataloader, device):
    y_true = []
    y_pred = []
    model.eval()

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(predictions)
        y_true.extend(labels.tolist())

    return y_true, y_pred


def evaluate_dataset(model, data_path, transform, batch_size, device, num_workers):
    dataset = ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    y_true, y_pred = collect_predictions(model, dataloader, device)
    return dataset, y_true, y_pred, accuracy_score(y_true, y_pred)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device, require_cuda=args.require_cuda)
    print_device_summary(device)

    with open(args.class_names_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = build_model(num_classes=len(class_names), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    data_path = args.data_dir / args.split
    clean_dataset, clean_true, clean_pred, clean_accuracy = evaluate_dataset(
        model, data_path, get_val_transforms(), args.batch_size, device, args.num_workers
    )
    _, stress_true, stress_pred, stress_accuracy = evaluate_dataset(
        model, data_path, get_stress_test_transforms(), args.batch_size, device, args.num_workers
    )

    report = classification_report(
        clean_true,
        clean_pred,
        target_names=clean_dataset.classes,
        zero_division=0,
    )
    print(f"clean_accuracy={clean_accuracy:.4f}")
    print(f"stress_test_accuracy={stress_accuracy:.4f}")
    print(report)

    ConfusionMatrixDisplay.from_predictions(
        clean_true,
        clean_pred,
        display_labels=clean_dataset.classes,
        xticks_rotation=45,
    )
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    results = {
        "split": args.split,
        "clean_accuracy": clean_accuracy,
        "stress_test_accuracy": stress_accuracy,
        "classification_report": report,
        "robustness_table": [
            {
                "model_variant": "Current trained model",
                "clean_accuracy": clean_accuracy,
                "stress_test_accuracy": stress_accuracy,
            }
        ],
    }
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with open(args.output_dir / "results.md", "w", encoding="utf-8") as file:
        file.write("# SortSnap Evaluation Results\n\n")
        file.write("| Model Variant | Clean Accuracy | Stress-Test Accuracy |\n")
        file.write("| --- | ---: | ---: |\n")
        file.write(
            f"| Current trained model | {clean_accuracy:.2%} | {stress_accuracy:.2%} |\n"
        )


if __name__ == "__main__":
    main()
