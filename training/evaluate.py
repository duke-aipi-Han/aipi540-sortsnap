import argparse
import json
import sys
import time
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


DATA_DIR = Path("data/processed")
SPLIT = "test"
MODEL_PATH = Path("models/resnet18_waste_classifier.pth")
CLASS_NAMES_PATH = Path("models/class_names.json")
OUTPUT_DIR = Path("outputs")


def log_line(message: str, log_lines: list[str]):
    print(message)
    log_lines.append(message)


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SortSnap classifier.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


@torch.inference_mode()
def collect_predictions(model, dataloader, device):
    y_true = []
    y_pred = []
    inference_seconds = 0.0
    image_count = 0
    model.eval()

    for images, labels in dataloader:
        images = images.to(device)
        synchronize_if_cuda(device)
        batch_start_time = time.perf_counter()
        logits = model(images)
        synchronize_if_cuda(device)
        inference_seconds += time.perf_counter() - batch_start_time

        predictions = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(predictions)
        y_true.extend(labels.tolist())
        image_count += images.size(0)

    return y_true, y_pred, inference_seconds, image_count


def evaluate_dataset(model, data_path, transform, batch_size, device, num_workers):
    dataset = ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    y_true, y_pred, inference_seconds, image_count = collect_predictions(
        model, dataloader, device
    )
    average_inference_seconds_per_image = inference_seconds / max(image_count, 1)
    return (
        dataset,
        y_true,
        y_pred,
        accuracy_score(y_true, y_pred),
        inference_seconds,
        average_inference_seconds_per_image,
    )


def main():
    args = parse_args()
    log_lines = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto", require_cuda=False)
    print_device_summary(device)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = build_model(num_classes=len(class_names), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    data_path = DATA_DIR / SPLIT
    (
        clean_dataset,
        clean_true,
        clean_pred,
        clean_accuracy,
        clean_inference_seconds,
        clean_average_inference_seconds_per_image,
    ) = evaluate_dataset(
        model, data_path, get_val_transforms(), args.batch_size, device, args.num_workers
    )
    (
        _,
        stress_true,
        stress_pred,
        stress_accuracy,
        stress_inference_seconds,
        stress_average_inference_seconds_per_image,
    ) = evaluate_dataset(
        model, data_path, get_stress_test_transforms(), args.batch_size, device, args.num_workers
    )

    report = classification_report(
        clean_true,
        clean_pred,
        target_names=clean_dataset.classes,
        zero_division=0,
    )
    log_line(f"clean_accuracy={clean_accuracy:.4f}", log_lines)
    log_line(f"stress_test_accuracy={stress_accuracy:.4f}", log_lines)
    log_line(
        "clean_average_inference_seconds_per_image="
        f"{clean_average_inference_seconds_per_image:.6f}",
        log_lines,
    )
    log_line(
        "stress_average_inference_seconds_per_image="
        f"{stress_average_inference_seconds_per_image:.6f}",
        log_lines,
    )
    log_line(report, log_lines)

    ConfusionMatrixDisplay.from_predictions(
        clean_true,
        clean_pred,
        display_labels=clean_dataset.classes,
        xticks_rotation=45,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    results = {
        "split": SPLIT,
        "clean_accuracy": clean_accuracy,
        "stress_test_accuracy": stress_accuracy,
        "clean_inference_seconds": clean_inference_seconds,
        "clean_average_inference_seconds_per_image": clean_average_inference_seconds_per_image,
        "stress_inference_seconds": stress_inference_seconds,
        "stress_average_inference_seconds_per_image": stress_average_inference_seconds_per_image,
        "classification_report": report,
        "robustness_table": [
            {
                "model_variant": "Current trained model",
                "clean_accuracy": clean_accuracy,
                "stress_test_accuracy": stress_accuracy,
                "clean_average_inference_seconds_per_image": clean_average_inference_seconds_per_image,
                "stress_average_inference_seconds_per_image": stress_average_inference_seconds_per_image,
            }
        ],
    }
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with open(OUTPUT_DIR / "results.md", "w", encoding="utf-8") as file:
        file.write("# SortSnap Evaluation Results\n\n")
        file.write(
            "| Model Variant | Clean Accuracy | Stress-Test Accuracy | "
            "Clean Inference/Image | Stress Inference/Image |\n"
        )
        file.write("| --- | ---: | ---: | ---: | ---: |\n")
        file.write(
            f"| Current trained model | {clean_accuracy:.2%} | "
            f"{stress_accuracy:.2%} | "
            f"{clean_average_inference_seconds_per_image:.6f}s | "
            f"{stress_average_inference_seconds_per_image:.6f}s |\n"
        )

    with open(OUTPUT_DIR / "evaluation_output.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(log_lines))
        file.write("\n")


if __name__ == "__main__":
    main()
