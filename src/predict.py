import json
from pathlib import Path

import torch
from PIL import Image

from src.model import build_model
from src.preprocess import get_inference_transform


class WasteClassifierPredictor:
    def __init__(self, model: torch.nn.Module, class_names: list[str]):
        self.model = model
        self.class_names = class_names
        self.transform = get_inference_transform()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_artifacts(cls, model_path: Path, class_names_path: Path):
        if not model_path.exists() or not class_names_path.exists():
            raise FileNotFoundError("Model artifacts are missing.")

        with open(class_names_path, "r", encoding="utf-8") as file:
            class_names = json.load(file)

        if isinstance(class_names, dict):
            class_names = [class_names[str(index)] for index in range(len(class_names))]

        model = build_model(num_classes=len(class_names), freeze_backbone=False)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return cls(model=model, class_names=class_names)

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        k = min(top_k, len(self.class_names))
        scores, indices = torch.topk(probabilities, k=k)

        top_predictions = [
            {
                "category": self.class_names[index.item()],
                "confidence": float(score.item()),
            }
            for score, index in zip(scores, indices)
        ]

        return {
            "top_category": top_predictions[0]["category"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "mode": "trained_model",
        }
