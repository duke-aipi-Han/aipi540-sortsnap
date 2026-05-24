import random

from PIL import Image

# mock classifier for demo purposes; Just randomly determine the category.

CATEGORIES = [
    "cardboard",
    "paper",
    "plastic_bottle",
    "plastic_wrap",
    "metal_can",
    "glass_bottle",
    "food_waste",
    "other_trash",
]


class MockWasteClassifier:
    """Drop-in classifier used when trained model artifacts are unavailable."""

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        del image

        selected = random.sample(CATEGORIES, k=min(top_k, len(CATEGORIES)))
        top_score = random.uniform(0.62, 0.93)
        remaining_score = max(0.01, 1.0 - top_score)
        trailing_scores = []

        if len(selected) > 1:
            for index in range(1, len(selected)):
                if index == len(selected) - 1:
                    trailing_scores.append(remaining_score)
                else:
                    score = random.uniform(0.01, remaining_score)
                    trailing_scores.append(score)
                    remaining_score = max(0.01, remaining_score - score)

        scores = [top_score, *trailing_scores]
        top_predictions = [
            {"category": category, "confidence": score}
            for category, score in zip(selected, scores)
        ]

        return {
            "top_category": top_predictions[0]["category"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "mode": "demo",
        }
