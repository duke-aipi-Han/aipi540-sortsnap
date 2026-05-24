from typing import Protocol

from PIL import Image

# define base class for image classifiers
# To support different classifiers based on the underlying model or approach used.
# Also for the mock/demo fallback in case the main classifier is not available.

class ImageClassifier(Protocol):
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        """Return top predictions for a PIL image."""
