from io import BytesIO
from pathlib import Path

import streamlit as st

from src.classifier import ImageClassifier
from src.image_utils import open_uploaded_image, overlay_result_on_image, resize_for_display
from src.mock_model import MockWasteClassifier
from src.rules import get_disposal_recommendation

# Trained model artifacts expected at:
MODEL_PATH = Path("models/resnet18_waste_classifier.pth")
CLASS_NAMES_PATH = Path("models/class_names.json")

# Load the trained predictor from the model artifacts.
@st.cache_resource(show_spinner=False)
def load_trained_predictor():
    from src.predict import WasteClassifierPredictor

    return WasteClassifierPredictor.from_artifacts(
        model_path=MODEL_PATH,
        class_names_path=CLASS_NAMES_PATH,
    )


# Check if the model artifacts are available.
def model_artifacts_available() -> bool:
    return MODEL_PATH.exists() and CLASS_NAMES_PATH.exists()


# Load the mock predictor for demo purposes.
@st.cache_resource(show_spinner=False)
def load_mock_predictor():
    return MockWasteClassifier()


# Load the appropriate classifier based on whether the trained model artifacts are available.
def load_classifier() -> ImageClassifier:
    if model_artifacts_available():
        try:
            return load_trained_predictor()
        except Exception as exc:
            st.warning(
                "The trained model could not be loaded, so SortSnap used demo mode. "
                f"Details: {exc}"
            )

    return load_mock_predictor()

# take provided photo and run the classification.
def analyze_image(image):
    with st.spinner("Analyzing image..."):
        classifier = load_classifier()
        prediction = classifier.predict(image)
        recommendation = get_disposal_recommendation(prediction["top_category"])

    display_image = resize_for_display(image)
    annotated_image = overlay_result_on_image(
        display_image,
        category=prediction["top_category"],
        decision=recommendation["decision"],
        confidence=prediction["confidence"],
    )
    return annotated_image, prediction, recommendation

# streamlit run app.py to run locally
# minimal streamlit app that allows users to upload/take a photo, and then will process it, classify it, and provide disposal recommendations.
def main() -> None:
    st.set_page_config(page_title="SortSnap", page_icon="SS", layout="centered")

    st.title("SortSnap")
    st.caption("See it. Classify it. Sort it.")

    st.write(
        "SortSnap helps you decide how to dispose of your household waste."
    )

    if model_artifacts_available():
        st.caption("Using trained model artifacts from the local models folder.")
    else:
        st.caption("Demo mode: trained model artifacts are not present.")

    image_source = st.radio(
        "Image source",
        ["Upload image", "Take photo"],
        horizontal=True,
    )

    image_file = None
    image_caption = "Uploaded image"

    if image_source == "Upload image":
        st.session_state.pop("captured_photo_bytes", None)
        image_file = st.file_uploader(
            "Upload a photo of one household waste item",
            type=["jpg", "jpeg", "png", "webp"],
        )
    else:
        image_caption = "Captured image"
        captured_photo = st.session_state.get("captured_photo_bytes")

        if captured_photo is None:
            camera_file = st.camera_input(
                "Take a photo of one household waste item",
                help=(
                    "On a mobile browser, this uses the phone camera when camera "
                    "permissions are allowed. HTTPS is usually required after deployment."
                ),
            )
            if camera_file is not None:
                st.session_state["captured_photo_bytes"] = camera_file.getvalue()
                st.rerun()
        else:
            image_file = BytesIO(captured_photo)

    if image_file is None:
        st.info("Upload an image or take a photo to classify a household waste item.")
        return

    try:
        image = open_uploaded_image(image_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    annotated_image, prediction, recommendation = analyze_image(image)
    st.image(annotated_image, caption=image_caption)

    if image_source == "Take photo" and st.button("Retake photo"):
        st.session_state.pop("captured_photo_bytes", None)
        st.rerun()

    st.caption(
        f"{recommendation['explanation']} {recommendation['local_rules_note']} "
        f"Mode: {prediction['mode'].replace('_', ' ')}."
    )


if __name__ == "__main__":
    main()
