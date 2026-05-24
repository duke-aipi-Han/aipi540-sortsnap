from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# keep it simple on types supported
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# helper funcitons for handling images

def validate_uploaded_file(uploaded_file) -> None:
    if hasattr(uploaded_file, "type") and uploaded_file.type:
        if uploaded_file.type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError("Unsupported file type. Please upload a JPG, PNG, or WEBP image.")


def open_uploaded_image(uploaded_file) -> Image.Image:
    validate_uploaded_file(uploaded_file)
    try:
        image = Image.open(uploaded_file)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Could not open the uploaded image.") from exc


def resize_for_display(image: Image.Image, max_size: int = 800) -> Image.Image:
    display_image = image.copy()
    display_image.thumbnail((max_size, max_size))
    return display_image

# helper function to overlay recommendations on image.
def overlay_result_on_image(
    image: Image.Image,
    category: str,
    decision: str,
    confidence: float,
) -> Image.Image:
    annotated = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    category_label = category.replace("_", " ").title()
    lines = [
        f"{category_label} ({confidence:.0%})",
        decision,
    ]

    padding = 12
    line_gap = 6
    text_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max(box[2] - box[0] for box in text_boxes)
    text_height = sum(box[3] - box[1] for box in text_boxes) + line_gap * (len(lines) - 1)
    box_width = min(annotated.width - 20, text_width + padding * 2)
    box_height = text_height + padding * 2
    x0 = 10
    y0 = max(10, annotated.height - box_height - 10)
    x1 = x0 + box_width
    y1 = y0 + box_height

    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(0, 0, 0, 180))

    y = y0 + padding
    for line in lines:
        draw.text((x0 + padding, y), line, fill=(255, 255, 255, 255), font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap

    return annotated.convert("RGB")
