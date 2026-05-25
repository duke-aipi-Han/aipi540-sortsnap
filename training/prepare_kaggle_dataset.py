import argparse
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# Data preparation for the Kaggle household waste dataset from: 
# https://www.kaggle.com/datasets/mostafaabla/garbage-classification

# This script expects the dataset already in data/raw, and will extract and most importantly map the folders into something we can use for our simplfied classification.

DATASET_SLUG = "alistairking/recyclable-and-household-waste-classification"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
SEED = 0

CLASS_MAPPING = {
    "aerosol_cans": "metal_can",
    "aluminum_food_cans": "metal_can",
    "aluminum_soda_cans": "metal_can",
    "cardboard_boxes": "cardboard",
    "cardboard_packaging": "cardboard",
    "clothing": "other_trash",
    "coffee_grounds": "food_waste",
    "disposable_plastic_cutlery": "other_trash",
    "eggshells": "food_waste",
    "food_waste": "food_waste",
    "glass_beverage_bottles": "glass_bottle",
    "glass_cosmetic_containers": "glass_bottle",
    "glass_food_jars": "glass_bottle",
    "magazines": "paper",
    "newspaper": "paper",
    "office_paper": "paper",
    "paper_cups": "other_trash",
    "plastic_cup_lids": "plastic_wrap",
    "plastic_detergent_bottles": "plastic_bottle",
    "plastic_food_containers": "plastic_bottle",
    "plastic_shopping_bags": "plastic_wrap",
    "plastic_soda_bottles": "plastic_bottle",
    "plastic_straws": "other_trash",
    "plastic_trash_bags": "plastic_wrap",
    "plastic_water_bottles": "plastic_bottle",
    "shoes": "other_trash",
    "steel_food_cans": "metal_can",
    "styrofoam_cups": "other_trash",
    "styrofoam_food_containers": "other_trash",
    "tea_bags": "food_waste",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the Kaggle household waste dataset from data/raw/archive.zip."
    )
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--default_only",
        action="store_true",
        help="Use only Kaggle default images for a random train/val/test split.",
    )
    parser.add_argument(
        "--real_world_test",
        action="store_true",
        help="Use default images for train/val and real_world images for test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace output_dir if it already exists.",
    )
    return parser.parse_args()


def resolve_split_strategy(args):
    if args.default_only and args.real_world_test:
        raise RuntimeError("Use only one of --default_only or --real_world_test.")
    if args.real_world_test:
        return "real_world_test"
    if args.default_only:
        return "default_only"
    return "mixed"


def get_archive_path(raw_dir: Path):
    archive_path = raw_dir / "archive.zip"
    if not archive_path.exists():
        raise RuntimeError(
            f"Expected manually downloaded Kaggle archive at {archive_path}."
        )
    return archive_path


def extract_dataset(archive_path: Path, raw_dir: Path):
    extract_dir = raw_dir / "recyclable-and-household-waste-classification"
    images_dir = find_dataset_images_dir(extract_dir)
    if images_dir is not None:
        return images_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    images_dir = find_dataset_images_dir(extract_dir)
    if images_dir is None:
        raise RuntimeError(f"Could not find an images directory under {extract_dir}.")
    return images_dir


def find_dataset_images_dir(extract_dir: Path):
    if not extract_dir.exists():
        return None

    required_source_classes = set(CLASS_MAPPING)
    candidates = [extract_dir, *extract_dir.rglob("images")]
    for candidate in candidates:
        if not candidate.is_dir():
            continue

        child_names = {path.name for path in candidate.iterdir() if path.is_dir()}
        if required_source_classes.issubset(child_names):
            return candidate

    return None


def collect_images(images_dir: Path):
    grouped = defaultdict(list)
    for source_class, target_class in CLASS_MAPPING.items():
        source_dir = images_dir / source_class
        if not source_dir.exists():
            raise RuntimeError(f"Expected source class folder is missing: {source_dir}")

        for image_path in source_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                source_variant = image_path.parent.name
                grouped[target_class].append((source_class, source_variant, image_path))

    return grouped


def split_items(items, train_ratio: float, val_ratio: float, rng: random.Random):
    shuffled = list(items)
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def split_real_world_test(items, rng: random.Random):
    default_items = [item for item in items if item[1] == "default"]
    real_world_items = [item for item in items if item[1] == "real_world"]
    if not default_items:
        raise RuntimeError("No default images found for a target class.")
    if not real_world_items:
        raise RuntimeError("No real_world images found for a target class.")

    rng.shuffle(default_items)
    train_end = int(len(default_items) * (TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO)))
    return {
        "train": default_items[:train_end],
        "val": default_items[train_end:],
        "test": real_world_items,
    }


def split_default_only(items, rng: random.Random):
    default_items = [item for item in items if item[1] == "default"]
    if not default_items:
        raise RuntimeError("No default images found for a target class.")
    return split_items(default_items, TRAIN_RATIO, VAL_RATIO, rng)


def split_for_strategy(items, strategy: str, rng: random.Random):
    if strategy == "real_world_test":
        return split_real_world_test(items, rng)
    if strategy == "default_only":
        return split_default_only(items, rng)
    return split_items(items, TRAIN_RATIO, VAL_RATIO, rng)


def prepare_output_dir(output_dir: Path, force: bool):
    if output_dir.exists():
        if not force:
            raise RuntimeError(
                f"{output_dir} already exists. Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_split(output_dir: Path, grouped, strategy: str):
    rng = random.Random(SEED)
    manifest = []
    summary = {split: Counter() for split in ("train", "val", "test")}

    for target_class, items in sorted(grouped.items()):
        splits = split_for_strategy(items, strategy, rng)
        for split, split_items_for_class in splits.items():
            target_dir = output_dir / split / target_class
            target_dir.mkdir(parents=True, exist_ok=True)

            for source_class, source_variant, source_path in split_items_for_class:
                destination_name = f"{source_class}__{source_path.name}"
                destination_path = target_dir / destination_name
                shutil.copy2(source_path, destination_path)
                summary[split][target_class] += 1
                manifest.append(
                    {
                        "split": split,
                        "target_class": target_class,
                        "source_class": source_class,
                        "source_variant": source_variant,
                        "source_path": str(source_path),
                        "destination_path": str(destination_path),
                    }
                )

    return manifest, summary


def write_metadata(output_dir: Path, manifest, summary, split_strategy: str):
    metadata = {
        "dataset": DATASET_SLUG,
        "dataset_url": f"https://www.kaggle.com/datasets/{DATASET_SLUG}",
        "license": "MIT",
        "split_strategy": split_strategy,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "seed": SEED,
        "class_mapping": CLASS_MAPPING,
        "summary": {
            split: dict(sorted(counter.items())) for split, counter in summary.items()
        },
    }
    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def main():
    args = parse_args()
    split_strategy = resolve_split_strategy(args)

    archive_path = get_archive_path(args.raw_dir)
    images_dir = extract_dataset(archive_path, args.raw_dir)

    grouped = collect_images(images_dir)
    prepare_output_dir(args.output_dir, args.force)
    manifest, summary = copy_split(
        output_dir=args.output_dir,
        grouped=grouped,
        strategy=split_strategy,
    )
    write_metadata(args.output_dir, manifest, summary, split_strategy)

    print(f"source_images_dir={images_dir}")
    print(f"processed_dir={args.output_dir}")
    print(f"split_strategy={split_strategy}")
    for split, counts in summary.items():
        print(f"{split}={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
