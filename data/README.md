# SortSnap Data

Not committed to GitHub.

## Selected Dataset

SortSnap uses the Kaggle Recyclable and Household Waste Classification dataset:

- URL: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification
- License: MIT
- Size: 15,000 PNG images, 30 item-level categories
- Structure: `images/<source_class>/{default,real_world}/*.png`

Download the Kaggle archive manually as `data/raw/archive.zip`. The prep script
unzips it if needed and converts it into SortSnap's application-level classes.
If the extracted folder already exists under `data/raw`, the script reuses it
automatically.

## Expected Folder Structure

Use a torchvision `ImageFolder` layout:

```text
data/
  raw/
  processed/
    train/
      cardboard/
      paper/
      plastic_bottle/
      plastic_wrap/
      metal_can/
      glass_bottle/
      food_waste/
      other_trash/
    val/
      cardboard/
      paper/
      plastic_bottle/
      plastic_wrap/
      metal_can/
      glass_bottle/
      food_waste/
      other_trash/
    test/
      cardboard/
      paper/
      plastic_bottle/
      plastic_wrap/
      metal_can/
      glass_bottle/
      food_waste/
      other_trash/
```

## Build the Dataset

Download the dataset zip from Kaggle and place it here:

```text
data/raw/archive.zip
```

Then run:

```bash
python training/prepare_kaggle_dataset.py --force
```

The default split mixes `default` and `real_world` images, shuffles with seed
`0`, and uses a fixed `70/10/20` train/val/test split.

For a more realistic PoC evaluation, train and validate on controlled
`default` images and reserve `real_world` images for test:

```bash
python training/prepare_kaggle_dataset.py --real_world_test --force
```

To ignore `real_world` images and split only controlled `default` images:

```bash
python training/prepare_kaggle_dataset.py --default_only --force
```

This writes:

```text
data/raw/
  archive.zip
  recyclable-and-household-waste-classification/
data/processed/
  dataset_metadata.json
  manifest.json
  train/
  val/
  test/
```

## Class Mapping

The 30 Kaggle item classes are mapped to SortSnap classes before training:

- `cardboard`: `cardboard_boxes`, `cardboard_packaging`
- `paper`: `magazines`, `newspaper`, `office_paper`
- `plastic_bottle`: `plastic_detergent_bottles`, `plastic_food_containers`, `plastic_soda_bottles`, `plastic_water_bottles`
- `plastic_wrap`: `plastic_cup_lids`, `plastic_shopping_bags`, `plastic_trash_bags`
- `metal_can`: `aerosol_cans`, `aluminum_food_cans`, `aluminum_soda_cans`, `steel_food_cans`
- `glass_bottle`: `glass_beverage_bottles`, `glass_cosmetic_containers`, `glass_food_jars`
- `food_waste`: `coffee_grounds`, `eggshells`, `food_waste`, `tea_bags`
- `other_trash`: `clothing`, `disposable_plastic_cutlery`, `paper_cups`, `plastic_straws`, `shoes`, `styrofoam_cups`, `styrofoam_food_containers`
