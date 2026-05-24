---
title: AIPI540 Sortsnap
emoji: 📸
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Machine Vision app that detects if trash if recyclable
license: apache-2.0
---

# SortSnap

**See it. Classify it. Sort it.**
*Duke AIPI540 - Han Wang*

SortSnap is a mini-hackathon computer vision project. 

The app lets a user take/upload a photo of a household waste item, classifies the item or material, and then applies disposal rules to recommend recycle, trash, compost, or special.

As recycling rules vary by city, county, item condition, and collection program, SortSnap separates the system into two layers:

1. **Vision classifier:** predicts a visual/material category such as `cardboard`, `paper`, `plastic_bottle`, `metal_can`, or `food_waste`.
2. **Rules engine:** maps that category to a disposal recommendation using rules.

## Why Robustness Matters

For real household use, robustness matters more than raw benchmark accuracy. User photos may have poor lighting, cluttered backgrounds, motion blur, odd angles, partial objects, shadows, or confusing packaging. The project includes realistic augmentation and a evaluation path so we can show whether a model stays stable under messy real-world conditions.

## Project Architecture
This app has 2 components:
1) A Streamlit UI that can be run locally or on HuggingFace Spaces, and performs inference only.
2) A local trainer for the model for offline transfer learning.

The mock classifier and trained classifier implement the same simple interface: `.predict(image) -> dict`.

## Project Structure

```text
aipi540-sortsnap/
  README.md
  requirements.txt
  .gitignore
  app.py
  config/
    disposal_rules.json
  src/
    __init__.py
    classifier.py
    mock_model.py
    model.py
    preprocess.py
    predict.py
    rules.py
    image_utils.py
  training/
    train.py
    evaluate.py
    transforms.py
  data/
    README.md
  models/
    README.md
```

## Local Setup

Use Python 3.13
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app supports two classifiers:

- **Demo mode:** uses a mock classifier when model files are not present.
- **Trained model mode:** automatically uses `models/resnet18_waste_classifier.pth` and `models/class_names.json` when available.

For image input, the app supports both file upload and browser camera capture. On a mobile phone, the camera option can use the phone camera when the browser grants camera permission. 

After an image is uploaded or captured, SortSnap analyzes it. The predicted class and recommended action are then displayed on the photo.
Disposal recommendations are loaded from `config/disposal_rules.json`, which maps model classes to actions such as recycle, trash, compost, check local rules, or special drop-off.

## Model Artifacts

Expected local model files:

```text
models/resnet18_waste_classifier.pth
models/class_names.json
```

If these files are missing, SortSnap falls back to demo mode.

## Training

Data loaded as an ImageFolder-style directory:

```text
data/processed/
  train/
    cardboard/
    paper/
    plastic/
    metal/
    glass/
    trash/
  val/
    cardboard/
    paper/
    plastic/
    metal/
    glass/
    trash/
  test/
    cardboard/
    paper/
    plastic/
    metal/
    glass/
    trash/
```

Train a ResNet18 transfer-learning model:

```bash
python training/train.py --data_dir data/processed --epochs 5 --batch_size 32 --lr 0.001 --freeze_backbone
```

Prefer local CUDA training and fail fast if PyTorch cannot access the GPU:

```bash
python training/train.py --data_dir data/processed --epochs 5 --batch_size 32 --lr 0.001 --freeze_backbone --device cuda --require_cuda
```

Evaluate clean and stress-test performance:

```bash
python training/evaluate.py --data_dir data/processed --split test
```

Check PyTorch CUDA access:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

The training script saves:

```text
models/resnet18_waste_classifier.pth
models/class_names.json
```

The evaluation script writes summary artifacts to `outputs/`.
