# SortSnap Models

This folder is for small model artifacts used by the deployed Streamlit app.

Expected files after training:

```text
models/resnet18_waste_classifier.pth
models/class_names.json
```

For this hack, assume the trained model is small enough to keep with the deployed app. 
The app checks this folder for local artifacts and falls back to demo mode when they are missing.
