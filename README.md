```markdown
# Brain Tumor Detection Using MRI

## Project Title

Hybrid deep learning network enabled brain tumor detection
with shape-aware loss-based structure correcting adversarial
network using MRI.

## Project Pipeline

MRI Image
    ↓
Midpoint Filter
    ↓
BSE
    ↓
CLAHE
    ↓
SCAN
    ↓
Tumor Region
    ↓
LARK + GLCM + GDP
    ↓
DHKFHNet
    ↓
Tumor Prediction

## Input

The application accepts one MRI image.

Supported formats:

- JPG
- JPEG
- PNG
- BMP
- TIFF

## Preprocessing

The MRI image passes through:

1. Midpoint Filter
2. Brain Surface Extraction (BSE)
3. CLAHE

CLAHE is the additional preprocessing stage
introduced in this project.

## Segmentation

SCAN is used to generate the tumor-region
segmentation mask.

## Feature Extraction

Three feature descriptors are used:

- LARK
- GLCM
- GDP

The extracted features are combined into a
13-dimensional feature vector.

## Classification

The classifier uses:

- Segmented MRI image features
- LARK features
- GLCM features
- GDP features

These are combined and passed to DHKFHNet
for binary classification.

Classes:

0 - No Tumor
1 - Tumor

## Model Training

Deep-learning models are trained using
Google Colab.

The trained models are stored locally as:

models/
├── scan/
│   └── scan_model.pth
│
└── dhkfhnet/
    └── dhkfhnet_model.pth

## Website

The web application is developed using
Streamlit.

Run the application with:

streamlit run app.py

## Important

The SCAN and DHKFHNet architectures used for
training must be identical to the architectures
defined in this project.

The trained model files must exist before
running prediction.
```
