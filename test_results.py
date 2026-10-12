import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from utils.config import SCAN_MODEL_PATH
from preprocessing.midpoint_filter import midpoint_filter
from preprocessing.bse import brain_surface_extractor
from preprocessing.clahe import apply_clahe
from segmentation.predict_scan import (
    load_scan_model,
    predict_scan,
    create_soft_attention,
)
from feature_extraction.extract_features import extract_features
from classification.predict_dhkfhnet import load_dhkfhnet



# SETTINGS

DATASET_DIR = Path("data/br35h")

TEST_PER_CLASS = 225
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}



# REPRODUCIBLE TEST SET


random.seed(SEED)

no_images = sorted([
    p for p in (DATASET_DIR / "no").iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
])

yes_images = sorted([
    p for p in (DATASET_DIR / "yes").iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
])

if len(no_images) < TEST_PER_CLASS:
    raise RuntimeError("Not enough No Tumor images.")

if len(yes_images) < TEST_PER_CLASS:
    raise RuntimeError("Not enough Tumor images.")

test_no = random.sample(no_images, TEST_PER_CLASS)
test_yes = random.sample(yes_images, TEST_PER_CLASS)

test_images = [(p, 0) for p in test_no] + [(p, 1) for p in test_yes]

random.shuffle(test_images)



# LOAD MODELS


print()
print("=" * 60)
print("LOADING TRAINED MODELS")
print("=" * 60)

scan_model, scan_device = load_scan_model(SCAN_MODEL_PATH)
dhkfhnet_model, dhkfhnet_device = load_dhkfhnet()

print("SCAN model loaded.")
print("DHKFHNet model loaded.")



# EVALUATION


y_true = []
y_pred = []
y_probability = []

total = len(test_images)

print()
print("=" * 60)
print("TESTING")
print("=" * 60)

for index, (image_path, true_class) in enumerate(test_images, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    # SAME PIPELINE USED BY THE FINAL MODEL
    
    midpoint = midpoint_filter(image)

    bse, _ = brain_surface_extractor(midpoint)

    clahe = apply_clahe(bse)

    scan_probability = predict_scan(
        scan_model,
        clahe
    )

    attention = create_soft_attention(
        scan_probability
    )

    enhanced = (
        clahe.astype(np.float32)
        * attention[..., None]
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    features = np.asarray(
        extract_features(enhanced),
        dtype=np.float32
    )

    
    # DHKFHNet
    

    gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        (224, 224)
    )

    image_tensor = (
        torch.tensor(
            gray,
            dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    ).to(dhkfhnet_device)

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(dhkfhnet_device)

    with torch.no_grad():

        output = dhkfhnet_model(
            image_tensor,
            feature_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    probability_tumor = (
        probabilities[0, 1]
        .item()
    )

    predicted_class = int(
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )

    y_true.append(true_class)
    y_pred.append(predicted_class)
    y_probability.append(probability_tumor)

    if index % 25 == 0 or index == total:
        print(
            f"Processed: {index}/{total}"
        )



# METRICS

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_probability
)

cm = confusion_matrix(
    y_true,
    y_pred
)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "No Tumor",
        "Tumor"
    ],
    zero_division=0
)



# FINAL REPORT

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test samples: {len(y_true)}")
print()
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print()
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)