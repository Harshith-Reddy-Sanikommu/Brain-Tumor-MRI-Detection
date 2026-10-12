import sys
from pathlib import Path
import json
import random


# PROJECT PATH


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))



# IMPORTS


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.midpoint_filter import midpoint_filter
from preprocessing.bse import brain_surface_extractor
from preprocessing.clahe import apply_clahe

from segmentation.scan_model import SCAN
from segmentation.shape_aware_loss import ShapeAwareLoss



# PATHS


TRAIN_DIR = (
    PROJECT_DIR
    / "data"
    / "br35h"
    / "Br35H-Mask-RCNN"
    / "TRAIN"
)

VAL_DIR = (
    PROJECT_DIR
    / "data"
    / "br35h"
    / "Br35H-Mask-RCNN"
    / "VAL"
)

TRAIN_JSON = TRAIN_DIR / "annotations_train.json"
VAL_JSON = VAL_DIR / "annotations_val.json"

MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "scan"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "scan_model.pth"



# SETTINGS


IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 60

LEARNING_RATE = 0.0003

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# REPRODUCIBILITY


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)



# DATASET


class BR35HScanDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file
    ):

        self.image_dir = Path(image_dir)

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as file:

            self.data = json.load(file)

        self.records = []

        for _, record in self.data.items():

            if not isinstance(record, dict):
                continue

            filename = record.get("filename")
            regions = record.get("regions", [])

            if filename is None:
                continue

            image_path = self.image_dir / filename

            if not image_path.exists():
                continue

            self.records.append(
                {
                    "filename": filename,
                    "regions": regions
                }
            )

        print(
            f"Loaded {len(self.records)} valid images "
            f"from {annotation_file.name}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = (
            self.image_dir
            / record["filename"]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_height = image.shape[0]
        original_width = image.shape[1]

        
        # MRI → MIDPOINT → BSE → CLAHE
        

        midpoint_image = midpoint_filter(image)

        bse_image, _ = brain_surface_extractor(
            midpoint_image
        )

        clahe_image = apply_clahe(
            bse_image
        )

        
        # RESIZE
        

        clahe_image = cv2.resize(
            clahe_image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        
        # IMAGE TENSOR
        

        image_array = (
            clahe_image.astype(np.float32)
            / 255.0
        )

        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        image_tensor = torch.tensor(
            image_array,
            dtype=torch.float32
        )

        
        # CREATE MASK
       

        mask = np.zeros(
            (
                original_height,
                original_width
            ),
            dtype=np.uint8
        )

        for region in record["regions"]:

            if not isinstance(region, dict):
                continue

            shape_attributes = region.get(
                "shape_attributes",
                {}
            )

            if not isinstance(
                shape_attributes,
                dict
            ):
                continue

            if shape_attributes.get("name") != "polygon":
                continue

            all_points_x = shape_attributes.get(
                "all_points_x",
                []
            )

            all_points_y = shape_attributes.get(
                "all_points_y",
                []
            )

            if (
                len(all_points_x) < 3
                or
                len(all_points_y) < 3
            ):
                continue

            points = np.column_stack(
                (
                    all_points_x,
                    all_points_y
                )
            ).astype(np.int32)

            cv2.fillPoly(
                mask,
                [points],
                255
            )

       
        # RESIZE MASK
        

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask.astype(np.float32)
            / 255.0
        )

        mask_tensor = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image_tensor, mask_tensor



# CREATE DATASETS


print("=" * 60)
print("SCAN TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Training directory:", TRAIN_DIR)
print("Validation directory:", VAL_DIR)

train_dataset = BR35HScanDataset(
    TRAIN_DIR,
    TRAIN_JSON
)

val_dataset = BR35HScanDataset(
    VAL_DIR,
    VAL_JSON
)

print()
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))



# DATA LOADERS


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)



# MODEL


model = SCAN().to(DEVICE)

criterion = ShapeAwareLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



# TRAINING


best_val_loss = float("inf")
best_epoch = 0

print()
print("=" * 60)
print("STARTING SCAN TRAINING")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):


    # TRAIN


    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # VALIDATION


    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # DISPLAY
   

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


    # SAVE BEST MODEL


    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best SCAN model saved "
            f"(epoch {epoch})"
        )



# COMPLETE


print()
print("=" * 60)
print("SCAN TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Model saved at:"
)

print(MODEL_PATH)

print("=" * 60)