import sys
from pathlib import Path
import random


# PROJECT PATH


PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))



# IMPORTS


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from classification.dhkfhnet import DHKFHNet
from utils.config import DHKFHNET_MODEL_PATH



# PATHS


DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "br35h"
)

TRAIN_CACHE = (
    DATASET_DIR
    / "dhkfhnet_softscan_train_cache.npz"
)

VAL_CACHE = (
    DATASET_DIR
    / "dhkfhnet_softscan_val_cache.npz"
)

TEST_CACHE = (
    DATASET_DIR
    / "dhkfhnet_softscan_test_cache.npz"
)



# SETTINGS


IMAGE_SIZE = 224

FEATURE_SIZE = 13

NUM_CLASSES = 2

BATCH_SIZE = 32

EPOCHS = 40

LEARNING_RATE = 0.0002

WEIGHT_DECAY = 0.0001

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 8

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 3



# REPRODUCIBILITY


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)



# DEVICE


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# LOAD CACHE


def load_cache(path):

    print()
    print("Loading:")
    print(path)

    data = np.load(path)

    images = data["images"]
    features = data["features"]
    labels = data["labels"]

    print("Images:", images.shape)
    print("Features:", features.shape)
    print("Labels:", labels.shape)

    return images, features, labels



# PREPARE DATA


def prepare_dataset(
    images,
    features,
    labels
):

    
    # Cache images are stored as uint8.
    # Convert to [0,1].
    

    images = (
        images.astype(np.float32)
        / 255.0
    )

  
    # Add channel dimension
    #
    # [N,224,224]
    # →
    # [N,1,224,224]
    

    images = np.expand_dims(
        images,
        axis=1
    )

    features = features.astype(
        np.float32
    )

    labels = labels.astype(
        np.int64
    )

    return TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(features),
        torch.from_numpy(labels)
    )



# TRAIN ONE EPOCH


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, features, labels in loader:

        images = images.to(DEVICE)
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            images,
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )



# VALIDATION


def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, features, labels in loader:

            images = images.to(DEVICE)
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                images,
                features
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )



# MAIN


def main():

    print("=" * 60)
    print("DHKFHNET SOFT-SCAN TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Seed:", RANDOM_SEED)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Feature size:", FEATURE_SIZE)

   
    # CHECK CACHE FILES
    
    if not TRAIN_CACHE.exists():

        raise FileNotFoundError(
            f"Training cache not found:\n{TRAIN_CACHE}"
        )

    if not VAL_CACHE.exists():

        raise FileNotFoundError(
            f"Validation cache not found:\n{VAL_CACHE}"
        )

    
    # LOAD TRAINING CACHE
   

    train_images, train_features, train_labels = (
        load_cache(TRAIN_CACHE)
    )

    
    # LOAD VALIDATION CACHE
    
    val_images, val_features, val_labels = (
        load_cache(VAL_CACHE)
    )

    
    # VERIFY FEATURES
   
    if train_features.shape[1] != FEATURE_SIZE:

        raise RuntimeError(
            f"Expected {FEATURE_SIZE} features, "
            f"but found {train_features.shape[1]}"
        )

    if val_features.shape[1] != FEATURE_SIZE:

        raise RuntimeError(
            f"Expected {FEATURE_SIZE} features, "
            f"but found {val_features.shape[1]}"
        )

    
    # DATASETS
    
    train_dataset = prepare_dataset(
        train_images,
        train_features,
        train_labels
    )

    val_dataset = prepare_dataset(
        val_images,
        val_features,
        val_labels
    )

    print()
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    
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
   
    model = DHKFHNet(
        feature_size=FEATURE_SIZE,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print()
    print("DHKFHNet created.")

    
    # LOSS
    
    criterion = nn.CrossEntropyLoss()

    
    # OPTIMIZER
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    
    # SCHEDULER
    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )

    
    # BEST MODEL
    
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    
    # TRAIN
    
    print()
    print("=" * 60)
    print("STARTING DHKFHNET TRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_accuracy
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy * 100:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy * 100:.2f}% "
            f"LR: {current_lr:.6f}"
        )

        
        # SAVE BEST VALIDATION MODEL
        

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "feature_size":
                        FEATURE_SIZE,

                    "num_classes":
                        NUM_CLASSES,

                    "image_size":
                        IMAGE_SIZE,

                    "epoch":
                        epoch,

                    "best_validation_accuracy":
                        best_val_accuracy
                },
                DHKFHNET_MODEL_PATH
            )

            print(
                f"Best model saved "
                f"(epoch {epoch})"
            )

        else:

            epochs_without_improvement += 1

        
        # EARLY STOPPING
        
        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

   
    # COMPLETE
   

    print()
    print("=" * 60)
    print("DHKFHNET TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print()
    print("Model saved at:")
    print(DHKFHNET_MODEL_PATH)

    print("=" * 60)



# RUN


if __name__ == "__main__":
    main()