from pathlib import Path



# BASE PROJECT DIRECTORY


BASE_DIR = Path(__file__).resolve().parent.parent



# MODEL DIRECTORY


MODEL_DIR = BASE_DIR / "models"



# TRAINED MODEL PATHS


SCAN_MODEL_PATH = (
    MODEL_DIR
    / "scan"
    / "scan_model.pth"
)

DHKFHNET_MODEL_PATH = (
    MODEL_DIR
    / "dhkfhnet"
    / "dhkfhnet_softscan_seed42.pth"
)



# IMAGE SETTINGS


IMAGE_SIZE = 224



# CLAHE SETTINGS


CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (
    8,
    8
)



# SUPPORTED IMAGE EXTENSIONS


SUPPORTED_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
]