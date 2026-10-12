from pathlib import Path

from utils.image_utils import load_image
from inference.pipeline import run_pipeline


IMAGE_PATH = Path(
    r"D:\Brain_Tumor_Detection\data\br35h\yes\y4.jpg"
)


print("Loading image...")

image = load_image(
    IMAGE_PATH
)

print("Running complete pipeline...")

result = run_pipeline(
    image
)

print()
print("================================")
print("PIPELINE TEST SUCCESSFUL")
print("================================")

print(
    "Actual Class : Tumor"
)

print(
    "Prediction   :",
    result["prediction"]
)

print(
    "Confidence   :",
    f"{result['confidence'] * 100:.2f}%"
)

print(
    "Features     :",
    len(result["features"])
)

print(
    "Image shape  :",
    result["enhanced_image"].shape
)

print("================================")