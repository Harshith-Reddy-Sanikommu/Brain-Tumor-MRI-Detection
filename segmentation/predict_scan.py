import torch
import cv2
import numpy as np

from segmentation.scan_model import SCAN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_scan_model(model_path):

    model = SCAN().to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model, DEVICE


def predict_scan(model, image):

    if image is None:
        raise ValueError(
            "Invalid image"
        )

    original_height, original_width = (
        image.shape[:2]
    )

    # SCAN input size
    resized = cv2.resize(
        image,
        (224, 224)
    )

    resized = resized.astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        resized
    ).permute(
        2, 0, 1
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(
            tensor
        )

        probability = torch.sigmoid(
            output
        )

    probability = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    # Return probability map
    # to original image size
    probability = cv2.resize(
        probability,
        (
            original_width,
            original_height
        )
    )

    return probability


def create_soft_attention(
    probability
):

    # Soft SCAN attention used in
    # the successful DHKFHNet pipeline

    attention = (
        0.50 +
        0.50 * probability
    )

    return attention