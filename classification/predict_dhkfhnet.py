import torch

from .dhkfhnet import DHKFHNet

from utils.config import DHKFHNET_MODEL_PATH

from utils.image_utils import (
    prepare_grayscale_for_model
)


def load_dhkfhnet(device=None):

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = DHKFHNet(
        num_classes=2,
        feature_size=13
    ).to(device)

    checkpoint = torch.load(
        DHKFHNET_MODEL_PATH,
        map_location=device,
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

    return model, device


def predict_dhkfhnet(
    model,
    image,
    features,
    device
):
    """
    Predict using DHKFHNet.

    Inputs:
        image    = segmented MRI
        features = 13 handcrafted features

    Image is converted to:
        1 x 224 x 224
    """

    image_tensor = prepare_grayscale_for_model(
        image
    )

    image_tensor = torch.tensor(
        image_tensor,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            image_tensor,
            feature_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return (
        predicted_class,
        confidence
    )