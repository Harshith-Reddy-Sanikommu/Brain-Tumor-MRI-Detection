import cv2
import numpy as np
from PIL import Image


def load_image(file):
    """
    Load an image and return it as RGB NumPy array.
    """
    image = Image.open(file).convert("RGB")
    return np.array(image)


def rgb_to_gray(image):
    """
    Convert RGB image to grayscale.
    """
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )


def normalize_image(image):
    """
    Min-max normalize image to range [0, 1].
    """
    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum == minimum:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    return (
        image - minimum
    ) / (
        maximum - minimum
    )


def resize_image(image, size=(224, 224)):
    """
    Resize image to the required model size.
    """
    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )


def prepare_for_model(image, size=(224, 224)):
    """
    Prepare image for SCAN.

    Output:
        3 x 224 x 224
    """

    image = resize_image(
        image,
        size
    )

    # SCAN requires 3-channel RGB input
    if len(image.shape) == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    return image


def prepare_grayscale_for_model(
    image,
    size=(224, 224)
):
    """
    Prepare image for DHKFHNet.

    DHKFHNet requires:
        1-channel grayscale image

    Output:
        1 x 224 x 224
    """

    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    image = resize_image(
        image,
        size
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image