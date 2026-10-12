
import cv2
import numpy as np


def midpoint_filter(image, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    image = image.astype(np.float32)

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8
    )

    minimum = cv2.erode(
        image,
        kernel
    )

    maximum = cv2.dilate(
        image,
        kernel
    )

    result = (minimum + maximum) / 2.0

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

