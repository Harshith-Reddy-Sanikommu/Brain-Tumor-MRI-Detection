
import cv2
import numpy as np


def brain_surface_extractor(image):
    """
    BSE-style brain-region extraction.

    Produces:
    1. Brain-region image
    2. Brain-region mask
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(mask)

        cv2.drawContours(
            clean_mask,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED
        )

        mask = clean_mask

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return result, mask

