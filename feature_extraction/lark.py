
import cv2
import numpy as np


def extract_lark_features(image):
    """
    Lightweight LARK-style local structure features.

    Returns 5 numerical features.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image.copy()

    gray = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    features = [
        np.mean(magnitude),
        np.std(magnitude),
        np.mean(np.abs(gx)),
        np.mean(np.abs(gy)),
        np.max(magnitude)
    ]

    return np.array(
        features,
        dtype=np.float32
    )

