
import cv2
import numpy as np


def extract_glcm_features(image):
    """
    Lightweight GLCM texture features.

    Returns:
        contrast
        homogeneity
        energy
    """

    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    image = (
        image.astype(np.float32) / 255.0 * 7
    ).astype(np.uint8)

    levels = 8

    glcm = np.zeros(
        (levels, levels),
        dtype=np.float64
    )

    # Horizontal relationship
    for y in range(image.shape[0]):
        for x in range(image.shape[1] - 1):

            i = image[y, x]
            j = image[y, x + 1]

            glcm[i, j] += 1

    # Vertical relationship
    for y in range(image.shape[0] - 1):
        for x in range(image.shape[1]):

            i = image[y, x]
            j = image[y + 1, x]

            glcm[i, j] += 1

    total = glcm.sum()

    if total > 0:
        glcm /= total

    contrast = 0.0
    homogeneity = 0.0
    energy = 0.0

    for i in range(levels):
        for j in range(levels):

            contrast += (
                (i - j) ** 2
            ) * glcm[i, j]

            homogeneity += (
                glcm[i, j] /
                (1 + abs(i - j))
            )

            energy += glcm[i, j] ** 2

    return np.array(
        [
            contrast,
            homogeneity,
            energy
        ],
        dtype=np.float32
    )

