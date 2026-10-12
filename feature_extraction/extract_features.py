import numpy as np

from .lark import extract_lark_features
from .glcm import extract_glcm_features
from .gdp import extract_gdp_features


def extract_features(image):
    """
    Extract and combine:

    LARK = 5 features
    GLCM = 3 features
    GDP  = 5 features

    Total = 13 features.
    """

    lark_features = extract_lark_features(
        image
    )

    glcm_features = extract_glcm_features(
        image
    )

    gdp_features = extract_gdp_features(
        image
    )

    features = np.concatenate(
        [
            lark_features,
            glcm_features,
            gdp_features
        ]
    )

    return features.astype(
        np.float32
    )