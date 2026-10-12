
import cv2


def apply_clahe(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    if len(image.shape) == 3:

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        l_channel = clahe.apply(
            l_channel
        )

        enhanced = cv2.merge(
            (
                l_channel,
                a_channel,
                b_channel
            )
        )

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_LAB2RGB
        )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)

