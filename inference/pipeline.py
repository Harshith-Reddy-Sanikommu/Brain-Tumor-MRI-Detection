import numpy as np

from utils.config import (
    SCAN_MODEL_PATH,
    DHKFHNET_MODEL_PATH
)

from preprocessing.midpoint_filter import (
    midpoint_filter
)

from preprocessing.bse import (
    brain_surface_extractor
)

from preprocessing.clahe import (
    apply_clahe
)

from segmentation.predict_scan import (
    load_scan_model,
    predict_scan,
    create_soft_attention
)

from classification.predict_dhkfhnet import (
    load_dhkfhnet,
    predict_dhkfhnet
)

from feature_extraction.extract_features import (
    extract_features
)


def run_pipeline(image):

    
    # 1. MIDPOINT FILTER
    

    midpoint = midpoint_filter(
        image
    )


    
    # 2. BSE SKULL REMOVAL
   

    bse, bse_mask = brain_surface_extractor(
        midpoint
    )


    
    # 3. CLAHE
   

    clahe = apply_clahe(
        bse
    )


   
    # 4. SCAN SEGMENTATION
    

    scan_model, scan_device = load_scan_model(
        SCAN_MODEL_PATH
    )

    scan_probability = predict_scan(
        scan_model,
        clahe
    )

    attention = create_soft_attention(
        scan_probability
    )


   
    # 5. SOFT SCAN ATTENTION
    

    enhanced = (
        clahe.astype(np.float32)
        * attention[..., None]
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)


   
    # 6. SCAN PROBABILITY MAP
    

    segmentation = (
        scan_probability * 255.0
    )

    segmentation = np.clip(
        segmentation,
        0,
        255
    ).astype(np.uint8)


    
    # 7. FEATURE EXTRACTION
   

    features = np.asarray(
        extract_features(
            enhanced
        ),
        dtype=np.float32
    )


    
    # 8. DHKFHNET CLASSIFICATION
   

    dhkfhnet_model, dhkfhnet_device = (
        load_dhkfhnet()
    )

    predicted_class, confidence = (
        predict_dhkfhnet(
            dhkfhnet_model,
            enhanced,
            features,
            dhkfhnet_device
        )
    )


   
    # 9. FINAL PREDICTION
    

    if predicted_class == 1:

        prediction = "Tumor Detected"

    else:

        prediction = "No Tumor Detected"


    
    # 10. RETURN RESULTS
   

    return {

        "preprocessing": {

            "midpoint": midpoint,

            "bse": bse,

            "clahe": clahe

        },

        "bse_mask": bse_mask,

        "segmentation": segmentation,

        "segmented_image": enhanced,

        "attention": attention,

        "enhanced_image": enhanced,

        "features": features,

        "prediction": prediction,

        "class": predicted_class,

        "confidence": confidence

    }