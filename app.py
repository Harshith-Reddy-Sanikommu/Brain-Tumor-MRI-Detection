import streamlit as st

from utils.image_utils import load_image
from inference.pipeline import run_pipeline



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Brain Tumor Detection Using MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CUSTOM UI STYLE


st.markdown(
    """
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 10px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    border: 2px solid rgba(128,128,128,0.30);
    text-align: center;
    margin: 10px 0 20px 0;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
}

.small-text {
    font-size: 14px;
    opacity: 0.7;
}


/* ============================================================
   EQUAL OVERVIEW BOXES
   ============================================================ */

.overview-card {
    height: 145px;
    padding: 24px;
    border-radius: 12px;
    background-color: #e7f0fb;
    box-sizing: border-box;
}

.overview-title {
    color: #0056a6;
    font-size: 21px;
    font-weight: 600;
    margin-bottom: 22px;
}

.overview-text {
    color: #0056a6;
    font-size: 17px;
    line-height: 1.5;
}

</style>
""",
    unsafe_allow_html=True
)



# SIDEBAR


with st.sidebar:

    st.header("🧠 Project Information")

    st.write(
        "**Hybrid Deep Learning Network**"
    )

    st.write(
        "MRI-based brain tumor detection "
        "using Shape-Aware SCAN and DHKFHNet."
    )

    st.divider()

    st.subheader("Proposed Pipeline")

    st.write("MRI Image")
    st.write("↓ Midpoint Filter")
    st.write("↓ BSE Skull Removal")
    st.write("↓ CLAHE")
    st.write("↓ Shape-Aware SCAN")
    st.write("↓ Soft SCAN Attention")
    st.write("↓ LARK + GLCM + GDP")
    st.write("↓ DHKFHNet")
    st.write("↓ Final Prediction")

    st.divider()

    st.subheader("Model Components")

    st.write("• Shape-Aware SCAN")
    st.write("• Soft SCAN Attention")
    st.write("• LARK Features")
    st.write("• GLCM Features")
    st.write("• GDP Features")
    st.write("• DHKFHNet Classifier")

    st.divider()

    st.caption(
        "Major Project | Brain Tumor Detection"
    )



# HEADER

st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection Using MRI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Hybrid deep learning pipeline for automated MRI-based brain tumor analysis'
    '</div>',
    unsafe_allow_html=True
)



# PROJECT OVERVIEW

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)


with overview_col1:

    st.markdown(
        '<div class="overview-card">'
        '<div class="overview-title">Preprocessing</div>'
        '<div class="overview-text">Midpoint + BSE + CLAHE</div>'
        '</div>',
        unsafe_allow_html=True
    )


with overview_col2:

    st.markdown(
        '<div class="overview-card">'
        '<div class="overview-title">Segmentation</div>'
        '<div class="overview-text">Shape-Aware SCAN</div>'
        '</div>',
        unsafe_allow_html=True
    )


with overview_col3:

    st.markdown(
        '<div class="overview-card">'
        '<div class="overview-title">Features</div>'
        '<div class="overview-text">LARK + GLCM + GDP</div>'
        '</div>',
        unsafe_allow_html=True
    )


with overview_col4:

    st.markdown(
        '<div class="overview-card">'
        '<div class="overview-title">Classification</div>'
        '<div class="overview-text">DHKFHNet</div>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()



# IMAGE UPLOAD

st.markdown(
    '<div class="section-title">📤 Upload MRI</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Select an MRI image for analysis",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ],
    accept_multiple_files=False,
    width="stretch"
)



# PROCESS IMAGE

if uploaded_file is not None:

    image = load_image(uploaded_file)

    st.markdown(
        '<div class="section-title">🖼️ Input MRI</div>',
        unsafe_allow_html=True
    )

    input_col1, input_col2 = st.columns(
        [2, 1]
    )

    with input_col1:

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch"
        )

    with input_col2:

        st.markdown(
            '<div class="info-box">',
            unsafe_allow_html=True
        )

        st.write("### Image Information")

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**Dimensions:** "
            f"{image.shape[1]} × {image.shape[0]}"
        )

        if len(image.shape) == 3:

            st.write(
                f"**Channels:** {image.shape[2]}"
            )

        st.write(
            "**Status:** Ready for analysis"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


   
    # ANALYZE BUTTON
    
    analyze = st.button(
        "🔍 Analyze MRI",
        type="primary",
        width="stretch"
    )


    if analyze:

        with st.spinner(
            "Running the complete hybrid deep learning pipeline..."
        ):

            try:

                result = run_pipeline(image)


               
                # RESULTS TABS
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "🔬 Preprocessing",
                        "🎯 SCAN",
                        "🧩 Features",
                        "🧠 Prediction",
                        "📋 Summary"
                    ]
                )


               
                # TAB 1 — PREPROCESSING
                
                with tab1:

                    st.subheader(
                        "Preprocessing Stages"
                    )

                    st.write(
                        "The input MRI passes through the "
                        "following preprocessing operations."
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.image(
                            result["preprocessing"]["midpoint"],
                            caption="1. Midpoint Filter",
                            width="stretch"
                        )

                    with col2:

                        st.image(
                            result["preprocessing"]["bse"],
                            caption="2. BSE Skull Removal",
                            width="stretch"
                        )

                    with col3:

                        st.image(
                            result["preprocessing"]["clahe"],
                            caption="3. CLAHE",
                            width="stretch"
                        )


                
                # TAB 2 — SCAN
                

                with tab2:

                    st.subheader(
                        "Shape-Aware SCAN Segmentation"
                    )

                    st.write(
                        "SCAN generates a probability map that is "
                        "used to provide soft attention to the MRI."
                    )

                    st.image(
                        result["segmentation"],
                        caption="SCAN Probability Map",
                        width="stretch"
                    )

                    st.subheader(
                        "Soft SCAN Attention"
                    )

                    attention_display = (
                        result["attention"] * 255.0
                    ).clip(
                        0,
                        255
                    ).astype(
                        "uint8"
                    )

                    st.image(
                        attention_display,
                        caption="Soft SCAN Attention Map",
                        width="stretch"
                    )

                    st.subheader(
                        "Enhanced MRI"
                    )

                    st.image(
                        result["enhanced_image"],
                        caption="MRI after Soft SCAN Attention",
                        width="stretch"
                    )


                
                # TAB 3 — FEATURES
                

                with tab3:

                    st.subheader(
                        "LARK + GLCM + GDP Feature Extraction"
                    )

                    features = result["features"]

                    st.write(
                        "The enhanced MRI is converted into a "
                        "13-dimensional feature representation."
                    )

                    st.metric(
                        "Total Extracted Features",
                        len(features)
                    )

                    feature_names = [

                        "LARK Mean Magnitude",
                        "LARK Std Magnitude",
                        "LARK Mean |GX|",
                        "LARK Mean |GY|",
                        "LARK Max Magnitude",

                        "GLCM Contrast",
                        "GLCM Homogeneity",
                        "GLCM Energy",

                        "GDP Mean Magnitude",
                        "GDP Std Magnitude",
                        "GDP Mean Cosine",
                        "GDP Mean Sine",
                        "GDP Max Magnitude"

                    ]

                    feature_data = {
                        "Feature": feature_names,
                        "Value": features
                    }

                    st.dataframe(
                        feature_data,
                        width="stretch",
                        hide_index=True
                    )


                
                # TAB 4 — PREDICTION
                
                with tab4:

                    st.subheader(
                        "Final DHKFHNet Prediction"
                    )

                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    predicted_class = result["class"]


                    if predicted_class == 1:

                        st.error(
                            f"⚠️ {prediction}"
                        )

                    else:

                        st.success(
                            f"✅ {prediction}"
                        )


                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:

                        st.metric(
                            "Predicted Class",
                            "Tumor"
                            if predicted_class == 1
                            else "No Tumor"
                        )

                    with metric_col2:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )


                    st.progress(
                        float(confidence)
                    )


                    st.info(
                        "The final classification is generated "
                        "by DHKFHNet using the enhanced MRI and "
                        "the extracted LARK + GLCM + GDP features."
                    )


                
                # TAB 5 — SUMMARY
                
                with tab5:

                    st.subheader(
                        "Analysis Summary"
                    )

                    summary_data = {

                        "Stage": [

                            "Input MRI",
                            "Midpoint Filter",
                            "BSE Skull Removal",
                            "CLAHE",
                            "Shape-Aware SCAN",
                            "Soft SCAN Attention",
                            "Feature Extraction",
                            "DHKFHNet"

                        ],

                        "Status": [

                            "Completed",
                            "Completed",
                            "Completed",
                            "Completed",
                            "Completed",
                            "Completed",
                            "13 features extracted",
                            "Completed"

                        ]

                    }

                    st.dataframe(
                        summary_data,
                        width="stretch",
                        hide_index=True
                    )

                    st.divider()

                    st.subheader(
                        "Final Result"
                    )

                    st.write(
                        f"**Prediction:** {prediction}"
                    )

                    st.write(
                        f"**Confidence:** "
                        f"{confidence * 100:.2f}%"
                    )


           
            # ERROR HANDLING
            

            except FileNotFoundError:

                st.error(
                    "Trained model files were not found."
                )

                st.info(
                    "Check the following model files:\n\n"
                    "models/scan/scan_model.pth\n\n"
                    "models/dhkfhnet/"
                    "dhkfhnet_softscan_seed42.pth"
                )


            except RuntimeError as error:

                st.error(
                    "Model loading or prediction failed."
                )

                st.code(
                    str(error)
                )


            except Exception as error:

                st.error(
                    "An unexpected error occurred."
                )

                st.code(
                    str(error)
                )


else:

    
    # EMPTY STATE
    

    st.info(
        "👆 Upload an MRI image above to begin the analysis."
    )

    st.markdown(
        """
### 🔬 Analysis Workflow

**1. Preprocessing**  
Midpoint Filtering → BSE Skull Removal → CLAHE

**2. Segmentation**  
Shape-Aware SCAN

**3. Attention**  
Soft SCAN Attention

**4. Feature Extraction**  
LARK + GLCM + GDP

**5. Classification**  
DHKFHNet

**6. Final Output**  
Tumor / No Tumor with confidence
"""
    )



# FOOTER


st.divider()

st.caption(
    "Brain Tumor Detection Using MRI | "
    "Hybrid Deep Learning Pipeline"
)