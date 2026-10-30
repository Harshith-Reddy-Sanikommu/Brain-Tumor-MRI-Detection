import streamlit as st

from pathlib import Path
from utils.image_utils import load_image
from inference.pipeline import run_pipeline


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="NeuroVista | Brain MRI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# LOAD EXTERNAL CSS
# Keep style.css in the same folder as this app.py file.
# ------------------------------------------------------------

CSS_PATH = Path(__file__).parent / "style.css"

if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning("style.css was not found. Place it in the same folder as app.py.")


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">N</div>
            <div>
                <div class="brand-name">NEUROVISTA</div>
                <div class="brand-caption">MRI INTELLIGENCE</div>
            </div>
        </div>
        <div class="sidebar-intro">
            A machine learning and deep learning-based interface for brain MRI image analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown('<div class="sidebar-label">PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("**Hybrid Deep Learning Network**")
    st.caption(
        "MRI-based brain tumor classification using Shape-Aware SCAN "
        "and DHKFHNet."
    )

    st.divider()

    st.markdown('<div class="sidebar-label">ANALYSIS PIPELINE</div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("01", "MRI image input"),
        ("02", "Midpoint filtering"),
        ("03", "BSE skull removal"),
        ("04", "CLAHE enhancement"),
        ("05", "Shape-Aware SCAN"),
        ("06", "Soft SCAN attention"),
        ("07", "LARK + GLCM + GDP"),
        ("08", "DHKFHNet prediction"),
    ]

    for number, label in pipeline_steps:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <span class="pipeline-number">{number}</span>
                <span>{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown('<div class="sidebar-label">MODEL COMPONENTS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="component-list">
            <span>Shape-Aware SCAN</span>
            <span>Soft SCAN Attention</span>
            <span>LARK feature descriptors</span>
            <span>GLCM texture features</span>
            <span>GDP feature descriptors</span>
            <span>DHKFHNet classifier</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()


# ------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            <span class="eyebrow-dot"></span>
            INTELLIGENT MEDICAL IMAGE ANALYSIS
        </div>
        <div class="hero-title">Brain MRI <span>Intelligence.</span></div>
        <div class="hero-subtitle">
            A hybrid deep learning pipeline designed to analyze MRI scans
            and classify brain tumor patterns.
        </div>
        <div class="hero-meta">
            <span>SHAPE-AWARE SCAN</span>
            <span class="meta-separator">/</span>
            <span>SOFT ATTENTION</span>
            <span class="meta-separator">/</span>
            <span>DHKFHNET</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# PIPELINE OVERVIEW CARDS
# ------------------------------------------------------------

st.markdown('<div class="section-kicker">THE FRAMEWORK</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">A closer look at the pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Four connected stages transform the input MRI into a final classification.</div>',
    unsafe_allow_html=True
)

overview = [
    ("01", "Preprocessing", "Midpoint filter · BSE · CLAHE", "◈"),
    ("02", "Segmentation", "Shape-Aware SCAN", "◎"),
    ("03", "Feature extraction", "LARK · GLCM · GDP", "⌘"),
    ("04", "Classification", "DHKFHNet", "✳"),
]

overview_cols = st.columns(4, gap="medium")

for col, (number, title, description, symbol) in zip(overview_cols, overview):
    with col:
        st.markdown(
            f"""
            <div class="overview-card">
                <div class="overview-card-top">
                    <span class="overview-symbol">{symbol}</span>
                    <span class="overview-number">{number}</span>
                </div>
                <div class="overview-card-title">{title}</div>
                <div class="overview-card-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# MRI UPLOAD
# ------------------------------------------------------------

st.markdown('<div class="section-kicker">START AN ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Upload your MRI scan</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Choose a brain MRI image to run it through the complete hybrid deep learning pipeline.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=False,
    width="stretch"
)


# ------------------------------------------------------------
# PROCESS IMAGE
# ------------------------------------------------------------

if uploaded_file is not None:

    image = load_image(uploaded_file)

    st.markdown('<div class="section-rule compact-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">INPUT REVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">MRI scan preview</div>', unsafe_allow_html=True)

    input_col1, input_col2 = st.columns([1.65, 1], gap="large")

    with input_col1:
        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch"
        )

    with input_col2:
        channels = image.shape[2] if len(image.shape) == 3 else 1

        st.markdown(
            f"""
            <div class="file-info-card">
                <div class="file-info-label">SELECTED FILE</div>
                <div class="file-name">{uploaded_file.name}</div>
                <div class="file-info-divider"></div>
                <div class="file-info-row">
                    <span>Image dimensions</span>
                    <strong>{image.shape[1]} × {image.shape[0]}</strong>
                </div>
                <div class="file-info-row">
                    <span>Channels</span>
                    <strong>{channels}</strong>
                </div>
                <div class="file-info-row">
                    <span>File format</span>
                    <strong>{uploaded_file.type or "Image"}</strong>
                </div>
                <div class="ready-status">
                    <span class="status-dot"></span>
                    Image ready for analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="analysis-action">', unsafe_allow_html=True)
    analyze = st.button(
        "Run MRI analysis  →",
        type="primary",
        width="stretch"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze:

        with st.spinner("Running the complete hybrid deep learning pipeline..."):

            try:
                result = run_pipeline(image)

                st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-kicker">ANALYSIS OUTPUT</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-heading">Explore the model results</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-description">Review each stage of the processing pipeline and the final model output.</div>',
                    unsafe_allow_html=True
                )

                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "01  Preprocessing",
                        "02  SCAN",
                        "03  Features",
                        "04  Prediction",
                        "05  Summary"
                    ]
                )

                # TAB 1 — PREPROCESSING
                with tab1:
                    st.subheader("Preprocessing stages")
                    st.write(
                        "The input MRI passes through the following preprocessing operations."
                    )

                    col1, col2, col3 = st.columns(3, gap="medium")

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
                    st.subheader("Shape-Aware SCAN segmentation")
                    st.write(
                        "SCAN generates a probability map that is used to provide soft attention to the MRI."
                    )

                    st.image(
                        result["segmentation"],
                        caption="SCAN Probability Map",
                        width="stretch"
                    )

                    st.subheader("Soft SCAN attention")
                    attention_display = (
                        result["attention"] * 255.0
                    ).clip(0, 255).astype("uint8")

                    st.image(
                        attention_display,
                        caption="Soft SCAN Attention Map",
                        width="stretch"
                    )

                    st.subheader("Enhanced MRI")
                    st.image(
                        result["enhanced_image"],
                        caption="MRI after Soft SCAN Attention",
                        width="stretch"
                    )

                # TAB 3 — FEATURES
                with tab3:
                    st.subheader("LARK + GLCM + GDP feature extraction")
                    features = result["features"]

                    st.write(
                        "The enhanced MRI is converted into a 13-dimensional feature representation."
                    )

                    st.metric("Total Extracted Features", len(features))

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
                    st.subheader("Final DHKFHNet prediction")

                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    predicted_class = result["class"]

                    result_label = "Tumor" if predicted_class == 1 else "No Tumor"
                    result_style = "tumor-result" if predicted_class == 1 else "clear-result"
                    result_symbol = "!" if predicted_class == 1 else "✓"

                    st.markdown(
                        f"""
                        <div class="prediction-results">
                            <div class="prediction-result-card {result_style}">
                                <div class="result-card-label">PREDICTED CLASS</div>
                                <div class="result-main-line">
                                    <span class="result-symbol">{result_symbol}</span>
                                    <span class="result-class-name">{result_label}</span>
                                </div>
                                <div class="result-card-note">DHKFHNet classification output</div>
                            </div>
                            <div class="confidence-result-card">
                                <div class="result-card-label">MODEL CONFIDENCE</div>
                                <div class="confidence-number">{confidence * 100:.2f}<span>%</span></div>
                                <div class="result-card-note">Confidence score for the predicted class</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(float(confidence))

                    st.info(
                        "The final classification is generated by DHKFHNet using "
                        "the enhanced MRI and the extracted LARK + GLCM + GDP features. "
                        "This output is a research demonstration and is not a medical diagnosis."
                    )

                # TAB 5 — SUMMARY
                with tab5:
                    st.subheader("Analysis summary")

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
                    st.subheader("Final result")
                    st.write(f"**Prediction:** {prediction}")
                    st.write(f"**Confidence:** {confidence * 100:.2f}%")

            except FileNotFoundError:
                st.error("Trained model files were not found.")
                st.info(
                    "Check the following model files:\n\n"
                    "models/scan/scan_model.pth\n\n"
                    "models/dhkfhnet/dhkfhnet_softscan_seed42.pth"
                )

            except RuntimeError as error:
                st.error("Model loading or prediction failed.")
                st.code(str(error))

            except Exception as error:
                st.error("An unexpected error occurred.")
                st.code(str(error))

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">◎</div>
            <div class="empty-title">Your analysis workspace is ready</div>
            <div class="empty-description">
                Upload a brain MRI image above to begin. The image will pass through
                preprocessing, SCAN attention, feature extraction and DHKFHNet classification.
            </div>
            <div class="empty-flow">
                <span>Preprocessing</span>
                <b>→</b>
                <span>SCAN attention</span>
                <b>→</b>
                <span>Feature extraction</span>
                <b>→</b>
                <span>Prediction</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="app-footer">
        <div class="footer-brand">NEUROVISTA <span>·</span> MRI INTELLIGENCE</div>
        <div class="footer-note">
            Brain Tumor Detection Using MRI · Hybrid Deep Learning Pipeline
        </div>
        <div class="footer-disclaimer">
            Research and educational use only. Not intended to replace professional medical advice.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
