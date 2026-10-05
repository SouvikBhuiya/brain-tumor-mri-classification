import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

/* Subtitle */
.main-subtitle {
    text-align: center;
    font-size: 17px;
    color: #94a3b8;
    margin-bottom: 35px;
}

/* Section headings */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: white;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Cards */
.card {
    background-color: #111827;
    border: 1px solid #263244;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

/* Card title */
.card-title {
    color: #38bdf8;
    font-size: 18px;
    font-weight: 600;
}

/* Card value */
.card-value {
    color: #cbd5e1;
    font-size: 15px;
    margin-top: 12px;
}

/* Prediction */
.prediction-title {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-result {
    text-align: center;
    color: #38bdf8;
    font-size: 34px;
    font-weight: 700;
    margin: 8px 0 20px 0;
    text-transform: capitalize;
}

.confidence-title {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    text-transform: uppercase;
}

.confidence-value {
    text-align: center;
    color: #4ade80;
    font-size: 25px;
    font-weight: 600;
    margin-top: 8px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #263244;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "best_efficientnetb0.keras"
    )


model = load_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    image = image.resize((224, 224))

    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        predictions[0][predicted_index] * 100
    )

    return predicted_class, confidence


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Brain Tumor MRI Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered MRI image classification using EfficientNetB0'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INFORMATION CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">🧠 Model</div>'
        '<div class="card-value">EfficientNetB0</div>'
        '</div>',
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">📐 Image Size</div>'
        '<div class="card-value">224 × 224 pixels</div>'
        '</div>',
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">🏷️ Classes</div>'
        '<div class="card-value">4 MRI categories</div>'
        '</div>',
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">⚡ Framework</div>'
        '<div class="card-value">TensorFlow + Streamlit</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload Brain MRI</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# IMAGE SECTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.markdown(
        "### 🖼️ Uploaded MRI Image"
    )


    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )


    with col2:

        st.image(
            image,
            use_container_width=True
        )


        st.write("")


        predict_button = st.button(
            "🔍 Predict Tumor Type",
            use_container_width=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        with st.spinner(
            "Analysing MRI image..."
        ):

            predicted_class, confidence = predict_image(
                image
            )


        st.markdown("---")


        st.markdown(
            '<div class="prediction-title">'
            'Predicted Classification'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f'<div class="prediction-result">'
            f'{predicted_class}'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="confidence-title">'
            'Model Confidence'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f'<div class="confidence-value">'
            f'{confidence:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )


        st.progress(
            min(int(confidence), 100)
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "---"
)

st.caption(
    "🧠 Brain Tumor MRI Classification System"
)

st.caption(
    "Developed using Python, TensorFlow, EfficientNetB0 and Streamlit"
)