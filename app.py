import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_my_model():
    return load_model("cnn_model.h5", compile=False)

model = load_my_model()

# =========================
# Class Labels
# =========================
class_names = ["Early Blight", "Late Blight", "Healthy"]

# =========================
# Disease Data
# =========================
disease_data = {
    "Early Blight": {
        "info": "Fungal disease causing dark spots with rings.",
        "prevention": [
            "Use disease-free seeds",
            "Maintain spacing",
            "Avoid overhead watering"
        ],
        "treatment": [
            "Remove infected leaves",
            "Apply fungicides"
        ]
    },
    "Late Blight": {
        "info": "Serious disease caused by fungus.",
        "prevention": [
            "Use resistant varieties",
            "Avoid moisture",
            "Good drainage"
        ],
        "treatment": [
            "Remove infected plants",
            "Use copper fungicides"
        ]
    },
    "Healthy": {
        "info": "No disease detected.",
        "prevention": [
            "Maintain watering",
            "Use fertilizers"
        ],
        "treatment": ["No treatment needed"]
    }
}

# =========================
# Sidebar
# =========================
st.sidebar.title("🌿 About")
st.sidebar.write("This application identifies potato leaf diseases using an image classification model and provides real-time predictions along with disease details, prevention, and treatment suggestions.")
st.sidebar.markdown("""
### 📌 Features
📷 Image-based disease detection  
🧠 Classification model  
⚡ Real-time prediction  

### 🩺 Support
📊 Confidence & charts  
🛡️ Prevention tips  
💊 Treatment suggestions  

### ⚙️ Tech
💻 Python & Streamlit  
📱 Works on mobile & desktop  
""")

# =========================
# Title
# =========================
st.title("🌿 Potato Leaf Disease Detection")
st.write("Upload or capture an image to detect disease.")

# =========================
# Input Section
# =========================
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])

# 🔥 Camera Toggle Button
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if st.button("📸 Open Camera", use_container_width=True):
    st.session_state.camera_on = not st.session_state.camera_on

camera_image = None

if st.session_state.camera_on:
    camera_image = st.camera_input("Take a photo")

# =========================
# Image Selection
# =========================
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)

elif camera_image is not None:
    image = Image.open(camera_image)

# =========================
# Prediction
# =========================
if image is not None:

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict(img_array)

    probs = prediction[0]
    predicted_class = class_names[np.argmax(probs)]
    confidence = np.max(probs)

    # Result
    st.success(f"✅ Prediction: {predicted_class}")

    # Confidence
    if confidence > 0.8:
        st.success(f"High Confidence: {confidence:.2f}")
    elif confidence > 0.5:
        st.warning(f"Medium Confidence: {confidence:.2f}")
    else:
        st.error(f"Low Confidence: {confidence:.2f}")

    st.progress(float(confidence))

    # Chart
    df = pd.DataFrame({
        "Disease": class_names,
        "Probability": probs
    })

    st.subheader("📊 Prediction Probability")
    st.bar_chart(df.set_index("Disease"))

    # Info
    st.subheader("🩺 Disease Info")
    st.write(disease_data[predicted_class]["info"])

    # Prevention
    with st.expander("🛡️ Prevention Tips"):
        for tip in disease_data[predicted_class]["prevention"]:
            st.write(f"✅ {tip}")

    # Treatment
    with st.expander("💊 Treatment"):
        for tip in disease_data[predicted_class]["treatment"]:
            st.write(f"🔹 {tip}")

    # Warning
    if predicted_class == "Late Blight":
        st.error("⚠️ Serious disease! Take action immediately.")

    # Download
    result_text = f"""
Prediction: {predicted_class}
Confidence: {confidence:.2f}
"""

    st.download_button(
        label="📥 Download Result",
        data=result_text,
        file_name="result.txt"
    )

else:
    st.warning("⚠️ Upload image or open camera to continue.")