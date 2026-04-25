import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("cnn_model.h5", compile=False)

# Class labels (must match training)
class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# App title
st.title("🌿 Potato Leaf Disease Detection")

st.write("Upload a potato leaf image to predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")