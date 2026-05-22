import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown

# Google Drive model URL
MODEL_URL = "https://drive.google.com/uc?id=1bD1VlzRukO_2yM4WOV4X_FOgvT_nivT7"

MODEL_PATH = "pneumonia_model.keras"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Title
st.title("AI Medical Diagnosis System")

st.write("Upload a Chest X-ray Image")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    # Resize image
    img = img.resize((224, 224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    st.subheader("Prediction Result")

    if prediction[0][0] > 0.5:

        confidence = prediction[0][0] * 100

        st.error("PNEUMONIA DETECTED")

        st.write(f"Confidence: {confidence:.2f}%")

    else:

        confidence = (1 - prediction[0][0]) * 100

        st.success("NORMAL")

        st.write(f"Confidence: {confidence:.2f}%")
