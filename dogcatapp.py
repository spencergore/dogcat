import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import subprocess

try:
    import requests
except ImportError:
    subprocess.call(["pip", "install", "requests"])
    import requests

MODEL_URL = "https://drive.google.com/uc?export=download&id=1HLhTsifwbG_AK7D29WDObXoYpFeTkL9D"

# Download model
@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL, stream=True)
    with open("cat_dog_classifier.h5", "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return tf.keras.models.load_model("cat_dog_classifier.h5")

# Load the model
model = load_model()
# Load the trained model

# Define image size (same as CNN input size)
IMG_SIZE = (150, 150)

# Title of the app
st.title("🐶🐱 Spencer's Cat vs. Dog Classifier")

st.write("Upload an image of a **cat or dog**, and this AI model will predict what it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = np.array(image)
    img = cv2.resize(img, IMG_SIZE)  # Resize to 150x150
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img)[0][0]

    # Display result
    if prediction > 0.5:
        st.success("🐶 This is a **dog!**")
    else:
        st.success("🐱 This is a **cat!**")
