import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Define image size (same as CNN input size)
IMG_SIZE = (150, 150)

# Title of the app
st.title("🐶🐱 Cat vs. Dog Classifier")

st.write("Upload an image of a **cat or dog**, and this AI model will predict what it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

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
