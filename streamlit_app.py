import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

# Load the model
model = load_model("plant_disease_model.h5")
classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']  # Replace with your actual class names

st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(128, 128))
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    st.subheader(f"Predicted Disease: {predicted_class}")
    st.write(f"Confidence: {np.max(prediction)*100:.2f}%")
