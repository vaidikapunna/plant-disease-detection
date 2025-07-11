import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import json
import gtts
import tempfile

# Load trained model
model = tf.keras.models.load_model("improved_plant_disease_model.h5")

# Get class labels
class_labels = sorted(os.listdir("dataset/train"))

# Load cure info
with open("cures_multilang.json", "r", encoding="utf-8") as f:
    cure_data = json.load(f)

# Language codes for gTTS
languages = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Marathi": "mr"
}

# UI
st.title(" Plant Disease Detection")
st.write("Upload a leaf image, choose your language, and get the prediction with simple cure tips!")

uploaded_file = st.file_uploader(" Choose a leaf image", type=["jpg", "jpeg", "png"])
selected_lang = st.selectbox(" Select your language", list(languages.keys()))
lang_code = languages[selected_lang]

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((128, 128))
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Check if it's a healthy leaf
    if "healthy" in predicted_class.lower():
        st.success(" No disease detected. The plant is healthy!")

        disease_info = cure_data.get(predicted_class, {}).get(selected_lang) or \
                       cure_data.get(predicted_class, {}).get("English")

        if isinstance(disease_info, dict):
            tips = disease_info.get("tips", [])
            images = disease_info.get("images", [])
            videos = disease_info.get("videos", [])

            st.subheader(f" Care Tips in {selected_lang}:")
            for tip in tips:
                st.markdown(f"- {tip}")

            if images:
                st.subheader(" Tip Images:")
                for img_url in images:
                    st.image(img_url, use_column_width=True)

            if videos:
                st.subheader(" Tip Videos:")
                for video_url in videos:
                    st.video(video_url)

            if tips:
                if st.button(" Listen to tips"):
                    tts = gtts.gTTS(" ".join(tips), lang=lang_code)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        tts.save(f.name)
                        st.audio(f.name, format='audio/mp3')
        elif isinstance(disease_info, str):
            st.info(disease_info)
    else:
        st.success(f" Predicted Disease: **{predicted_class}**")

        disease_info = cure_data.get(predicted_class, {}).get(selected_lang) or \
                       cure_data.get(predicted_class, {}).get("English")

        if disease_info:
            if isinstance(disease_info, dict):
                tips = disease_info.get("tips", [])
                images = disease_info.get("images", [])
                videos = disease_info.get("videos", [])

                st.subheader(f" Cure Tips in {selected_lang}:")
                for tip in tips:
                    st.markdown(f"- {tip}")

                if images:
                    st.subheader(" Tip Images:")
                    for img_url in images:
                        st.image(img_url, use_column_width=True)

                if videos:
                    st.subheader(" Tip Videos:")
                    for video_url in videos:
                        st.video(video_url)

                if tips:
                    if st.button(" Listen to tips"):
                        tts = gtts.gTTS(" ".join(tips), lang=lang_code)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                            tts.save(f.name)
                            st.audio(f.name, format='audio/mp3')
            else:
                st.info(disease_info)
        else:
            st.warning(" No cure info found for this disease. Please update the database.")