import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown

model_url = 'https://drive.google.com/uc?id=17O6WtPID3z9SQtdSXdL9GfRIEF60mg3T'
model_filename = 'trained_model.h5'
gdown.download(model_url, model_filename, quiet=False)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_filename)
    image = Image.open(io.BytesIO(test_image.read()))
    image = image.resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Streamlit Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Pages
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                result_index = model_prediction(test_image)
                class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', ...]
                st.success(f"Our Model is Predicting it's a {class_names[result_index]}")
