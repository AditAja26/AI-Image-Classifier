import streamlit as st 
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Set title
st.title("Simple Image Classifier")
st.write("Upload an image, and the AI will try to guess what it is!")

# Load model
@st.cache_resource
def load_model():
    """Loads the pre-trained MobileNetV2 model."""
    model = mobilenet_v2.MobileNetV2(weights='imagenet')
    return model

model = load_model()

# Image Preprocessing
def preprocess_image(img_pil):
    """
    Takes a PIL image, resizes it to 224x224,
    and formats it for the model.
    """

    img_resized = img_pil.resize((224,224))
    img_array = keras_image.img_to_array(img_resized)
    img_expanded = np.expand_dims(img_array, axis=0)

    return preprocess_input(img_expanded)

# Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    processed_image = preprocess_image(pil_image)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    #Display result

    st.image(pil_image, caption='Your uploaded image', use_container_width=True)
    st.subheader("Here's what I think it is:")

    for i, (imagenet_id, label, score) in enumerate (decoded_predictions):
        percentage = score*100
        st.write(f"{i+1}. **{label.replace('_', ' ')}** ({percentage:.2f}%)")

