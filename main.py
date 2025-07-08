import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('resnet_model.h5')

def preprocess_image(img):
    """Resize and normalize the uploaded image."""
    # Convert image to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((256, 256))  # Adjust size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required by your model
    return img_array

def predict_image(model, img_array):
    """Predict if the image is a deepfake or not."""
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title("Deepfake Detection App")
st.write("Upload an image to detect if it's a deepfake.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_array = preprocess_image(image_data)

    # Make prediction
    prediction = predict_image(model, img_array)

    # Display result
    st.write("Prediction:")
    confidence_score = float(prediction[0])  # Convert NumPy array to a float
    if confidence_score > 0.35:  # Adjust threshold
        st.write(f"This image has a Confidence Score of {confidence_score * 100:.2f}% of being Real.")
    else:
        st.write(f"This image has a Confidence Score of {100 - (confidence_score * 100):.2f}% of being a Deepfake.")