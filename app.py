import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model (ensure the model file is in the same directory or provide the correct path)
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("model.keras")  # model path
    return model

model = load_model()

# Function to make predictions
def cancerPrediction(img):
    # Resize image to match model's expected sizing
    img = img.resize((256, 256))
    
    # Convert image to array
    img_array = image.img_to_array(img) / 255.0  # Normalize
    input_arr_img = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch processing
    
    # Make prediction
    pred = (model.predict(input_arr_img) > 0.5).astype(int)[0][0]
    return "Cancer" if pred == 0 else "Non-Cancer"

# Streamlit UI
st.title("Oral Cancer Detection App")
st.write("Upload an image to check if it is cancerous or non-cancerous.")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)
    
    # Display image
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Analyzing")
    
    # Get prediction
    result = cancerPrediction(img)
    st.subheader(f"Prediction: {result}")
