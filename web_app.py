import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Set up the page layout
st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon=":eyeglasses:", layout="wide")

# Load the trained model
@st.cache_resource
def load_keras_model():
    return load_model("my_model (1).keras")

model = load_keras_model()

# Function to preprocess the image
def preprocess_image(image, target_size):
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)  # Resize and maintain aspect ratio
    image = np.array(image)  # Convert to numpy array
    if image.shape[2] == 4:  # If the image has an alpha channel, remove it
        image = image[..., :3]
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🩺 Diabetic Retinopathy Detection")
st.markdown("""
    Detect **diabetic retinopathy** from uploaded retinal images using a deep learning model.
    Upload a clear retinal image below, and the model will predict whether diabetic retinopathy is present.
""")

# Upload image section
st.sidebar.title("Upload Section")
st.sidebar.markdown("Please upload a retinal image in `.jpg`, `.jpeg`, or `.png` format.")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Retinal Image", use_column_width=True)
    st.write("")

    # Display a progress bar
    with st.spinner("Processing the image..."):
        # Preprocess the image
        processed_image = preprocess_image(Image.open(uploaded_file), target_size=(224, 224))

    st.success("Image successfully preprocessed!")

    # Predict using the trained model
    st.write("🤖 **Model Prediction**")
    prediction = model.predict(processed_image)
    

    # Since the output of model.predict is a numpy array, use indexing and threshold properly
    result = prediction[0][0]  # Access the first element of the prediction
    #st.write(f"Raw prediction value: `{result}`")

    # Display prediction result based on a threshold (e.g., 0.5 for binary classification)
    if result< 0.5:
        st.markdown("<h2 style='color:green;'>🟢 The person does not have diabetic retinopathy.</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:red;'>🔴 The person has diabetic retinopathy.</h2>", unsafe_allow_html=True)

    # Add a confidence bar for better UI experience
    st.progress(float(result))  # Progress bar showing prediction confidence

else:
    st.warning("Please upload an image to begin the prediction.")

# Footer section
st.markdown("""
    ---
    ### How to Use:
    - Upload a retinal image using the uploader in the sidebar.
    - The model will process the image and provide a prediction along with the confidence score.
    - This is a binary classification where:
        - A prediction below **0.5** means **No Diabetic Retinopathy**.
        - A prediction above **0.5** means **Diabetic Retinopathy Detected**.
    
    ### Model Info:
    - **Model Type**: Deep Learning CNN for Diabetic Retinopathy Detection.
    - **Input Size**: 224x224 Retinal Image.
""")



