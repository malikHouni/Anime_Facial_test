import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from modelCreation import predict_top_emotions  # Import the function from your prediction module

# Load your trained model
model = load_model('manga_face_generator_retrained.h5')

# Mapping emotional expressions to directory names
EMOTION_DIRS = {
    "Angry": "angry",
    "Crying": "crying",
    "Embarrassed": "embarrassed",
    "Happy": "happy",
    "Pleased": "pleased",
    "Sad": "sad",
    "Shock": "shock",
    "Blushed":"blushed",
    "Confused":"confused",
    "Normal":"normal",
    "Pout":"pout"  
}

# Load CSS file for styling
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Load the custom CSS
load_css('assets/style.css')
# Function to recommend images based on selected expression
def recommend_image(emotion):
    if emotion in EMOTION_DIRS:
        dir_path = os.path.join('img2', EMOTION_DIRS[emotion])  # Specify your data path
        images = os.listdir(dir_path)
        if images:
            random_image = np.random.choice(images)  # Random image from directory
            image_path = os.path.join(dir_path, random_image)
            return image_path
    return None

# Function to predict the emotion from the uploaded image
def predict_expression(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Adjust target size as needed
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Streamlit app
st.title("Manga Face Emotion Recommender")

# Create two columns
col1, col2 = st.columns(2)

# Left column for emotion selection
with col1:
    st.header("Select an Emotion")
    selected_emotion = st.selectbox(
        "Choose an emotion:",
        list(EMOTION_DIRS.keys())
    )
    
    if st.button("Recommend Image"):
        if selected_emotion:
            image_path = recommend_image(selected_emotion)
            if image_path:
                st.image(image_path, caption=f"{selected_emotion} face", use_column_width=True)
            else:
                st.error("No images found for this emotion.")

# Right column for image upload and prediction
with col2:
    st.header("Upload an Image to Predict Emotion")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict Emotion"):
            # Save uploaded file temporarily to predict
            temp_image_path = "temp_image.jpg"
            img.save(temp_image_path)

            top_classes, top_probs = predict_top_emotions(temp_image_path)
            for cls, prob in zip(top_classes, top_probs):
                st.write(f"The model predicts: **{cls}** with confidence {prob:.2f}.")


