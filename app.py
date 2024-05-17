import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define class names (ensure these match your training data)
class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']  # Replace with actual class names

# Function to make predictions
def predict(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch axis

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Streamlit app
st.title('Tomato Disease Classifier')
st.write("Upload an image of a tomato leaf to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predicted_class, confidence = predict(uploaded_file)
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
