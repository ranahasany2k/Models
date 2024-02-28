import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

import tensorflow as tf
import numpy as np
from PIL import Imag
import matplotlib.pyplot as plt

# Load the model
model1 = tf.keras.models.load_model('model1.h5')
model2 = tf.keras.models.load_model('model2.h5')
model3 = tf.keras.models.load_model('model3.h5')

# Load training history data
with open('history_norm.json', 'r') as file:
    history_norm = json.load(file)

with open('history_norm2.json', 'r') as file:
    history_norm2 = json.load(file)

with open('history_norm3.json', 'r') as file:
    history_norm3 = json.load(file)

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((28, 28))
    grayscale_image = resized_image.convert('L')
    img_array = np.array(grayscale_image)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, 28, 28))
    return img_array

# Functions to make predictions
def make_prediction(model, image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction)

# Function to plot training history
def plot_graph(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], color='b', label='Training Accuracy')
    plt.plot(history['val_accuracy'], color='r', label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], color='b', label='Training Loss')
    plt.plot(history['val_loss'], color='r', label='Validation Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")

    st.pyplot(plt)

def main():
    st.title('Digit Recognition App')

    # File uploader for user to upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions using all models
        prediction1 = make_prediction(model1, image)
        prediction2 = make_prediction(model2, image)
        prediction3 = make_prediction(model3, image)

        # Display the predictions
        st.write(f"Prediction according to model 1: {prediction1}")
        st.write(f"Prediction according to model 2: {prediction2}")
        st.write(f"Prediction according to model 3: {prediction3}")

    # Add tabs for plotting training history graphs
    st.sidebar.title('Model Training History')

    if st.sidebar.checkbox('Show Model 1 Graph'):
        plot_graph(history_norm, "Model 1")

    if st.sidebar.checkbox('Show Model 2 Graph'):
        plot_graph(history_norm2, "Model 2")

    if st.sidebar.checkbox('Show Model 3 Graph'):
        plot_graph(history_norm3, "Model 3")

if __name__ == '__main__':
    main()
