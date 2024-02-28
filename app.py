import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the digit recognition models
model1_digit = load_model('model1.h5')
model2_digit = load_model('model2.h5')
model3_digit = load_model('model3.h5')

# Load the fashion recognition models
model1_fashion = load_model('model1_2.h5')
model2_fashion = load_model('model2_2.h5')
model3_fashion = load_model('model3_2.h5')

# Load digit recognition training history data
import json

with open('history_norm.json', 'r') as file:
    history_norm = json.load(file)

with open('history_norm2.json', 'r') as file:
    history_norm2 = json.load(file)

with open('history_norm3.json', 'r') as file:
    history_norm3 = json.load(file)

# Load fashion recognition training history data
with open('history_norm_2.json', 'r') as file:
    history_norm_2 = json.load(file)

with open('history_norm2_2.json', 'r') as file:
    history_norm2_2 = json.load(file)

with open('history_norm3_2.json', 'r') as file:
    history_norm3_2 = json.load(file)

# Clothing labels dictionary
labels_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28
    resized_image = image.resize((28, 28))
    # Convert image to grayscale
    grayscale_image = resized_image.convert('L')
    # Convert image to array
    img_array = np.array(grayscale_image)
    # Normalize the pixel values
    img_array = img_array / 255.0
    # Reshape the array to match model input shape
    img_array = img_array.reshape((1, 28, 28))
    return img_array

# Function to make digit recognition prediction
def make_digit_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction1 = model1_digit.predict(preprocessed_image)
    prediction2 = model2_digit.predict(preprocessed_image)
    prediction3 = model3_digit.predict(preprocessed_image)
    return np.argmax(prediction1), np.argmax(prediction2), np.argmax(prediction3)

# Function to make fashion recognition prediction
def make_fashion_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction1 = model1_fashion.predict(preprocessed_image)
    prediction2 = model2_fashion.predict(preprocessed_image)
    prediction3 = model3_fashion.predict(preprocessed_image)
    return np.argmax(prediction1), np.argmax(prediction2), np.argmax(prediction3)

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
    st.title('Recognition App')

    # Choose recognition type
    recognition_type = st.radio("Choose Recognition Type:", ("Digit Recognition", "Fashion Recognition"))

    if recognition_type == "Digit Recognition":
        # File uploader for digit recognition
        uploaded_file = st.file_uploader("Choose a digit image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make digit recognition prediction
            prediction1, prediction2, prediction3 = make_digit_prediction(image)

            # Display the prediction
            st.write(f"Prediction according to model 1: {prediction1}")
            st.write(f"Prediction according to model 2: {prediction2}")
            st.write(f"Prediction according to model 3: {prediction3}")

        # Add tabs for plotting digit recognition training history graphs
        st.sidebar.title('Digit Recognition Model Training History')

        if st.sidebar.checkbox('Show Model 1 Graph'):
            plot_graph(history_norm, "Model 1")

        if st.sidebar.checkbox('Show Model 2 Graph'):
            plot_graph(history_norm2, "Model 2")

        if st.sidebar.checkbox('Show Model 3 Graph'):
            plot_graph(history_norm3, "Model 3")

    elif recognition_type == "Fashion Recognition":
        # File uploader for fashion recognition
        uploaded_file = st.file_uploader("Choose a fashion image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make fashion recognition prediction
            prediction1, prediction2, prediction3 = make_fashion_prediction(image)

            # Display the prediction
            st.write(f"Prediction according to model 1: {labels_dict[prediction1]}")
            st.write(f"Prediction according to model 2: {labels_dict[prediction2]}")
            st.write(f"Prediction according to model 3: {labels_dict[prediction3]}")

        # Add tabs for plotting fashion recognition training history graphs
        st.sidebar.title('Fashion Recognition Model Training History')

        if st.sidebar.checkbox('Show Model 1 Graph'):
            plot_graph(history_norm_2, "Model 1")

        if st.sidebar.checkbox('Show Model 2 Graph'):
            plot_graph(history_norm2_2, "Model 2")

        if st.sidebar.checkbox('Show Model 3 Graph'):
            plot_graph(history_norm3_2, "Model 3")

if __name__ == '__main__':
    main()
