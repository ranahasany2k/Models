import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import json
# Load the digit recognition models
model1_digit = keras.models.load_model('Model_Comparisons/model1.h5')
model2_digit = keras.models.load_model('Model_Comparisons/model2.h5')
model3_digit = keras.models.load_model('Model_Comparisons/model3.h5')

# Load the fashion recognition models
model1_fashion = keras.models.load_model('Model_Comparisons/model1_2.h5')
model2_fashion = keras.models.load_model('Model_Comparisons/model2_2.h5')
model3_fashion = keras.models.load_model('Model_Comparisons/model3_2.h5')

# Load the CIFAR-10 recognition models
model1_cifar = keras.models.load_model('Model_Comparisons/model1_3.h5')
model2_cifar = keras.models.load_model('Model_Comparisons/model2_3.h5')
model3_cifar = keras.models.load_model('Model_Comparisons/model3_3.h5')

# Load digit recognition training history data
with open('Model_Comparisons/history_norm.json', 'r') as file:
    history_norm = json.load(file)

with open('Model_Comparisons/history_norm2.json', 'r') as file:
    history_norm2 = json.load(file)

with open('Model_Comparisons/history_norm3.json', 'r') as file:
    history_norm3 = json.load(file)

# Load fashion recognition training history data
with open('Model_Comparisons/history_norm_2.json', 'r') as file:
    history_norm_2 = json.load(file)

with open('Model_Comparisons/history_norm2_2.json', 'r') as file:
    history_norm2_2 = json.load(file)

with open('Model_Comparisons/history_norm3_2.json', 'r') as file:
    history_norm3_2 = json.load(file)

# Load CIFAR-10 recognition training history data
with open('Model_Comparisons/history_norm_3.json', 'r') as file:
    history_norm_3 = json.load(file)

with open('Model_Comparisons/history_norm2_3.json', 'r') as file:
    history_norm2_3 = json.load(file)

with open('Model_Comparisons/history_norm3_3.json', 'r') as file:
    history_norm3_3 = json.load(file)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, linewidth=0.2, linecolor="black", fmt=".0f")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    st.pyplot(plt)

# Load confusion matrix data from JSON files
def load_confusion_matrix(json_file):
    with open(json_file, 'r') as file:
        cm_data = json.load(file)
    # Reshape 1D array to 2D matrix
    cm_matrix = np.array(cm_data).reshape((10, 10))
    return cm_matrix



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

# CIFAR-10 labels dictionary
cifar_labels_dict = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

# Function to preprocess CIFAR-10 image
def preprocess_cifar_image(image):
    # Convert image to RGB mode (if it's not already in RGB)
    image = image.convert('RGB')
    resized_image = image.resize((32, 32))
    img_array = np.array(resized_image)
    # Convert to float and normalize pixel values to range [0, 1]
    img_array = img_array.astype('float32') / 255.0
    # Reshape to match model input shape (32x32x3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array




# Function to preprocess the image for digit and fashion recognition
def preprocess_image(image):
    resized_image = image.resize((28, 28))
    grayscale_image = resized_image.convert('L')
    img_array = np.array(grayscale_image)
    img_array = img_array / 255.0
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

# Function to make CIFAR-10 recognition prediction
def make_cifar_prediction(image):
    preprocessed_image = preprocess_cifar_image(image)
    prediction1 = model1_cifar.predict(preprocessed_image)
    prediction2 = model2_cifar.predict(preprocessed_image)
    prediction3 = model3_cifar.predict(preprocessed_image)
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
    recognition_type = st.radio("Choose Recognition Type:", ("Digit Recognition", "Fashion Recognition", "CIFAR-10 Recognition"))
    labels1='''0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'''

    labels2='''0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'''
    if recognition_type == "Digit Recognition":
        # Digit recognition code
        uploaded_file = st.file_uploader("Choose a digit image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction1, prediction2, prediction3 = make_digit_prediction(image)
            st.write(f"Prediction according to model 1: {prediction1}")
            st.write(f"Prediction according to model 2: {prediction2}")
            st.write(f"Prediction according to model 3: {prediction3}")

        st.sidebar.title('Digit Recognition Model Training History')
        if st.sidebar.checkbox('Show Model 1 Graph'):
            plot_graph(history_norm, "Model 1")
        if st.sidebar.checkbox('Show Model 2 Graph'):
            plot_graph(history_norm2, "Model 2")
        if st.sidebar.checkbox('Show Model 3 Graph'):
            plot_graph(history_norm3, "Model 3")

        st.sidebar.title('Digit Recognition Confusion Matrix')
        if st.sidebar.checkbox('Show Model 1 Confusion Matrix'):
            cm1 = load_confusion_matrix('Model_Comparisons/cm1.json')
            plot_confusion_matrix(cm1, "Model 1")
        if st.sidebar.checkbox('Show Model 2 Confusion Matrix'):
            cm2 = load_confusion_matrix('Model_Comparisons/cm2.json')
            plot_confusion_matrix(cm2, "Model 2")
        if st.sidebar.checkbox('Show Model 3 Confusion Matrix'):
            cm3 = load_confusion_matrix('Model_Comparisons/cm3.json')
            plot_confusion_matrix(cm3, "Model 3")

    elif recognition_type == "Fashion Recognition":
        # Fashion recognition code
        uploaded_file = st.file_uploader("Choose a fashion image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction1, prediction2, prediction3 = make_fashion_prediction(image)
            st.write(f"Prediction according to model 1: {labels_dict[prediction1]}")
            st.write(f"Prediction according to model 2: {labels_dict[prediction2]}")
            st.write(f"Prediction according to model 3: {labels_dict[prediction3]}")

        st.sidebar.title('Fashion Recognition Model Training History')
        if st.sidebar.checkbox('Show Model 1 Graph'):
            plot_graph(history_norm_2, "Model 1")
        if st.sidebar.checkbox('Show Model 2 Graph'):
            plot_graph(history_norm2_2, "Model 2")
        if st.sidebar.checkbox('Show Model 3 Graph'):
            plot_graph(history_norm3_2, "Model 3")
        st.sidebar.title('Fastion Recognition Confusion Matrix')
        if st.sidebar.checkbox('Show Model 1 Confusion Matrix'):
            cm1 = load_confusion_matrix('Model_Comparisons/cm1_2.json')
            st.markdown(f"'''{labels1}'''")
            plot_confusion_matrix(cm1, "Model 1")

        if st.sidebar.checkbox('Show Model 2 Confusion Matrix'):
            cm2 = load_confusion_matrix('Model_Comparisons/cm2_2.json')
            st.markdown(f"'''{labels1}'''")
            plot_confusion_matrix(cm2, "Model 2")
        
        if st.sidebar.checkbox('Show Model 3 Confusion Matrix'):
            cm3 = load_confusion_matrix('Model_Comparisons/cm3_2.json')
            st.markdown(f"'''{labels1}'''")
            plot_confusion_matrix(cm3, "Model 3")

    elif recognition_type == "CIFAR-10 Recognition":
        # CIFAR-10 recognition code
        uploaded_file = st.file_uploader("Choose a CIFAR-10 image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction1, prediction2, prediction3 = make_cifar_prediction(image)
            st.write(f"Prediction according to model 1: {cifar_labels_dict[prediction1]}")
            st.write(f"Prediction according to model 2: {cifar_labels_dict[prediction2]}")
            st.write(f"Prediction according to model 3: {cifar_labels_dict[prediction3]}")

        st.sidebar.title('CIFAR-10 Recognition Model Training History')
        if st.sidebar.checkbox('Show Model 1 Graph'):
            plot_graph(history_norm_3, "Model 1")
        if st.sidebar.checkbox('Show Model 2 Graph'):
            plot_graph(history_norm2_3, "Model 2")
        if st.sidebar.checkbox('Show Model 3 Graph'):
            plot_graph(history_norm3_3, "Model 3")
        st.sidebar.title('Digit Recognition Confusion Matrix')
        if st.sidebar.checkbox('Show Model 1 Confusion Matrix'):
            cm1 = load_confusion_matrix('Model_Comparisons/cm1_3.json')
            st.markdown(f"'''{labels2}'''")
            plot_confusion_matrix(cm1, "Model 1")
        if st.sidebar.checkbox('Show Model 2 Confusion Matrix'):
            cm2 = load_confusion_matrix('Model_Comparisons/cm2_3.json')
            st.markdown(f"'''{labels2}'''")
            plot_confusion_matrix(cm2, "Model 2")
        if st.sidebar.checkbox('Show Model 3 Confusion Matrix'):
            cm3 = load_confusion_matrix('Model_Comparisons/cm3_3.json')
            st.markdown(f"'''{labels2}'''")
            plot_confusion_matrix(cm3, "Model 3")

if __name__ == '__main__':
    main()
