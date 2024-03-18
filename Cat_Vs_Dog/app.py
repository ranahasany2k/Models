import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the model
model = load_model('my_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((150, 150))
    img_array = np.array(resized_image)
    # Normalize pixel values to range [0, 1]
    img_array = img_array.astype('float32') / 255.0
    # Expand dimensions to match model input shape (150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def make_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

def main():
    st.title('Dog or Cat Recognition')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        prediction = make_prediction(image)
        if prediction[0][0] > 0.5:
            st.write("# Prediction: Dog")
        else:
            st.write("# Prediction: Cat")


if __name__ == '__main__':
    main()
