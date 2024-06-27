import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the model
model = load_model("dog_cat_class.h5")

# Define a function to make predictions
def predict_image(img_path):
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] > 0.5:
        return 'dog'
    else:
        return 'cat'

# Streamlit app
st.title("Dog and Cat Classifier")
# st.markdown("Upload an image to classify it as a dog or a cat.")

uploaded_file = st.file_uploader("Upload an image to classify it as a dog or a cat.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)  # Create two columns with equal width

    with col1:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    with col2:
        st.write("")
        st.write("Classifying...")
        label = predict_image(uploaded_file)
        st.subheader("Result")
        st.write(f"The image is classified as: **{label}**")

# Add some padding at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
