import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("cifar10_model.keras")

model = load_my_model()

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image. The app will resize it to 32x32 and predict the class using your TensorFlow CNN.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    resized_image = image.resize((32, 32))
    st.image(resized_image, caption="Resized to 32x32", width=150)

    img_array = np.array(resized_image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions)) * 100

    st.subheader("Prediction")
    st.write(f"Predicted class: **{predicted_label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("All class probabilities")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {predictions[0][i]:.4f}")