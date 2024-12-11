import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import gdown
import os
import kagglehub
import pickle

# Download latest version
path = kagglehub.dataset_download("quandang/vietnamese-foods")

model_url = "https://drive.google.com/uc?id=1xK3SpZGOwPDPqasMmrE5VaRDeSqXZRjz"

model_path = 'vietnamese_food_model.pkl'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

train_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join(path, 'Images/Train')
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())

st.title("Ứng dụng nhận diện món ăn Việt Nam")

st.write("Hãy tải lên một ảnh món ăn Việt Nam để nhận diện.")

uploaded_file = st.file_uploader("Chọn ảnh món ăn", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã tải lên", use_container_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    st.write(f"Dự đoán: {class_names[predicted_class[0]]}")
    st.write(f"Xác suất: {predictions[0][predicted_class[0]]:.2f}")
