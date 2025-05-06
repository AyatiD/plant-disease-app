from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model("plant_disease_model.h5")

classes = ['Early Blight', 'Late Blight', 'Healthy']

def preprocess_image(image):
    img = image.resize((128, 128))  # match model input shape
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file"
    file = request.files['file']
    img = Image.open(file.stream)
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = classes[np.argmax(prediction)]
    return f"Prediction: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
