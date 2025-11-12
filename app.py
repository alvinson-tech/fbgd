from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow import keras
import pickle

app = Flask(__name__)

# Configuration
IMG_SIZE = 128
MODEL_PATH = 'model/blood_group_model.h5'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'
UPLOAD_FOLDER = 'uploads'

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

def preprocess_image(image_path):
    """Preprocess the uploaded image"""
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Reshape for model input
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Preprocess image
        processed_img = preprocess_image(file_path)
        
        # Make prediction
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class]) * 100
        
        # Get blood group label
        blood_group = label_encoder.inverse_transform([predicted_class])[0]
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'blood_group': blood_group,
            'confidence': f"{confidence:.2f}%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)