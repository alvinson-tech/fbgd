# 🩸 Blood Group Detection from Fingerprints

A deep learning-based web application that predicts blood groups from fingerprint images using Convolutional Neural Networks (CNN).

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Code Explanation](#code-explanation)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project implements a machine learning system that can identify a person's blood group by analyzing their fingerprint patterns. The system uses a Convolutional Neural Network (CNN) trained on fingerprint images to classify them into one of eight blood groups: A+, A-, B+, B-, AB+, AB-, O+, and O-.

### Why Fingerprints for Blood Group Detection?

Research suggests that certain ridge patterns and characteristics in fingerprints may correlate with blood groups. This project explores this relationship using deep learning to extract and learn these patterns automatically.

## 📊 Dataset

**Source:** [Kaggle - Finger Print Based Blood Group Dataset](https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset/data)

### Dataset Structure
The dataset contains fingerprint images organized by blood group:
```
dataset/
├── A+/
│   ├── fingerprint1.bmp
│   ├── fingerprint2.bmp
│   └── ...
├── A-/
├── B+/
├── B-/
├── AB+/
├── AB-/
├── O+/
└── O-/
```

### Image Specifications
- **Format:** BMP, JPG, PNG, TIFF
- **Type:** Grayscale fingerprint images
- **Content:** Clear fingerprint patterns with visible ridges and valleys

## 🔬 How It Works

### 1. **Data Collection & Preprocessing**
- Fingerprint images are loaded from organized folders (one per blood group)
- Images are converted to grayscale (if not already)
- Resized to a uniform 128x128 pixels for consistency
- Pixel values normalized to 0-1 range for better model training

### 2. **Model Training**
- A Convolutional Neural Network (CNN) learns to identify patterns
- The network extracts features like ridge patterns, minutiae points, and spatial relationships
- Multiple layers progressively learn from simple edges to complex patterns
- Training occurs over 50 epochs with 80-20 train-test split

### 3. **Prediction**
- User uploads a fingerprint image through the web interface
- Image undergoes the same preprocessing (grayscale, resize, normalize)
- CNN analyzes the fingerprint and predicts the blood group
- Results displayed with confidence percentage

## 📁 Project Structure

```
blood-group-detection/
│
├── app.py                      # Flask web application
├── train_model.py              # Model training script
├── index.html                  # Web interface (in templates/)
│
├── dataset/                    # Training data
│   ├── A+/
│   ├── A-/
│   ├── B+/
│   ├── B-/
│   ├── AB+/
│   ├── AB-/
│   ├── O+/
│   └── O-/
│
├── model/                      # Trained model files
│   ├── blood_group_model.h5    # Trained CNN model
│   └── label_encoder.pkl       # Blood group label encoder
│
├── uploads/                    # Temporary upload folder
└── templates/                  # HTML templates
    └── index.html
```

## 🏗️ Technical Architecture

### Convolutional Neural Network (CNN) Architecture

```
Input Layer (128x128x1 grayscale image)
    ↓
Conv2D Layer (32 filters, 3x3) + ReLU
    ↓
MaxPooling (2x2)
    ↓
Conv2D Layer (64 filters, 3x3) + ReLU
    ↓
MaxPooling (2x2)
    ↓
Conv2D Layer (128 filters, 3x3) + ReLU
    ↓
MaxPooling (2x2)
    ↓
Flatten Layer
    ↓
Dense Layer (128 neurons) + ReLU
    ↓
Dropout (50%)
    ↓
Output Layer (8 neurons - softmax)
```

### Why This Architecture?

- **Convolutional Layers:** Extract spatial features from fingerprints (ridges, valleys, patterns)
- **MaxPooling:** Reduces dimensionality while retaining important features
- **Multiple Filters:** Progressively increase (32→64→128) to learn complex patterns
- **Dropout:** Prevents overfitting by randomly deactivating neurons during training
- **Softmax Output:** Provides probability distribution across 8 blood groups

## 💻 Code Explanation

### 1. `train_model.py` - Model Training Script

**Purpose:** Trains the CNN model on fingerprint dataset

**Key Functions:**

#### `load_dataset(dataset_path)`
```python
def load_dataset(dataset_path):
    images = []
    labels = []
    # Loops through each blood group folder
    # Loads all fingerprint images
    # Converts to grayscale and resizes to 128x128
    # Normalizes pixel values (0-1 range)
    return np.array(images), np.array(labels)
```
- **What it does:** Reads all fingerprint images from organized folders
- **Processing:** Grayscale conversion → Resize → Normalize
- **Output:** Arrays of images and corresponding blood group labels

#### `create_model(num_classes)`
```python
def create_model(num_classes):
    model = keras.Sequential([
        # Input layer for 128x128 grayscale images
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # 3 Convolutional blocks (extract features)
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # ... more layers
        
        # Classification layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```
- **What it does:** Builds the neural network architecture
- **Layers Explained:**
  - **Conv2D:** Detects patterns in images (edges, ridges, etc.)
  - **MaxPooling:** Reduces image size while keeping important features
  - **Flatten:** Converts 2D features to 1D array
  - **Dense:** Fully connected layer for final classification
  - **Dropout:** Prevents overfitting
  - **Softmax:** Outputs probability for each blood group

#### `main()` - Training Pipeline
1. Loads dataset from folders
2. Encodes blood group labels (A+ → 0, A- → 1, etc.)
3. Splits data into training (80%) and testing (20%)
4. Creates and compiles the CNN model
5. Trains for 50 epochs with batch size of 32
6. Evaluates accuracy on test data
7. Saves trained model and label encoder

**Training Process:**
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Categorical crossentropy (multi-class classification)
- **Metric:** Accuracy
- **Epochs:** 50 iterations through entire dataset
- **Batch Size:** 32 images processed simultaneously

### 2. `app.py` - Flask Web Application

**Purpose:** Serves the web interface and handles predictions

**Key Functions:**

#### `preprocess_image(image_path)`
```python
def preprocess_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 128x128
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize (0-1 range)
    img = img / 255.0
    # Reshape for model input
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img
```
- **What it does:** Prepares uploaded image for prediction
- **Steps:** Same preprocessing as training (consistency is crucial)
- **Output:** Formatted array ready for CNN

#### `@app.route('/predict', methods=['POST'])`
```python
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Receive uploaded file
    file = request.files['file']
    
    # 2. Save temporarily
    file.save(file_path)
    
    # 3. Preprocess image
    processed_img = preprocess_image(file_path)
    
    # 4. Make prediction
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[0][predicted_class]) * 100
    
    # 5. Convert prediction to blood group
    blood_group = label_encoder.inverse_transform([predicted_class])[0]
    
    # 6. Return result as JSON
    return jsonify({
        'blood_group': blood_group,
        'confidence': f"{confidence:.2f}%"
    })
```

**Prediction Flow:**
1. User uploads fingerprint image
2. Image saved temporarily to disk
3. Image preprocessed (grayscale, resize, normalize)
4. CNN analyzes image and outputs 8 probabilities
5. Highest probability selected as prediction
6. Numeric prediction converted back to blood group label
7. Result sent back to web interface

### 3. `index.html` - Web Interface

**Purpose:** User-friendly interface for uploading fingerprints

**Key Features:**
- **Drag & Drop Upload:** Intuitive file selection
- **Image Preview:** Shows uploaded fingerprint before analysis
- **Real-time Processing:** Loading indicator during prediction
- **Results Display:** Blood group with confidence percentage
- **Responsive Design:** Works on desktop and mobile devices

**JavaScript Functionality:**
```javascript
// Handle file upload
function handleFile(file) {
    // Display preview of uploaded image
    // Enable prediction button
}

// Send to backend for prediction
predictBtn.addEventListener('click', async () => {
    // Create form data with image
    // Send POST request to /predict endpoint
    // Display results
});
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Step 1: Clone or Download Project
```bash
git clone <repository-url>
cd blood-group-detection
```

### Step 2: Install Dependencies
```bash
pip install flask tensorflow opencv-python numpy scikit-learn
```

### Step 3: Download Dataset
1. Download from [Kaggle](https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset/data)
2. Extract and place in project root as `dataset/`
3. Ensure folder structure matches:
```
dataset/
├── A+/
├── A-/
├── B+/
├── B-/
├── AB+/
├── AB-/
├── O+/
└── O-/
```

### Step 4: Train the Model
```bash
python train_model.py
```
This will:
- Load all fingerprint images
- Train the CNN model (takes 10-30 minutes depending on hardware)
- Save trained model to `model/blood_group_model.h5`
- Save label encoder to `model/label_encoder.pkl`

### Step 5: Run Web Application
```bash
python app.py
```
Access at: `http://localhost:5001`

## 📖 Usage

1. **Open Web Browser:** Navigate to `http://localhost:5001`
2. **Upload Fingerprint:** Click "Choose File" or drag & drop image
3. **Preview:** Verify the uploaded fingerprint image
4. **Detect:** Click "🔍 Detect Blood Group" button
5. **View Results:** Blood group displayed with confidence percentage

### Tips for Best Results
- Use clear, high-quality fingerprint images
- Ensure good lighting and contrast
- Capture complete fingerprint (not partial)
- Supported formats: BMP, JPG, PNG, TIFF

## 📊 Model Performance

### Training Configuration
- **Epochs:** 50
- **Batch Size:** 32
- **Train-Test Split:** 80-20
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

### Expected Accuracy
- Training accuracy varies based on dataset quality and size
- Typical range: 70-90% depending on data
- Performance displayed after training completes

### Improving Accuracy
- Add more training images per blood group
- Use data augmentation (rotation, flipping)
- Adjust model architecture (more layers, different filters)
- Increase training epochs
- Fine-tune hyperparameters

## 🛠️ Technologies Used

### Backend
- **Flask:** Web framework for Python
- **TensorFlow/Keras:** Deep learning framework
- **OpenCV:** Image processing library
- **NumPy:** Numerical computing
- **scikit-learn:** Machine learning utilities (LabelEncoder, train_test_split)

### Frontend
- **HTML5:** Structure and markup
- **CSS3:** Styling and animations
- **JavaScript:** Interactive functionality
- **Fetch API:** Asynchronous HTTP requests

### Machine Learning
- **CNN (Convolutional Neural Network):** Image classification
- **Supervised Learning:** Labeled training data
- **Multi-class Classification:** 8 blood group categories

## 🔍 Understanding Key Concepts

### What is a CNN?
A Convolutional Neural Network is a type of artificial intelligence specifically designed for image analysis. It mimics how the human visual system works by:
1. Detecting simple features (edges, lines)
2. Combining them into complex patterns (shapes, textures)
3. Making decisions based on learned patterns

### How Does Training Work?
1. **Forward Pass:** Image fed through network, prediction made
2. **Loss Calculation:** How wrong was the prediction?
3. **Backward Pass:** Adjust network weights to reduce error
4. **Repeat:** Process thousands of images many times
5. **Result:** Network learns to recognize fingerprint patterns

### What is Confidence Score?
The percentage indicates how certain the model is about its prediction. For example:
- 95% confidence: Very sure about the prediction
- 60% confidence: Less certain, might need better image quality

## 🎓 Learning Outcomes

This project demonstrates:
- Image classification using deep learning
- CNN architecture design and implementation
- Data preprocessing and normalization
- Model training and evaluation
- Web application development with Flask
- Integration of ML models into web interfaces

## 📝 Notes & Considerations

### Important Disclaimers
⚠️ **This is an educational project for learning machine learning concepts**
- Not intended for medical diagnosis
- Accuracy depends on dataset quality and size
- Should not replace laboratory blood tests
- Use only for educational and research purposes

### Limitations
- Requires substantial training data for high accuracy
- Performance varies with image quality
- May not generalize to all fingerprint types
- Environmental factors affect fingerprint capture

### Future Enhancements
- Add data augmentation for better generalization
- Implement ensemble models for higher accuracy
- Add user authentication and history tracking
- Mobile app development
- Real-time camera capture
- Multi-model comparison

## 🤝 Contributing

This project is for educational purposes. Feel free to:
- Experiment with different architectures
- Try various preprocessing techniques
- Implement data augmentation
- Improve the user interface
- Add new features

## 📧 Support

For questions or issues related to:
- **Dataset:** Visit [Kaggle dataset page](https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset/data)
- **Code:** Review comments in source files
- **Concepts:** Research CNN and image classification tutorials

## 📜 License

Educational project - check dataset license on Kaggle for usage terms.

---

**Built with 🩸 for learning Deep Learning and Computer Vision**