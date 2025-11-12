import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import pickle

# Configuration
IMG_SIZE = 128
DATASET_PATH = 'dataset'
MODEL_PATH = 'model'

def load_dataset(dataset_path):
    """Load fingerprint images and labels from dataset folder"""
    images = []
    labels = []
    
    print("Loading dataset...")
    blood_groups = os.listdir(dataset_path)
    
    for blood_group in blood_groups:
        blood_group_path = os.path.join(dataset_path, blood_group)
        
        if not os.path.isdir(blood_group_path):
            continue
            
        print(f"Loading {blood_group} images...")
        
        for img_file in os.listdir(blood_group_path):
            # Support multiple image formats (case-insensitive for .BMP, .bmp, etc.)
            if img_file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                img_path = os.path.join(blood_group_path, img_file)
                
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    # Normalize pixel values
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(blood_group)
    
    print(f"Total images loaded: {len(images)}")
    return np.array(images), np.array(labels)

def create_model(num_classes):
    """Create a simple CNN model"""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Load dataset
    images, labels = load_dataset(DATASET_PATH)
    
    # Check if images were loaded
    if len(images) == 0:
        print("\n❌ ERROR: No images were loaded!")
        print("\nPlease check:")
        print("1. The 'dataset' folder exists")
        print("2. Blood group folders (A+, A-, B+, etc.) are inside 'dataset'")
        print("3. Image files (.BMP, .bmp, etc.) are inside each blood group folder")
        print("\nRun 'python check_dataset.py' to diagnose the issue.")
        return
    
    # Reshape images for CNN (add channel dimension)
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    
    # Save label encoder
    with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Blood groups: {label_encoder.classes_}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_categorical, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    model = create_model(len(label_encoder.classes_))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save model
    model.save(os.path.join(MODEL_PATH, 'blood_group_model.h5'))
    print(f"\nModel saved to {MODEL_PATH}/blood_group_model.h5")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()