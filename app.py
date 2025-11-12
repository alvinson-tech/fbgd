from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import os
import cv2
import numpy as np
from tensorflow import keras
import pickle
from models import db, User

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

IMG_SIZE = 128
MODEL_PATH = 'model/blood_group_model.h5'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'
UPLOAD_FOLDER = 'uploads'

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def preprocess_image(image_path):
    """Preprocess the uploaded image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

# ============== AUTHENTICATION ROUTES ==============

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        mobile = request.form.get('mobile', '').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not (email or mobile):
            flash('Please provide either email or mobile number', 'error')
            return render_template('signup.html')
        
        if not password or len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        # Check if user exists
        if email and User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('signup.html')
        
        if mobile and User.query.filter_by(mobile=mobile).first():
            flash('Mobile number already registered', 'error')
            return render_template('signup.html')
        
        # Create new user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            email=email if email else None,
            mobile=mobile if mobile else None,
            password=hashed_password
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()  # email or mobile
        password = request.form.get('password')
        
        if not identifier or not password:
            flash('Please provide all fields', 'error')
            return render_template('login.html')
        
        # Check if identifier is email or mobile
        user = User.query.filter(
            (User.email == identifier) | (User.mobile == identifier)
        ).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

# ============== MAIN APPLICATION ROUTES ==============

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/predict', methods=['POST'])
@login_required
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

# ============== DATABASE INITIALIZATION ==============

def create_tables():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    create_tables()
    app.run(debug=True, port=5001)