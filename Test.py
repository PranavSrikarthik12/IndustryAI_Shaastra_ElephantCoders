import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract

# Load the saved model
model = load_model('/kaggle/working/creditworthiness_model.h5')

# Function to extract text features (same as during training)
def extract_text_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        text = pytesseract.image_to_string(img)
        return [len(text)]  # Example: length of text as a placeholder feature
    return [0]

# Function to preprocess a new image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img_resized = cv2.resize(img, (224, 224))
        return img_resized
    return None

# Test data
test_image_path = "/kaggle/input/personal-financial-dataset-for-india/Bank Statement/1.jpg"  # Replace with the actual path to a test image

# Preprocess the test image
test_image = preprocess_image(test_image_path)
if test_image is not None:
    test_visual_features = np.expand_dims(test_image, axis=0)  # Add batch dimension
    
    # Extract text features
    test_text_features = np.expand_dims(extract_text_features(test_image_path), axis=0)
    
    # Predict using the model
    predicted_score = model.predict([test_visual_features, test_text_features])
    print(f"Predicted Creditworthiness Score: {predicted_score[0][0]}")
else:
    print("Error: Unable to read the test image.")
