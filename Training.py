import os
import cv2
import pytesseract
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Dataset path
data_dir = "/kaggle/input/personal-financial-dataset-for-india/"

# OCR setup
!apt-get install tesseract-ocr -y
!pip install pytesseract

# Function to extract features using OCR
def extract_text_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        text = pytesseract.image_to_string(img)
        # Convert text to numeric features (you can customize this part)
        return [len(text)]  # Example: length of text as a placeholder feature
    return [0]  # Return a default value if the image is invalid

# Function to load images and features
def load_data():
    labels = []  # Replace with actual creditworthiness scores if available
    visual_features = []
    text_features = []
    
    # Loop through each folder in the dataset
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a directory
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                
                # Check if the file is an image
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"Processing: {img_path}")  # Debugging: Check which image is being processed
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize the image to the model's expected input size (224x224)
                        img_resized = cv2.resize(img, (224, 224))
                        visual_features.append(img_resized)
                        
                        # Extract text features
                        text_features.append(extract_text_features(img_path))
                        
                        # Here you can assign the labels based on the folder or image name
                        labels.append(50)  # Replace this with actual labels (e.g., based on folder or external data)
                    else:
                        print(f"Error reading image: {img_path}")  # Debugging: Print failed image paths
    return np.array(visual_features), np.array(text_features), np.array(labels)

# Load dataset
visual_data, text_data, labels = load_data()

# Train-test split
X_train_vis, X_val_vis, X_train_txt, X_val_txt, y_train, y_val = train_test_split(
    visual_data, text_data, labels, test_size=0.2, random_state=42
)

# Define the CNN model for images
cnn_base = ResNet50(include_top=False, input_shape=(224, 224, 3))
x = Flatten()(cnn_base.output)
visual_model = Model(inputs=cnn_base.input, outputs=x)

# Define the model for text features
text_input = Input(shape=(text_data.shape[1],))
text_dense = Dense(64, activation='relu')(text_input)

# Combine visual and text features
combined = Concatenate()([visual_model.output, text_dense])
x = Dense(128, activation='relu')(combined)
output = Dense(1, activation='linear')(x)  # Predicting a continuous score
model = Model(inputs=[visual_model.input, text_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(
    [X_train_vis, X_train_txt],
    y_train,
    validation_data=([X_val_vis, X_val_txt], y_val),
    epochs=10,
    batch_size=32
)

# Save the trained model
model.save("/kaggle/working/creditworthiness_model.h5")
