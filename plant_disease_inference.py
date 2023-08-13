import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Function to load class indices
def load_class_indices(path):
    with open(path, 'r') as f:
        return json.load(f)

# Load saved model
model_path = "plant_disease_model.h5"
model = load_model(model_path)

# Load class indices
class_indices = load_class_indices('class_indices.json')
class_labels = list(class_indices.keys())

# Function for prediction
def predict_disease(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Return the label of the predicted class
    return class_labels[predicted_class]

# Example of using the function
image_path = 'Algodão (Cotton) - Mancha de Mirotecio (Myrothecium leaf spot) - 1/DSC_0103.jpg'
print("Predicted class: ", predict_disease(image_path))

