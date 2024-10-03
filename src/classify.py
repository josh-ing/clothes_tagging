import numpy as np
import cv2 as cv
import os
from PIL import Image
from rembg import remove

def classify_clothing(model, image_path):
    # Load and preprocess the image
    print("Classifying clothes")
    # img = preprocess_foreground(image_path)
    img = Image.open(image_path)
    remove_bg = remove(img).convert('L')

    remove_bg = remove_bg.resize((28, 28))  # Resize to match model input size
    img_array = np.array(remove_bg) / 255.0  # Normalize pixel values
    print("Normalized values")

    # Add batch dimension and reshape for model input
    img_input = img_array.reshape((1, 28, 28, 1))
    print("Reshape value")

    # Make prediction
    print("Predicting")
    predictions = model.predict(img_input)[0]

    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    

    return class_labels[predicted_class]

