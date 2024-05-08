import cv2
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import numpy as np

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5)) # Adding dropout for regularization
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (28, 28))  # Resize to match Fashion MNIST image size
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0  # Normalize pixel values

def classify_clothing(image_path):
    img = load_and_preprocess_image(image_path)

    # Load Fashion MNIST dataset
    (_, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = np.expand_dims(x_test, axis=-1)
    x_test = x_test.astype('float32') / 255.0

    input_shape = x_test.shape[1:]
    num_classes = 10  # Number of classes in Fashion MNIST dataset

    # Create the CNN model
    model = create_cnn_model(input_shape, num_classes)

    # Load pre-trained weights (optional)
    # model.load_weights('path_to_pretrained_weights.h5')

    # Predict the clothing type
    predictions = model.predict(img)
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class


def recognize_color(image_path):
    # Load the image
    img = cv2.imread(image_path)
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Reshape the image to 1D array
    pixel_values = img_rgb.reshape((-1, 3))
        
    # Convert to float32
    pixel_values = np.float32(pixel_values)
        
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
    # Convert back to 8 bit values
    centers = np.uint8(centers)
        
    # Flatten the labels array
    labels = labels.flatten()
        
    # Find dominant color
    max_label = max(set(labels), key=labels.tolist().count)
    dominant_color = centers[max_label]
        
    return dominant_color