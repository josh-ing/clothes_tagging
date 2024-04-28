import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

class train_model:

    def classify_clothing(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.resnet.preprocess_input(img)

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