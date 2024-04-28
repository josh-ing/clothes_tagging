import cv2
import tensorflow as tf
import numpy as np
import train_model

def main():
    # Path to the image
    image_path = 'resources\IMG_0416.JPEG'
    # cnn = train_model()

    # Classify clothing type
    clothing_type = train_model.classify_clothing(image_path)
    print("Clothing Type:", clothing_type)

    # Recognize color
    color = train_model.recognize_color(image_path)
    print("Dominant Color (RGB):", color)

if __name__ == "__main__":
    main()
