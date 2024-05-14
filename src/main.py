import tensorflow as tf
import numpy as np
import classify

def main():
    print("Entry point")
    image_path = r"src\resources\511AN5bcB4L.jpeg"
    model_path = r"model\fashion_mnist_cnn_model.h5"
    trained_model = tf.keras.models.load_model(model_path)
    print("Loaded model")
    print(str(classify.classify_clothing(trained_model, image_path)))


if __name__ == "__main__":
    main()
