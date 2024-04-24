import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

def load():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print('Shape of training cloth images: ',train_images.shape)
    print('Shape of training label: ',train_labels.shape)
    print('Shape of test cloth images: ',test_images.shape)
    print('Shape of test labels: ', test_labels.shape)