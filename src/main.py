import tensorflow as tf
import numpy as np
import classify
from parse_labels import parse_labels
from process_images import process_images
from model.CNNModel import CNNModel
from training.model_trainer import model_trainer
from evaluation.model_evalulator import model_evalulator

def main():
    # old src
    # print("Entry point")
    # image_path = r"src\resources\511AN5bcB4L.jpeg"
    # model_path = r"model\fashion_mnist_cnn_model.h5"
    # trained_model = tf.keras.models.load_model(model_path)
    # print("Loaded model")
    # print(str(classify.classify_clothing(trained_model, image_path)))

    data_prep = parse_labels('src/resources/dataset/fashion.json', 'src/resources/dataset/fashion-cat.json')
    data_prep.load_data()
    classes = data_prep.encode_labels()
    images, labels = data_prep.get_data()

    # Preprocess images
    preprocessor = process_images()
    processed_images = [preprocessor.load_image(img) for img in images]

    # Setup model
    cnn_model = CNNModel(num_labels=len(classes))
    model = cnn_model.get_model()

    # Train model
    trainer = model_trainer(model)
    trainer.train(processed_images, labels, num_epochs=10)

    # Evaluate model
    evaluator = model_evalulator(model)
    evaluator.evaluate(processed_images, labels)


if __name__ == "__main__":
    main()
