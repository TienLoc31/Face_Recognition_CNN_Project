import os
import cv2
import numpy as np
import tensorflow as tf
import logging
from mtcnn.mtcnn import MTCNN

logging.basicConfig(level=logging.DEBUG)

model = tf.keras.models.load_model('trained_cnn_model_final4.h5')
class_names_path = 'class_names.txt'

def load_class_names():
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f]
        logging.info(f"Class names loaded: {class_names}")
        return class_names
    else:
        logging.error("class_names.txt file not found.")
        return []

class_names = load_class_names()

def detect_and_crop_face(image, padding=0, target_size=(64, 64)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(image.shape[1], x + w + padding)
            h_pad = min(image.shape[0], y + h + padding)
            cropped_face = image[y_pad:h_pad, x_pad:w_pad]
            resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
            return resized_face
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def preprocess_image(image, target_size=(64, 64)):
    face = detect_and_crop_face(image, target_size=target_size)
    if face is None:
        return None
    normalized_image = face / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image
def test_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image from path: {image_path}")
        return
    
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        logging.error("Preprocessing failed: preprocessed_image is None")
        return
    
    predictions = model.predict(preprocessed_image)
    logging.info(f"Predictions: {predictions}")
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    logging.info(f"Probabilities: {probabilities}")
    recognized_face_index = np.argmax(probabilities)
    recognized_face_name = class_names[recognized_face_index] if recognized_face_index < len(class_names) else "Unknown"
    confidence = probabilities[recognized_face_index]
    logging.info(f"Recognized: {recognized_face_name} with confidence {confidence}")

if __name__ == "__main__":
    test_image('Data\Data_test\Vuong Dinh Hue\VuongDinhHue (1).jpg')
