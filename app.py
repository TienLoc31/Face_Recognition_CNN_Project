import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, json, render_template, request, jsonify
import logging
import base64
import joblib

from camera import VideoCamera

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

model_h5 = None
model_pkl = None
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

# Load the .h5 model
try:
    model_h5 = tf.keras.models.load_model('trainned_data_6_class.h5')
    logging.info("Model .h5 loaded successfully.")
    model_h5.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dummy_input = np.zeros((1, 64, 64, 3))
    model_h5.evaluate(dummy_input, np.array([0]), verbose=0)
    logging.info("Model .h5 metrics built.")
except Exception as e:
    logging.error(f"Error loading .h5 model: {e}")

# Load the .pkl model
try:
    model_pkl = joblib.load('train_model_svc.pkl')
    logging.info("Model .pkl loaded successfully.")
except Exception as e:
    logging.error(f"Error loading .pkl model: {e}")

def preprocess_face_for_h5(face, target_size=(64, 64)):
    resized_face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    normalized_face = resized_face / 255.0
    preprocessed_face = np.expand_dims(normalized_face, axis=0)
    return preprocessed_face

def preprocess_face_for_svm(face, target_size=(64, 64)):
    resized_face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
    normalized_face = gray_face / 255.0
    preprocessed_face_flat = normalized_face.flatten().reshape(1, -1)
    return preprocessed_face_flat

def recognize_faces(image):
    if image is None or image.size == 0:
        logging.error("Empty image provided to recognize_faces.")
        return []

    preprocessed_face_h5 = preprocess_face_for_h5(image)
    preprocessed_face_svm = preprocess_face_for_svm(image)

    recognized_faces = []

    try:
        if model_h5 and model_pkl:
            predictions_h5 = model_h5.predict(preprocessed_face_h5)
            probabilities_h5 = tf.nn.softmax(predictions_h5[0]).numpy()
            recognized_face_index_h5 = np.argmax(probabilities_h5)
            confidence_h5 = probabilities_h5[recognized_face_index_h5]

            predictions_pkl = model_pkl.predict_proba(preprocessed_face_svm)
            recognized_face_index_pkl = np.argmax(predictions_pkl)
            confidence_pkl = predictions_pkl[0][recognized_face_index_pkl]

            if confidence_h5 > confidence_pkl:
                recognized_face_name = class_names[recognized_face_index_h5] if recognized_face_index_h5 < len(class_names) else "Unknown"
                recognized_faces.append((0, 0, image.shape[1], image.shape[0], recognized_face_name, confidence_h5))
            else:
                recognized_face_name = class_names[recognized_face_index_pkl] if recognized_face_index_pkl < len(class_names) else "Unknown"
                recognized_faces.append((0, 0, image.shape[1], image.shape[0], recognized_face_name, confidence_pkl))
        else:
            logging.error("One or both models are not loaded.")
    except Exception as e:
        logging.error(f"Error in prediction: {e}")

    return recognized_faces

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='application/json')

@app.route('/upload', methods=['POST'])
def upload_file():
    image = None

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            logging.error("No file selected for uploading.")
            return jsonify({'error': 'No selected file'}), 400

        try:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Image decoding failed.")
        except Exception as e:
            logging.error(f"Error decoding image: {e}")
            return jsonify({'error': 'Error decoding image'}), 500
    else:
        try:
            data = request.get_json()
            image_data = data['image']
            image_data = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Image decoding failed.")
        except Exception as e:
            logging.error(f"Error processing JSON image: {e}")
            return jsonify({'error': 'Error processing JSON image'}), 500

    recognized_faces = recognize_faces(image)

    json_serializable_faces = [
        (int(x), int(y), int(w), int(h), label, float(confidence))
        for (x, y, w, h, label, confidence) in recognized_faces
    ]

    _, jpeg = cv2.imencode('.jpg', image)

    return jsonify({
        'faces': json_serializable_faces,
        'image': base64.b64encode(jpeg.tobytes()).decode('utf-8')
    })

def gen(camera):
    while True:
        frame = camera.get_frame()
        recognized_faces = recognize_faces(frame)

        json_serializable_faces = [
            {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'label': label,
                'confidence': float(confidence)
            }
            for (x, y, w, h, label, confidence) in recognized_faces
        ]

        yield json.dumps({'faces': json_serializable_faces}).encode()

if __name__ == '__main__':
    app.run(debug=True)
