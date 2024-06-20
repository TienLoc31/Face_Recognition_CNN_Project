import os
import cv2
import dlib
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pickle

# Đường dẫn đến thư mục dữ liệu huấn luyện
data_train_dir = 'Data/Data_train'

# Khởi tạo dlib HOG face detector, shape predictor và face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Hàm để trích xuất đặc trưng khuôn mặt từ một hình ảnh
def get_face_encodings(image):
    detections = detector(image, 1)
    encodings = []
    for k, d in enumerate(detections):
        shape = sp(image, d)
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        encodings.append(np.array(face_descriptor))
    return encodings

# Danh sách lưu các đặc trưng khuôn mặt và nhãn tương ứng
X_train = []
y_train = []

# Duyệt qua từng thư mục con trong thư mục dữ liệu huấn luyện
for person_name in os.listdir(data_train_dir):
    person_dir = os.path.join(data_train_dir, person_name)
    if os.path.isdir(person_dir):
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is not None:  # Kiểm tra hình ảnh đã đọc thành công
                encodings = get_face_encodings(image)
                if encodings:  # Đảm bảo có đặc trưng được trích xuất
                    X_train.append(encodings[0])
                    y_train.append(person_name)  # Sử dụng tên thư mục làm nhãn

# Chuyển đổi danh sách đặc trưng và nhãn thành numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

# Mã hóa các nhãn bằng LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Huấn luyện mô hình SVM
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train_encoded)

# Lưu mô hình đã huấn luyện và LabelEncoder
with open('face_recognition_model.pkl', 'wb') as f:
    pickle.dump((clf, le), f)
