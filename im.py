import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import imutils
import time
from fer import FER

def extract_features(image, method="SIFT"):
    """Extracts features from an image using SIFT or ORB."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "SIFT":
        descriptor = cv2.SIFT_create()
    elif method == "ORB":
        descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(gray, None)
    if features is None:
        print("No features detected.")
        return ([], None)
    print(f"Extracted {len(features)} features.")
    return (kps, features)

def load_images_and_labels(image_paths):
    """Loads images and their corresponding labels from a folder."""
    data = []
    labels = []
    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2] 
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        (kps, features) = extract_features(image)
        if features is not None:
            data.append(features)
            labels.extend([label] * features.shape[0])  
    print(f"Loaded {len(data)} images and {len(labels)} labels.")
    return (data, labels)

def train_knn(data, labels, k=1):
    """Trains a k-nearest neighbors classifier."""
    data = np.vstack(data)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(data, labels)
    print("Model trained successfully.")
    return model, le

def recognize_image(camera_image, model, le, feature_method="SIFT"):
    """Recognizes an image using the trained model."""
    (kps, features) = extract_features(camera_image, feature_method)
    if features is None or len(features) == 0:
        return "No features detected."

    predictions = model.predict(features)  

    counts = {}
    for prediction in predictions:
        label = le.inverse_transform([prediction])[0]
        counts[label] = counts.get(label, 0) + 1

    if counts: 
        predicted_label = max(counts, key=counts.get)
        confidence = counts[predicted_label] / len(predictions)  # Calculate confidence
        print(f"Predicted label: {predicted_label} with confidence: {confidence}")
        if confidence < 0.5:  # Set a threshold for confidence
            return "Stranger"
        
        # Emotion detection
        emotion_detector = FER()
        emotions = emotion_detector.detect_emotions(camera_image)
        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            print(f"Detected emotion: {dominant_emotion}")
            return predicted_label, dominant_emotion
        return predicted_label
    else:
        return "Face not recognized. Please add it to the database."  

from cv2 import CascadeClassifier

def capture_and_recognize(image_folder, feature_method="SIFT"):
    face_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    """Captures an image from the camera and recognizes it."""
    image_paths = list(paths.list_images(image_folder))
    (data, labels) = load_images_and_labels(image_paths)

    if not data:
        return "No training images found."

    model, le = train_knn(data, labels)

    while True:
        cap = cv2.VideoCapture(0) 
        for _ in range(5):  # Retry opening the camera 5 times
            time.sleep(1)  # Wait for 1 second before retrying
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("Camera opened successfully.")
                break
            print("Failed to open camera, retrying...")
            time.sleep(1)  # Wait for 1 second before retrying
        else:
            return "Cannot open camera after multiple attempts."

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "Can't receive frame (stream end?). Exiting ..."

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            result = recognize_image(frame[y:y + h, x:x + w], model, le, feature_method)
            if isinstance(result, tuple):
                recognized_label, dominant_emotion = result
            else:
                recognized_label = result
                dominant_emotion = "Unknown"  # Default value if emotion is not detected
            cv2.putText(frame, f"{recognized_label} - {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not cap.isOpened():
        return "Cannot open camera"

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Can't receive frame (stream end?). Exiting ..."

    cap.release()
    cv2.destroyAllWindows()
    recognized_label, dominant_emotion = recognize_image(frame, model, le, feature_method)
    return recognized_label, dominant_emotion


image_folder = "images" 
feature_method = "SIFT" 

if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Created folder '{image_folder}'. Please add images to subfolders.")
else:
    result = capture_and_recognize(image_folder, feature_method)
    print(f"Recognized: {result}")
