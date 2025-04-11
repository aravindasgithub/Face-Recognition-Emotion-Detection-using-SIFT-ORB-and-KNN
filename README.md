# 🧠 Face Recognition & Emotion Detection

Real-time face recognition and emotion detection using OpenCV, SIFT/ORB feature extraction, KNN classification, and the FER library.

## 💡 Features
- Face recognition using SIFT or ORB features
- KNN classifier for person identification
- Real-time camera input with face detection
- Emotion detection using FER (Facial Expression Recognition)
- Handles unknown faces with a "Stranger" label

## 📂 Folder Structure
Organize your dataset as follows:

images/ ├── Alice/ │ ├── img1.jpg │ ├── img2.jpg ├── Bob/ │ ├── img1.jpg │ ├── img2.jpg

Each subfolder name becomes the label for the person in it.

## 🛠 Requirements

Install dependencies with:

##pip install opencv-python imutils scikit-learn fer numpy
▶️ How to Run
Add images to the images/ folder as shown above.

Run the script:

bash
Copy
Edit
python your_script_name.py
A webcam window will open. Press q to quit.
