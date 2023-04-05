# each anchorbox will have the following format
# assuming 3 classes
# y = [pc, bx, by, bh, bw, c1, c2, c3]
# c defines classes
import os
import sys
import cv2
import numpy as np

import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

path = os.getcwd()
file_sys = 'emotion-recognition-neural-networks'
emotion_recognition_path = os.path.join(path, file_sys)

# Load the Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the expression label list
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the FER2013 model for facial expression recognition
fer_model = load_model(os.path.join(emotion_recognition_path, 'fer2013_mini_XCEPTION.102-0.66.hdf5'))

def preprocess_face(face, input_size):
    face = cv2.resize(face, (input_size, input_size))
    face = face.astype("float32") / 255
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture a frame. Check your camera connection.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Process the detected faces
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = gray[y:y+h, x:x+w]

        # Preprocess the face for the FER2013 model
        face_preprocessed = preprocess_face(face, 64)  # Use 64x64 input size for the FER2013 model

        # Predict the facial expression
        expression_probs = fer_model.predict(face_preprocessed)
        expression_idx = np.argmax(expression_probs[0])

        # Get the label of the predicted expression
        expression_label = expression_labels[expression_idx]

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the facial expression label
        cv2.putText(frame, expression_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detected faces and expressions
    cv2.imshow('Face Detection and Expression Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()



#python detect.py --source 0