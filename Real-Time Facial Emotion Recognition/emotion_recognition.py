import cv2
from keras.models import load_model
import numpy as np

class EmotionRecognizer:
    def __init__(self, model_path):
        """
        Initializes the emotion recognizer with a pre-trained model.
        """
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def preprocess_image(self, face):
        """
        Preprocesses the face image for the model.
        """
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)
        return face

    def predict_emotion(self, face):
        """
        Predicts the emotion of a detected face.
        """
        processed_face = self.preprocess_image(face)
        predictions = self.model.predict(processed_face)
        return self.emotion_labels[np.argmax(predictions)]

    def start_camera(self):
        """
        Starts the webcam and performs emotion detection in real time.
        """
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                emotion = self.predict_emotion(face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imshow('Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Example Usage
# recognizer = EmotionRecognizer("emotion_model.h5")
# recognizer.start_camera()
