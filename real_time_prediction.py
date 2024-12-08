import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.mediapipe_utils import extract_landmarks
from gesture_labels import get_label, press_key

model = load_model("models/hand_gesture_model.h5")

def predict_gesture(landmarks):
    prediction = model.predict(np.expand_dims(landmarks, axis=0))
    return np.argmax(prediction)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    landmarks = extract_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    gesture_index = None

    if landmarks is not None:
        gesture_index = predict_gesture(landmarks / np.max(landmarks))
                
        cv2.putText(frame, get_label(gesture_index), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    #press_key(gesture_index)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()