import cv2
import numpy as np
import mediapipe as mp
import os
from utils.mediapipe_utils import extract_landmarks
from gesture_labels import gesture_labels, draw_landmarks

DATA_DIR = "gesture_data"
os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands

def collect_data(label):
    with mp_hands.Hands() as hands:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)
                    
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    landmarks = np.array(landmarks).flatten()
        
                    filename = os.path.join(DATA_DIR, f"{label}_{len(os.listdir(DATA_DIR))}.npy")
                    np.save(filename, landmarks)

            cv2.putText(frame, f"Collecting: {gesture_labels[label]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    label = 6
    collect_data(label)
