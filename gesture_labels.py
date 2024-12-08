import pydirectinput
import mediapipe as mp

KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

gesture_labels = {
    1: "Hello",
    2: "Okay",
    3: "Like",
    4: "Dislike",
    5: "1",
    6: "+",
}

def get_label(index):
    return gesture_labels.get(index, "Unknown")

def press_key(index):
    if index in gesture_labels.keys():
        if index == 5 or index == 6:
            pydirectinput.press(gesture_labels[index])

def draw_landmarks(frame, hand_landmarks):
    mp_drawing.draw_landmarks(frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                            color=(0,0,255),
                            thickness=1,
                            circle_radius=1),
                            mp_drawing.DrawingSpec(
                            color=(0,255,0),
                            thickness=1,
                            circle_radius=1))