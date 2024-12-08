{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c535f52b-0161-480d-9cca-88fc61af2f05",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import mediapipe as mp\n",
    "import os\n",
    "from utils.mediapipe_utils import extract_landmarks\n",
    "from gesture_labels import gesture_labels, draw_landmarks\n",
    "\n",
    "DATA_DIR = \"gesture_data\"\n",
    "os.makedirs(DATA_DIR, exist_ok=True)\n",
    "\n",
    "mp_hands = mp.solutions.hands\n",
    "\n",
    "def collect_data(label):\n",
    "    with mp_hands.Hands() as hands:\n",
    "        cap = cv2.VideoCapture(0)\n",
    "        while True:\n",
    "            ret, frame = cap.read()\n",
    "            if not ret:\n",
    "                return\n",
    "            \n",
    "            frame = cv2.flip(frame, 1)\n",
    "            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "            \n",
    "            frame.flags.writeable = False\n",
    "            results = hands.process(frame)\n",
    "            frame.flags.writeable = True\n",
    "\n",
    "            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)\n",
    "            \n",
    "            if results.multi_hand_landmarks:\n",
    "                for hand_landmarks in results.multi_hand_landmarks:\n",
    "                    draw_landmarks(frame, hand_landmarks)\n",
    "                    \n",
    "                    landmarks = []\n",
    "                    for lm in hand_landmarks.landmark:\n",
    "                        landmarks.append([lm.x, lm.y, lm.z])\n",
    "                    landmarks = np.array(landmarks).flatten()\n",
    "        \n",
    "                    filename = os.path.join(DATA_DIR, f\"{label}_{len(os.listdir(DATA_DIR))}.npy\")\n",
    "                    np.save(filename, landmarks)\n",
    "\n",
    "            cv2.putText(frame, f\"Collecting: {gesture_labels[label]}\", (10, 50),\n",
    "                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)\n",
    "            cv2.imshow(\"Data Collection\", frame)\n",
    "            if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "                break\n",
    "        \n",
    "        cap.release()\n",
    "        cv2.destroyAllWindows()\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    label = 6\n",
    "    collect_data(label)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c568a35f-1d8e-4020-be41-06800d3dd5cd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
