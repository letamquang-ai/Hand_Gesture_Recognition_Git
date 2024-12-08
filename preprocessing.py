import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_DIR = "gesture_data"

def load_data():
    data = []
    labels = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            label = int(file.split("_")[0])
            labels.append(label)
            data.append(np.load(os.path.join(DATA_DIR, file)))
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def preprocess_data():
    data, labels = load_data()
    data = data / np.max(data)  # Normalize
    labels = to_categorical(labels)  # One-hot encode
    return train_test_split(data, labels, test_size=0.2, random_state=42)
