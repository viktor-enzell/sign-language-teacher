
import numpy as np

# function that converts the handladmarks to a list of coordinates
def flatten_landmarks(hand_landmarks):
    keypoints = []
    for data_point in hand_landmarks.landmark:
        keypoints.append(data_point.x)
        keypoints.append(data_point.y)
        keypoints.append(data_point.z)
    return keypoints

# Make sure that the array is on a scale of [0,1]
def transform_array(array):
    return list(np.array(array) * (1 / max(array)))

# Executes both the previous functions
def preprocess_keypoints(land_marks):
    res = flatten_landmarks(land_marks)
    res = transform_array(res)
    return res