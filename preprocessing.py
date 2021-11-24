import numpy as np

# function that converts the handladmarks to a list of coordinates
def flatten_landmarks(hand_landmarks):
    keypoints = []
    distances = []
    for data_point in hand_landmarks.landmark:
        keypoints.append(data_point.x)
        keypoints.append(data_point.y)
        keypoints.append(data_point.z)
        distances.append(np.linalg.norm([data_point.x, data_point.y, data_point.z], ord=None, axis=None, keepdims=False))
    return keypoints, distances

# Make sure that the array is on a scale of [0,1]
def normalize_array(array):
    return list(np.array(array) * (1 / max(array)))

# Executes both the previous functions
def preprocess_keypoints(land_marks):
    keypoints, distances = flatten_landmarks(land_marks)
    keypoints = normalize_array(keypoints)
    distances = normalize_array(distances)
    return keypoints + distances