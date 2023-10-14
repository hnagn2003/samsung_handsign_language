import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from timeit import default_timer as timer
import tflite_runtime.interpreter as tflite
import tflite_runtime
import mediapipe as mp
import cv2
import numpy as np
import math

# import tensorflow as tf
import pandas as pd
import json

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
interpreter = tflite.Interpreter('model/model.tflite')
prediction_fn = interpreter.get_signature_runner('serving_default')

def read_json_file(file_path):
    try:
        # Open the file and load the JSON data into a Python object
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        # Raise an error if the file path does not exist
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        # Raise an error if the file does not contain valid JSON data
        raise ValueError(f"Invalid JSON data in file: {file_path}")

s2p_map = {k.lower():v for k,v in read_json_file(os.path.join("data/sign_to_prediction_index_map.json")).items()}
p2s_map = {v:k for k,v in read_json_file(os.path.join("data/sign_to_prediction_index_map.json")).items()}

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    keypoints = {
        'face': results.face_landmarks,
        'left_hand': results.left_hand_landmarks,
        'pose': results.pose_landmarks,
        'right_hand': results.right_hand_landmarks
    }

    frame_keypoints = []

    for keypoint_type in ['face', 'left_hand', 'pose', 'right_hand']:
        if keypoints[keypoint_type]:
            for landmark in keypoints[keypoint_type].landmark:
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])
        else:
            num_keypoints = 468 if keypoint_type == 'face' else 21
            frame_keypoints.extend([[math.nan, math.nan, math.nan]] * num_keypoints)

    return frame_keypoints
def predict_result(kp):
    # exit(0)
    indata = kp.astype(np.float32)
    output_test = prediction_fn(inputs=indata)
    result = output_test['outputs'].reshape(-1) 
    # print('results: ', result.shape)
    sign  = np.stack(result)
    predict = np.argsort(-sign, -1)
    confident = sign[predict[0]]
    print("label:", p2s_map[predict[0]], ",confident:", confident)

#---------------------utils-----------------------#
def nested_3dims_list2array(three_dim_list):
    numpy_arrays:np.array = []
    for dim1 in three_dim_list:
        dim1_array:np.array = []  # Store arrays for the second dimension
        for dim2 in dim1:
            dim2_array = np.array(dim2)  # Convert the innermost dimension to a NumPy array
            dim1_array.append(np.array(dim2_array))
        dim1_array = np.array(dim1_array)
        #padding
        if (dim1_array.shape != (543,3)):
            row_pad = 543 - dim1_array.shape[0]
            col_pad = 3 - dim1_array.shape[1]
            dim1_array = np.pad(dim1_array, ((0, row_pad), (0, col_pad)), mode='constant', constant_values=0)
        numpy_arrays.append(dim1_array)
    res = np.array(numpy_arrays)
    return res

cap = cv2.VideoCapture('/home/hngan/Desktop/Project/samsung/output/frame_process_log/video_1.avi')
keypoint_arrays = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_keypoints = process_frame(frame)

    keypoint_arrays.append(frame_keypoints)

    # Continue processing or displaying the frame

cap.release()
cv2.destroyAllWindows()

# total_frames = len(keypoint_arrays)
keypoint_arrays = nested_3dims_list2array(keypoint_arrays)
# keypoint_arrays = np.asarray(keypoint_arrays, dtype=object)
# print('-----', ((keypoint_arrays).shape))
# print((type(keypoint_arrays)))
# print((type(keypoint_arrays[0])))
# print((type(keypoint_arrays[0][0])))
print(predict_result(keypoint_arrays))