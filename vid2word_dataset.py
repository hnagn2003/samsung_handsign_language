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
import pandas as pd
import json
from pathlib import Path
import logging
import time
import sys

#import model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
interpreter = tflite.Interpreter('model/model.tflite')
prediction_fn = interpreter.get_signature_runner('serving_default')

#Write inference results to file
saved_folder = 'data'
inf_res_path = 'inference_results'+ str(time.time()) + '.json'

# sign to prediction map
s2p_map_path = "data/sign_to_prediction_index_map.json"

# wlsal data (2000 labels)
wlasl_json_file = 'WLASL/start_kit/WLASL_v0.3.json'
raw_vid_path = 'WLASL/start_kit/raw_videos'

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
    
s2p_map = {k.lower():v for k,v in read_json_file(os.path.join(s2p_map_path)).items()}
p2s_map = {v:k for k,v in read_json_file(os.path.join(s2p_map_path)).items()}

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
    indata = kp.astype(np.float32)
    output_test = prediction_fn(inputs=indata)
    result = output_test['outputs'].reshape(-1) 
    # print('results: ', result.shape)
    sign  = np.stack(result)
    predict = np.argsort(-sign, -1)
    confident = sign[predict[0]]
    pred_label = p2s_map[predict[0]]
    if (confident == 'nan'):
        print("Confident score could not be NAN! May be a error from reading video or skeleton detection model")
        exit(0)
    return pred_label, float(confident)

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

#read all 250 labels
label = s2p_map.keys()
# for label in range (NUM_LABEL):
    
# read labeled data json file, include image name and their label 

content = json.load(open(wlasl_json_file))
results = []
pred_count = 0
incorrect_count = 0
for entry in content:
    gloss = entry['gloss']
    if (gloss not in label):
        print('not contains', gloss)
        continue
    instances = entry['instances']
    for inst in instances:
        video_id = inst['video_id']
        video_path = os.path.join(raw_vid_path, video_id)+'.mp4'
        
        video_file = Path(video_path)

        if not video_file.is_file():
            continue
        print(video_path)
        # run inference for each video
        cap = cv2.VideoCapture(video_path)
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
        # keypoint_arrays = np.array(keypoint_arrays, ndmin=3, dtype=object)
        keypoint_arrays = nested_3dims_list2array(keypoint_arrays)
        pred, confident = predict_result(keypoint_arrays)
        result = {'id':video_id, 'label': gloss, 'pred':pred, 'confident':confident, 'sourecorrect?':gloss==pred}
        results.append(result)
        print(result)
        pred_count = pred_count+1
        if (gloss!=pred):
            incorrect_count = incorrect_count + 1
acc = (pred_count - incorrect_count) / pred_count
logging.basicConfig(filename='log/inference_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))           
logging.info('Total inferences {} , Number of wrong: {}. Accuracy {}'.format(pred_count, incorrect_count, acc))
        


if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)
results_path = os.path.join(saved_folder, inf_res_path)
with open(results_path, 'w') as json_file:
    # Use json.dump() to write the data to the file
    json.dump(results, json_file, indent=4)
        
