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

# input: keypoints as stream
# output: print the pred in the end of word

#import model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
interpreter = tflite.Interpreter('model/model.tflite')
prediction_fn = interpreter.get_signature_runner('serving_default')
s2p_map_path = "data/sign_to_prediction_index_map.json"
frame_log_path = '/home/hngan/Desktop/Project/samsung/output/frame_process_log'

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
    return results

def get_kp(detected_kp):
    keypoints = {
        'face': detected_kp.face_landmarks,
        'left_hand': detected_kp.left_hand_landmarks,
        'pose': detected_kp.pose_landmarks,
        'right_hand': detected_kp.right_hand_landmarks
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
def predict_result(kp):
    indata = kp.astype(np.float32)
    output_test = prediction_fn(inputs=indata)
    result = output_test['outputs'].reshape(-1) 
    # print('results: ', result.shape)
    sign  = np.stack(result)
    predict = np.argsort(-sign, -1)
    predict_prob = sign[predict]
    
    confident = predict_prob[0]
    pred_label = p2s_map[predict[0]]
    # if (confident != math.nan):
    print('pred_label: ', pred_label, ', confident: ', confident)
def have_hand(detected_kp):
    return True
def main():
    # read video
    import cv2
    HAND_NOT_FOUND_THRESHHOLD = 10
    FRAME_TO_PRED = 5
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it to the camera you want to use
    prev_frame_time = 0
    keypoint_arrays = []
    hand_flag = 0
    fps = 10
    delay = 1 / fps
    frames = []
    video_count = 1
    video_writer = None
    while True:
        start_time = time.time()
        ret, image = cap.read()
        if not ret:
            break
        frames.append(image)
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        image = cv2.putText(image, str(fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        detected_keypoints = process_frame(image)
            
        if detected_keypoints.left_hand_landmarks == None and detected_keypoints.right_hand_landmarks == None:                
            if (hand_flag > HAND_NOT_FOUND_THRESHHOLD or hand_flag==-1):
                hand_flag = -1
            else:
                hand_flag+=1
            # print("can not detect hand")
        else: 
            hand_flag = 0
            # print("frame added")
            keypoint_arrays.append(get_kp(detected_keypoints))

        # if (tmp % no_downsample) == 0:
        if (hand_flag >= HAND_NOT_FOUND_THRESHHOLD):
            # print(len(keypoint_arrays[0], keypoint_arrays[0][0]))
            print("no hand found, ", len(keypoint_arrays))
            if (len(keypoint_arrays) > FRAME_TO_PRED):
                predict_result(nested_3dims_list2array(keypoint_arrays))
                keypoint_arrays = []
                
                #logging
                output_filename = os.path.join(frame_log_path, f'video_{video_count}.avi')
                frame_height, frame_width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
                for f in frames:
                    video_writer.write(f)
                video_count += 1
                # print('video_count: ', video_count)
                frames.clear()
                frames = []
                # print("keypoint_arrays: ", len(frames))
        # print(len(keypoint_arrays), no_frames / no_downsample)
        
        cv2.imshow("ASL", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < delay:
            time.sleep(delay - elapsed_time)

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
  


if __name__ == "__main__":
    main()