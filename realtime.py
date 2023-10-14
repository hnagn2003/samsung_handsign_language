import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import mediapipe as mp
import tflite_runtime.interpreter as tflite 

import json
import os

import cv2
import numpy as np
import math

import gc
import time
LABEL = 250
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity = 0
)
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
    #normalize predict_prob
    # mu = predict_prob[LABEL-1]
    # predict_prob = predict_prob - mu

    # original_sum = np.sum(predict_prob)

    # # Define the desired sum (e.g., 100)
    # desired_sum = 10000

    # # Normalize the array so that its sum is equal to the desired sum
    # nor_predict_prob = (predict_prob / original_sum) * desired_sum
    # print(nor_predict_prob)
    
    
    confident = predict_prob[0]
    pred_label = p2s_map[predict[0]]
    # if (confident):
    print('pred_label: ', pred_label, ', confident: ', confident)

def main():
    cap = cv2.VideoCapture('data/good morning_cut.mp4')

    tmp = 0

    prev_frame_time = 0
    new_frame_time = 0
    keypoint_arrays = []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    predicted_frame_per_second = 40
    no_seconds = 0.7 # the number of second will be batched for one prediction
    no_overlap = 20

    no_downsample = math.ceil(fps / predicted_frame_per_second)
    no_frames = int(no_seconds * fps)

    while True:
        ret, image = cap.read()
        tmp = (tmp + 1) % no_downsample
        if not ret:
            break
        # image = cv2.flip(image, 1)

        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # fps = cap.get(cv2.CAP_PROP_FPS)

        image = cv2.putText(image, str(fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

        # if (tmp % no_downsample) == 0:
        keypoint_arrays.append(process_frame(image))

        if len(keypoint_arrays) > no_frames / no_downsample:
            predict_result(nested_3dims_list2array(keypoint_arrays))

        keypoint_arrays = keypoint_arrays[-no_overlap:]

        cv2.imshow("ASL", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    