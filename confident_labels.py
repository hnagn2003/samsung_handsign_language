import json
import os

file_log = "/home/hngan/Desktop/Project/samsung/data/inference_results_finalfull.json"
s2p_map_path = "data/sign_to_prediction_index_map.json"
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
s2p_map_list = list(s2p_map)
with open(file_log, "r") as json_file:
    data = json.load(json_file)
num_label = len(s2p_map)
true_pred_count = [[0, 0] for _ in range(num_label)]

for item in data:
    label = item["label"]
    confident = item["confident"]
    ind_label = s2p_map[label]
    true_pred_count[ind_label][1] += 1
    if (item["sourecorrect?"]):
        true_pred_count[ind_label][0] +=1
# print(true_pred_count)
confidents = [0 if num_label[1] == 0 else num_label[0]/num_label[1] for i, num_label in enumerate(true_pred_count)]
# print("confidents", confidents)
# confidents_map = {k:v for k, v in zip(s2p_map.keys(), [[confident for confident in confidents, a[1] for a in true_pred_count] for _ in range(num_label)])}
confidents_map = {k:v for k, v in zip(s2p_map.keys(), [[conf, num_word[1]] for conf, num_word in zip(confidents, true_pred_count)])}
# print(confidents_map)
res = dict(sorted(confidents_map.items(), key=lambda item: item[1], reverse=True))
json_file_name = "output/top_acc_labels.json"

# Write the dictionary to a JSON file
with open(json_file_name, "w") as json_file:
    json.dump(res, json_file, indent=4)
print(res.keys())