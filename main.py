import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import csv
from sklearn.utils import Bunch  # Import Bunch
import random
import frameextractor as fe
import handshape_feature_extractor as hfe


def get_frame_count(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return None

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    return frame_count

def extract_feature(location, input_file, mid_frame_counter):
    middle_image = cv2.imread(
        fe.frameExtractor(location + input_file, location + "frames/", mid_frame_counter),
        cv2.IMREAD_GRAYSCALE
    )
    response = hfe.HandShapeFeatureExtractor.get_instance().extract_feature(middle_image)

    return response

def decide_gesture_by_file_name(gesture_file_name):
    return next((x for x in gesture_details if x.gesture_key == gesture_file_name.split('_')[0]), None)

def decide_gesture_by_name(lookup_gesture_name):
    return next((x for x in gesture_details if x.gesture_name.replace(" ", "").lower() == lookup_gesture_name.lower()), None)

def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)

    gesture_detail = Bunch(gesture_key="", gesture_name="", output_label="")
    cos_sin, gesture_detail = min(
        ((tf.keras.losses.cosine_similarity(video_feature, fv.extracted_feature, axis=-1).numpy(), fv.gesture_detail)
         for fv in featureVectorList),
        key=lambda x: x[0], default=(1.0, gesture_detail)
    )

    print(f"{gesture_file_name} calculated gesture {gesture_detail.gesture_name}")
    return gesture_detail


# import collections
#
#
# def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter):
#     video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)
#
#     gesture_detail = Bunch(gesture_key="", gesture_name="", output_label="")
#     similarities = [
#         (tf.keras.losses.cosine_similarity(video_feature, fv.extracted_feature, axis=-1).numpy(), fv.gesture_detail)
#         for fv in featureVectorList
#     ]
#     similarities_sorted = sorted(similarities, key=lambda x: x[0])
#     top_10_similarities = similarities_sorted[:10]
#     gesture_names = [gesture_detail.gesture_name for _, gesture_detail in top_10_similarities]
#     gesture_name_mode = collections.Counter(gesture_names).most_common(1)[0][0]
#     gesture_detail_mode = next(
#         gesture_detail for _, gesture_detail in top_10_similarities if gesture_detail.gesture_name == gesture_name_mode
#     )
#
#     return gesture_detail_mode


gesture_keys = [
    "Num0", "Num1", "Num2", "Num3", "Num4", "Num5",
    "Num6", "Num7", "Num8", "Num9", "FanDown",
    "FanOn", "FanOff", "FanUp", "LightOff", "LightOn", "SetThermo"
]

gesture_names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "Decrease Fan Speed", "FanOn", "FanOff", "Increase Fan Speed",
    "LightOff", "LightOn", "SetThermo"
]

output_labels = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16"
]

gesture_details = [
    Bunch(gesture_key=key, gesture_name=name, output_label=label)
    for key, name, label in zip(gesture_keys, gesture_names, output_labels)
]

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
featureVectorList = []
path_to_train_data = "traindata/"
for ct, file in enumerate(os.listdir(path_to_train_data)):
    if not file.startswith(('.', 'frames', 'results')):
        gesture = decide_gesture_by_file_name(file)
        if gesture:
            featureVectorList.append(Bunch(gesture_detail=gesture, extracted_feature=extract_feature(path_to_train_data, file, ct)))

# =============================================================================
# Get the penultimate layer for test data and recognize gestures
# =============================================================================
print(f'ct:{ct}')
video_locations, test_count = ["test/"], 0 + ct
with open('Results.csv', 'w', newline='') as results_file:
    csv_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for video_location in video_locations:
        for test_file in os.listdir(video_location):
            if not test_file.startswith('.') and not test_file.startswith('frames') and not test_file.startswith('results'):
                recognized_gesture_detail = determine_gesture(video_location, test_file, test_count)
                test_count += 1
                csv_writer.writerow([recognized_gesture_detail.output_label])
