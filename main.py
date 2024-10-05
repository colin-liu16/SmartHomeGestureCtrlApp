import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import csv
import re as regex
from sklearn.utils import Bunch  # Import Bunch

import frameextractor as fe
import handshape_feature_extractor as hfe


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

    re_run = True
    max_mutations = 0
    gesture_detail = Bunch(gesture_key="", gesture_name="", output_label="")
    while re_run and max_mutations < 5:
        cos_sin = 1.0
        position = 0
        cursor = 0
        for featureVector in featureVectorList:
            calc_cos_sin = tf.keras.losses.cosine_similarity(
                video_feature,
                featureVector.extracted_feature,
                axis=-1
            ).numpy()  # Convert tensor to a number
            if calc_cos_sin < cos_sin:
                cos_sin = calc_cos_sin
                position = cursor
            cursor += 1

        gesture_detail = featureVectorList[position].gesture_detail
        print(f"{gesture_file_name} calculated gesture {gesture_detail.gesture_name}")
        re_run = False
        max_mutations += 1
    return gesture_detail


def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)

    # Initialize default gesture detail
    gesture_detail = Bunch(gesture_key="", gesture_name="", output_label="")

    # Find the closest match using cosine similarity
    cos_sin, gesture_detail = min(
        ((tf.keras.losses.cosine_similarity(video_feature, fv.extracted_feature, axis=-1).numpy(), fv.gesture_detail)
         for fv in featureVectorList),
        key=lambda x: x[0], default=(1.0, gesture_detail)
    )

    print(f"{gesture_file_name} calculated gesture {gesture_detail.gesture_name}")
    return gesture_detail



# Create the gesture details using Bunch instead of a custom class
gesture_details = [
    Bunch(gesture_key="Num0", gesture_name="0", output_label="0"),
    Bunch(gesture_key="Num1", gesture_name="1", output_label="1"),
    Bunch(gesture_key="Num2", gesture_name="2", output_label="2"),
    Bunch(gesture_key="Num3", gesture_name="3", output_label="3"),
    Bunch(gesture_key="Num4", gesture_name="4", output_label="4"),
    Bunch(gesture_key="Num5", gesture_name="5", output_label="5"),
    Bunch(gesture_key="Num6", gesture_name="6", output_label="6"),
    Bunch(gesture_key="Num7", gesture_name="7", output_label="7"),
    Bunch(gesture_key="Num8", gesture_name="8", output_label="8"),
    Bunch(gesture_key="Num9", gesture_name="9", output_label="9"),
    Bunch(gesture_key="FanDown", gesture_name="Decrease Fan Speed", output_label="10"),
    Bunch(gesture_key="FanOn", gesture_name="FanOn", output_label="11"),
    Bunch(gesture_key="FanOff", gesture_name="FanOff", output_label="12"),
    Bunch(gesture_key="FanUp", gesture_name="Increase Fan Speed", output_label="13"),
    Bunch(gesture_key="LightOff", gesture_name="LightOff", output_label="14"),
    Bunch(gesture_key="LightOn", gesture_name="LightOn", output_label="15"),
    Bunch(gesture_key="SetThermo", gesture_name="SetThermo", output_label="16")
]

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
featureVectorList = []
path_to_train_data = "traindata/"
count = 0
for file in os.listdir(path_to_train_data):
    if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
        gesture = decide_gesture_by_file_name(file)
        if gesture:
            # Instead of GestureFeature, we use Bunch to store both gesture_detail and extracted_feature
            featureVectorList.append(Bunch(gesture_detail=gesture, extracted_feature=extract_feature(path_to_train_data, file, count)))
            count += 1

# =============================================================================
# Get the penultimate layer for test data and recognize gestures
# =============================================================================
video_locations = ["test/"]
test_count = 0

with open('Results.csv', 'w', newline='') as results_file:
    csv_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for video_location in video_locations:
        for test_file in os.listdir(video_location):
            if not test_file.startswith('.') and not test_file.startswith('frames') and not test_file.startswith('results'):
                # Recognize the gesture (use cosine similarity for comparing the vectors)
                recognized_gesture_detail = determine_gesture(video_location, test_file, test_count)
                test_count += 1
                csv_writer.writerow([recognized_gesture_detail.output_label])
