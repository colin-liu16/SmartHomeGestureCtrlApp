# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
#code to get the key frame from the video and save it as a png file.

import cv2
import os
import random

def frameExtractor(videopath, frames_path, count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)

    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no = int(video_length / 2) # + random.randint(-10, 10)

    cap.set(1, frame_no)
    ret, frame = cap.read()

    if ret:
        frame_filename = os.path.join(frames_path, f"{count + 1:05d}.png")

        # print(videopath, frame_filename, frame_no, video_length)

        cv2.imwrite(frame_filename, frame)
        if not os.path.exists(frame_filename):
            raise Exception(f"Failed to save frame: {frame_filename}")
        return frame_filename
    else:
        raise Exception(f"Failed to extract frame from {videopath}")

