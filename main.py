'''
Author : Dracoon Havoc
Copyright 2019@Havoc
'''
'''Import Packages'''
from yourai import Detection
import os
import numpy as np
import cv2
import string
from time import gmtime, strftime
import json
execution_path = os.getcwd()
'''Setup Input Device'''
protocal = None  #!'rstp' or None
#camera = os.path.join(execution_path, 'video.mp4')
camera = 0
# camera = 'rtsp://admin:0pen0pen@10.10.99.12/Streaming/Channels/101?tcp'
speed = None  #!None, fast, faster, fastest, flash

# def forSecond(second_number, output_arrays, count_arrays,
#               average_output_count):
#     print(second_number)
#     print(output_arrays)
#     print(count_arrays)
#     print(average_output_count)

compute = Detection()
compute.UseYOLOv3()
compute.LinkModel(
    os.path.join(execution_path,
                 'yolo.h5'))  #! Change this as Model name in Root directory
compute.LoadModel(speed)

objects = compute.ObjectList(person=True,
                             bicycle=True,
                             car=True,
                             motorcycle=True)  #!More objects in README.md
print("Starting Detection: ")

compute.detectAboveGivenObjects(
    custom_objects=objects,  #!Pass custom objects
    camera_input=camera,  #!Pass feed
    protocal_use=protocal,
    frame_size=speed,  #!Change speed of detection
    frames_per_second=14,  #!CAUTION effects detection rate
    frame_detection_interval=2,  #!CAUTION effects detection rate
    # data_per_second=forSecond,
    ui_enable=True,  #! To disable Output frame
    minimum_percentage_probability=
    60,  #! Minimum probability required for detection
    return_detected_frame=False,  #!Gives datapoint of detected object in video
    display_percentage_probability=True,  #!Displays percentage in output frame
    display_object_name=True,
    log_progress=False)