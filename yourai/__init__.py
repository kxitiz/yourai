'''
Author : Dracoon Havoc
Copyright 2019@Havoc
'''
import cv2
import numpy as np
import tensorflow as tf
import os
from keras import backend as K
from keras.layers import Input
from PIL import Image
from .YOLOv3.visualization import draw_box, draw_caption
import colorsys
from .YOLOv3.models import yolo_main, tiny_yolo_main
from .YOLOv3.utils import letterbox_image, yolo_eval
from .udpopencv import UdpVideoCapture

import time
import sys


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class Detection:
    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_min = 1333
        self.__input_image_max = 800
        self.__detection_storage = None

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair dryer',
                                 79: 'toothbrush'}

        # Unique instance variables for YOLOv3 model
        self.__yolo_iou = 0.45
        self.__yolo_score = 0.1
        self.__yolo_anchors = np.array(
            [[10., 13.], [16., 30.], [33., 23.], [30., 61.], [62., 45.], [59., 119.], [116., 90.], [156., 198.],
             [373., 326.]])
        self.__yolo_model_image_size = (416, 416)
        self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = "", "", ""
        self.sess = K.get_session()

        # Unique instance variables for TinyYOLOv3.
        self.__tiny_yolo_anchors = np.array(
            [[10., 14.], [23., 27.], [37., 58.], [81., 82.], [135., 169.], [344., 319.]])

    def UseYOLOv3(self):
        self.__modelType = "yolov3"

    def UseModelAsTinyYOLOv3(self):
        self.__modelType = "tinyyolov3"

    def LinkModel(self, model_path):
        if (self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True

    def LoadModel(self, detection_speed="normal"):

        if (self.__modelType == "yolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (208, 208)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (128, 128)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (96, 96)

        elif (self.__modelType == "tinyyolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (832, 832)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (576, 576)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (272, 272)

        if (self.__modelLoaded == False):
            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif (self.__modelType == "yolov3"):
                model = yolo_main(Input(shape=(None, None, 3)), len(self.__yolo_anchors) // 3,
                                  len(self.numbers_to_names))
                model.load_weights(self.modelPath)

                hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                              for x in range(len(self.numbers_to_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)
                np.random.shuffle(self.colors)
                np.random.seed(None)

                self.__yolo_input_image_shape = K.placeholder(shape=(2,))
                self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output,
                                                                                       self.__yolo_anchors,
                                                                                       len(self.numbers_to_names),
                                                                                       self.__yolo_input_image_shape,
                                                                                       score_threshold=self.__yolo_score,
                                                                                       iou_threshold=self.__yolo_iou)

                self.__model_collection.append(model)
                self.__modelLoaded = True

            elif (self.__modelType == "tinyyolov3"):
                model = tiny_yolo_main(Input(shape=(None, None, 3)), len(self.__tiny_yolo_anchors) // 2,
                                       len(self.numbers_to_names))
                model.load_weights(self.modelPath)

                hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                              for x in range(len(self.numbers_to_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)
                np.random.shuffle(self.colors)
                np.random.seed(None)

                self.__yolo_input_image_shape = K.placeholder(shape=(2,))
                self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output,
                                                                                       self.__tiny_yolo_anchors,
                                                                                       len(self.numbers_to_names),
                                                                                       self.__yolo_input_image_shape,
                                                                                       score_threshold=self.__yolo_score,
                                                                                       iou_threshold=self.__yolo_iou)

                self.__model_collection.append(model)
                self.__modelLoaded = True


    def ObjectList(self,
                   person=False,
                   bicycle=False,
                   car=False,
                   motorcycle=False,
                   airplane=False,
                   bus=False,
                   train=False,
                   truck=False,
                   boat=False,
                   traffic_light=False,
                   fire_hydrant=False,
                   stop_sign=False,
                   parking_meter=False,
                   bench=False,
                   bird=False,
                   cat=False,
                   dog=False,
                   horse=False,
                   sheep=False,
                   cow=False,
                   elephant=False,
                   bear=False,
                   zebra=False,
                   giraffe=False,
                   backpack=False,
                   umbrella=False,
                   handbag=False,
                   tie=False,
                   suitcase=False,
                   frisbee=False,
                   skis=False,
                   snowboard=False,
                   sports_ball=False,
                   kite=False,
                   baseball_bat=False,
                   baseball_glove=False,
                   skateboard=False,
                   surfboard=False,
                   tennis_racket=False,
                   bottle=False,
                   wine_glass=False,
                   cup=False,
                   fork=False,
                   knife=False,
                   spoon=False,
                   bowl=False,
                   banana=False,
                   apple=False,
                   sandwich=False,
                   orange=False,
                   broccoli=False,
                   carrot=False,
                   hot_dog=False,
                   pizza=False,
                   donut=False,
                   cake=False,
                   chair=False,
                   couch=False,
                   potted_plant=False,
                   bed=False,
                   dining_table=False,
                   toilet=False,
                   tv=False,
                   laptop=False,
                   mouse=False,
                   remote=False,
                   keyboard=False,
                   cell_phone=False,
                   microwave=False,
                   oven=False,
                   toaster=False,
                   sink=False,
                   refrigerator=False,
                   book=False,
                   clock=False,
                   vase=False,
                   scissors=False,
                   teddy_bear=False,
                   hair_dryer=False,
                   toothbrush=False):
        custom_objects_dict = {}
        input_values = [person, bicycle, car, motorcycle, airplane,
                        bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
                        parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
                        giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
                        sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
                        bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
                        broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed,
                        dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
                        oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
                        toothbrush]
        actual_labels = ["person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

        for input_value, actual_label in zip(input_values, actual_labels):
            if (input_value == True):
                custom_objects_dict[actual_label] = "valid"
            else:
                custom_objects_dict[actual_label] = "invalid"

        return custom_objects_dict

    def detectAboveGivenObjects(self,
                       custom_objects=None,
                       camera_input=None,
                       frames_per_second=20,
                       frame_detection_interval=1,
                       minimum_percentage_probability=50,
                       log_progress=False,
                       display_percentage_probability=True,
                       display_object_name=True,
                       data_per_frame=None,
                       data_per_second=None,
                       data_per_minute=None,
                       frame_size=None,
                       protocal_use=None,
                       ui_enable=None,
                       video_complete_function=None,
                       return_detected_frame=False,
                       detection_timeout=None):

        if (camera_input == None):
            raise ValueError(
                "You must set 'input_file_path' to a valid video file, or set 'camera_input' to a valid camera")
        else:
            try:
                if (self.__modelType == "yolov3" or self.__modelType == "tinyyolov3"):
                    output_frames_dict = {}
                    output_frames_count_dict = {}

                    if (protocal_use == None):
                        print("TCP Protocal Used")
                        input_video = cv2.VideoCapture(camera_input)
                        if (frame_size == None):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
                            frame_width = 416
                            frame_height = 416
                        elif (frame_size == 'fast'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
                            frame_width = 320
                            frame_height = 320
                        elif (frame_size == 'faster'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 208)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 208)
                            frame_width = 208
                            frame_height = 208
                        elif (frame_size == 'fastest'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
                            frame_width = 128
                            frame_height = 128
                        elif (frame_size == 'flash'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 96)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
                            frame_width = 96
                            frame_height = 96

                    #*other imputs only UDP protocal*#
                    elif (protocal_use == 'rstp'):
                        print('RSTP Protocal Used')
                        input_video = UdpVideoCapture(camera_input)
                        if (frame_size == None):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
                            frame_width = 416
                            frame_height = 416
                        elif (frame_size == 'fast'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
                            frame_width = 320
                            frame_height = 320
                        elif (frame_size == 'faster'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 208)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 208)
                            frame_width = 208
                            frame_height = 208
                        elif (frame_size == 'fastest'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
                            frame_width = 128
                            frame_height = 128
                        elif (frame_size == 'flash'):
                            input_video.set(cv2.CAP_PROP_FRAME_WIDTH, 96)
                            input_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
                            frame_width = 96
                            frame_height = 96
                        input_video.start()

                    else:
                        print('No Input Video Found')

                    counting = 0
                    out_boxes = None
                    out_scores = None
                    out_classes = None

                    model = self.__model_collection[0]

                    detection_timeout_count = 0
                    video_frames_count = 0
                    while True:
                        # while (input_video.isOpened()):
                        ret, image = input_video.read()
                        frame = cv2.resize(image, (frame_width, frame_height),
                                           fx=0.5,
                                           fy=0.5,
                                           interpolation=cv2.INTER_LINEAR)
                        # time.sleep(0.2)
                        if (ret):
                            video_frames_count += 1
                            if (detection_timeout != None):
                                if ((video_frames_count % frames_per_second) == 0):
                                    detection_timeout_count += 1

                                print(detection_timeout_count)
                                print(video_frames_count)
                                if (detection_timeout_count >= detection_timeout):
                                    break

                            output_objects_array = []

                            counting += 1

                            if (log_progress == True):
                                print("Processing Frame : ", str(counting))

                            detected_copy = frame.copy()
                            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                            frame = Image.fromarray(np.uint8(frame))

                            new_image_size = (self.__yolo_model_image_size[0] - (self.__yolo_model_image_size[0] % 32),
                                              self.__yolo_model_image_size[1] - (self.__yolo_model_image_size[1] % 32))
                            boxed_image = letterbox_image(frame, new_image_size)
                            image_data = np.array(boxed_image, dtype="float32")

                            image_data /= 255.
                            image_data = np.expand_dims(image_data, 0)

                            check_frame_interval = counting % frame_detection_interval

                            if (counting == 1 or check_frame_interval == 0):
                                out_boxes, out_scores, out_classes = self.sess.run(
                                    [self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes],
                                    feed_dict={
                                        model.input: image_data,
                                        self.__yolo_input_image_shape: [frame.size[1], frame.size[0]],
                                        K.learning_phase(): 0
                                    })

                            min_probability = minimum_percentage_probability / 100

                            for a, b in reversed(list(enumerate(out_classes))):
                                predicted_class = self.numbers_to_names[b]
                                box = out_boxes[a]
                                score = out_scores[a]

                                if score < min_probability:
                                    continue

                                if (custom_objects != None):
                                    if (custom_objects[predicted_class] == "invalid"):
                                        continue

                                label = "{} {:.2f}".format(predicted_class, score)

                                top, left, bottom, right = box
                                top = max(0, np.floor(top + 0.5).astype('int32'))
                                left = max(0, np.floor(left + 0.5).astype('int32'))
                                bottom = min(frame.size[1], np.floor(bottom + 0.5).astype('int32'))
                                right = min(frame.size[0], np.floor(right + 0.5).astype('int32'))

                                try:
                                    color = label_color(b)
                                except:
                                    color = (255, 0, 0)

                                detection_details = (left, top, right, bottom)
                                draw_box(detected_copy, detection_details, color=color)

                                if (display_object_name == True and display_percentage_probability == True):
                                    draw_caption(detected_copy, detection_details, label)
                                elif (display_object_name == True):
                                    draw_caption(detected_copy, detection_details, predicted_class)
                                elif (display_percentage_probability == True):
                                    draw_caption(detected_copy, detection_details, str(score * 100))

                                each_object_details = {}
                                each_object_details["name"] = predicted_class
                                each_object_details["percentage_probability"] = score * 100
                                each_object_details["box_points"] = detection_details
                                output_objects_array.append(each_object_details)

                            output_frames_dict[counting] = output_objects_array

                            output_objects_count = {}
                            for eachItem in output_objects_array:
                                eachItemName = eachItem["name"]
                                try:
                                    output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                                except:
                                    output_objects_count[eachItemName] = 1

                            output_frames_count_dict[counting] = output_objects_count

                            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                            if (data_per_frame != None):
                                if (return_detected_frame == True):
                                    data_per_frame(counting,
                                                   output_objects_array,
                                                   output_objects_count,
                                                   detected_copy)
                                elif (return_detected_frame == False):
                                    data_per_frame(counting,
                                                   output_objects_array,
                                                   output_objects_count)

                            if (data_per_second != None):
                                if (counting != 1 and (counting % frames_per_second) == 0):

                                    this_second_output_object_array = []
                                    this_second_counting_array = []
                                    this_second_counting = {}

                                    for aa in range(counting):
                                        if (aa >= (counting - frames_per_second)):
                                            this_second_output_object_array.append(output_frames_dict[aa + 1])
                                            this_second_counting_array.append(output_frames_count_dict[aa + 1])

                                    for eachCountingDict in this_second_counting_array:
                                        for eachItem in eachCountingDict:
                                            try:
                                                this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                                                                 eachCountingDict[eachItem]
                                            except:
                                                this_second_counting[eachItem] = eachCountingDict[eachItem]

                                    for eachCountingItem in this_second_counting:
                                        this_second_counting[eachCountingItem] = this_second_counting[
                                                                                     eachCountingItem] / frames_per_second

                                    if (return_detected_frame == True):
                                        data_per_second(
                                            int(counting / frames_per_second),
                                            this_second_output_object_array,
                                            this_second_counting_array,
                                            this_second_counting,
                                            detected_copy)

                                    elif (return_detected_frame == False):
                                        data_per_second(
                                            int(counting / frames_per_second),
                                            this_second_output_object_array,
                                            this_second_counting_array,
                                            this_second_counting)

                            if (data_per_minute != None):

                                if (counting != 1 and (counting % (frames_per_second * 60)) == 0):

                                    this_minute_output_object_array = []
                                    this_minute_counting_array = []
                                    this_minute_counting = {}

                                    for aa in range(counting):
                                        if (aa >= (counting - (frames_per_second * 60))):
                                            this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                            this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                                    for eachCountingDict in this_minute_counting_array:
                                        for eachItem in eachCountingDict:
                                            try:
                                                this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                                 eachCountingDict[eachItem]
                                            except:
                                                this_minute_counting[eachItem] = eachCountingDict[eachItem]

                                    for eachCountingItem in this_minute_counting:
                                        this_minute_counting[eachCountingItem] = this_minute_counting[
                                                                                     eachCountingItem] / (
                                                                                     frames_per_second * 60)

                                    if (return_detected_frame == True):
                                        data_per_minute(
                                            int(counting /
                                                (frames_per_second * 60)),
                                            this_minute_output_object_array,
                                            this_minute_counting_array,
                                            this_minute_counting,
                                            detected_copy)

                                    elif (return_detected_frame == False):
                                        data_per_minute(
                                            int(counting /
                                                (frames_per_second * 60)),
                                            this_minute_output_object_array,
                                            this_minute_counting_array,
                                            this_minute_counting)
                            if(ui_enable == True):
                                cv2.imshow('Frame', detected_copy)
                                if (protocal_use == None):
                                    if cv2.waitKey(5) == 27:  # ESC key press
                                        cv2.destroyAllWindows()
                                        input_video.release()
                                        break
                                else:
                                    if cv2.waitKey(5) == 27:  # ESC key press
                                        input_video.stop()
                                        cv2.destroyAllWindows()
                                        break

                        else:
                            print("frame is not ready")
                            break

                    if (video_complete_function != None):

                        this_video_output_object_array = []
                        this_video_counting_array = []
                        this_video_counting = {}

                        for aa in range(counting):
                            this_video_output_object_array.append(output_frames_dict[aa + 1])
                            this_video_counting_array.append(output_frames_count_dict[aa + 1])

                        for eachCountingDict in this_video_counting_array:
                            for eachItem in eachCountingDict:
                                try:
                                    this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                                    eachCountingDict[eachItem]
                                except:
                                    this_video_counting[eachItem] = eachCountingDict[eachItem]

                        for eachCountingItem in this_video_counting:
                            this_video_counting[eachCountingItem] = this_video_counting[
                                                                        eachCountingItem] / counting

                        video_complete_function(this_video_output_object_array, this_video_counting_array,
                                                this_video_counting)

            except:
                raise ValueError(
                    "An error occured. It may be that your input source is invalid or is not properly configured to receive the right parameters. ")
