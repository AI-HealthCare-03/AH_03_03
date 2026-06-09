# -*- encoding: utf-8 -*-
# ---------------------------- Food Classifier ---------------------------
# This code is written to test the food classifier using customized YOLO model
# It supports YOLO v3 and v4 as of 20Dec. 11, 2020
# For options in detail, please refer to /config/confi.py

# Usage example:  python3 food_classifier_yolo.py --video=run.mp4
#                 python3 food_classifier_yolo.py --image=bird.jpg
# -------------------------------------------------------------------------

import os.path
import cv2 as cv
import argparse
import sys
import numpy as np
import json
from PIL import ImageFont, ImageDraw, Image

from config import config

parser = argparse.ArgumentParser(description='Food Classification and Localization ver. 0.9')
parser.add_argument('--image', help='Full path to image file.')
parser.add_argument('--video', help='Full path to video file.')
parser.add_argument('--showText', type=int, default=1, help='show text in the output.')
parser.add_argument('--ps', type=int, default=1, help='stop each image in the screen.')
args = parser.parse_args()

# Initialize the parameters
args.image      = config.TEST_IMAGE_PATH  # image path
args.video      =  config.TEST_VIDEO_PATH # video path
args.showText = config.SHOW_TEXT_FLAG #1
args.ps = config.PS_FLAG # 1

# refine the inferences
confThreshold   = config.CONF_THRES # 0.1 #0.5  # Confidence threshold
nmsThreshold    = config.NMS_THRES #0.1 #0.4  # Non-maximum suppression threshold

# modes inference size regardless of input image size
inpWidth        = config.INPWIDTH # 32*10  # 608     #Width of network's input image # 320(32*10)
inpHeight       = config.INPHEIGHT # 32*9 # 608     #Height of network's input image # 288(32*9) best

# model base directory
modelBaseDir    = config.ModelBaseDir # "C:/Users/mmc/workspace/yolo"

# Load names of classes from a file
classesFile = os.path.sep.join([modelBaseDir, config.CLASSES_FILE])
classes = None
with open(classesFile, 'rt', encoding='utf-8') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load codes of classes from a file
classes_File_Codes = os.path.sep.join([modelBaseDir, config.CLASSES_FILE_CODE])
classes_codes = None
with open(classes_File_Codes, 'rt', encoding='utf-8') as f:
    classes_codes = f.read().rstrip('\n').split('\n')

assert (len(classes) == len(classes_codes))

# model configuration and weights paths
modelConfiguration = os.path.sep.join([modelBaseDir, config.Model_Configuration])
modelWeights = os.path.sep.join([modelBaseDir, config.Model_Weights])

# load a given model
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        #label = '%s:%s' % (classes[classId], label)
        label = u'%s' % (classes[classId])
        #label = u'%s' % (classId)
        print('label:{}, class_id:{}'.format(label, classId))


    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    if args.showText:
        #cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
        #        (0, 255, 255), cv.FILLED)
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (0, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        #fontpath = "./font/gulim.ttc"
        #font_ = ImageFont.truetype(fontpath, 14)
        #img_pil = Image.fromarray(frame)
        #draw = ImageDraw.Draw(img_pil)
        #draw.text((left, top), label, font=font_, fill=(0, 0, 0, 0))
        #frame = np.array(img_pil)
        #cv.imshow('pil', frame)


def postprocess(frame, outs, showimg=False):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        if(args.showText):
            print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] >= confThreshold:
                if(args.showText):
                    print('obj score: ', detection[4], " - confidence:", scores[classId], " - thres : ", confThreshold)
                    #print(detection)
            if confidence >= confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                # cv.rectangle(frame, (left, top), (left+width, top+height), (255, 0, 255),2)
                # cv.imshow('test', frame)
                # cv.waitKey(1)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    rests =[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        rests.append([classIds[i], left, top, width, height, frameWidth, frameHeight])
        if(showimg):
            drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

    return rests

def food_classifier_Json(image):
    # do somthing
    print(args.showText)
    locations = food_classifier_pipeline(frame=image) #[(2321, 0, 0, 10, 10)] # list of (id, rect) from classfication
    jsons = []
    for j,location in enumerate(locations):
        class_id, x, y, width, height, framewidth, frameheight =location
        res_json = {}
        res_json["ClassID"] = classes_codes[class_id] # code , class_id (training class)
        res_json["ClassName"] = classes[class_id]
        res_json["x"] = int(x)
        res_json["y"] = int(y)
        res_json["w"] = int(width)
        res_json["h"] = int(height)
        res_json["framewidth"] = int(framewidth)
        res_json["frameheight"]= int(frameheight)
        jsons.append(res_json)
    print(json.dumps(jsons,ensure_ascii=False))

    return json.dumps(jsons,ensure_ascii=False)

def food_classifier_pipeline(frame):

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    final_infos = postprocess(frame, outs)

    return final_infos

# Process inputs
def main(main_args):
    winName = 'Food Classification Results'
    #cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    m_startFrame = np.maximum(0, config.Video_Start_Frame)

    outputFile = "yolo_out_py.avi"
    if (main_args.image):
        # Open the image file
        if not os.path.isfile(main_args.image):
            print("Input image file ", main_args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(main_args.image)
        outputFile = args.image[:-4] + '_yolo_out_py.jpg'
    elif (main_args.video):
        # Open the video file
        if not os.path.isfile(main_args.video):
            print("Input video file ", main_args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(main_args.video)
        cap.set(cv.CAP_PROP_POS_FRAMES, m_startFrame)
        outputFile = main_args.video[:-4] + '_yolo_out_py.avi'
    else:
        # Webcam input
        cap = cv.VideoCapture(0)

    # Get the video writer initialized to save the output video
    if (not main_args.image):
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    pcontinue = True
    while pcontinue:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            # if(main_args.ps):
            #    cv.waitKey(0)
            #else:
            #    cv.waitKey(1)

            #break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        if main_args.showText:
            print(getOutputsNames(net))

        postprocess(frame, outs, showimg=True)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        if main_args.showText:
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            print(label)
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if (main_args.image):
            cv.imwrite(outputFile, frame.astype(np.uint8));
        else:
            vid_writer.write(frame.astype(np.uint8))

        #cv.imshow(winName, frame)
        #cv.waitKey(1)
        pcontinue=False

if __name__ == "__main__":
    main(main_args=args)
