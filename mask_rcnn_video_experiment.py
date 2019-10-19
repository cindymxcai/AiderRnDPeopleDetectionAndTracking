# USAGE
# python mask_rcnn_video.py --input videos/IMG_1324.MOV --output output/cats_and_dogs_output.avi --mask-rcnn mask-rcnn-coco

# import the necessary packages
import logging
import math

import numpy as np
import argparse
import imutils
import time
import cv2
import os
import os
import shutil

frameCounter = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video file")
ap.add_argument("-o", "--output", required=False,
                help="path to output video file")
ap.add_argument("-m", "--mask-rcnn", required=True,
                help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
                               "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
                                "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
                               "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["input"])
writer = None
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
    last_frame = total



# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    total = -1
# 上一帧的人位置列表
pre_postion = []
# 上一帧人的颜色列表
pre_color = []
# 偏移量
offset = 30
# 日志文件目录
tracker_log_dir = './output/log/'
# 帧的宽
frame_width = 0
# 帧的高
frame_height = 0
# 每个格子有多少点的集合
every_square_have_how_much_tracker_points = []

processFrames = 2  # Process each 5th frame

# 所有的轨迹，用来计算热点区域
all_track_points = []
# last_frame = 192


# if os.path.exists(tracker_log_dir):
#     try:
#         shutil.rmtree(tracker_log_dir)
#         os.makedirs(tracker_log_dir)
#     except:
#         print("有点异常哎")

# LOG_LEVEL = logging.NOTSET
# LOGFORMAT = "[%(log_color)s%(levelname)s] [%(log_color)s%(asctime)s] %(log_color)s%(filename)s [line:%(log_color)s%(lineno)d] : %(log_color)s%(message)s%(reset)s"
# import colorlog
#
# logging.root.setLevel(LOG_LEVEL)
# ############
# # 此配置是将日志输出到myapp.log
# colorlog.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', filename='myapp.log',
#                      filemode='w', datefmt='%a, %d %b %Y %H:%M:%S', )
# ##############
# formatter = colorlog.ColoredFormatter(LOGFORMAT)
# stream = logging.StreamHandler()
# stream.setLevel(LOG_LEVEL)
# stream.setFormatter(formatter)
# log = logging.getLogger()
# log.setLevel(LOG_LEVEL)
# log.addHandler(stream)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    frameCounter += 1

    (grabbed, frame) = vs.read()
    if frameCounter == (last_frame - 1):
        cv2.imwrite("final.jpg", frame)
        print("保存final.jpg成功！")
    #
    #if frameCounter < 191:
     #   continue
    #if frameCounter > 193:
        #break

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    if frameCounter % 2 == 0:
        # construct a blob from the input frame and then perform a
        # forward pass of the Mask R-CNN, giving us (1) the bounding box
        # coordinates of the objects in the image along with (2) the
        # pixel-wise segmentation for each specific object
        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        (boxes, masks) = net.forward(["detection_out_final",
                                      "detection_masks"])
        end = time.time()

        people_counter_in_current_frame = 0
        # loop over the number of detected objects
        for i in range(0, boxes.shape[2]):
            # extract the class ID of the detection along with the
            # confidence (i.e., probability) associated with the
            # prediction
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]
            if classID != 0:
                continue
            people_counter_in_current_frame += 1
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the frame and then compute the width and the
                # height of the bounding box
                (H, W) = frame.shape[:2]
                frame_width = W
                frame_height = H
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY

                # extract the pixel-wise segmentation for the object,
                # resize the mask such that it's the same dimensions of
                # the bounding box, and then finally threshold to create
                # a *binary* mask
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH),
                                  interpolation=cv2.INTER_NEAREST)
                mask = (mask > args["threshold"])

                # extract the ROI of the image but *only* extracted the
                # masked region of the ROI
                roi = frame[startY:endY, startX:endX][mask]

                # grab the color used to visualize this particular class,
                # then create a transparent overlay by blending the color
                # with the ROI

                # 首先要找找有没有符合自己当前坐标偏移范围内的颜色文件
                lengthOfBox = endX - startX
                middleX = int(startX + (lengthOfBox / 2))
                currentCoords = [middleX, endY + 5]

                # color = COLORS[classID]
                color = COLORS[people_counter_in_current_frame]

                # draw the bounding box of the instance on the frame
                color = [int(c) for c in color]

                # draw the predicted label and associated probability of
                # the instance segmentation on the frame
                text = "{}: {:.4f}".format(LABELS[classID], confidence)

                tracker = str(".")

                # 如果当前的最新坐标数组大于或者等于当前帧的检测人数，说明已经添加了，没有则添加

                if len(pre_postion) >= people_counter_in_current_frame:
                    # 那么我们要找最近的符合偏移量的坐标
                    for idx, val in enumerate(pre_postion):
                        print("[INFO] " + "坐标列表: " + str(pre_postion) + "颜色列表:" + str(pre_color))

                        temp = int(
                            abs(math.sqrt(
                                math.pow(val[0] - currentCoords[0], 2) + math.pow(val[1] - currentCoords[1], 2))))
                        # print("[INFO] " + "当前位置: " + str(currentCoords) + " 比对位置: " + str(val))
                        # print("[INFO] " + "距离是: " + str(temp))
                        if temp < offset:
                            color = pre_color[idx]
                            cv2.putText(frame, text, (startX, startY - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY),
                                          color, 2)
                            cv2.putText(frame, tracker, (val[0], val[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            # store the blended ROI in the original frame
                            blended = ((0.4 * np.array(color)) + (0.6 * roi)).astype("uint8")
                            # store the blended ROI in the original frame
                            frame[startY:endY, startX:endX][mask] = blended
                            print(
                                "[INFO] " + "符合偏移量设定，选中颜色是 " + str(pre_color[idx]) + " 对应坐标为:" + str(pre_postion[idx]))
                            # # 把当前的轨迹点写到文件里去
                            path = tracker_log_dir + str(idx) + '_track_points.log'

                            f = open(path, 'a+', encoding='utf8')  # 在最后一行追加,以|分隔，便于后面处理
                            f.write(str(currentCoords) + '|')
                            f.close()

                            # 读取轨迹文件,并绘画
                            f = open(tracker_log_dir + str(idx) + '_track_points.log',
                                     encoding='utf8')
                            for val in f.readline().split("|"):
                                if val.strip() is not '':
                                    val = eval(val.strip())
                                    cv2.putText(frame, tracker, (val[0], val[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                pre_color[idx], 2)
                                f.close()

                            # 更新列表
                            pre_postion[idx] = currentCoords
                            pre_color[idx] = color
                        else:
                            print("[INFO] " + "超过了设定偏移量，不选该颜色")

                else:
                    pre_postion.append(currentCoords)
                    pre_color.append(color)
                    cv2.putText(frame, text, (startX, startY - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  color, 2)
                    # store the blended ROI in the original frame
                    blended = ((0.4 * np.array(color)) + (0.6 * roi)).astype("uint8")
                    # store the blended ROI in the original frame
                    frame[startY:endY, startX:endX][mask] = blended
                    cv2.putText(frame, tracker, (currentCoords[0], currentCoords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color,
                                2)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    (elap * total)/4))

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            print("writer......")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f} seconds".format(
                    (elap * total) / processFrames))

        # write the output frame to disk
        writer.write(frame)
        # cv2.imshow('Result', frame)
        # cv2.imwrite('Result.jpg', frame)
        if cv2.waitKey(1) == ord('c'):
            break
# write the output frame to disk

# release the file pointers
print("[INFO] cleaning up...")
# 统计所有的轨迹点
for parent, dirnames, filenames in os.walk(tracker_log_dir, followlinks=True):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        f = open(file_path,
                 encoding='utf8')
        for val in f.readline().split("|"):
            if val.strip() is not '':
                val = eval(val.strip())
            f.close()

# 最后我们来计算热点ROI区域
# 4*3 12个格 标记单元格内点最多的 
# 单个格子的宽度
single_width = frame_width / 4
# 单个格子的高度
single_height = frame_height / 3
# 画12个格子   divided 3*4 sections

# 读取所有的点 get dots
for parent, dirnames, filenames in os.walk(tracker_log_dir, followlinks=True):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        f = open(file_path,
                 encoding='utf8')
        for val in f.readline().split("|"):
            if val.strip() is not '':
                val = eval(val.strip())
                all_track_points.append(val)
            f.close()

print(all_track_points)
 
# 对所有的点分配到12个格子里  Assign all points to 12 grids
every_square_x_and_y = []
for val in all_track_points:
    x = 0
    y = 0
    for i in range(1, 5):
        for j in range(1, 4):
            every_square_have_how_much_tracker_points.append(0)
            every_square_x_and_y.append(0)
            if val[0] > i * single_width:
                x = i
            if val[1] > i * single_height:
                y = j
    print("x: " + str(x) + "y: " + str(y))
    # 将点打入对应x,y所属的格子里 Put the point into the grid corresponding to x, y
    idx = (y - 1) * 4 + x
    print(idx)
    # 先获取当前多少 再对应的加一 First get the current number, then add the corresponding one
    try:
        every_square_have_how_much_tracker_points[idx] += 1
        every_square_x_and_y[idx] = [x, y]
    except:
        every_square_have_how_much_tracker_points.insert(idx, 1)
        every_square_x_and_y.insert(idx, [x, y])

print(every_square_have_how_much_tracker_points)
# 在12个格子中找出最多点的那一个
print(max(every_square_have_how_much_tracker_points))
# 找到最多的那个索引
max_roi = 1
for idx, val in enumerate(every_square_have_how_much_tracker_points):
    if val == max(every_square_have_how_much_tracker_points):
        max_roi = idx
        print("最大的数量的索引是：" + str(idx))

print(every_square_x_and_y)
print(every_square_x_and_y[max_roi])

print("single_width: " + str(single_width) + " single_height:" + str(single_height))

x = every_square_x_and_y[max_roi][0]
y = every_square_x_and_y[max_roi][1]

print("x: " + str(x) + " y:" + str(y))

startX = int((x - 1) * single_width)
startY = int((y - 1) * single_height)
endX = int(x * single_width)
endY = int(y * single_height)

print("frame_width: " + str(frame_width) + " frame_height:" + str(frame_height))
print("startx: " + str(startX) + " starty:" + str(startY) + " endx:" + str(endX) + " endy:" + str(endY))

# # 绘画 mark the section
img = cv2.imread("final.jpg")
colors = (0, 154, 255)
cv2.rectangle(img, (startX, startY), (endX, endY), colors, 5)
cv2.imwrite("final.jpg", img)

writer.release()
vs.release()
cv2.destroyAllWindows()

# to run the program use python mask_rcnn_video.py --input videos/Test.mp4 --mask-rcnn mask-rcnn-coco in terminal
