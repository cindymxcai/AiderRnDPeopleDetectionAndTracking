# USAGE
# python mask_rcnn_video.py --input videos/peopleWalking.mp4 --output output/peopleWalking_output.avi --mask-rcnn mask-rcnn-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Internal functions #


#   Find the section width based on the frame width / number of columns
def calculate_section_width(f_width, n_columns):

    return f_width / n_columns


# Find the height of the section based on the frame height / number of rows
def calculate_section_height(f_height, n_rows):

    return f_height / n_rows


# Calculate section boundaries (sectionNum = [[startX, startY],[endX. endY]])
def set_section_boundaries(section_number):
    if section_number == 1:
        return [[0, 0], [section_width, section_height]]
    elif section_number == 2:
        return [[section_width + 1, 0], [section_width * 2, section_height]]
    elif section_number == 3:
        return [[(section_width * 2) + 1, 0], [section_width * 3, section_height]]
    elif section_number == 4:
        return [[(section_width * 3) + 1, 0], [frame_width, section_height]]
    elif section_number == 5:
        return [[0, section_height + 1], [section_width, section_height * 2]]
    elif section_number == 6:
        return [[section_width + 1, section_height + 1], [section_width * 2, section_height * 2]]
    elif section_number == 7:
        return [[(section_width * 2) + 1, section_height + 1], [section_width * 3, section_height * 2]]
    elif section_number == 8:
        return [[(section_width * 3) + 1, section_height + 1], [frame_width, section_height * 2]]
    elif section_number == 9:
        return [[0, (section_height * 2) + 1], [section_width, frame_height]]
    elif section_number == 10:
        return [[section_width + 1, (section_height * 2) + 1], [section_width * 2, frame_height]]
    elif section_number == 11:
        return [[(section_width * 2) + 1, (section_height * 2) + 1], [section_width * 3, frame_height]]
    elif section_number == 12:
        return [[(section_width * 3) + 1, (section_height * 2) + 1], [frame_width, frame_height]]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video file")
ap.add_argument("-o", "--output", required=True,
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

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    total = -1

# Declare and initialize counters
frameCounter = 0
trackerCounter = 1

#  Declare and initialize frame processing limiter
processFrames = 2  # Process each nth frame

#  Delcare and initialize a frame, instance segmentation and tracking marker color
frameColor = 2  # 2 = Blue

# Declare and initialize array to store x and y coordinates for tracking markers
coords = []
currentCoords = []
counter = []
currentFrame = []
timeFrame = []
gotWH = False
frame_width = 0
frame_height = 0

# Declare and initialize timer
frameTime = 0.03333  # One frames time spans
seconds = 0
minutes = 0
hours = 0


# loop over frames from the video file stream
while True:
    frameCounter += 1

    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # Declare and initialize a person counter and number
    personCounter = 0
    personNumber = 0

    # Check for the first OR every nth OR the last frame
    if frameCounter == 1 or frameCounter % processFrames == 0 or frameCounter == total:

        if frameCounter != 1 and frameCounter != total:
            seconds += frameTime*processFrames
        else:
            seconds += frameTime
        if seconds >= 60:
            minutes += 1
            seconds = 0
        if minutes >= 60:
            hours += 1
            minutes = 0

        print("Frame number: ", frameCounter)
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

        # Count the number of people detected in the frame
        for i in range(0, boxes.shape[2]):
            label = int(boxes[0, 0, i, 1])
            if label == 0:
                personCounter += 1

        # loop over the number of detected objects
        for i in range(0, boxes.shape[2]):
            # extract the class ID of the detection along with the
            # confidence (i.e., probability) associated with the
            # prediction
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            # Exclude all non-person objects
            if classID == 0:
                personNumber += 1
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the frame and then compute the width and the
                    # height of the bounding box

                    (H, W) = frame.shape[:2]

                    if not gotWH:
                        frame_width = W
                        frame_height = H
                        gotWH = True
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
                    color = COLORS[frameColor]
                    # if personNumber < personCounter:
                    #    frameColor += 1
                    # else:
                    #    frameColor = 0
                    blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

                    # store the blended ROI in the original frame
                    frame[startY:endY, startX:endX][mask] = blended

                    # draw the bounding box of the instance on the frame
                    color = [int(c) for c in color]
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  color, 2)

                    # draw the predicted label and associated probability of
                    # the instance segmentation on the frame
                    # text = "{}: {:.4f} {} {}".format(LABELS[classID], confidence, "Person number: ", personNumber)
                    text = "{}: {:.4f}".format(LABELS[classID], confidence)
                    cv2.putText(frame, text, (startX, startY - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    #  Set tracking marker symbol
                    tracker = "."
                    startX1 = startX
                    endX1 = endX
                    #  Set the length of the bounding box
                    lengthOfBox = endX - startX

                    #  Find middle point of bounding box for tracking marker placement
                    middleX = int(startX + (lengthOfBox/2))

                    #  Assign the current marker coordinates to the currentCoords array
                    currentCoords = [middleX, endY+5]

                    #  Append the current tracking marker coordinates to the coordinates array
                    coords.append(currentCoords)

                    #  Place the tracking marker at the bottom center of the bounding box
                    cv2.putText(frame, tracker, (middleX, endY + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    trackerCounter = 1
                    trackerString = str(trackerCounter)

            # Print all tracking coordinates to the current frame
            for c in coords:
                cv2.putText(frame, tracker, (c[0], c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                trackerCounter += 1
                trackerString = str(trackerCounter)

            # check if the video writer is None
            if writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)

                # some information on processing a single frame
                if total > 0:
                    elap = (end - start)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f} seconds".format(
                        (elap * total) / processFrames))

            # write the output frame to disk
            writer.write(frame)

            #  Save the final frame of the video as a jpg file for hot zone detection visual
            if frameCounter == total:
                cv2.imwrite("final1.jpg", frame)
                print("final1.jpg saved to output！")

        counter.append(personCounter)
        for c in counter:
            with open("peopleCount.csv", "w") as csv_file:
                for line in str(counter):
                    csv_file.write(line)

        currentFrame.append(frameCounter)
        for c in currentFrame:
            with open("frameCounter.csv", "w") as csv_file:
                for line in str(currentFrame):
                    csv_file.write(line)

        timeString = str(hours)
        timeString += ":"
        timeString += str(minutes)
        timeString += ":"
        timeString += str(round(seconds, 3))

        timeFrame.append(timeString)
        for c in timeFrame:
            with open("timeFrames.csv", "w") as csv_file:
                for line in str(timeFrame):
                    csv_file.write(line)

        #  Print out the number of people detected in the frame, the current frame number and the video time stamp
        print("People detected: ", personCounter, " - frame number: ", frameCounter, " - time: ", timeString)
        personNumber = 0

#  Declare and initialize number of columns and rows
columns = 4
rows = 3
#  Find the section width based on the frame width / number of columns
section_width = calculate_section_width(frame_width, columns)
#  Find the section height based on the frame height / number of rows
section_height = calculate_section_height(frame_height, rows)
#  Set number of sections
numOfSections = columns*rows

# Calculate all section boundaries (sectionNum = [[startX, startY],[endX. endY]])
section1 = set_section_boundaries(1)
section2 = set_section_boundaries(2)
section3 = set_section_boundaries(3)
section4 = set_section_boundaries(4)
section5 = set_section_boundaries(5)
section6 = set_section_boundaries(6)
section7 = set_section_boundaries(7)
section8 = set_section_boundaries(8)
section9 = set_section_boundaries(9)
section10 = set_section_boundaries(10)
section11 = set_section_boundaries(11)
section12 = set_section_boundaries(12)

#   Declare and initialize section total counters
section1Total = 0; section2Total = 0; section3Total = 0; section4Total = 0
section5Total = 0; section6Total = 0; section7Total = 0; section8Total = 0
section9Total = 0; section10Total = 0; section11Total = 0; section12Total = 0

#  Declare and initialize a tracking marker counter
markerCount = 0

#  Find which section the tracking marker is in and add 1 to the section total counter
for p in coords:
    # Check if the marker is in the first column
    if p[0] >= section1[0][0] and p[0] <= section1[1][0]:
        # Check if the marker is in the first row
        if p[1] >= section1[0][1] and p[1] < section1[1][1]:
            section1Total += 1
        # Check if the marker is in the second row
        elif p[1] >= section5[0][1] and p[1] < section5[1][1]:
            section5Total += 1
        # Check if the marker is in the thirdrow
        elif p[1] >= section9[0][1] and p[1] <= section9[1][1]:
            section9Total += 1
    # Check if the marker is in the second column
    elif p[0] >= section2[0][0] and p[0] <= section2[1][0]:
        # Check if the marker is in the first row
        if p[1] >= section2[0][1] and p[1] < section2[1][1]:
            section2Total += 1
        # Check if the marker is in the second row
        elif p[1] >= section6[0][1] and p[1] < section6[1][1]:
            section6Total += 1
        # Check if the marker is in the third row
        elif p[1] >= section10[0][1] and p[1] <= section10[1][1]:
            section10Total += 1

    # Check if the marker is in the third column
    elif p[0] >= section3[0][0] and p[0] <= section3[1][0]:
        # Check if the marker is in the first row
        if p[1] >= section3[0][1] and p[1] < section3[1][1]:
            section3Total += 1
        # Check if the marker is in the second row
        elif p[1] >= section7[0][1] and p[1] < section7[1][1]:
            section7Total += 1
        # Check if the marker is in the third row
        elif p[1] >= section11[0][1] and p[1] <= section11[1][1]:
            section11Total += 1
    # Check if the marker is in the fourth column
    elif p[0] >= section4[0][0] and p[0] <= section4[1][0]:
        # Check if the marker is in the first row
        if p[1] >= section4[0][1] and p[1] < section4[1][1]:
            section4Total += 1
        # Check if the marker is in the second row
        elif p[1] >= section8[0][1] and p[1] < section8[1][1]:
            section8Total += 1
        # Check if the marker is in the third row
        elif p[1] >= section12[0][1] and p[1] <= section12[1][1]:
            section12Total += 1

    markerCount += 1

    #  Write all coordinates to a text file
    with open("output.txt", "w") as txt_file:
        for line in str(coords):
            txt_file.write(" ".join(line))  # works with any number of elements in a line

#  Find tracking marker average per section
markerAverage = markerCount / numOfSections

sectionTotals = [section1Total, section2Total, section3Total, section4Total,
                 section5Total, section6Total, section7Total, section8Total,
                 section9Total, section10Total, section11Total, section12Total]

#  Print out total markers, marker average and each sections marker totals
print("Total Markers: ", markerCount)
print("Marker Count: ", markerCount)
print("Marker Average: ", markerAverage, " per section")
print("Section 1 Total: ", section1Total)
print("Section 2 Total: ", section2Total)
print("Section 3 Total: ", section3Total)
print("Section 4 Total: ", section4Total)
print("Section 5 Total: ", section5Total)
print("Section 6 Total: ", section6Total)
print("Section 7 Total: ", section7Total)
print("Section 8 Total: ", section8Total)
print("Section 9 Total: ", section9Total)
print("Section 10 Total: ", section10Total)
print("Section 11 Total: ", section11Total)
print("Section 12 Total: ", section12Total)

#  Open the last frame of the video file
img = cv2.imread("final1.jpg")

# Set colors for section boxes - rgb values are reversed (not sure why)
green = (0, 255, 0)  #50 - 74 tracking markers
orange = (0, 165, 255)  #75 - 100 tracking markers
red = (0, 0, 255)  #100+ tracking markers

#  Set section box line thickness
lineSize = 2

# Section with most foot traffic
first = 0
# Section with second most foot traffic
second = 0
# Section with third most foot traffic
third = 0
# 3 most visited sections
top3 = [None]*3
# hot zone sections
topZone = ""
secondZone = ""
thirdZone = ""

# Find section with most markers
for total in sectionTotals:
    if total >= section1Total and first < section1Total:
        first = total
        top3[0] = section1
        topZone = "Section 1"
    if total >= section2Total and first < section2Total:
        first = total
        top3[0] = section2
        topZone = "Section 2"
    if total >= section3Total and first < section3Total:
        first = total
        top3[0] = section3
        topZone = "Section 3"
    if total >= section4Total and first < section4Total:
        first = total
        top3[0] = section4
        topZone = "Section 4"
    if total >= section5Total and first < section5Total:
        first = total
        top3[0] = section5
        topZone = "Section 5"
    if total > section6Total and first < section6Total:
        first = total
        top3[0] = section6
        topZone = "Section 6"
    if total >= section7Total and first < section7Total:
        first = total
        top3[0] = section7
        topZone = "Section 7"
    if total >= section8Total and first < section8Total:
        first = total
        top3[0] = section8
        topZone = "Section 8"
    if total >= section9Total and first < section9Total:
        first = total
        top3[0] = section9
        topZone = "Section 9"
    if total >= section10Total and first < section10Total:
        first = total
        top3[0] = section10
        topZone = "Section 10"
    if total >= section11Total and first < section11Total:
        first = total
        top3[0] = section11
        topZone = "Section 11"
    if total >= section12Total and first < section12Total:
        first = total
        top3[0] = section12
        topZone = "Section 12"


# Find section with second most markers
for total in sectionTotals:
    if total < first:
        if total >= section1Total and second <= section1Total:
            second = total
            top3[1] = section1
            secondZone = "Section 1"
        if total >= section2Total and second <= section2Total:
            second = total
            top3[1] = section2
            secondZone = "Section 2"
        if total >= section3Total and second <= section3Total:
            second = total
            top3[1] = section3
            secondZone = "Section 3"
        if total >= section4Total and second <= section4Total:
            second = total
            top3[1] = section4
            secondZone = "Section 4"
        if total >= section5Total and second <= section5Total:
            second = total
            top3[1] = section5
            secondZone = "Section 5"
        if total >= section6Total and second <= section6Total:
            second = total
            top3[1] = section6
            secondZone = "Section 6"
        if total >= section7Total and second <= section7Total:
            second = total
            top3[1] = section7
            secondZone = "Section 7"
        if total >= section8Total and second <= section8Total:
            second = total
            top3[1] = section8
            secondZone = "Section 8"
        if total >= section9Total and second <= section9Total:
            second = total
            top3[1] = section9
            secondZone = "Section 9"
        if total >= section10Total and second <= section10Total:
            second = total
            top3[1] = section10
            secondZone = "Section 10"
        if total >= section11Total and second <= section11Total:
            second = total
            top3[1] = section11
            secondZone = "Section 11"
        if total >= section12Total and second <= section12Total:
            second = total
            top3[1] = section12
            secondZone = "Section 12"

# Find section with third most markers
for total in sectionTotals:
    if total < first:
        if total < second:
            if total >= section1Total and third <= section1Total:
                third = total
                top3[2] = section1
                thirdZone = "Section 1"
            if total >= section2Total and third <= section2Total:
                third = total
                top3[2] = section2
                thirdZone = "Section 2"
            if total >= section3Total and third <= section3Total:
                third = total
                top3[2] = section3
                thirdZone = "Section 3"
            if total >=  section4Total and third <= section4Total:
                third = total
                top3[2] = section4
                thirdZone = "Section 4"
            if total >=  section5Total and third <= section5Total:
                third = total
                top3[2] = section5
                thirdZone = "Section 5"
            if total >=  section6Total and third <= section6Total:
                third = total
                top3[2] = section6
                thirdZone = "Section 6"
            if total >= section7Total and third <= section7Total:
                third = total
                top3[2] = section7
                thirdZone = "Section 7"
            if total >=  section8Total and third <= section8Total:
                third = total
                top3[2] = section8
                thirdZone = "Section 8"
            if total >=  section9Total and third <= section9Total:
                third = total
                top3[2] = section9
                thirdZone = "Section 9"
            if total >= section10Total and third <= section10Total:
                third = total
                top3[2] = section10
                thirdZone = "Section 10"
            if total >= section11Total and third <= section11Total:
                third = total
                top3[2] = section11
                thirdZone = "Section 11"
            if total >= section12Total and third <= section12Total:
                third = total
                top3[2] = section12
                thirdZone = "Section 12"

# Write red hot zone to the final output frame
startX = int(top3[0][0][0])
startY = int(top3[0][0][1])
endX = int(top3[0][1][0])
endY = int(top3[0][1][1])
#  Write the section box to screen
cv2.rectangle(img, (startX, startY), (endX, endY-lineSize), red, lineSize)
cv2.imwrite("final1.jpg", img)

# Write orange hot zone to the final output frame
startX = int(top3[1][0][0])
startY = int(top3[1][0][1])
endX = int(top3[1][1][0])
endY = int(top3[1][1][1])
#  Write the section box to screen
cv2.rectangle(img, (startX, startY), (endX, endY-lineSize), orange, lineSize)
cv2.imwrite("final1.jpg", img)

# Write green hot zone to the final output frame
startX = int(top3[2][0][0])
startY = int(top3[2][0][1])
endX = int(top3[2][1][0])
endY = int(top3[2][1][1])
#  Write the section box to screen
cv2.rectangle(img, (startX, startY), (endX, endY-lineSize), green, lineSize)
cv2.imwrite("final1.jpg", img)

print("Top section is: ", topZone, first, "    Second zone: ", secondZone, second, "   Third zone: ", thirdZone, third)

# Add section totals to an array for printing to csv file
sections = [[section1Total, section2Total, section3Total, section4Total],
            [section5Total, section6Total, section7Total, section8Total],
            [section9Total, section10Total, section11Total, section12Total]]

sect = np.array(sections)
with open("sections.csv", "w") as csv_file:
    for line in str(sections):
        csv_file.write(line)

print('People detected marker section totals:')
print(sect)

# release the file pointers
print("[INFO] cleaning up...")

writer.release()
vs.release()


