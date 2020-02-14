
# ————————————————————————————————
# 对银行卡号码的定位效果非常差！！！  对身份证文字的定位也不怎么好
# ————————————————————————————————

import numpy as np
import cv2
from imutils.object_detection import non_max_suppression    # pip install imutils

image1 = cv2.imread("images\\1.bmp")     # image1是源图像
(height1, width1) = image1.shape[:2]
size = 320
(height2, width2) = (size, size)  # 模型输入图像的宽度和高度必须是32的倍数，否则会报错
image2 = cv2.resize(image1, (width2, height2))  # image2是现在的图像

net = cv2.dnn.readNet("frozen_east_text_detection.pb")
blob = cv2.dnn.blobFromImage(image2, 1.0, (width2, height2), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)

(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
(rows, cols) = scores.shape[2:4]  # grab the rows and columns from score volume
rects = []  # stores the bounding box coordiantes for text regions
confidences = []  # stores the probability associated with each bounding box region in rects

for y in range(rows):
    scoresdata = scores[0, 0, y]
    xdata0 = geometry[0, 0, y]
    xdata1 = geometry[0, 1, y]
    xdata2 = geometry[0, 2, y]
    xdata3 = geometry[0, 3, y]
    angles = geometry[0, 4, y]

    for x in range(cols):

        if scoresdata[x] < 0.5:  # if score is less than min_confidence, ignore
            continue
        # print(scoresdata[x])
        offsetx = x * 4.0
        offsety = y * 4.0
        # EAST detector automatically reduces volume size as it passes through the network
        # extracting the rotation angle for the prediction and computing their sine and cos

        angle = angles[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = xdata0[x] + xdata2[x]
        w = xdata1[x] + xdata3[x]
        #  print(offsetx,offsety,xdata1[x],xdata2[x],cos)
        endx = int(offsetx + (cos * xdata1[x]) + (sin * xdata2[x]))
        endy = int(offsety + (sin * xdata1[x]) + (cos * xdata2[x]))
        startx = int(endx - w)
        starty = int(endy - h)

        # appending the confidence score and probabilities to list
        rects.append((startx, starty, endx, endy))
        confidences.append(scoresdata[x])

# applying non-maxima suppression to supppress weak and overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

rW = width1 / float(width2)
rH = height1 / float(height2)
for (startx, starty, endx, endy) in boxes:
    startx = int(startx * rW)
    starty = int(starty * rH)
    endx = int(endx * rW)
    endy = int(endy * rH)

    cv2.rectangle(image1, (startx, starty), (endx, endy), (0, 255, 0), 2)

cv2.imshow("text Detection", image1)
cv2.waitKey(0)
