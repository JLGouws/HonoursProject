import argparse
import cv2 as cv
import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt

bbs = []
tmpbbs = []

predictedBbs = []
def IoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2]) * (boxA[3])
	boxBArea = (boxB[2]) * (boxB[3])
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

groundT = open("fixedLabels.txt", "r")
for line in groundT:
    if line == '\n':
        bbs.append(tmpbbs)
        tmpbbs = []
        continue
    else:
        line = line.strip()
        rest = line.split(" ")[1]
        q = tuple(map(np.int32, rest.split(",")))
        tmpbbs.append(q)
if len(tmpbbs) > 0:
    bbs.append(tmpbbs)
predictions = open("output.txt", "r")
tmpbbs = []
classPredictions = [[], [], []]
maxFrame = 0
for line in predictions:
    if line == '\n':
        predictedBbs.append(tmpbbs)
        tmpbbs = []
        maxFrame += 1
        continue
    else:
        line = line.strip()
        frameNum = int(line.split(" ")[0])
        rest = line.split(" ")[1]
        numList = rest.split(",")
        q = list(map(np.int32, numList[:-1]))
        classPredictions[q[0]].append((frameNum, q[1:], np.float64(numList[-1])))
        tmpbbs.append(q)
if len(tmpbbs) > 0:
    predictedBbs.append(tmpbbs)

print(maxFrame)
for group in classPredictions:
    group.sort(key = lambda x : x[2], reverse = True)
#print(classPredictions[0])

colors = ["cyan", "firebrick", "olivedrab"]
fig, ax = plt.subplots(1, 2, figsize = (8, 4.5), tight_layout = True)
mAP = 0
for clid in range(3):
    tP = 0
    totalPredictions = 0
    precision = []
    recall = []
    interpPrecision = []
    interpRecall = []

    groundTruths = 0
    for k in range(maxFrame):
        for bb in bbs[k]:
            if bb[0] == clid:
                groundTruths += 1

    for pred in classPredictions[clid]:
        gtBBs = bbs[pred[0]]
        gtBB = [0, 0, 0, 0]
        for bb in gtBBs:
            if bb[0] == clid:
                gtBB = list(bb)[1:]
        if(IoU(pred[1], gtBB)) > 0.5:
            tP += 1
        totalPredictions += 1
        precision += [tP / totalPredictions]
        recall += [tP]
    precision.append(0.)
    precision = np.array(precision)
    recall = np.array(recall + [groundTruths ]) / groundTruths
    k = 0
    AP = 0
    for rec in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        while k < len(recall) and recall[k] < rec:
            k += 1
        if k != len(recall):
            AP += np.amax(precision[k:])
            interpPrecision += [np.amax(precision[k:])]
        else:
            AP += 0.
            interpPrecision += [0.]
        print(np.amax(precision[k:]))
        interpRecall += [rec]
    AP = AP / 11
    mAP += AP
    print()
        
    ax[0].plot(recall, precision, label = clid, c = colors[clid])
    ax[1].plot(interpRecall, interpPrecision, label = clid, c = colors[clid])

mAP /= 3
print(mAP)
ax[0].legend(title = "Face ID")
ax[1].legend(title = "Face ID")
ax[0].set(xlabel = "recall", ylabel = "precision", title = "Raw")
ax[1].set(xlabel = "recall", ylabel = "precision", title = "Interpolated")
fig.savefig("precRecallCurvesTakiTaki.pdf")
plt.show()
