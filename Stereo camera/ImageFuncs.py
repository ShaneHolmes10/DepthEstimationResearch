
import cv2
import numpy as np
import cv2
from cv2 import aruco

def CompareImages(image1, image2):
    errorL2 = cv2.norm(image1, image2, cv2.NORM_L2)
    similarity = 1 - errorL2 / (image1.shape[0] * image1.shape[1])
    return similarity


def findArucoMarkers(img, draw=True, getTrackers=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray,
                                               arucoDict,
                                               parameters=arucoParam)


    center = [0, 0]
    ret = {}
    
    if(bboxs != [] and getTrackers):
        num = 0
        for obj in bboxs:

            center[0] = (obj[0][0][0] + obj[0][1][0] + obj[0][2][0] + obj[0][3][0])/4
            center[1] = (obj[0][0][1] + obj[0][1][1] + obj[0][2][1] + obj[0][3][1])/4

            if draw:
                img = cv2.circle(img, (int(center[0]), int(center[1])), 2, (255, 255, 0), 2)    
            ret[ids[num][0]] = center[:]
            num+=1
    
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)


    return ret