
import cv2
import numpy as np

def CompareImages(image1, image2):
    errorL2 = cv2.norm(image1, image2, cv2.NORM_L2)
    similarity = 1 - errorL2 / (image1.shape[0] * image1.shape[1])
    return similarity

