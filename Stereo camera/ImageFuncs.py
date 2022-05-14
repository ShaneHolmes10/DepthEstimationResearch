
import cv2
import numpy as np

def CompareImages(image1, image2):
    errorL2 = cv2.norm(image1, image2, cv2.NORM_L2)
    similarity = 1 - errorL2 / (image1.shape[0] * image1.shape[1])
    return similarity

def PartitionIntoSections(img, squareDim):
  sectionsList = []
  for x in range(squareDim, squareDim*int(img.shape[0]/squareDim)+1, squareDim):
    for y in range(squareDim, squareDim*int(img.shape[1]/squareDim)+1, squareDim):
      sectionsList.append(img[x-squareDim:x, y-squareDim:y])
  return sectionsList