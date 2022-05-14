
import cv2
import numpy as np
import ImageFuncs as imf
from matplotlib import pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

def ExtractImageFromDevice(frame, index):
    xValue = int(index*(frame.shape[1]/2))
    return frame[0:frame.shape[0],xValue:xValue+int(frame.shape[1]/2)]

def SuperImposeImages(frame1, frame2):
    a_channel = np.ones(frame1.shape, dtype=np.float)*0.5
    frame1 = frame1 / 255.0
    frame1 = frame1*a_channel

    a_channel = np.ones(frame1.shape, dtype=np.float)*0.5
    frame2 = frame2 / 255.0
    frame2 = frame2*a_channel

    return frame1 + frame2

# Start a video capture object from the camera device
cap = cv2.VideoCapture(0)

numDisparities_ = 6*16
blockSize_ = 5*2 + 5
minDisparity_ = 0

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)


while(True):

    # Get the frame from the camera device
    ret, frame = cap.read()

    
    frame1 = ExtractImageFromDevice(frame, 1)
    frame2 = ExtractImageFromDevice(frame, 0)

    #frame1 = frame1[0:15, 0:15]
    #frame2 = frame2[0:15, 0:15]

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


    stereo = cv2.StereoBM_create(numDisparities=numDisparities_, blockSize=blockSize_)
    depth = stereo.compute(frame2, frame1)

    frame = SuperImposeImages(frame1, frame2)

    normal = np.zeros((frame.shape[0], frame.shape[1]))

    depth = depth.astype(np.float32)
    depth = (depth/16.0 - minDisparity_)/numDisparities_

    '''
    for x in range(len(depth)):
        for y in range(len(depth[0])):
            normal[x][y] = depth[x][y]
    '''

    #print(depth[221][509])
    #print()
    #print()
    # Display the frame and then wait
    cv2.imshow('frame', depth)
    
    #plt.imshow(depth)

    #plt.show()
    cv2.waitKey(100)

    # If the window isn't visible i.e. not there then break out of loop
    if(cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1):
        break

# Destroy all the windows
cv2.destroyAllWindows()


