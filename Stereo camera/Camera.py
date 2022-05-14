
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

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)


def nothing(x):
    pass


cv2.createTrackbar('numDisparities','disp',6,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',0,1,nothing)
cv2.createTrackbar('preFilterSize','disp',16,25,nothing)
cv2.createTrackbar('preFilterCap','disp',31,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',15,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',5,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',6,25,nothing)
cv2.createTrackbar('minDisparity','disp',0,25,nothing)



stereo = cv2.StereoBM_create()

while(True):

    # Get the frame from the camera device
    ret, frame = cap.read()

    
    frame1 = ExtractImageFromDevice(frame, 1)
    frame2 = ExtractImageFromDevice(frame, 0)


    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    '''
    numDisparities_ = 6*16
    blockSize_ = 5*2 + 5
    preFilterType_ = 0
    preFilterSize_ = 16
    preFilterCap_ = 31
    textureThreshold_ = 10
    uniquenessRatio_ = 15
    speckleRange_ = 15
    speckleWindowSize_ = 5
    disp12MaxDiff_ = 6
    minDisparity_ = 0
    '''

    
    numDisparities_ = cv2.getTrackbarPos('numDisparities', 'disp')*16
    blockSize_ = cv2.getTrackbarPos('blockSize', 'disp')*2 + 5
    preFilterType_ = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize_ = cv2.getTrackbarPos('preFilterSize', 'disp')*2 + 5
    preFilterCap_ = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold_ = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio_ = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange_ = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize_ = cv2.getTrackbarPos('speckleWindowSize', 'disp')*2
    disp12MaxDiff_ = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity_ = cv2.getTrackbarPos('minDisparity', 'disp')
    


    stereo.setNumDisparities(numDisparities_)
    stereo.setBlockSize(blockSize_)
    stereo.setPreFilterType(preFilterType_)
    stereo.setPreFilterSize(preFilterSize_)
    stereo.setPreFilterCap(preFilterCap_)
    stereo.setTextureThreshold(textureThreshold_)
    stereo.setUniquenessRatio(uniquenessRatio_)
    stereo.setSpeckleRange(speckleRange_)
    stereo.setSpeckleWindowSize(speckleWindowSize_)
    stereo.setDisp12MaxDiff(disp12MaxDiff_)
    stereo.setMinDisparity(minDisparity_)


    
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


