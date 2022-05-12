
import cv2
import numpy as np
import math
import ImageFuncs as imf

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

base = 12
focus = 302

while(True):

    # Get the frame from the camera device
    ret, frame = cap.read()

    
    frame1 = ExtractImageFromDevice(frame, 1)
    frame2 = ExtractImageFromDevice(frame, 0)
    
    centerMarker0R = (int(frame1.shape[1]/2), int(frame1.shape[0]/2))
    centerMarker0L = (int(frame2.shape[1]/2), int(frame2.shape[0]/2))

    
    try:
        centerMarker0R = imf.findArucoMarkers(frame1)[0]
        centerMarker0L = imf.findArucoMarkers(frame2)[0]
    except:
        pass    

    frame = SuperImposeImages(frame1, frame2)

    cv2.line(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int(centerMarker0R[0]), int(centerMarker0R[1])), (255, 255, 0), 2)
    cv2.line(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int(centerMarker0L[0]), int(centerMarker0L[1])), (255, 255, 0), 2)
    
    
    disparity = math.dist(centerMarker0R, centerMarker0L)+0.0001
    depth = base*focus / disparity
    if depth < 1000:
        print( base*focus / disparity)


    # Display the frame and then wait
    cv2.imshow('frame', frame)
    cv2.waitKey(10)

    # If the window isn't visible i.e. not there then break out of loop
    if(cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1):
        break

# Destroy all the windows
cv2.destroyAllWindows()


