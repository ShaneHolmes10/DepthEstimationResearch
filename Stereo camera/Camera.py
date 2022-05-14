
import cv2
import numpy as np
import ImageFuncs as imf
from matplotlib import pyplot as plt
import time

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

def 

# Start a video capture object from the camera device
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

time.sleep(5)

timer = 0
count = 0
while(True):

    # Get the frame from the camera device
    ret, frame = cap.read()


    
    
    frame1 = ExtractImageFromDevice(frame, 1)
    frame2 = ExtractImageFromDevice(frame, 0)


    print("x shift: %s, " % count, end="")
    xshift = count
    if timer > 100:
        count += 1
    timer += 1
    '''
    frame1 = frame1[144:144+101, 232:232+146] 
    frame2 = frame2[144:144+101, 232+xshift:232+xshift+146]

    print(imf.CompareImages(frame1, frame2))



    frame = SuperImposeImages(frame1, frame2)
    '''
    print(frame1.shape)

    sections1 = imf.PartitionIntoSections(frame1, 75)
    sections2 = imf.PartitionIntoSections(frame2, 75)
    

    plt.rcParams['figure.dpi'] = 100

    fig = plt.figure(figsize=(10, 7))
    rows = 10
    columns = 8

    for x in range(0, 40):
        fig.add_subplot(rows, columns, x+1)
        plt.imshow(sections1[x])
    
    for x in range(0, 40):
        fig.add_subplot(rows, columns, x+1+40)
        plt.imshow(sections2[x])

    #plt.imshow(sections1[0])
    plt.show()

    # Display the frame and then wait
    #cv2.imshow('frame', sections1[0])
    #cv2.waitKey(10)

    # If the window isn't visible i.e. not there then break out of loop
    if(cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1):
        break

# Destroy all the windows
cv2.destroyAllWindows()


