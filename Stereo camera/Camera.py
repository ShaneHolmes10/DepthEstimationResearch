
import cv2
import numpy


# Start a video capture object from the camera device
cap = cv2.VideoCapture(1)

while(True):

    # Get the frame from the camera device
    ret, frame = cap.read()

    # Display the frame and then wait
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    # If the window isn't visible i.e. not there then break out of loop
    if(cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1):
        break

# Destroy all the windows
cv2.destroyAllWindows()


