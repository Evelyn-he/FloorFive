import cv2
import numpy as np

# start video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret is None:
        continue

    # convert frame to RGB for detection model
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)


    # TODO:
    # insert detection model here


    # convert frame back to BGR to displaying
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # display window
    window = 'Distraction Detection'
    cv2.namedWindow(window)
    cv2.moveWindow(window, 100, 50)
    cv2.imshow(window, frame)

    # check for escape key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()