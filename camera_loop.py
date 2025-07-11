import cv2
import numpy as np

IS_RECORDING = False

# start video
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# create buffer for how long a user should stay distracted for
buffer_idx = 0
buffer_size = int(4 * fps)
distracted_buffer = np.zeros(buffer_size, dtype=np.int8)

# placeholder function for distraction detection
def is_distracted():
    """Test distraction by pressing space bar"""
    return int(cv2.waitKey(1) == ord(' '))


while cap.isOpened():
    ret, frame = cap.read()
    if ret is None:
        continue

    # convert frame to RGB for detection model
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # TODO: replace with call to detection models
    distracted = is_distracted()

    distracted_buffer[buffer_idx] = distracted
    distraction_avg = np.average(distracted_buffer)
    buffer_idx = (buffer_idx + 1) % buffer_size

    # Check when to start/stop recording
    if distraction_avg >= 0.8 and not IS_RECORDING:
        IS_RECORDING = True
        print('START RECORDING')

    if distraction_avg <= 0.1 and IS_RECORDING:
        IS_RECORDING = False
        print('STOP RECORDING')

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