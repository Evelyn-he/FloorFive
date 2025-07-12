import keyboard
import cv2
import numpy as np
from detection_model import is_distracted
from popup.chatbot_launcher import launch_chatbot, kill_chatbot

# State variables
IS_RECORDING = False
MANUAL_OVERRIDE = False

# start video
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# create video window
window = 'Distraction Detection'
cv2.namedWindow(window)
cv2.moveWindow(window, 100, 50)

# create buffer for how long a user should stay distracted for
buffer_idx = 0
buffer_size = int(3 * fps)
distracted_buffer = np.zeros(buffer_size, dtype=np.int8)

# launch chatbot
chatbot = launch_chatbot()

while cap.isOpened():
    ret, frame = cap.read()
    if ret is None:
        continue

    # convert frame to RGB for detection model
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # TODO: replace with call to detection models
    distracted = is_distracted(frame)

    distracted_buffer[buffer_idx] = distracted
    distraction_avg = np.average(distracted_buffer)
    buffer_idx = (buffer_idx + 1) % buffer_size

    # Check when to start/stop recording
    if not IS_RECORDING and distraction_avg >= 0.8:
        IS_RECORDING = True
        print('START RECORDING')

    if IS_RECORDING and not MANUAL_OVERRIDE and distraction_avg <= 0.1:
        IS_RECORDING = False
        print('STOP RECORDING')

    # convert frame back to BGR to displaying
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Show recording message
    if IS_RECORDING:
        cv2.putText(frame, "RECORDING", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow(window, frame)

    # allow manual start/stop recording override
    if keyboard.is_pressed('r'):
        while keyboard.is_pressed('r'):
            pass
        IS_RECORDING ^= True
        MANUAL_OVERRIDE ^= True

    # check for escape key
    if keyboard.is_pressed('esc'):
        break

    # require waitKey to update frames
    cv2.waitKey(1)

# kill chatbot
kill_chatbot(chatbot)

cap.release()
cv2.destroyAllWindows()