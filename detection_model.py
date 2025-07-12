import cv2
import keyboard
import numpy as np
from utils.onnx_utils import create_ort_session

MP_landmark = create_ort_session("models/mediapipe_face-facelandmarkdetector-w8a8.onnx")


def is_distracted(frame):
    # Actual key check
    space_pressed = int(keyboard.is_pressed('space'))
    #print(space_pressed) # uncomment to test keyboard input detection
    return space_pressed

def get_model_ouptut(image):
    # resize image to model input shape
    image = cv2.resize(image, (192,192))
    # convert to uint 8
    image = image.astype(np.uint8)
    # move to NCHW format
    image = image.transpose(2,0,1)
    image = np.reshape(image, (1,3,192,192))

    return MP_landmark.run(None, {"image": image})

