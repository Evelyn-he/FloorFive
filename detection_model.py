import cv2
import numpy as np
from utils.onnx_utils import create_ort_session

MP_landmark = create_ort_session("models/mediapipe_face-facelandmarkdetector-w8a8.onnx")

def is_distracted(frame):
    score, landmarks = get_model_ouptut(frame)
    # TODO: implement more complicated distraction detection based on face landmarks
    # print(score)
    return int(score < 0.8)


def get_model_ouptut(image):
    # resize image to model input shape
    image = cv2.resize(image, (192,192))
    # convert to uint 8
    image = image.astype(np.uint8)
    # move to NCHW format
    image = image.transpose(2,0,1)
    image = np.reshape(image, (1,3,192,192))

    return MP_landmark.run(None, {"image": image})

