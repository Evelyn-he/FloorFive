import cv2
import numpy as np
from utils.onnx_utils import create_ort_session

MP_landmark = create_ort_session("models/mediapipe_face-facelandmarkdetector-w8a8.onnx")

def is_distracted(frame):
    # dummy return for now. Can trigger by pressing spacebar
    return int(cv2.waitKey(1) == ord(' '))

    # TODO: use model output to calculate if "distracted"
    output = get_model_ouptut(frame)


def get_model_ouptut(image):
    # resize image to model input shape
    image = cv2.resize(image, (192,192))
    # convert to uint 8
    image = image.astype(np.uint8)
    # move to NCHW format
    image = image.transpose(2,0,1)
    image = np.reshape(image, (1,3,192,192))

    return MP_landmark.run(None, {"image": image})

