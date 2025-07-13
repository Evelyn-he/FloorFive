import cv2
import numpy as np
from utils.onnx_utils import create_ort_session

MP_landmark = create_ort_session("models/mediapipe_face-facelandmarkdetector-w8a8.onnx")

def is_distracted(frame):
    score, landmarks = get_model_ouptut(frame)

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []
    for idx in [33, 263, 1, 61, 291, 199]:
        lm = landmarks[0][idx] / 255
        lm_x, lm_y, lm_z = lm[1], lm[0], lm[2]
        x, y = int(lm_x*img_w),  int(lm_y*img_h)

        face_2d.append([x,y])
        face_3d.append([x, y, lm_z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0,0,1]])

    dist_coeffs = np.zeros((4,1), dtype = np.float64)

    (success, rotation_vector, translation_vector) = cv2.solvePnP(face_3d, face_2d, camera_matrix, dist_coeffs)
    rmat = cv2.Rodrigues(rotation_vector)[0] # rotation matrix

    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    x_angle = angles[0] * 360
    y_angle = angles[1] * 360

    looking_away = 0
    # See where the user's head tilting
    if y_angle < -15:
        text = "Not a straight face"
        looking_away = 1 #L
    elif y_angle > 15:
        looking_away = 1 #R
        text = "Not a straight face"
    elif x_angle < -7:
        looking_away = 1 #D
        text = "Not a straight face"
    elif x_angle > 20:
        looking_away = 1 #U
        text = "Not a straight face"
    else:
        text = "Straight face"

    # print(text)
    return looking_away


def get_model_ouptut(image):
    # resize image to model input shape
    image = cv2.resize(image, (192,192))
    # convert to uint 8
    image = image.astype(np.uint8)
    # move to NCHW format
    image = image.transpose(2,0,1)
    image = np.reshape(image, (1,3,192,192))

    return MP_landmark.run(None, {"image": image})

