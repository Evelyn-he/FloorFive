import cv2
import numpy as np
from utils.onnx_utils import create_ort_session
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

MP_landmark = create_ort_session("models/mediapipe_face-facelandmarkdetector-w8a8.onnx")
INPUT_NAME  = MP_landmark.get_inputs()[0].name

POSE_IDX = [33, 263, 1, 61, 291, 199]

def get_model_output(frame: np.ndarray) -> np.ndarray | None:
    """Run the ONNX face-mesh; return a (478,3) np.float32 array or None."""
    # 1) BGR→RGB  2) resize to 192×192  3) uint8 NCHW batch
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (192,192)).astype(np.uint8)
    batch = small.transpose(2,0,1)[None, ...]  # (1,3,192,192)

    outs = MP_landmark.run(None, {INPUT_NAME: batch})
    # pick the output whose last dim == 3
    for o in outs:
        arr = np.array(o)
        if arr.ndim >= 3 and arr.shape[-1] == 3:
            lm = arr
            break
    else:
        return None

    # collapse to (478,3)
    if   lm.ndim == 4: lm = lm.reshape(-1,3)
    elif lm.ndim == 3: lm = lm[0]
    return lm.astype(np.float32)


def is_distracted(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]

    lm = get_model_output(frame)
    if lm is None:
        head_text = "No face detected"
        distracted = False
    else:
        pts2d = []
        pts3d = []
        for i in POSE_IDX:
            x, y, z = lm[i]
            pts2d.append([x * w, y * h])
            pts3d.append([x * w, y * h, z])
        pts2d = np.array(pts2d, dtype=np.float64)
        pts3d = np.array(pts3d, dtype=np.float64)

        focal = w
        center = (w/2, h/2)
        cam_m = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist = np.zeros((4,1), dtype=np.float64)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(pts3d, pts2d, cam_m, dist)
        rmat = cv2.Rodrigues(rotation_vector)[0] # rotation matrix

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        
        # decide text
        if y_angle < -15:
            head_text   = "Turned Left"
        elif y_angle > 15:
            head_text   = "Looking Right"
        elif x_angle < -7:
            head_text   = "Turning Down"
        elif x_angle > 20:
            head_text   = "Looking Up"
        else:
            head_text   = "Straight"
            
        distracted = head_text != "Straight"

    # overlay head-pose status
    #print(head_text)
    #cv2.putText(frame, head_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # ——— 2) EYE CLOSURE ———
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100)
    )

    eyes_closed = False
    if len(faces) > 0:
        x,y,fw,fh = faces[0]
        roi_gray  = gray[y:y+fh, x:x+fw]
        roi_color = frame[y:y+fh, x:x+fw]

        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(30,30)
        )
        if len(eyes) < 2:
            eyes_closed = True
        else:
            for (ex,ey,ew,eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

    elif len(faces) == 0:
        eyes_closed = True

    eye_text  = "Eyes Closed" if eyes_closed else "Eyes Open"
    eye_color = (0,0,255) if eyes_closed else (0,255,0)
    #print(eye_text)
    #cv2.putText(frame, eye_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)\

    return distracted or eyes_closed
