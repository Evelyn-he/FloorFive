import keyboard
import threading
import cv2
import numpy as np
from detection_model import is_distracted
from popup.chatbot_launcher import launch_chatbot, kill_chatbot
from audio_pipeline.record_system_audio import SystemAudioRecorder, resample_to_16k
from whisper_pipeline.transcriber import transcribe

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

# launch audio recording
recorder = SystemAudioRecorder()

# launch chatbot
chatbot = launch_chatbot()


def process_audio(filename):
    resample_to_16k(filename)
    transcript = transcribe(filename)
    with open("popup/text.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + transcript)

while cap.isOpened():
    ret, frame = cap.read()
    if ret is None:
        continue

    # convert frame to RGB for detection model
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    distracted = is_distracted(frame)

    distracted_buffer[buffer_idx] = distracted
    distraction_avg = np.average(distracted_buffer)
    buffer_idx = (buffer_idx + 1) % buffer_size

    # Check when to start/stop recording

    if not IS_RECORDING and distraction_avg >= 0.6:
        IS_RECORDING = True
        recorder.start_recording()
        print('START RECORDING')

    if IS_RECORDING and not MANUAL_OVERRIDE and distraction_avg <= 0.5:
        IS_RECORDING = False
        wav_file = recorder.stop_recording()
        threading.Thread(target=process_audio, args=(wav_file, ), daemon=True).start()

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

        if IS_RECORDING:
            recorder.start_recording()
        else:
            wav_file = recorder.stop_recording()
            threading.Thread(target=process_audio, args=(wav_file, ), daemon=True).start()

    # check for escape key
    if keyboard.is_pressed('esc'):
        break

    # require waitKey to update frames
    cv2.waitKey(1)

# kill chatbot
kill_chatbot(chatbot)

# Clean up .wav files
recorder.stop_recording()
recorder.audio_cleanup()

cap.release()
cv2.destroyAllWindows()