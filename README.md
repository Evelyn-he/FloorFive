# Never Miss a Moment Again with LoQ-in AI
LoQ-in AI is an intelligent desktop application that automatically captures and summarizes audio content when you're away or distracted. Whether you are watching your favorite movie, following an online lecture, or participating in Zoom meetings, LoQ-in AI will intelligently monitor your attention and status to ensure you don't miss any important content.

## How It Works
LoQ-in AI is a background application that uses computer vision models to detect when you are away or no longer actively viewing your screen. When it determines a user has been away or distracted for an extended period of time, the application will automatically begin recording the computer's audio output. Upon returning, a convenient pop-up appears with an AI-generated summary of what was missed.

## Use Cases
- Remote Work: Stay updated during work meetings when you need to step away briefly
- Education: Never miss important content during lecture recordings
- Entertainment: Catch up on live streams and shows after bathroom breaks or phone calls

## Privacy and Security
LoQ-in AI operates entirely offline with no data transmitted to external servers. Camera
input and audio recordings are processed locally and deleted after summarization. This
ensures wherever you are and whatever you are watching, your activity stays private and secure.


# Setup Instructions
1. Clone the project

    `git clone https://github.com/Evelyn-he/FloorFive.git`

2. Setup and activate a python virtual environment

    `python -m venv venv`

    Note: we require python `3.11.x` to be used for building the virutal environment.


3. Activate the virtual environement

    `source venv/bin/activate` or `.\venv\Scripts\activate` depending on your operating system.

4. Install project dependencies

    `pip install -r requirements.txt`

5. Update sound settings

   a. In your device's settings, go to `System > Sound > More Sound Settings`

   b. Click on the `Recording` tab

   c. Right click `Stereo Mix` and click `Enable`

6. Download whipser models

   a. Go to `https://aihub.qualcomm.com/compute/models/whisper_base_en?domain=Audio` and download both the decoder and encoder models for Onnx Runtime.

   b. Copy the models to this repository's `models/` folder with the following names:
           `whisper_base_en-whisperdecoderinf.onnx`
           `whisper_base_en-whisperencoderinf.onnx`

7. Download gemma model

    a. Go to https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/blob/main/onnx/model_q4f16.onnx and download the gemma-3 1b model.

    b. Copy the model to `models/gemma-3-1b-it-ONNX-GQA/` with the following name: `model_q4f15.onnx`

## Run and Usage Instructions

Within the python virtual environment run: `python start.py`

# Contributers
Evelyn He:

Haolin Ye:

Nathan Han:

Yathusan Koneswararajah:

Kevin Zhu:
