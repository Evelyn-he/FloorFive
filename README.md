# FloorFive

# Setup
1. Clone the repo

    `git clone https://github.com/Evelyn-he/FloorFive.git`

2. Setup python virtual environment
    
    `python -m venv venv`

3. Activate virtual environment

    `source venv/bin/activate` 

4. Install project dependencies

    `pip install -r requirements.txt`

5. Download whipser models

   a. Go to `https://aihub.qualcomm.com/compute/models/whisper_base_en?domain=Audio` and download both the decoder and encoder models for Onnx Runtime.

   b. Copy the models to this repository's `models/` folder with the following names:
           `whisper_base_en-whisperdecoderinf.onnx`
           `whisper_base_en-whisperencoderinf.onnx`

7. Update sound settings

   a. In your device's settings, go to `System > Sound > More Sound Settings`

   b. Click on the `Recording` tab

   c. Right click `Stereo Mix` and click `Enable`
