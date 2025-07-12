import sounddevice as sd
import soundfile as sf
import threading
import time
import os

AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), "audio_recordings")

class SystemAudioRecorder:
    def __init__(self):
        self.recording = False
        self.stream = None
        self.file = None
        self.thread = None
        self.device_index = None
        self.samplerate = None
        self.recording_number = 0
        
    def find_loopback_device(self):
        """Find the WASAPI loopback device for system audio capture."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for idx, device in enumerate(devices):
            if 'stereo mix' in device['name'].lower():
                hostapi_name = hostapis[device['hostapi']]['name'].lower()
                if 'wasapi' in hostapi_name:
                    return idx
        raise RuntimeError("No WASAPI loopback device found. Ensure you're on Windows 10+ and using a compatible sound driver.")
    
    def start_recording(self, channels=2):
        """Start recording system audio."""
        if self.recording:
            print("Already recording!")
            return False

        filename = os.path.join(AUDIO_FOLDER, f'audio_{self.recording_number}.wav')
            
        try:
            self.device_index = self.find_loopback_device()
            device_info = sd.query_devices(self.device_index)
            self.samplerate = int(device_info['default_samplerate'])
            
            print(f"Using device: {sd.query_devices(self.device_index)['name']}")
            
            # Open the sound file for writing
            self.file = sf.SoundFile(filename, mode='w', samplerate=self.samplerate, 
                                   channels=channels, subtype='PCM_16')

            self.recording_number += 1

            # Create the input stream
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                device=self.device_index,
                channels=channels,
                dtype='int16',
                callback=self._audio_callback
            )
            
            self.stream.start()
            self.recording = True
            print(f"Recording started. Saving to {filename}")
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            if self.file:
                self.file.close()
            return False
    
    def stop_recording(self):
        """Stop recording system audio."""
        if not self.recording:
            print("Not currently recording!")
            return False
            
        try:
            self.recording = False
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            if self.file:
                self.file.close()
                filename = self.file.name
                self.file = None
                print(f"Recording stopped and saved to {filename}")
            
            return True
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function to write audio data to file."""
        if status:
            print(f"Audio callback status: {status}")
        if self.recording and self.file:
            self.file.write(indata)
    
    def is_recording(self):
        """Check if currently recording."""
        return self.recording

    def audio_cleanup(self):
        """Remove all files from the 'audio recordings' directory"""
        for filename in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
