import sounddevice as sd
import soundfile as sf

def find_loopback_device():
    """Find the WASAPI loopback device for system audio capture."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for idx, device in enumerate(devices):
        if 'stereo mix' in device['name'].lower():
            hostapi_name = hostapis[device['hostapi']]['name'].lower()
            if 'wasapi' in hostapi_name:
                return idx
    raise RuntimeError("No WASAPI loopback device found. Ensure you're on Windows 10+ and using a compatible sound driver.")

def record_system_audio(filename='system_audio.wav', channels=2):
    device_index = find_loopback_device()
    device_info = sd.query_devices(device_index)
    samplerate = int(device_info['default_samplerate'])
    print(f"Using device: {sd.query_devices(device_index)['name']}")
    with sf.SoundFile(filename, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_16') as file:
        print("Recording system audio... Press Enter to stop.")
        try:
            with sd.InputStream(samplerate=samplerate, device=device_index, channels=channels, dtype='int16',
                                callback=lambda indata, frames, time, status: file.write(indata)):
                input()  # Wait for user to press Enter
        except KeyboardInterrupt:
            print("\nRecording interrupted by user.")
        print("Recording stopped and saved to", filename)

# Run the recording function

record_system_audio()