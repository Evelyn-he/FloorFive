import numpy as np
import os
import queue
import sounddevice as sd
import sys
import threading
import yaml
import wave

from concurrent.futures import ThreadPoolExecutor
from whisper_pipeline.model import WhisperBaseEnONNX
from qai_hub_models.models.whisper_base_en import App as WhisperApp

def load_wav_file(path: str, sample_rate: int) -> np.ndarray:
    """
    Load a WAV file and return it as a normalized numpy float32 array.
    Assumes mono channel WAV file.
    """
    with wave.open(path, 'rb') as wf:
        if wf.getframerate() != sample_rate:
            raise ValueError(f"Expected sample rate {sample_rate}, got {wf.getframerate()}")
        if wf.getnchannels() != 1:
            raise ValueError("Only mono-channel WAV files are supported")

        audio_bytes = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0  # normalize to [-1.0, 1.0]
    return audio_np

def process_transcription(
    whisper: WhisperApp,
    chunk: np.ndarray,
    silence_threshold: float,
    sample_rate: int
) -> None:
    """
    Process a chunk of audio data and transcribe it using the Whisper model.
    This function is run in a separate thread to allow for concurrent processing.

    Inputs:
    - whisper: WhisperApp instance for transcription
    - chunk: Audio data chunk to be transcribed (numpy array)
    - silence_threshold: Threshold for silence detection
    - sample_rate: Sample rate for audio recording
    """
    
    if np.abs(chunk).mean() > silence_threshold:
        transcript = whisper.transcribe(chunk, sample_rate)
        if transcript.strip():    
            return transcript.strip()


def transcribe_wav_file(
    wav_path: str,
    whisper: WhisperApp,
    chunk_duration: float = 4.0,
    silence_threshold: float = 0.001,
    max_workers: int = 4,
    sample_rate: int = 16000
) -> None:
    """
    Transcribe a WAV file using the Whisper model in chunks.

    - wav_path: Path to the .wav file
    - whisper: WhisperApp instance
    - chunk_duration: Duration of each chunk in seconds
    - silence_threshold: Skip chunks with mean volume below this threshold
    - max_workers: Number of parallel workers
    - sample_rate: Audio sample rate
    """
    audio = load_wav_file(wav_path, sample_rate)
    chunk_samples = int(sample_rate * chunk_duration)
    print(f"Loaded {len(audio)} samples from '{wav_path}', processing in {chunk_samples}-sample chunks...")

    full_transcript = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            futures.append(executor.submit(
                process_transcription,
                whisper,
                chunk,
                silence_threshold,
                sample_rate
            ))

        for future in futures:
            transcript = future.result()
            if transcript:
                full_transcript.append(transcript)

    # Return as a single line
    return " ".join(full_transcript)

class LiveTranscriber:
    def __init__(self):

        
        # audio settings
        self.sample_rate = 16000 # config.get("sample_rate", 16000)
        self.chunk_duration = 4 # config.get("chunk_duration", 4)
        self.channels = 1 # config.get("channels", 1)
        
        # processing settings
        self.max_workers = 4 # config.get("max_workers", 4)
        self.silence_threshold = 0.001 # config.get("silence_threshold", 0.001)
        self.queue_timeout = 1.0 # config.get("queue_timeout", 1.0)
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # model paths
        self.encoder_path = "models/whisper_base_en-whisperencoderinf.onnx" # config.get("encoder_path", "models/whisper_base_en-whisperencoderinf.onnx")
        self.decoder_path = "models/whisper_base_en-whisperdecoderinf.onnx" # config.get("decoder_path", "models/whisper_base_en-whisperdecoderinf.onnx")

        # check that the model paths exist
        if not os.path.exists(self.encoder_path):
            sys.exit(f"Encoder model not found at {self.encoder_path}.")
        if not os.path.exists(self.decoder_path):
            sys.exit(f"Decoder model not found at {self.decoder_path}.")

        # initialize the model
        print("Loading model...")
        self.model = WhisperApp(WhisperBaseEnONNX(self.encoder_path, self.decoder_path))

        # initialize the audio queue and stop event
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

    def run(self):
        """
        Run the live transcription.
        """
        
        # launch the audio processing and recording threads
        process_thread = threading.Thread(
            target=process_audio, 
            args=(
                self.model,
                self.audio_queue,
                self.stop_event,
                self.max_workers,
                self.queue_timeout,
                self.chunk_samples,
                self.silence_threshold,
                self.sample_rate
            )
        )
        process_thread.start()

        record_thread = threading.Thread(
            target=record_audio, 
            args=(
                self.audio_queue,
                self.stop_event,
                self.sample_rate,
                self.channels
            )
        )
        record_thread.start()

        # wait for threads to finish
        try:
            while True:
                record_thread.join(timeout=0.1)
                if not record_thread.is_alive():
                    break
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        finally:
            self.stop_event.set()
            record_thread.join()
            process_thread.join()
            print("Transcription stopped.")


def transcribe(wav_file):
    # --- Configuration ---
    encoder_path = "models/whisper_base_en-whisperencoderinf.onnx"
    decoder_path = "models/whisper_base_en-whisperdecoderinf.onnx"
    sample_rate = 16000
    chunk_duration = 4.0
    silence_threshold = 0.001
    max_workers = 4

    # --- Checks ---
    if not os.path.exists(wav_file):
        sys.exit(f"Error: WAV file not found at: {wav_file}")
    if not os.path.exists(encoder_path):
        sys.exit(f"Error: Encoder model not found at: {encoder_path}")
    if not os.path.exists(decoder_path):
        sys.exit(f"Error: Decoder model not found at: {decoder_path}")

    # --- Load Model ---
    print("Loading Whisper ONNX model...")
    whisper_model = WhisperApp(WhisperBaseEnONNX(encoder_path, decoder_path))

    # --- Transcribe ---
    transcript = transcribe_wav_file(
        wav_path=wav_file,
        whisper=whisper_model,
        chunk_duration=chunk_duration,
        silence_threshold=silence_threshold,
        max_workers=max_workers,
        sample_rate=sample_rate
    )

    return transcript
