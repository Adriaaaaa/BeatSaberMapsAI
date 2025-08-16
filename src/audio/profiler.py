import numpy as np
from typing import Dict, Any, Tuple


class AudioProfiler:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def describe_audio(self, audio_data: np.ndarray):
        """
        Analyze the audio data and return a description.
        :param audio_data: Audio data as a numpy array.
        :return: Dictionary with audio analysis results.
        """
        duration = len(audio_data) / self.sample_rate
        mean_amplitude = np.mean(np.abs(audio_data))
        max_amplitude = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(np.square(audio_data)))

        print(f"Audio Duration: {duration:.2f} seconds")
        print(f"Mean Amplitude: {mean_amplitude:.4f}")
        print(f"Max Amplitude: {max_amplitude:.4f}")
        print(f"RMS: {rms:.4f}")
