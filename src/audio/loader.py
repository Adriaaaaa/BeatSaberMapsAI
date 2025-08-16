from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import librosa
import soundfile as sf
import os

AUDIO_LIB = "librosa"  # or "soundfile"


class AudioLoader:
    """
    Class for loading audio files.
    """

    def __init__(
        self, audio_lib: str = AUDIO_LIB, sample_rate: int = 44100, mono: bool = True
    ):
        """
        Initialize the AudioLoader with the specified audio library.
        :param audio_lib: The audio library to use for loading audio files.
        """
        self.audio_lib = audio_lib
        self.sample_rate = sample_rate
        self.mono = mono

    def load_audio(self, map_folder: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the audio data and sample rate.
        :param file_path: Path to the audio file.
        :param sr: Sample rate for loading the audio, i.e. 44100 for 44.1 kHz it is the number of samples (echantillon) per second, i.e we record 44000 points of the sound curve per second (CD quality).
        :param mono: If True, convert the audio to mono, sufficient to extract spectral features, energy, RMS, etc.
        :return: Tuple containing the audio data as a numpy array and the sample rate.
        """
        audio_files = list(Path(map_folder).glob("*.ogg")) + list(
            Path(map_folder).glob("*.egg")
        )
        if not audio_files or len(audio_files) == 0:
            print(f"No audio files found in {map_folder}.")
            return np.array([]), self.sample_rate

        # Use the first audio file found
        song_path = audio_files[0]

        p = Path(song_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file {song_path} does not exist.")

        if self.audio_lib == "librosa":
            return self.load_audio_librosa(p)
        else:
            raise ValueError(f"Unsupported audio library: {self.audio_lib}")

    def load_audio_librosa(self, path: Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the audio data and sample rate.
        :param file_path: Path to the audio file.
        :param sr: Sample rate for loading the audio, i.e. 44100 for 44.1 kHz it is the number of samples (echantillon) per second, i.e we record 44000 points of the sound curve per second (CD quality).
        :param mono: If True, convert the audio to mono, sufficient to extract spectral features, energy, RMS, etc.
        :return: Tuple containing the audio data as a numpy array and the sample rate.
        """
        # librosa.load returns audio data in float32 format by default
        # and resamples to the specified sample rate
        audio_data, sample_rate = librosa.load(
            path, sr=self.sample_rate, mono=self.mono
        )
        return (_ensure_dtype(audio_data, "float32"), self.sample_rate)


def _ensure_dtype(data: np.ndarray, type: str) -> np.ndarray:
    """
    Ensure the numpy array is of the specified type.
    :param data: Input numpy array.
    :param type: Desired numpy type.
    :return: Numpy array with the specified type.
    """
    return data.astype(type, copy=False)
