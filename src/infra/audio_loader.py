from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from domain.audio_features import AudioFeatures

import numpy as np
import librosa
import soundfile as sf
import os

from infra.cache import NpzCache
from infra.logger import LoggerManager
from infra.constants import *

log = LoggerManager.get_logger(__name__)


class AudioLoader:
    """
    Class for loading audio files.
    """

    def __init__(
        self,
        audio_lib: str = AUDIO_LIB,
        sample_rate: int = AUDIO_DEFAULT_SAMPLE_RATE,
        mono: bool = True,
    ):
        """
        Initialize the AudioLoader with the specified audio library.
        :param audio_lib: The audio library to use for loading audio files.
        """
        self.audio_lib = audio_lib
        self.sample_rate = sample_rate
        self.mono = mono

    def load_audio(self, map_folder: str) -> np.ndarray:
        """
        Load an audio file and return the audio data and sample rate.
        :param file_path: Path to the audio file.
        :param sr: Sample rate for loading the audio, i.e. 44100 for 44.1 kHz it is the number of samples (echantillon) per second, i.e we record 44000 points of the sound curve per second (CD quality).
        :param mono: If True, convert the audio to mono, sufficient to extract spectral features, energy, RMS, etc.
        :return: Tuple containing the audio data as a numpy array and the sample rate.
        """
        self.map_folder = map_folder
        audio_files = list(Path(map_folder).glob("*.ogg")) + list(
            Path(map_folder).glob("*.egg")
        )
        if not audio_files or len(audio_files) == 0:
            log.warning(f"No audio files found in {map_folder}.")
            return np.array([])

        # Use the first audio file found
        self.song_path = audio_files[0]

        p = Path(self.song_path)
        if not p.exists():
            log.error(f"Audio file {self.song_path} does not exist.")
            raise FileNotFoundError(f"Audio file {self.song_path} does not exist.")

        if self.audio_lib == "librosa":
            audio_data = self.load_audio_librosa(p)
            return audio_data
        else:
            log.error(f"Unsupported audio library: {self.audio_lib}")
            raise ValueError(f"Unsupported audio library: {self.audio_lib}")

    def load_audio_librosa(self, path: Path) -> np.ndarray:
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
        return _ensure_dtype(audio_data, "float32")


def _ensure_dtype(data: np.ndarray, type: str) -> np.ndarray:
    """
    Ensure the numpy array is of the specified type.
    :param data: Input numpy array.
    :param type: Desired numpy type.
    :return: Numpy array with the specified type.
    """
    return data.astype(type, copy=False)
