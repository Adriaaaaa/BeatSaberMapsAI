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

log = LoggerManager.get_logger(__name__)

VERSION = 2
AUDIO_LIB = "librosa"  # or "soundfile"


class AudioLoader:
    """
    Class for loading audio files.
    """

    def __init__(
        self,
        audio_lib: str = AUDIO_LIB,
        sample_rate: int = 44100,
        mono: bool = True,
        hop_length: int = 512,
    ):
        """
        Initialize the AudioLoader with the specified audio library.
        :param audio_lib: The audio library to use for loading audio files.
        """
        self.audio_lib = audio_lib
        self.sample_rate = sample_rate
        self.mono = mono
        self.hop = hop_length
        self.kind = f"audio_feature_sr{self.sample_rate}_hop{self.hop}_mono{self.mono}"
        self.cache = NpzCache(base_dir=".cache")

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

    def describe_audio(self, audio_data: np.ndarray):

        cached = self.cache.load(self.map_folder, self.kind, VERSION)
        if cached is None:
            feats = AudioFeatures.from_signal(audio_data, self.sample_rate, self.hop)
            dict_from_feats = feats.to_dict()
            self.cache.save(self.map_folder, dict_from_feats, self.kind, VERSION)
        else:
            log.info(f"Loaded cached audio features for {self.map_folder}.")
            # If cached, we can directly create the AudioFeatures instance from the cached data
            feats = AudioFeatures.from_dict(cached)

        print(feats.summary_str())


def _ensure_dtype(data: np.ndarray, type: str) -> np.ndarray:
    """
    Ensure the numpy array is of the specified type.
    :param data: Input numpy array.
    :param type: Desired numpy type.
    :return: Numpy array with the specified type.
    """
    return data.astype(type, copy=False)
