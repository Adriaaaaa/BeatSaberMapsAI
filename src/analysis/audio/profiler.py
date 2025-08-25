import numpy as np
from domain import AudioFeatures
from domain import TrackVector

from typing import Dict, Any, Tuple, Optional, List

from pathlib import Path

from infra.cache import NpzCache
from infra.logger import LoggerManager
from infra.constants import *

log = LoggerManager.get_logger(__name__)


class AudioProfiler:
    def __init__(
        self,
        sample_rate: int = AUDIO_DEFAULT_SAMPLE_RATE,
        hop_length: int = AUDIO_DEFAULT_HOP,
        mono: bool = True,
    ):
        self.sample_rate = sample_rate
        self.audio_features = None
        self.mono = mono
        self.hop = hop_length
        self.cache = NpzCache(base_dir=CACHE_DIR)
        self.kind_audio_features = (
            KIND_AUDIO_FEATURE + f"_sr{self.sample_rate}_hop{self.hop}_mono{self.mono}"
        )

    def extract_audio_features(
        self, audio_data: np.ndarray, map_folder: str | Path
    ) -> Optional[AudioFeatures]:
        """
        Analyze the audio data and return a description.
        :param audio_data: Audio data as a numpy array.
        :return: Dictionary with audio analysis results.
        """
        if audio_data.size == 0:
            print("No audio data provided.")
            return None

        cached = self.cache.load(
            map_folder, self.kind_audio_features, AUDIO_FEATURES_VERSION
        )
        if cached is None:
            feats = AudioFeatures.from_signal(audio_data, self.sample_rate, self.hop)
            dict_from_feats = feats.to_dict()
            self.cache.save(
                map_folder,
                dict_from_feats,
                self.kind_audio_features,
                AUDIO_FEATURES_VERSION,
            )
        else:
            log.info(f"Loaded cached audio features for {map_folder}.")
            # If cached, we can directly create the AudioFeatures instance from the cached data
            feats = AudioFeatures.from_dict(cached)

        return feats

    def convert_features_to_vector(
        self, track_id: str, features: AudioFeatures, map_folder: str | Path
    ) -> Optional["TrackVector"]:
        """
        Convert AudioFeatures to a TrackVector.
        :param track_id: Unique identifier for the track.
        :param features: AudioFeatures instance.
        :return: TrackVector instance or None if features is None.
        """

        if features is None:
            return None

        cached = self.cache.load(map_folder, KIND_TRACK_VECTOR, TRACK_VECTOR_VERSION)
        if cached is None:
            vector = TrackVector.from_features(track_id, features)
            dict_from_vector = vector.to_dict()
            self.cache.save(
                map_folder,
                dict_from_vector,
                KIND_TRACK_VECTOR,
                TRACK_VECTOR_VERSION,
            )
        else:
            log.info(f"Loaded cached track vector for {map_folder}.")
            # If cached, we can directly create the AudioFeatures instance from the cached data
            vector = TrackVector.from_dict(cached)

        return vector
