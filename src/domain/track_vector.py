from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict
import numpy as np
from dataclasses import dataclass, asdict

from pyparsing import Path

from domain.audio_features import AudioFeatures
from domain.vector_metadata import VectorMetadata

from utils.logger import LoggerManager
from utils.constants import *

log = LoggerManager.get_logger(__name__)


@dataclass
class TrackVector:
    # A vector representation of a music track for analysis.
    values: np.ndarray | None  # Numerical feature vector representing the track
    names: List[str] | None  # Names of the features corresponding to the values

    # metadata
    metadata: VectorMetadata | None

    def __init__(
        self,
        values: np.ndarray | None = None,
        names: List[str] | None = None,
        metadata: VectorMetadata | None = None,
    ):
        self.values = values if values is not None else np.array([], dtype=np.float32)
        self.names = names if names is not None else []
        self.metadata = metadata

    def to_dict(self) -> Dict:
        # Convert the values and names of the instance to a dictionary in order to be dumped in a npz
        # metadata are saved in a json in another function
        d = {}
        if self.values is not None:
            d["values"] = self.values.astype(np.float32)
        if self.names is not None:
            d["names"] = self.names
        return d

    @staticmethod
    def from_dict(d: Dict) -> TrackVector:
        # Create a TrackVector instance from a dictionary.
        d = dict(d)  # Make a copy to avoid modifying the original
        d["values"] = np.array(d["values"], dtype=np.float32)
        return TrackVector(**d)

    def complete_vector_with_features(
        self,
        features: AudioFeatures,
        oned_features_to_keep: List[str],
        twod_features_to_keep: List[str],
        stats_to_keep: List[str],
    ) -> None:
        # Create a TrackVector instance from AudioFeatures.
        if features is None:
            raise ValueError("features cannot be None")

        if self.values is None:
            log.error(
                "values is None, should call constructor with values=np.array([]) first"
            )
            raise ValueError(
                "values is None, should call constructor with values=np.array([]) first"
            )
        if not isinstance(self.values, np.ndarray):
            log.error("values should be a numpy array")
            raise TypeError("values should be a numpy array")

        if self.names is None:
            log.error("names is None, should call constructor with names=[] first")
            raise ValueError(
                "names is None, should call constructor with names=[] first"
            )

        if self.metadata is not None:
            log.info(
                f"Converting AudioFeatures to TrackVector for track_id={self.metadata.track_name}"
            )

        for feature_name in oned_features_to_keep:
            stats = features.feature_stats_1d(feature_name)
            for stat in stats_to_keep:
                if stat in stats:
                    self.values = np.append(self.values, stats[stat])
                    self.names.append(f"{feature_name}_{stat}")

        for feature_name in twod_features_to_keep:
            stats_2d = features.feature_stats_2d(feature_name)
            # stats is a list of array for each frequency bin
            # ex : chroma_stats = [chroma_bin_1_stats, chroma_bin_2_stats, ...]
            for i, stat_bin in enumerate(stats_2d):
                # Each stat is a dictionary of statistics for a specific frequency bin
                # stats_bin = chroma_bin_i_stats
                for sub_stat in stats_to_keep:
                    if sub_stat in stats_2d[stat_bin]:
                        self.values = np.append(
                            self.values, stats_2d[stat_bin][sub_stat]
                        )
                        self.names.append(f"{stat_bin}_{sub_stat}_{i}")

    def complete_vector_with_random_stuff(
        self, features: AudioFeatures
    ) -> None:  # manage other stats : tempo, density of beats and onsets

        if self.values is None:
            log.error(
                "values is None, should call constructor with values=np.array([]) first"
            )
            raise ValueError(
                "values is None, should call constructor with values=np.array([]) first"
            )

        if self.names is None:
            log.error("names is None, should call constructor with names=[] first")
            raise ValueError(
                "names is None, should call constructor with names=[] first"
            )

        if features.tempo is not None:
            self.values = np.append(self.values, features.tempo)
            self.names.append("tempo")

        duration = features.__getattribute__("duration_sec")

        if features.beat_times_sec is not None and len(features.beat_times_sec) > 1:
            if duration > 0:
                beat_density = len(features.beat_times_sec) / duration
                self.values = np.append(self.values, beat_density)
                self.names.append("beat_density")
        if features.onset_times_sec is not None and len(features.onset_times_sec) > 1:
            if duration > 0:
                onset_density = len(features.onset_times_sec) / duration
                self.values = np.append(self.values, onset_density)
                self.names.append("onset_density")

        # compute std onset interval and beat interval
        if features.onset_times_sec is not None and len(features.onset_times_sec) > 2:
            onset_intervals = np.diff(features.onset_times_sec)
            onset_interval_std = np.std(onset_intervals)
            self.values = np.append(self.values, onset_interval_std)
            self.names.append("onset_interval_std")

        if features.beat_times_sec is not None and len(features.beat_times_sec) > 2:
            beat_intervals = np.diff(features.beat_times_sec)
            beat_interval_std = np.std(beat_intervals)
            self.values = np.append(self.values, beat_interval_std)
            self.names.append("beat_interval_std")

    def complete_vector_with_cached_values(self, cached_values: Dict) -> None:
        if cached_values is None:
            log.error("Cached vector is None")
            return

        cached_vector = TrackVector.from_dict(cached_values)

        self.names = cached_vector.names
        self.values = cached_vector.values

    def summarize(self) -> str:
        # String representation of the TrackVector instance
        # on each line, print the name and the values
        if self.metadata is None:
            return "TrackVector(metadata=None)"

        lines = [f"TrackVector(track_id={self.metadata.track_name})"]

        if self.names is None or self.values is None:
            return "TrackVector(names=None, values=None)"

        for name, value in zip(self.names, self.values):
            name = str(name)
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)

    def validate(self) -> bool:
        # Validate the TrackVector instance.
        if self.metadata is None or not self.metadata.track_name:
            log.error("Invalid track_name")
            return False
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 1:
            log.error("Values must be a 1D numpy array")
            return False
        if not isinstance(self.names, list) or len(self.names) != len(self.values):
            log.error("Names must be a list with the same length as values")
            return False
        return True
