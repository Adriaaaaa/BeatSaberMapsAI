from __future__ import annotations
from typing import List, Dict
import numpy as np
from dataclasses import dataclass, asdict
from domain.audio_features import AudioFeatures

STATS_order = ["mean", "std", "min", "p10", "p50", "p90", "max"]

from infra.logger import LoggerManager

log = LoggerManager.get_logger(__name__)


@dataclass
class TrackVector:
    # A vector representation of a music track for analysis.
    track_id: str  # Unique identifier for the track
    values: np.ndarray  # Numerical feature vector representing the track
    names: List[str]  # Names of the features corresponding to the values

    def to_dict(self) -> Dict:
        # Convert the TrackVector instance to a dictionary.
        d = asdict(self)
        d["values"] = self.values.astype(np.float32)
        d["names"] = list(self.names)
        return d

    @staticmethod
    def from_dict(d: Dict) -> TrackVector:
        # Create a TrackVector instance from a dictionary.
        d = dict(d)  # Make a copy to avoid modifying the original
        d["values"] = np.array(d["values"], dtype=np.float32)
        d["names"] = [str(name) for name in d["names"]]
        return TrackVector(**d)

    @classmethod
    def from_features(
        cls, track_id: str, features: AudioFeatures, version: int = 1
    ) -> TrackVector:
        # Create a TrackVector instance from AudioFeatures.
        if features is None:
            return None

        vector = []
        names = []

        log.info(f"Converting AudioFeatures to TrackVector for track_id={track_id}")

        # Basic features 1D statistics
        one_d_features = [
            "rms_values",
            "centroid_values",
            "flux_values",
            "bandwidth_values",
            "rollof_values",
            "rms_harmonic_values",
            "rms_percussive_values",
        ]

        for feature_name in one_d_features:
            log.info(f"Processing feature: {feature_name}")
            stats = features.feature_stats_1d(feature_name)
            for stat in STATS_order:
                if stat in stats:
                    vector.append(stats[stat])
                    names.append(f"{feature_name}_{stat}")

        for feature_name in ["chroma", "mfcc_values"]:
            log.info(f"Processing feature: {feature_name}")
            stats = features.feature_stats_2d(feature_name)
            for stat in STATS_order:
                if stat in stats:
                    # For 2D features, we flatten the stats array
                    flattened = stats[stat].flatten()
                    vector.extend(flattened.tolist())
                    names.extend(
                        [f"{feature_name}_{stat}_{i}" for i in range(len(flattened))]
                    )

        # manage other stats : tempo, density of beats and onsets
        if features.tempo is not None:
            vector.append(features.tempo)
            names.append("tempo")

        duration = features.__getattribute__("duration_sec")

        if features.beat_times_sec is not None and len(features.beat_times_sec) > 1:
            if duration > 0:
                beat_density = len(features.beat_times_sec) / duration
                vector.append(beat_density)
                names.append("beat_density")
        if features.onset_times_sec is not None and len(features.onset_times_sec) > 1:
            if duration > 0:
                onset_density = len(features.onset_times_sec) / duration
                vector.append(onset_density)
                names.append("onset_density")

        # compute std onset interval and beat interval
        if features.onset_times_sec is not None and len(features.onset_times_sec) > 2:
            onset_intervals = np.diff(features.onset_times_sec)
            onset_interval_std = np.std(onset_intervals)
            vector.append(onset_interval_std)
            names.append("onset_interval_std")

        if features.beat_times_sec is not None and len(features.beat_times_sec) > 2:
            beat_intervals = np.diff(features.beat_times_sec)
            beat_interval_std = np.std(beat_intervals)
            vector.append(beat_interval_std)
            names.append("beat_interval_std")

        return cls(
            track_id=track_id, values=np.array(vector, dtype=np.float32), names=names
        )

    def summarize(self) -> str:
        # String representation of the TrackVector instance
        # on each line, print the name and the values
        lines = [f"TrackVector(track_id={self.track_id})"]
        for name, value in zip(self.names, self.values):
            name = str(name)
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)

    def validate(self) -> bool:
        # Validate the TrackVector instance.
        if not self.track_id or not isinstance(self.track_id, str):
            log.error("Invalid track_id")
            return False
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 1:
            log.error("Values must be a 1D numpy array")
            return False
        if not isinstance(self.names, list) or len(self.names) != len(self.values):
            log.error("Names must be a list with the same length as values")
            return False
        return True
