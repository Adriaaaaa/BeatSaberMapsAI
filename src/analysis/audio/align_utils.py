from __future__ import annotations
from typing import List, Tuple
import numpy as np

from infra.logger import LoggerManager
from domain import AudioFeatures

log = LoggerManager.get_logger(__name__)


features_attributes_to_align_1D = [
    "rms_values",
    "centroid_values",
    "flux_values",
    "bandwidth_values",
    "rollof_values",
    "rms_harmonic_values",
    "rms_percussive_values",
]

features_attributes_to_align_2D = ["mfcc_values", "chroma"]


def _analyze_lengths(features: AudioFeatures) -> Tuple[int, int]:
    """
    Analyze the lengths of the audio features.
    :param features: AudioFeatures instance.
    :return: Tuple containing the minimum, maximum lengths.
    """
    if features is None:
        return 0, 0

    if features.frames_times_sec is None:
        return 0, 0

    lengths = [features.frames_times_sec.size]

    for attr in features_attributes_to_align_1D:
        values = getattr(features, attr)
        if values is not None:
            lengths.append(values.size)

    for attr in features_attributes_to_align_2D:
        values = getattr(features, attr)
        if values is not None:
            lengths.append(values.shape[1])

    min_length = np.min(lengths)
    max_length = np.max(lengths)

    return min_length, max_length


def align_audio_features(features: AudioFeatures) -> AudioFeatures:
    """
    Align the audio features to the same length.
    :param features: AudioFeatures instance.
    :return: Aligned AudioFeatures instance.
    """
    if features is None:
        return None

    min_length, max_length = _analyze_lengths(features)
    log.info(f"Aligning audio features to length {min_length} (max was {max_length})")

    for attr in features_attributes_to_align_1D:
        values = getattr(features, attr)
        if values is not None and values.size > min_length:
            setattr(features, attr, values[:min_length])

    for attr in features_attributes_to_align_2D:
        values = getattr(features, attr)
        if values is not None and values.shape[1] > min_length:
            setattr(features, attr, values[:, :min_length])

    if (
        features.frames_times_sec is not None
        and features.frames_times_sec.size > min_length
    ):
        features.frames_times_sec = features.frames_times_sec[:min_length]

    return features
