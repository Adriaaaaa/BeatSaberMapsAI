import numpy as np
import pytest
import domain

from domain.track_vector import TrackVector
from domain.audio_features import AudioFeatures


# ---------------- Test 1 : forme - Nan ----------------#
def test_vector_propre_forme_nan(tone_440, sample_rate):

    feats = AudioFeatures.from_signal(tone_440, sample_rate, hop_length=512)
    vector = TrackVector()
    vector.complete_vector_with_features(
        features=feats,
        oned_features_to_keep=[],
        twod_features_to_keep=["mfcc_values", "chroma"],
        stats_to_keep=["mean", "std"],
    )
    print(vector.summarize())

    assert isinstance(vector.values, np.ndarray)
    assert vector.values is not None
    assert vector.names is not None
    assert not np.any(np.isnan(vector.values)), "Vector contains NaN values"
    assert vector.values.shape[0] == len(vector.names)
    assert vector.values.shape[0] > 0
    assert vector.values.shape[0] == 28  # 13 mfcc*2stats + 12*chroma*2stats = 50
