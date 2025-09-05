from enum import Enum

AUDIO_LIB = "librosa"  # or "soundfile"

AUDIO_DEFAULT_SAMPLE_RATE = 44_100
AUDIO_DEFAULT_HOP = 512
AUDIO_DEFAULT_NFFT = 2048

TRACK_VECTOR_VERSION = 6
AUDIO_FEATURES_VERSION = 5

GRID_COLS = 4
GRID_ROWS = 3

LOGS_DIR = "logs"
MAPS_DIR = "maps"
CACHE_DIR = ".cache"

KIND_TRACK_VECTOR = "track_vector"
KIND_AUDIO_FEATURE = "audio_feature"


DIRECTION_TO_VECTOR = {
    0: (0, 1),  # Up
    1: (0, -1),  # Bas
    2: (-1, 0),  # Left
    3: (1, 0),  # Right
    4: (-1, 1),  # Left Up
    5: (1, 1),  # Right Up
    6: (-1, -1),  # Left Down
    7: (1, -1),  # Right Down
    8: (0, 0),  # Free
}


DIRECTION_TO_ANGLE = {
    0: 90,  # Up
    1: 270,  # Down
    2: 180,  # Left
    3: 0,  # Right
    4: 135,  # UpLeft
    5: 45,  # UpRight
    6: 225,  # DownLeft
    7: 315,  # DownRight
}

NUM_CLUSTERS = 6

VERSION_MODEL_CLUSTERING = 3


STATS_ORDER = [
    "mean",
    "std",
    # "p10",
    # "p50",
    # "p90",
    # "skew",
    # "kurt"
]

ONED_FEATURES = [
    # "rms_values",
    # "centroid_values",
    # "flux_values",
    # "bandwidth_values",
    # "rollof_values",
    # "rms_harmonic_values",
    # "rms_percussive_values",
]

TWOD_FEATURES = ["chroma", "mfcc_values"]
