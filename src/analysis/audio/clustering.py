from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import csv

from domain import TrackVector

from infra.cache import NpzCache
from infra.logger import LoggerManager
from infra.constants import *

log = LoggerManager.get_logger(__name__)


class TrackCluster:

    def __init__(self) -> None:
        self.cache = NpzCache(base_dir=".cache")

    def build_track_matrix(
        self, track_folders: Iterable[str], cache: NpzCache, vec_kind: str
    ) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
        """
        Build a matrix from a list of TrackVector objects.
        :param track_ids: Iterable of track IDs to process.
        :param cache: NpzCache object for caching the feature matrix.
        :param vec_kind: Kind of vector (e.g., "track_vector").
        :return: A tuple containing the feature matrix and a list of track IDs.
        """
        rows: List[np.ndarray] = []  # one row = one track vector
        kept_ids: List[str] = []  # return track ids that were kept (because in cache)
        names_ref: List[str] = []  # reference names of features (from first vector)

        for folder in track_folders:
            cached = self.cache.load(folder, KIND_TRACK_VECTOR, TRACK_VECTOR_VERSION)
        if cached is None:
            log.warning(f"TrackVector is not cached for {folder}")
        else:
            log.info(f"Loaded cached track vector for {folder}.")
            # If cached, we can directly create the AudioFeatures instance from the cached data
            vector = TrackVector.from_dict(cached)
