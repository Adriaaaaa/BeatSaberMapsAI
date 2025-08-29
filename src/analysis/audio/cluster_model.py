import numpy as np
import joblib

from typing import Iterable

from .track_cluster import TrackCluster
from infra.cache import NpzCache
from infra.logger import LoggerManager

from infra.constants import *

log = LoggerManager.get_logger(__name__)


class ClusterModel:

    def __init__(self):
        self.cache = NpzCache(base_dir=".cache")
        self.clusterer = TrackCluster()

    def train(self, track_folders: Iterable[str]) -> None:
        """
        Train the clustering model on the track vectors.
        :param track_folders: Iterable of track folders to process.
        :return: Silhouette score or None if fitting fails.
        """
        matrix, kept_folders = self.clusterer.build_track_matrix(track_folders)

        if not matrix.size or not kept_folders:
            log.warning("No valid tracks found.")
            return None

        silhouette_score = self.clusterer.fit(matrix)

        if silhouette_score is not None:
            log.info(
                f"Training completed successfully. Silhouette score: {silhouette_score}"
            )
            log.info(f"Kept folders: {kept_folders}")

        else:
            log.warning("Training failed.")

        log.info(f"Number of clusters: \n")
        log.info(self.clusterer.cluster_size(matrix))

        labels = self.clusterer.predict(matrix)

        if labels is None:
            log.warning("Prediction failed.")
            return None

        self.clusterer.save_clusters_csv(
            f"models/track_clusters_v{VERSION_MODEL_CLUSTERING}.csv",
            kept_folders,
            labels,
        )

        self.clusterer.save_model(
            f"models/track_cluster_model_v{VERSION_MODEL_CLUSTERING}"
        )

    def predict(self, track_folder: str) -> tuple[list[str], np.ndarray] | None:
        """
        Predict the cluster for a given track folder.
        :param track_folder: Path to the track folder.
        :return: Cluster label or None if prediction fails.
        """

        if track_folder is None:
            log.warning("No track folder provided.")
            return None

        matrix, kept_folders = self.clusterer.build_track_matrix([track_folder])

        self.clusterer.load_model(
            f"models/track_cluster_model_v{VERSION_MODEL_CLUSTERING}"
        )

        if self.clusterer.feature_names is None:
            log.warning("Feature extraction failed.")
            return None

        if len(self.clusterer.feature_names) != matrix.shape[1]:
            log.warning("Feature mismatch.")
            return None

        labels = self.clusterer.predict(matrix)

        if labels is None:
            log.warning("Prediction failed.")
            return None

        return kept_folders, labels
