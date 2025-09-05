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
from utils.logger import LoggerManager
from utils.constants import *

log = LoggerManager.get_logger(__name__)


class TrackCluster:

    # Scaler for feature normalization
    # a standard scaler is used to standardize features by removing the mean and scaling to unit variance
    # in simple words, it makes sure all features are on the same scale
    # it will be used to preprocess the feature matrix before clustering
    # it is important to fit the scaler only on the training data
    # the fitted scaler can then be used to transform the training data and any new data
    # this ensures that the model is not biased by the scale of the features

    scaler: StandardScaler | None = None  # same as Optional[StandardScaler]

    # KMeans clustering model
    # KMeans is a popular clustering algorithm that partitions data into K distinct clusters
    # It works by iteratively assigning data points to the nearest cluster centroid and updating the centroids based on the assigned points
    # The algorithm aims to minimize the within-cluster variance, making the clusters as compact and well-separated as possible
    # it stops when the centroids no longer change significantly or a maximum number of iterations is reached
    kmeans: KMeans | None = None  # same as Optional[KMeans]

    # Feature names for the track vectors
    # These are the names of the features used in the track vectors
    feature_names: List[str] = []

    cache: NpzCache

    nb_clusters: int = NUM_CLUSTERS

    def __init__(self, nb_clusters: int = NUM_CLUSTERS) -> None:
        self.cache = NpzCache(base_dir=".cache")

        # The KMeans clustering algorithm
        # initialization the KMeans object
        # with the number of clusters and other parameters
        self.kmeans = KMeans(n_clusters=nb_clusters, random_state=42, n_init="auto")

    def build_track_matrix(
        self, track_folders: Iterable[str], unique_id: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a matrix from a list of TrackVector objects.
        :param track_ids: Iterable of track IDs to process.
        :param cache: NpzCache object for caching the feature matrix.
        :param vec_kind: Kind of vector (e.g., "track_vector").
        :return: A tuple containing the feature matrix and a list of track IDs.
        """
        if track_folders is None:
            return None, []

        rows: List[np.ndarray] = []  # one row = one track vector
        kept_folders: List[str] = (
            []
        )  # return track ids that were kept (because in cache)

        for folder in track_folders:
            cached = self.cache.load(
                folder, KIND_TRACK_VECTOR, unique_id, TRACK_VECTOR_VERSION
            )
            if cached is None:
                log.warning(f"TrackVector is not cached for {folder}")
            else:
                log.info(f"Loaded cached track vector for {folder}.")
                # If cached, we can directly create the AudioFeatures instance from the cached data
                vector = TrackVector.from_dict(cached)
                vector.validate()
                if self.feature_names == [] and vector.names:
                    self.feature_names = vector.names
                if vector.values is not None:
                    rows.append(vector.values.astype(np.float32))
                kept_folders.append(folder)

        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []

        if self.feature_names == []:
            log.warning("No feature names found, chelou.")

        matrix = np.vstack(rows).astype(np.float32)

        return matrix, kept_folders

    def fit(self, matrix: np.ndarray) -> float | None:
        """
        Fit the clustering model to the track vectors.
        :param matrix: Feature matrix where each row corresponds to a track vector."""

        if matrix is None:
            log.warning("No track matrix provided.")
            return

        if matrix.ndim != 2:
            log.warning("Invalid matrix shape.")
            return

        if matrix.shape[0] < NUM_CLUSTERS:
            log.warning(
                f"Not enough track vectors to form {NUM_CLUSTERS} clusters. Only {matrix.shape[0]} vectors found."
            )
            return

        # Fit the scaler to the track matrix
        # This ensures that the features are normalized before clustering
        # in simple words, it makes sure all features contribute equally
        # so that no single feature dominates the clustering process
        # this is especially important for distance-based algorithms like KMeans
        self.scaler = StandardScaler().fit(matrix)

        # Transform the matrix using the fitted scaler
        # This ensures that the features are on the same scale
        matrix_scaled = self.scaler.transform(matrix)

        if self.kmeans is None:
            log.error("KMeans model is not initialized.")
            return

        # Fit the KMeans model to the scaled matrix
        # This groups the track vectors into clusters
        self.kmeans.fit(matrix_scaled)

        # Get the cluster labels for each track vector
        labels = self.kmeans.labels_

        # Calculate the silhouette score to evaluate clustering quality
        # This metric indicates how similar an object is to its own cluster
        # A higher silhouette score indicates better-defined clusters
        silhouette = silhouette_score(matrix_scaled, labels)

        log.info(f"Clustering completed. Labels: {labels}")
        log.info(f"Silhouette Score: {silhouette}")

        if silhouette < 0:
            log.warning("Clustering quality is poor.")
        elif silhouette < 0.5:
            log.info("Clustering quality is acceptable.")
        else:
            log.info("Clustering quality is good.")

        return float(silhouette)

    def predict(self, matrix: np.ndarray) -> np.ndarray | None:
        """
        Predict the cluster labels for the given track vectors.
        :param matrix: Feature matrix where each row corresponds to a track vector.
        :return: Array of cluster labels or None if prediction fails.
        """
        if matrix is None:
            log.warning("No track matrix provided.")
            return None

        if self.kmeans is None:
            log.error("KMeans model is not initialized.")
            return None

        if self.scaler is None:
            log.error("Model not trained. Scaler is not fitted.")
            return None

        # Transform the matrix using the fitted scaler
        matrix_scaled = self.scaler.transform(matrix)

        # Predict the cluster labels for the scaled matrix
        labels = self.kmeans.predict(matrix_scaled)

        return labels

    def save_model(self, model_path: str) -> None:
        """
        Save the trained clustering model to a file.
        :param model_path: Path to the file where the model should be saved.
        """
        if self.kmeans is None:
            log.error("KMeans model is not initialized.")
            return

        if self.scaler is None:
            log.error("Model not trained. Scaler is not fitted.")
            return

        joblib.dump((self.kmeans, self.scaler), model_path)
        log.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """
        Load the trained clustering model from a file.
        :param model_path: Path to the file from which the model should be loaded.
        """
        try:
            self.kmeans, self.scaler = joblib.load(model_path)
            log.info(f"Model loaded from {model_path}")
        except Exception as e:
            log.error(f"Failed to load model from {model_path}: {e}")

    def cluster_size(self, matrix: np.ndarray | None) -> dict[int, int] | None:
        """
        Get the size of each cluster.
        :return: Array of cluster sizes or None if model is not trained.
        """
        if self.kmeans is None:
            log.error("KMeans model is not initialized.")
            return None

        if self.scaler is None:
            log.error("Model not trained. Scaler is not fitted.")
            return None

        # Get the cluster labels
        labels = self.kmeans.labels_ if matrix is None else self.predict(matrix)
        n_clusters = getattr(self.kmeans, "n_clusters", NUM_CLUSTERS)

        return {i: np.sum(labels == i) for i in range(n_clusters)}

    def save_clusters_csv(
        self,
        file_path: str,
        track_folders: list[str],
        labels: np.ndarray,
        extras: dict[str, list | np.ndarray] | None = None,
    ) -> None:
        """
        Save the cluster sizes to a CSV file.
        :param file_path: Path to the CSV file.
        :param track_folders: List of track folder paths.
        :param labels: Array of cluster labels for each track.
        :param extras: Optional dictionary of additional data to include in the CSV.
        """
        with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
            cols = ["Cluster Label"] + [i for i in range(NUM_CLUSTERS)]
            writer = csv.writer(csvfile)
            writer.writerow(cols)
            for i, folder in enumerate(track_folders):
                row = [""]
                for j in range(NUM_CLUSTERS):
                    if labels[i] == j:
                        row.append(folder)
                    else:
                        row.append("")
                writer.writerow(row)
