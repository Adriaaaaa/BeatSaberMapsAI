from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import json, time

from infra.logger import LoggerManager

log = LoggerManager.get_logger(__name__)


class NpzCache:
    """
    A simple cache system that stores data in a .npz file.
    """

    def __init__(self, base_dir: str | Path = ".cache"):
        # Initialize the cache directory and file
        # base_dir can either be a string or a Path object
        self.base_dir = Path(base_dir)

        # Ensure the base directory exists
        # exist_ok=True allows the directory to be created if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)

    def save(
        self, source_path: str | Path, data: Dict[str, Any], kind: str, version: float
    ):
        """
        Save data to a .npz file.
        :param source_path: Path to the source file from which features were extracted.
        :param data: Dictionary containing the data to be saved.
        :param kind: Type of data being saved (e.g., "beatmap", "audio_features").
        :param version: Version of the data being saved.
        """
        src = Path(source_path)
        composite_path = self.base_dir / f"{src.stem}_{kind}_v{version}.npz"
        fingerprint = _fingerprint(src)
        # Create metadata for the cache
        metadata = {
            "fingerprint": fingerprint,
            "kind": kind,
            "version": version,
            "timestamp": time.time(),
        }
        # Convert metadata to a json string and then to a numpy array
        meta_arr = np.array(json.dumps(metadata)).astype("S")

        # Prepare the data to be saved
        # Convert all values in the data dictionary to numpy arrays
        dict_to_save = {
            "__meta__": meta_arr,
        }
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                dict_to_save[key] = value
            elif isinstance(value, (int, float, bool)):
                # If the value is a single value, convert it to a numpy array
                dict_to_save[key] = np.array(value)
            else:
                # If the value is of an unsupported type, convert it to a string
                # This is a fallback, but ideally, we should handle all types properly
                dict_to_save[key] = np.array(json.dumps(value)).astype("S")

        # Save the data to a .npz file
        # np.savez allows saving multiple arrays in a single file
        # allow_pickle=False is used to prevent saving objects that could lead to security issues
        np.savez_compressed(str(composite_path), **dict_to_save, allow_pickle=False)
        print(f"Data saved to {composite_path}")
        return composite_path

    def load(self, source_path: str | Path, kind: str, version: float):
        """
        Load data from a .npz file.
        :param source_path: Path to the source file from which features were extracted.
        :param kind: Type of data being loaded (e.g., "beatmap", "audio_features").
        :param version: Version of the data being loaded.
        :return: Dictionary containing the loaded data.
        """
        src = Path(source_path)
        file_path = self.base_dir / f"{src.stem}_{kind}_v{version}.npz"
        if not file_path.exists():
            # meaning this file is not cached
            log.info(f"Cache file {file_path} does not exist.")
            return None

        with np.load(file_path, allow_pickle=False) as data:
            # Load the data from the .npz file
            # allow_pickle=False is used to prevent loading objects that could lead to security issues
            if "__meta__" not in data:
                log.info(f"No metadata found in {file_path}.")
                return None

            # Convert the metadata back from bytes to a dictionary
            meta_data = json.loads(data["__meta__"].item().decode("utf-8"))
            if meta_data["fingerprint"] != _fingerprint(src):
                log.info(
                    f"Cache file {file_path} is outdated or corrupted. Fingerprint mismatch."
                )
                return None

            out = {}
            for key in data.files:
                if key == "__meta__":
                    continue
                a = data[key]
                # If the array is a single value, convert it to a scalar
                out[key] = a.item() if a.shape == () else a
            return out


def _fingerprint(src: Path) -> Dict[str, Any]:
    """
    Private method to generate a fingerprint for the data to be cached.
    Fingerprint is constituted of the file size and last modified time in nanoseconds.
    : param src: Path to the source file or directory.
    : return: A dictionary containing the fingerprint.
    """
    file_info = src.stat()
    return {"size": file_info.st_size, "mtime_ns": file_info.st_mtime_ns}
