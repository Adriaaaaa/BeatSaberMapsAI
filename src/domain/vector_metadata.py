import json, hashlib, base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from utils.logger import LoggerManager

log = LoggerManager.get_logger(__name__)


@dataclass()
class VectorMetadata:
    # Metadata for a vector representation
    track_name: str
    created_at: str
    features_included: Dict[str, List[str]]

    id: Optional[str] = None  # Unique identifier for the vector

    @staticmethod
    def build_id(features_included: Dict[str, List[str]]) -> str:
        # Create a unique identifier for the vector based on its metadata

        if len(features_included) == 0:
            raise ValueError("features_included is required to build ID")

        meta_str = json.dumps(
            {
                "features_included": features_included,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        # blake2b hash ensures uniqueness (same input gives same output)
        # it's also short and standard in python
        # digest is used to retrieve the raw bytes of the hash
        hash = hashlib.blake2b(meta_str.encode("utf-8"), digest_size=12).digest()

        # to avoid manipulating raw bytes, we encode in base 32
        # encode : convert non printable bytes of hash in bytes in base 32 (A-Z2-7)
        # decode ascii : convert base 32 bytes back to a string where each byte is an ascii caracter (i.e. 0-9, A-Z)
        # rstrip : remove any trailing '=' characters
        encoded_hash = base64.b32encode(hash).decode("ascii").rstrip("=")

        log.info(
            f"Generated vector ID: {encoded_hash} for features: {features_included}"
        )

        return encoded_hash

    def to_dict(self) -> Dict[str, Any]:
        # Convert the vector metadata to a dictionary
        return {
            "track_name": self.track_name,
            "created_at": self.created_at,
            "features_included": self.features_included,
        }
