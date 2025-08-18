from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

import librosa

from dataclasses import dataclass, asdict

from infra.logger import LoggerManager

log = LoggerManager.get_logger(__name__)


@dataclass
class AudioFeatures:
    """
    Class to handle audio features extraction and storage.
    """

    sample_num: int  # number of samples in the audio signal
    sample_rate: int  # Hz, number of samples per second
    mean_amplitude: float  # mean amplitude of the audio signal
    max_amplitude: float  # maximal amplitude in the audio signal
    min_amplitude: float  # minimal amplitude in the audio signal
    duration_sec: float  # duration of the audio signal in seconds

    # pas de saut entre deux fenetres d'analyse = de combien de samples on avance entre deux frames / fenetres
    hop_length: int

    # onset are "les moments d'attaques dans un morceau = quand on détecte le début de qqch ou un changement brutal d'intensité"
    onset_times_sec: np.ndarray

    # the two following attributes are used to compute the root mean square over time
    # rms_time_sec is the time moments at which the root mean square is computed and stored in rms_values
    # keeping them separated is better for use in graphs and to write easily in cache
    rms_times_sec: np.ndarray | None = None  # root mean square time moments
    rms_values: np.ndarray | None = (
        None  # root mean square values of the audio signal, i.e. the energy of the audio signal over time
    )

    # si on analyse le signal avec des fenêtres de 2048 samples
    # la premiere frame analyse [0, 2048]
    # la 2e [512, 2560] = 1ere frame + hop_length

    # cls is used to create an instance of the class with a different way that the constructor __init__
    @classmethod
    def from_signal(
        cls, audio_data: np.ndarray, sample_rate: int = 44100, hop_length: int = 512
    ):
        """
        Extract features from the audio data.
        :param audio_data: Audio data as a numpy array.
        :param sample_rate: Sample rate of the audio data.
        :return: An instance of AudioFeatures with extracted features.
        """
        # shape of audio_data is (n_channels, n_samples) if stereo or (n_samples,) if mono
        # audio_date.shape[-1] gives the number of samples in the audio signal for both cases
        sample_num = int(audio_data.shape[-1])
        duration_sec = sample_num / sample_rate if sample_num > 0 else 0.0
        rms = np.sqrt(np.mean(np.square(audio_data)))
        max_amplitude = float(np.max(audio_data)) if sample_num > 0 else 0.0
        min_amplitude = float(np.min(audio_data)) if sample_num > 0 else 0.0
        mean_amplitude = float(np.mean(audio_data)) if sample_num > 0 else 0.0

        # RMS glissantes par frames de taille 2048 et de pas hop_length
        # rms ci-dessous donne une série par frame
        rms = librosa.feature.rms(
            y=audio_data, frame_length=2048, hop_length=hop_length
        )
        # rms should then be normalized as it depends on the compression of the song
        # rms.ptp = peak to peak => max - min
        # add a little something to avoid division by zero (signal plat = no music at this frame)
        rms_norm = (rms - rms.min()) / (np.ptp(rms) + 1e-12)

        # the following convert indexes of frames to time of frames
        # hop_length is the distance between two frames, in samples
        # sample_rate is the number of samples per second
        # hop_length / sample_rate will then be the distance between two frames in second
        # frame 0 will start at 0 sec, frame 1 at 0 + hop_length/sample_rate
        # frame index n will then start at n*hop_length/sample_rate
        rms_times = librosa.frames_to_time(
            np.arange(len(rms_norm)), sr=sample_rate, hop_length=hop_length
        )

        # An audio signal is the evolution of the air pressure over time for a set of "bande de frequences"
        # for example : "basses / graves" : 50-200Hz => slow vibrations
        # medium for voices / guitars : 500-2000
        # aigus  : 5000 and more
        # To detect onset, i.e. the moments where something changes (a new instrument begins to play for example)
        # librosa will need to analyse the spectogram that is the evolution of energy for each bande de fréquences over time (by frame)
        # To do so, librosa will :
        # 1/ convert signal into a spectogramme
        # 2/ detect how the energy in each bande de fréquence varies by frames
        # 3/ keeps only brutal augmentations for each
        # 4/ Combine all these detected points to detect where it moves in several bandes
        onset_envelop = librosa.onset.onset_strength(
            y=audio_data, sr=sample_rate, hop_length=hop_length
        )

        # with this global envelop, librosa will then have to detect local peaks, i.e. where the envelop is over a given threshold and over is nearest neighbours
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelop, sr=sample_rate, hop_length=hop_length
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=sample_rate, hop_length=hop_length
        )

        return cls(
            sample_num=sample_num,
            sample_rate=sample_rate,
            hop_length=hop_length,
            duration_sec=duration_sec,
            max_amplitude=max_amplitude,
            min_amplitude=min_amplitude,
            mean_amplitude=mean_amplitude,
            rms_times_sec=rms_times.astype(np.float32),
            rms_values=rms_norm.astype(np.float32),
            onset_times_sec=onset_times.astype(np.float32),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the AudioFeatures instance to a dictionary.
        This is useful for saving to cache or serialization.
        :return: Dictionary representation of the AudioFeatures instance.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create an AudioFeatures instance from a dictionary.
        Used for loading from cache or other sources.
        :param data: Dictionary containing audio features.
        :return: An instance of AudioFeatures.
        """
        d = dict(data)
        for k in ("rms_times_sec", "rms_values", "onset_times_sec"):
            if not isinstance(d[k], np.ndarray):
                d[k] = np.asarray(d[k])
        return AudioFeatures(**d)

    @staticmethod
    def _fmt_time(sec: float) -> str:
        "time in mm:ss:mmm"
        m = int(sec // 60)
        s = sec - 60 * m
        return f"{m:02d}:{s:06.3f}"

    def frame_hop_seconds(self) -> float:
        # espacement temporel entre deux frames RMS en secondes
        return self.hop_length / float(self.sample_rate) if self.sample_rate else 0.0

    def frames_per_second(self) -> float:
        # frquence d'echantillonnage de la série RMS (pts par seconde)
        hop_sec = self.frame_hop_seconds()
        return (1.0 / hop_sec) if hop_sec > 0 else 0.0

    def rms_stats(self) -> dict[str, float]:
        r = self.rms_values
        r = np.asarray(r)
        if r.size == 0:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
                "p10": 0,
                "p50": 0,
                "p90": 0,
            }
        return {
            "min": float(r.min()),
            "max": float(r.max()),
            "mean": float(r.mean()),
            "std": float(np.std(r, ddof=0)),
            "p10": float(np.percentile(r, 10)),
            "p50": float(np.percentile(r, 50)),
            "p90": float(np.percentile(r, 90)),
        }

    def summary_str(self, max_onsets: int = 8) -> str:
        duration = self._fmt_time(self.duration_sec)
        hop_s = self.frame_hop_seconds()
        fps = self.frames_per_second()
        self.rms_values = np.asarray(self.rms_values)
        rms_n = len(self.rms_values)
        onset_n = self.onset_times_sec.size
        rms = self.rms_stats()

        if onset_n > 0:
            sample = ",".join(
                self._fmt_time(t) for t in self.onset_times_sec[:max_onsets]
            )
        else:
            sample = "_"
        lines = [
            "=== AudioFeatures ===",
            f"Sample rate      : {self.sample_rate}",
            f"Durée            : {duration} ({self.duration_sec:.3f} s) | N echantillons = {self.sample_num}",
            f"Amplitude  min/max : {self.min_amplitude:.4f} / {self.max_amplitude:.4f}",
            f"RMS (normalisee) points : {rms_n} | hop : {hop_s*1000 : 0.1f} ms (~{fps:.1f} fps)",
            f"RMS stats (0..1)    :             min = {rms['min']:.3f} p10 = {rms['p10']:.3f} p50 = {rms['p50']:.3f} p90 = {rms['p90']:.3f} max = {rms['max']:.3f} mean = {rms['mean']:.3f} std = {rms['std']:.3f}",
            f"Onsets detectes    : {onset_n}" f"Onsets extraits    : {sample}",
        ]
        return "\n".join(lines)
