from __future__ import annotations
import numpy as np
import scipy.stats as stats

from typing import Dict, Any, Tuple, Optional, Sequence

import librosa

from dataclasses import dataclass, asdict

from utils.logger import LoggerManager
from utils.constants import *

log = LoggerManager.get_logger(__name__)


@dataclass
class AudioFeatures:
    """
    Class to handle audio features extraction and storage.
    """

    # number of samples in the audio signal
    # for a stereo signal, it is the number of samples per channel
    # for a mono signal, it is the total number of samples
    # in simple words, a sample is a number that represents
    # the amplitude of the audio signal at a given point in time
    sample_num: int

    # the sample rate is the number of samples per second
    # the higher it is, the more accurate the audio signal is
    # common values are 44100 Hz (CD quality) and 48000 Hz (professional audio quality)
    sample_rate: int

    # mean amplitude of the audio signal
    # it is the average of the absolute values of the samples
    # it gives an idea of the overall loudness of the audio signal
    mean_amplitude: float

    # maximal amplitude in the audio signal
    # it is the maximum of the absolute values of the samples
    # it gives an idea of the peak loudness of the audio signal
    max_amplitude: float

    # minimal amplitude in the audio signal
    # it is the minimum of the absolute values of the samples
    # it gives an idea of the lowest loudness of the audio signal
    min_amplitude: float

    # duration of the audio signal in seconds
    duration_sec: float

    # onset are "moments d'attaques"
    # they are the time moments where something happens in the audio signal
    # for example, the moment when a new instrument starts playing or when a note is played
    # for beat saber and similar games, they are the moments where a new block appears
    # they are detected using a onset detection algorithm
    onset_times_sec: np.ndarray

    # time moments of the frames used to compute all the features that are frame-based
    # for example, the root mean square is computed over frames of 2048 samples with
    # a hop length of 512 samples
    # frames_times_sec will then be the time moments of the start of each frame

    frames_times_sec: np.ndarray | None = None

    # pas de saut entre deux fenetres d'analyse = de combien de samples on avance entre deux frames / fenetres
    hop_length: int = 512

    # si on analyse le signal avec des fenêtres de 2048 samples
    # la premiere frame analyse [0, 2048]
    # la 2e [512, 2560] = 1ere frame + hop_length
    # to convert hop_length in seconds, divide by sample_rate
    # for each frame, fraùes_times_sec will give the time in seconds of the start of the frame
    # frames_times_sec can be computed from hop_length and sample_rate

    # root mean square values of the audio signal, i.e. the energy of the audio signal over time
    # in simple terms, the RMS is a measure of how loud the audio signal is over time
    # it is computed by taking the square root of the average of the squared values of the samples
    # it is often used in audio signal processing to measure the loudness of a signal
    # it's better to use the normalized RMS values that are between 0 and 1
    # because the RMS values depend on the compression of the audio signal
    # a normalized RMS value of 0 means that the audio signal is silent
    # a normalized RMS value of 1 means that the audio signal is at its maximum loud
    # it's better than amplitude because it gives an idea of the loudness over time
    # whereas amplitude is just a snapshot of the loudness at a given point in time
    # RMS is also less sensitive to sudden changes in amplitude (like a pop or a click)
    # RMS is a more stable measure of loudness over time
    rms_values: np.ndarray | None = None

    # spectral centroid values of the audio signal
    # the spectral centroid indicates where the "center of mass" for a sound is located
    # it is a measure used in digital signal processing to characterize a spectrum
    # it is calculated as the weighted mean of the frequencies present in the signal, with their magnitudes as weights
    # it is often associated with the perception of "brightness" of a sound - higher values indicate a brighter sound
    # for example, a sound with a lot of high-frequency content (like a cymbal crash) will have a higher spectral centroid than a sound with more low-frequency content (like a bass drum)
    # brightness is a perceptual property that allows us to distinguish between sounds that are rich in high frequencies and those that are not (ie "brillance")
    # a sound with a high spectral centroid is often perceived as "brighter" or "sharper" or "lighter", while a sound with a low spectral centroid is perceived as "darker" or "mellower"
    # these values are normed between 0 and 1 by librosa because they depend on the sample rate of the audio signal
    centroid_values: Optional[np.ndarray] | None = None

    # spectral flux values of the audio signal
    # the spectral flux is a measure of how quickly the power spectrum of a signal is changing
    # it is calculated by comparing the power spectrum of the current frame to the power spectrum of the previous frame
    # it is often used in onset detection algorithms to identify the moments when new sounds begin to play in an audio signal
    # high spectral flux values indicate a rapid change in the spectrum, which is often associated with the onset of a new sound
    # low spectral flux values indicate a more stable spectrum, which is often associated with sustained sounds
    # spectral flux can be used to detect changes in timbre, rhythm, and other musical features
    # it is a useful feature for music information retrieval tasks such as beat tracking, genre classification
    # close to onsets values but a bit more continuous and less sparse than onsets
    # it can be used to complement onset detection algorithms
    flux_values: Optional[np.ndarray] | None = None

    # the short-time Fourier transform (STFT) is a mathematical technique used to analyze the frequency content of a signal over time
    # it is particularly useful for non-stationary signals, such as audio signals, where the frequency content can change rapidly over time
    # the STFT works by dividing the signal into short overlapping segments, or frames, and then applying the Fourier transform to each frame
    # this results in a time-frequency representation of the signal, where each point in the representation corresponds to a specific time and frequency
    # the STFT is commonly used in audio signal processing for tasks such as speech recognition, music analysis, and audio effects processing
    # the STFT can be used to extract features such as spectral centroid, spectral flux, and other time-frequency characteristics of the audio signal
    # the STFT is typically represented as a complex-valued matrix, where the rows correspond to frequency bins and the columns correspond to time frames
    # the magnitude of the STFT can be used to visualize the frequency content of the signal over time, while the phase information can be used for tasks such as audio synthesis and source separation
    # the STFT is a powerful tool for analyzing and processing audio signals, and is widely used in both research and industry
    # the STFT is often visualized using a spectrogram, which is a 2D representation of the magnitude of the STFT over time and frequency
    # spectrograms are commonly used in audio analysis and music information retrieval tasks
    # in short term, the STFT is the way to convert a signal into a spectrogram and allows to analyse how the energy in each frequency band evolves over time
    stft_magnitudes: Optional[np.ndarray] | None = None

    # the spectral bandwidth is a measure of the width of the frequency spectrum of a signal
    # it is calculated as the difference between the upper and lower frequency bounds of the spectrum
    # it provides information about the range of frequencies present in the signal
    # a signal with a wide spectral bandwidth contains a broad range of frequencies, while a signal
    # with a narrow spectral bandwidth contains a more limited range of frequencies
    # the spectral bandwidth is often used in audio signal processing to characterize the timbral qualities of
    # sounds and to analyze the frequency content of audio signals
    # it can be used to identify the presence of specific instruments or sound sources in a mix
    bandwidth_values: Optional[np.ndarray] | None = None

    # rolloff is a measure of the frequency below which a certain percentage of the total spectral energy is contained
    # it is often used in audio signal processing to characterize the spectral shape of sounds and to analyze the frequency content of audio signals
    # a high rolloff value indicates that a significant portion of the spectral energy is concentrated in the higher frequency range
    # while a low rolloff value indicates that the spectral energy is more evenly distributed across the frequency spectrum
    # rolloff can be used to identify the presence of specific instruments or sound sources in a mix
    # it can also be used to analyze the timbral qualities of sounds and to detect changes in the frequency content of audio signals over time
    rollof_values: Optional[np.ndarray] | None = None

    # chroma features are a representation of the spectral energy distribution of an audio signal
    # across the 12 different pitch classes (or chroma) in Western music
    # pitch classes correspond to the 12 different notes in an octave (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
    # they are often used in music information retrieval tasks such as chord recognition, key detection, and music similarity analysis
    # chroma features are calculated by mapping the frequency content of an audio signal onto the 12 pitch classes
    # this is typically done by applying a series of bandpass filters to the audio signal, each centered on one of the 12 pitch classes
    # the resulting chroma features can be represented as a 12-dimensional vector, where each
    # element corresponds to the energy in one of the pitch classes
    # chroma features are useful because they provide a compact representation of the harmonic content of an
    # audio signal, which can be used to analyze and compare different pieces of music
    chroma: Optional[np.ndarray] | None = None

    # harmonic component of the audio signal
    # the harmonic component represents the tonal content of the audio signal
    # harmonic sounds are those that have a clear pitch, such as musical notes played on an instrument or sung by a voice
    # harmonic means that the sound is made up of a fundamental frequency and its integer multiples (harmonics)
    # the fundamental frequency is the lowest frequency in the sound and determines the perceived pitch of the sound
    # a pitch in french is une "hauteur"
    # "à l'octave" means that the frequency is doubled or halved
    # for example, if the fundamental frequency is 440 Hz (the pitch of the musical note A4),
    # the first harmonic will be at 880 Hz (A5), the second harmonic at 1320 Hz (E6), and so on
    # it's the same note but at a higher pitch
    # rms of the harmonic component can give an idea of the loudness of the tonal content of the audio signal
    # tonal content in french is "contenu tonal", meaning that the sound has a pitch and is not just noise
    # tonal content is important in music because it provides the melody and harmony of a piece
    rms_harmonic_values: Optional[np.ndarray] | None = None

    # percussive component of the audio signal
    # the percussive component represents the transient content of the audio signal
    # percussive sounds are those that do not have a clear pitch, such as drum hits, claps, or other rhythmic elements
    # transient means that the sound is short and sudden, with a rapid attack and decay
    rms_percussive_values: Optional[np.ndarray] | None = None

    # mel-frequency cepstral coefficients (MFCCs) are a representation of the short-term power spectrum of an audio signal
    # they are commonly used in speech and audio processing tasks such as speech recognition, speaker identification, and music genre classification
    # MFCCs are calculated by applying a series of transformations to the audio signal, including a Fourier transform, a mel-scale filter bank, and a discrete cosine transform
    # the resulting MFCCs can be represented as a set of coefficients that capture the spectral characteristics of the audio signal
    # MFCCs are useful because they provide a compact representation of the spectral content of an audio signal, which can be used to analyze and compare different sounds
    # typically, the first 13 MFCCs are used, as they capture the most important spectral features of the audio signal
    # in simple words, MFCCs are a way to represent the timbral qualities of a sound
    mfcc_values: Optional[np.ndarray] | None = None

    # times of the detected beats in seconds
    # beats are the regular pulses in music that provide the rhythmic structure of a piece
    beat_times_sec: Optional[np.ndarray] | None = None

    # estimated tempo of the audio signal in beats per minute (BPM)
    # tempo is the speed of the music, typically measured in beats per minute (BPM)
    # a higher tempo indicates faster music, while a lower tempo indicates slower music
    tempo: Optional[float] | None = None

    # cls is used to create an instance of the class with a different way that the constructor __init__
    @classmethod
    def from_signal(
        cls,
        audio_data: np.ndarray,
        sample_rate: int = AUDIO_DEFAULT_SAMPLE_RATE,
        hop_length: int = AUDIO_DEFAULT_HOP,
        n_fft: int = AUDIO_DEFAULT_NFFT,
    ):
        """
        Extract features from the audio data.
        :param audio_data: Audio data as a numpy array.
        :param sample_rate: Sample rate of the audio data.
        :param hop_length: Hop length for frame-based features.
        :param n_fft: Number of FFT components for spectral features.
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

        # An audio signal is the evolution of the air pressure over time for a set of "bande de frequences"
        # for example : "basses / graves" : 50-200Hz => slow vibrations
        # medium for voices / guitars : 500-2000
        # aigus  : 5000 and more

        # compute the spectogram (STFT)
        # it will be a matrix wuere each column is a frame and each row is a frequency bin
        # a frequency bin is a small range of frequencies ("bande de fréquences")
        # for each frame, the STFT gives the amplitude of each frequency bin
        # the STFT is a complex matrix where each element has a real and an imaginary part
        # the real part represents the amplitude of the frequency bin (magnitude, in simple words, how strong is this frequency in the signal)
        # the imaginary part represents the phase of the frequency bin (in simple words, the timing of the wave or "décalage de phase")
        # the STFT is computed using the Fast Fourier Transform (FFT) algorithm
        stft = librosa.stft(y=audio_data, n_fft=n_fft, hop_length=hop_length)

        # magnitude of the STFT
        # it gives the amplitude of each frequency bin at each frame
        # in simple words, the magnitude is the strength of each frequency in the signal at each frame
        # it is a matrix where each column is a frame and each row is a frequency bin
        # difference between stft and stft_magnitudes is that stft is complex and stft_magnitudes is real
        stft_magnitudes = np.abs(stft)

        # number of frames in the STFT
        # it is the number of columns in the STFT matrix or stft_magnitudes matrix
        # it is also the number of frames used to compute frame-based features like RMS, spectral
        num_frames = stft_magnitudes.shape[1] if stft_magnitudes is not None else 0

        # times of each frame in seconds
        # it gives the time in seconds of the start of each frame
        # np arange(num_frames) gives an array of integers from 0 to num_frames-1
        # multiplying by hop_length gives the number of samples at the start of each frame
        # dividing by sample_rate converts samples to seconds
        # if num_frames is 0, we return an empty array of float32
        frame_times = (
            np.arange(num_frames) * hop_length / sample_rate
            if num_frames > 0
            else np.array([], dtype=np.float32)
        )

        # compute root mean square (RMS) for each frame
        # there are several ways to compute RMS
        # 1/ compute RMS directly from the audio signal for each frame
        # 2/ compute RMS from the STFT magnitudes for each frame
        # both methods give similar results
        # here we use the second method because we already have the STFT magnitudes
        rms = librosa.feature.rms(S=stft_magnitudes).ravel().astype(np.float32)

        # rms should then be normalized as it depends on the compression of the song
        # rms.ptp = peak to peak => max - min
        # add a little something to avoid division by zero (signal plat = no music at this frame)
        rms_norm = _normalize_feature(rms)

        # compute other spectral features
        # ravel() converts a 2D array to a 1D array
        # astype(np.float32) converts the array to float32 to save memory
        centroid = librosa.feature.spectral_centroid(
            S=stft_magnitudes, sr=sample_rate
        ).ravel()
        centroid_norm = _normalize_feature(centroid).astype(np.float32)

        bandwidth = librosa.feature.spectral_bandwidth(
            S=stft_magnitudes, sr=sample_rate
        ).ravel()
        bandwidth_norm = _normalize_feature(bandwidth).astype(np.float32)

        rollof = librosa.feature.spectral_rolloff(
            S=stft_magnitudes, sr=sample_rate, hop_length=hop_length
        ).ravel()
        rollof_norm = _normalize_feature(rollof).astype(np.float32)

        # there are several ways to compute spectral chroma
        # here we use the STFT magnitudes
        # it will be aligned with the other features as they are all computed from the same STFT
        # but it could be less accurate than using the audio signal directly and chroma_cqt
        # later we could compare both methods and improve if needed
        chroma = librosa.feature.chroma_stft(
            S=stft_magnitudes, sr=sample_rate, hop_length=hop_length
        ).astype(np.float32)

        # separate harmonic and percussive components
        harmonic, percussive = librosa.decompose.hpss(stft)

        rms_harmonic = librosa.feature.rms(S=harmonic).ravel().astype(np.float32)
        rms_percussive = librosa.feature.rms(S=percussive).ravel().astype(np.float32)

        rms_harmonic_norm = _normalize_feature(rms_harmonic).ravel().astype(np.float32)
        rms_percussive_norm = (
            _normalize_feature(rms_percussive).ravel().astype(np.float32)
        )

        # estimate tempo and beat times
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio_data, sr=sample_rate, hop_length=hop_length
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sample_rate, hop_length=hop_length
        )
        tempo = float(tempo) if tempo is not None else None
        beat_times = beat_times.astype(np.float32) if beat_times is not None else None

        # To detect onset, i.e. the moments where something changes (a new instrument begins to play for example)
        # librosa will need to analyse the spectogram that is the evolution of energy for each bande de fréquences over time (by frame)
        # first, we compute a global envelop of the onset strength over time
        # this envelop will then be used to detect local peaks that are the onsets
        # to compute the onset envelop, librosa will analyse the spectogram and compute the difference between two frames
        # if there is a big difference, it means that something changed in the audio signal
        # we don't use the spectogram directly because it is too detailed and noisy
        # the onset envelop is a simplified version of the spectogram that captures the most important changes
        # that's why we use the audio signal directly
        onset_envelop = librosa.onset.onset_strength(
            y=audio_data, sr=sample_rate, hop_length=hop_length
        )

        flux_norm = _normalize_feature(onset_envelop).astype(np.float32)

        # with this global envelop, librosa will then have to detect local peaks, i.e. where the envelop is over a given threshold and over is nearest neighbours
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelop, sr=sample_rate, hop_length=hop_length
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=sample_rate, hop_length=hop_length
        )

        mfcc_values = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=13,
        ).astype(np.float32)

        return cls(
            sample_num=sample_num,
            sample_rate=sample_rate,
            mean_amplitude=mean_amplitude,
            max_amplitude=max_amplitude,
            min_amplitude=min_amplitude,
            duration_sec=duration_sec,
            onset_times_sec=onset_times.astype(np.float32),
            frames_times_sec=frame_times.astype(np.float32),
            hop_length=hop_length,
            rms_values=rms_norm,
            centroid_values=centroid_norm,
            flux_values=flux_norm,
            stft_magnitudes=stft_magnitudes.astype(np.float32),
            bandwidth_values=bandwidth_norm,
            rollof_values=rollof_norm,
            chroma=chroma,
            rms_harmonic_values=rms_harmonic_norm,
            rms_percussive_values=rms_percussive_norm,
            beat_times_sec=beat_times,
            tempo=tempo,
            mfcc_values=mfcc_values,
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
        for k in (
            "frames_times_sec",
            "rms_values",
            "onset_times_sec",
            "centroid_values",
            "flux_values",
            "stft_magnitudes",
            "bandwidth_values",
            "rollof_values",
            "chroma",
            "rms_harmonic_values",
            "rms_percussive_values",
            "beat_times_sec",
            "mfcc_values",
        ):
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

    def feature_stats_1d(self, feature_to_summarize: str):
        if feature_to_summarize not in (
            "rms_values",
            "centroid_values",
            "flux_values",
            "bandwidth_values",
            "rollof_values",
            "rms_harmonic_values",
            "rms_percussive_values",
        ):
            log.error(
                f"feature_stats_1d: feature_to_summarize must be one of 'rms_values', 'centroid_values', 'flux_values', 'bandwidth_values', 'rollof_values', 'rms_harmonic_values', 'rms_percussive_values'"
            )
            log.error(f"feature_stats_1d: got {feature_to_summarize}")
            raise ValueError(
                f"feature_stats_1d: feature_to_summarize must be one of 'rms_values', 'centroid_values', 'flux_values', 'bandwidth_values', 'rollof_values', 'rms_harmonic_values', 'rms_percussive_values'"
            )

        feature = getattr(self, feature_to_summarize)
        if feature is None:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
                "p10": 0,
                "p50": 0,
                "p90": 0,
                "var": 0,
                "skew": 0,
                "kurt": 0,
            }

        feature = np.asarray(feature)
        return {
            "min": float(feature.min()),
            "max": float(feature.max()),
            "mean": float(feature.mean()),
            "std": float(np.std(feature, ddof=0)),
            "p10": float(np.percentile(feature, 10)),
            "p50": float(np.percentile(feature, 50)),
            "p90": float(np.percentile(feature, 90)),
            "var": float(np.var(feature, ddof=0)),
            "skew": float(stats.skew(feature)),
            "kurt": float(stats.kurtosis(feature)),
        }

    def feature_stats_2d(
        self,
        feature_to_summarize: str,
        stats_to_compute: Sequence[str] = (
            "mean",
            "std",
            "var",
            "min",
            "skew",
            "kurt",
            "max",
        ),
        quantiles: Sequence[int] = (10, 50, 90),
    ) -> Dict[str, np.ndarray]:
        if feature_to_summarize not in ("chroma", "mfcc_values"):
            raise ValueError(
                f"feature_stats_2d: feature_to_summarize must be one of 'chroma', 'mfcc_values'"
            )

        feature = getattr(self, feature_to_summarize)
        if feature is None:
            return {}

        summary = {}
        for i in range(feature.shape[0]):
            band = feature[i, :]
            band_stats = {}
            if "min" in stats_to_compute:
                band_stats["min"] = float(band.min())
            if "max" in stats_to_compute:
                band_stats["max"] = float(band.max())
            if "mean" in stats_to_compute:
                band_stats["mean"] = float(band.mean())
            if "std" in stats_to_compute:
                band_stats["std"] = float(np.std(band, ddof=0))
            if "var" in stats_to_compute:
                band_stats["var"] = float(np.var(band, ddof=0))
            if "skew" in stats_to_compute:
                band_stats["skew"] = float(stats.skew(band))
            if "kurt" in stats_to_compute:
                band_stats["kurt"] = float(stats.kurtosis(band))
            for q in quantiles:
                band_stats[f"p{q}"] = float(np.percentile(band, q))
            summary[f"{feature_to_summarize}_band_{i}"] = band_stats

        return summary

    def summary_str(self, max_onsets: int = 8) -> str:
        duration = self._fmt_time(self.duration_sec)
        hop_s = self.frame_hop_seconds()
        fps = self.frames_per_second()

        # ensure value series are numpy arrays
        rms_values = np.asarray(self.rms_values)
        centroid_values = np.asarray(self.centroid_values)
        flux_values = np.asarray(self.flux_values)
        stft_magnitudes = np.asarray(self.stft_magnitudes)
        bandwidth_values = np.asarray(self.bandwidth_values)
        rollof_values = np.asarray(self.rollof_values)
        chroma = np.asarray(self.chroma)
        mftcc_values = np.asarray(self.mfcc_values)
        rms_harmonic = np.asarray(self.rms_harmonic_values)
        rms_percussive = np.asarray(self.rms_percussive_values)
        beat_times_sec = np.asarray(self.beat_times_sec)

        # number of points in each feature series
        rms_n = len(rms_values)
        onset_n = self.onset_times_sec.size
        centroid_n = len(centroid_values)
        flux_n = len(flux_values)
        bandwidth_n = len(bandwidth_values)
        rollof_n = len(rollof_values)
        chroma_n = chroma.shape[1] if chroma.ndim == 2 else 0
        mfcc_n = mftcc_values.shape[1] if mftcc_values.ndim == 2 else 0
        rms_harmonic_n = len(rms_harmonic)
        rms_percussive_n = len(rms_percussive)
        beat_n = len(beat_times_sec)

        # stats for selected features
        rms_stats = self.feature_stats_1d("rms_values")
        centroid_stats = self.feature_stats_1d("centroid_values")
        flux_stats = self.feature_stats_1d("flux_values")
        bandwidth_stats = self.feature_stats_1d("bandwidth_values")
        rollof_stats = self.feature_stats_1d("rollof_values")
        rms_harmonic_stats = self.feature_stats_1d("rms_harmonic_values")
        rms_percussive_stats = self.feature_stats_1d("rms_percussive_values")
        chroma_stats = self.feature_stats_2d(
            "chroma",
            stats_to_compute=("min", "max", "mean", "std", "var", "skew", "kurt"),
            quantiles=(10, 50, 90),
        )
        mfcc_stats = self.feature_stats_2d(
            "mfcc_values",
            stats_to_compute=("min", "max", "mean", "std", "var", "skew", "kurt"),
            quantiles=(10, 50, 90),
        )

        if onset_n > 0:
            sample = ",".join(
                self._fmt_time(t) for t in self.onset_times_sec[:max_onsets]
            )
        else:
            sample = "_"
        lines = "\n".join(
            [
                "=== AudioFeatures ===",
                f"Samples         : {self.sample_num}",
                f"Sample rate      : {self.sample_rate}",
                f"Durée            : {duration} ({self.duration_sec:.3f} s)",
                f"Frame hop        : {self.hop_length} samples, {hop_s:.4f} sec, {fps:.1f} fps",
                f"Amplitude  min/mean/max : {self.min_amplitude:.4f} /  {self.mean_amplitude:.4f} / {self.max_amplitude:.4f}",
                f"RMS (normalisee) points : {rms_n} ",
                f"RMS stats (0..1)    :             min = {rms_stats['min']:.3f} p10 = {rms_stats['p10']:.3f} p50 = {rms_stats['p50']:.3f} p90 = {rms_stats['p90']:.3f} max = {rms_stats['max']:.3f} mean = {rms_stats['mean']:.3f} std = {rms_stats['std']:.3f} var = {rms_stats['var']:.3f}",
                f"Centroid (norm) points : {centroid_n}",
                f"Centroid stats (0..1) :             min = {centroid_stats['min']:.3f} p10 = {centroid_stats['p10']:.3f} p50 = {centroid_stats['p50']:.3f} p90 = {centroid_stats['p90']:.3f} max = {centroid_stats['max']:.3f} mean = {centroid_stats['mean']:.3f} std = {centroid_stats['std']:.3f} var = {centroid_stats['var']:.3f}",
                f"Spectral flux (norm) points : {flux_n}",
                f"Spectral flux stats (0..1) :             min = {flux_stats['min']:.3f} p10 = {flux_stats['p10']:.3f} p50 = {flux_stats['p50']:.3f} p90 = {flux_stats['p90']:.3f} max = {flux_stats['max']:.3f} mean = {flux_stats['mean']:.3f} std = {flux_stats['std']:.3f} var = {flux_stats['var']:.3f}",
                f"Spectral bandwidth (norm) points : {bandwidth_n}",
                f"Spectral bandwidth stats (0..1) :             min = {bandwidth_stats['min']:.3f} p10 = {bandwidth_stats['p10']:.3f} p50 = {bandwidth_stats['p50']:.3f} p90 = {bandwidth_stats['p90']:.3f} max = {bandwidth_stats['max']:.3f} mean = {bandwidth_stats['mean']:.3f} std = {bandwidth_stats['std']:.3f} var = {bandwidth_stats['var']:.3f}",
                f"Spectral rolloff (norm) points : {rollof_n}",
                f"Spectral rolloff stats (0..1) :             min = {rollof_stats['min']:.3f} p10 = {rollof_stats['p10']:.3f} p50 = {rollof_stats['p50']:.3f} p90 = {rollof_stats['p90']:.3f} max = {rollof_stats['max']:.3f} mean = {rollof_stats['mean']:.3f} std = {rollof_stats['std']:.3f} var = {rollof_stats['var']:.3f}",
                f"RMS harmonic (norm) points : {rms_harmonic_n}",
                f"RMS harmonic stats (0..1) :             min = {rms_harmonic_stats['min']:.3f} p10 = {rms_harmonic_stats['p10']:.3f} p50 = {rms_harmonic_stats['p50']:.3f} p90 = {rms_harmonic_stats['p90']:.3f} max = {rms_harmonic_stats['max']:.3f} mean = {rms_harmonic_stats['mean']:.3f} std = {rms_harmonic_stats['std']:.3f} var = {rms_harmonic_stats['var']:.3f}",
                f"RMS percussive (norm) points : {rms_percussive_n}",
                f"RMS percussive stats (0..1) :             min = { rms_percussive_stats['min']:.3f} p10 = {rms_percussive_stats['p10']:.3f} p50 = {rms_percussive_stats['p50']:.3f} p90 = {rms_percussive_stats['p90']:.3f} max = {rms_percussive_stats['max']:.3f} mean = {rms_percussive_stats['mean']:.3f} std = {rms_percussive_stats['std']:.3f} var = {rms_percussive_stats['var']:.3f}",
                f"Beats points    : {beat_n}",
                (
                    f"Estimated tempo : {self.tempo:.1f} BPM"
                    if self.tempo
                    else "Estimated tempo : _"
                ),
                f"Onsets detectes    : {onset_n}" f"Onsets extraits    : {sample}",
                f"Chroma points    : {chroma_n} (12 bins)",
                f"Chroma stats (0..1)   :",
                "    ",
                ", ".join(
                    f"bin_{i}: min=f{chroma_stats[f"chroma_band_{i}"]['min']:.3f} p10={chroma_stats[f'chroma_band_{i}']['p10']:.3f} p50={chroma_stats[f'chroma_band_{i}']['p50']:.3f} p90={chroma_stats[f'chroma_band_{i}']['p90']:.3f} max={chroma_stats[f'chroma_band_{i}']['max']:.3f} mean={chroma_stats[f'chroma_band_{i}']['mean']:.3f} std={chroma_stats[f'chroma_band_{i}']['std']:.3f} var={chroma_stats[f'chroma_band_{i}']['var']:.3f}"
                    for i in range(chroma.shape[0] if chroma.ndim == 2 else 0)
                ),
                f"MFCC points     : {mfcc_n} (typically 13 bins)",
                f"MFCC stats :",
                "    ",
                ", ".join(
                    f"bin_{i}: min={mfcc_stats[f'mfcc_values_band_{i}']['min']:.3f} p10={mfcc_stats[f'mfcc_values_band_{i}']['p10']:.3f} p50={mfcc_stats[f'mfcc_values_band_{i}']['p50']:.3f} p90={mfcc_stats[f'mfcc_values_band_{i}']['p90']:.3f} max={mfcc_stats[f'mfcc_values_band_{i}']['max']:.3f} mean={mfcc_stats[f'mfcc_values_band_{i}']['mean']:.3f} std={mfcc_stats[f'mfcc_values_band_{i}']['std']:.3f} var={mfcc_stats[f'mfcc_values_band_{i}']['var']:.3f}"
                    for i in range(
                        mftcc_values.shape[0] if mftcc_values.ndim == 2 else 0
                    )
                ),
            ]
        )
        return lines

    def validate_sizes(self) -> bool:
        """
        Validate that all frame-based features have the same number of frames.
        :return: True if all frame-based features have the same length, False otherwise.
        """
        expected_length = (
            self.frames_times_sec.size if self.frames_times_sec is not None else 0
        )

        frame_based_features = {
            "rms_values": self.rms_values,
            "centroid_values": self.centroid_values,
            "flux_values": self.flux_values,
            "stft_magnitudes": self.stft_magnitudes,
            "bandwidth_values": self.bandwidth_values,
            "rollof_values": self.rollof_values,
            "chroma": self.chroma,
            "rms_harmonic_values": self.rms_harmonic_values,
            "rms_percussive_values": self.rms_percussive_values,
            "mfcc_values": self.mfcc_values,
        }

        for feature_name, feature in frame_based_features.items():
            if feature is not None:
                feature_length = feature.shape[1] if feature.ndim == 2 else feature.size
                if feature_length != expected_length:
                    log.error(
                        f"Validation error: {feature_name} has length {feature_length}, expected {expected_length}"
                    )
                    return False

        return True


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    """Normalize a feature array to the range [0, 1]."""
    feature = np.asarray(feature)
    if feature.size == 0:
        return feature
    return (feature - feature.min()) / (np.ptp(feature) + 1e-12)
