
from __future__ import annotations
from typing import Literal, Optional

import shutil
import subprocess
import warnings

import numpy as np
import librosa


def load_and_resample(path: str, sample_rate: int = 22050) -> np.ndarray:
    if shutil.which("ffmpeg") is not None:
        cmd = [
            "ffmpeg", "-v", "error", "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate),
            "-",
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timed out decoding {path}")

        if result.returncode != 0 or len(result.stdout) == 0:
            stderr_tail = result.stderr.decode(errors="ignore")[-300:]
            raise RuntimeError(f"ffmpeg failed to decode {path} (exit {result.returncode}): {stderr_tail}")

        y = np.frombuffer(result.stdout, dtype=np.float32).copy()
        return y

    warnings.warn(
        "ffmpeg not found on PATH -- falling back to librosa's in-process decoder, which is "
        "vulnerable to hard crashes on some corrupted mp3s (not just clean exceptions). "
        "Installing ffmpeg is strongly recommended.",
        RuntimeWarning,
    )
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return y


def extract_log_mel(
    y: np.ndarray, sample_rate: int = 22050, n_mels: int = 128, hop_length: int = 512
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_mels=n_mels, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def extract_chroma(
    y: np.ndarray, sample_rate: int = 22050, n_chroma: int = 12, hop_length: int = 512
) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sample_rate, n_chroma=n_chroma, hop_length=hop_length
    )
    return chroma


def normalize_track(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return (features - mean) / (std + eps)


def segment_track(
    features: np.ndarray,
    sample_rate: int,
    hop_length: int,
    mode: Literal["fixed", "beat_sync"] = "fixed",
    segment_seconds: float = 7.5,
    y: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    n_frames = features.shape[1]

    if mode == "fixed":
        frames_per_segment = max(1, int(round(segment_seconds * sample_rate / hop_length)))
        segments = [
            features[:, start:start + frames_per_segment]
            for start in range(0, n_frames - frames_per_segment + 1, frames_per_segment)
        ]
        return segments

    elif mode == "beat_sync":
        if y is None:
            raise ValueError(
                "mode='beat_sync' requires the original waveform via the `y` argument "
                "(beat tracking runs on audio, not on the derived feature matrix)."
            )
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate, hop_length=hop_length)
        beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=n_frames)
        segments = [
            features[:, beat_frames[i]:beat_frames[i + 1]]
            for i in range(len(beat_frames) - 1)
            if beat_frames[i + 1] > beat_frames[i]
        ]
        return segments

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'fixed' or 'beat_sync'.")