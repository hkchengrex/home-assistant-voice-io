"""MFCC and PCEN feature extraction implemented with NumPy only."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filter_bank(sample_rate: int, n_fft: int, filter_count: int) -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(0.0), _hz_to_mel(sample_rate / 2.0), filter_count + 2
    )
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((filter_count, n_fft // 2 + 1), dtype=np.float32)
    for index in range(filter_count):
        left, center, right = bins[index : index + 3]
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2 + 1)
        for bin_index in range(left, center):
            filters[index, bin_index] = (bin_index - left) / max(1, center - left)
        for bin_index in range(center, right):
            filters[index, bin_index] = (right - bin_index) / max(1, right - center)
    return filters


@lru_cache(maxsize=8)
def _dct_basis(filter_count: int, coefficient_count: int) -> np.ndarray:
    filters = np.arange(filter_count, dtype=np.float32) + 0.5
    coefficients = np.arange(coefficient_count, dtype=np.float32)[:, None]
    return np.cos(np.pi * coefficients * filters / filter_count).astype(np.float32)


def _frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.size < frame_length:
        samples = np.pad(samples, (0, frame_length - samples.size))
    frame_count = 1 + int(np.ceil((samples.size - frame_length) / hop_length))
    total_length = (frame_count - 1) * hop_length + frame_length
    samples = np.pad(samples, (0, total_length - samples.size))
    shape = (frame_count, frame_length)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()


def _delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    denominator = 2 * sum(index * index for index in range(1, width + 1))
    result = np.zeros_like(features)
    for index in range(1, width + 1):
        result += index * (
            padded[width + index : width + index + features.shape[0]]
            - padded[width - index : width - index + features.shape[0]]
        )
    return result / denominator


def extract_mfcc(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    coefficient_count: int = 13,
    filter_count: int = 26,
) -> np.ndarray:
    """Return per-frame cepstral and delta features with utterance normalization."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty mono signal")

    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - 0.97 * samples[:-1]
    frame_length = round(sample_rate * 0.025)
    hop_length = round(sample_rate * 0.010)
    frames = _frame_signal(emphasized, frame_length, hop_length)
    frames *= np.hamming(frame_length).astype(np.float32)

    n_fft = 1 << (frame_length - 1).bit_length()
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2 / n_fft).astype(np.float32)
    mel_energy = power @ _mel_filter_bank(sample_rate, n_fft, filter_count).T
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    cepstra = log_mel @ _dct_basis(filter_count, coefficient_count).T

    # Per-utterance normalization reduces microphone gain and room coloration.
    mean = cepstra.mean(axis=0, keepdims=True)
    std = cepstra.std(axis=0, keepdims=True)
    normalized = (cepstra - mean) / np.maximum(std, 1e-4)
    combined = np.concatenate((normalized, _delta(normalized)), axis=1)
    return combined.astype(np.float32)


def extract_pcen_cepstra(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    coefficient_count: int = 13,
    filter_count: int = 40,
    smoothing: float = 0.05,
) -> np.ndarray:
    """Return PCEN-normalized cepstra for far-field command matching.

    Per-channel energy normalization adapts each frequency band to its recent
    energy. This suppresses steady room response, microphone gain, and distance
    changes before the compact cepstral projection used by DTW.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty mono signal")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must be between 0 and 1")

    frame_length = round(sample_rate * 0.025)
    hop_length = round(sample_rate * 0.010)
    frames = _frame_signal(samples, frame_length, hop_length)
    frames *= np.hanning(frame_length).astype(np.float32)

    n_fft = 1 << (frame_length - 1).bit_length()
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2 / n_fft).astype(np.float32)
    mel_energy = np.maximum(
        power @ _mel_filter_bank(sample_rate, n_fft, filter_count).T,
        1e-10,
    )

    smoother = np.empty_like(mel_energy)
    smoother[0] = mel_energy[0]
    for frame_index in range(1, mel_energy.shape[0]):
        smoother[frame_index] = (
            (1.0 - smoothing) * smoother[frame_index - 1]
            + smoothing * mel_energy[frame_index]
        )

    alpha = 0.98
    delta = 2.0
    root = 0.5
    pcen = (
        mel_energy / np.power(1e-6 + smoother, alpha) + delta
    ) ** root - delta**root
    cepstra = pcen @ _dct_basis(filter_count, coefficient_count).T
    mean = cepstra.mean(axis=0, keepdims=True)
    std = cepstra.std(axis=0, keepdims=True)
    return ((cepstra - mean) / np.maximum(std, 1e-4)).astype(np.float32)


def extract_command_features(
    samples: np.ndarray,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Return far-field PCEN cepstra and speech-motion features."""
    cepstra = extract_pcen_cepstra(samples, sample_rate)
    return np.concatenate((cepstra, _delta(cepstra)), axis=1).astype(np.float32)
