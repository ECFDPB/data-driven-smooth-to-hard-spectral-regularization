"""Complex Gaussian noise for single-image acquired k-space samples."""

from __future__ import annotations

import numpy as np


def standard_complex_noise(shape: tuple[int, ...], real_seed: int, imag_seed: int) -> np.ndarray:
    real = np.random.default_rng(real_seed).standard_normal(shape)
    imag = np.random.default_rng(imag_seed).standard_normal(shape)
    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def scale_noise_to_nominal_snr(
    clean: np.ndarray, standard: np.ndarray, mask: np.ndarray, snr_db: float
):
    if clean.shape != standard.shape:
        raise ValueError("clean signal and standard noise shapes differ")
    expanded = mask[:, None, :]
    measurement_count = int(mask.sum() * clean.shape[1])
    signal = clean * expanded
    signal_norm = float(np.linalg.norm(signal.ravel()))
    sigma = signal_norm / (np.sqrt(measurement_count) * 10 ** (float(snr_db) / 20.0))
    noise = (sigma * standard * expanded).astype(np.complex64)
    noise_norm = max(float(np.linalg.norm(noise.ravel())), np.finfo(float).eps)
    realized = 20.0 * np.log10(signal_norm / noise_norm)
    return noise, float(sigma), float(realized), measurement_count
