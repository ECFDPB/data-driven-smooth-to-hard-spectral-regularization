"""Case-level reconstruction metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def complex_nmse(reconstruction: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm((reconstruction - reference).ravel()) ** 2
        / max(np.linalg.norm(reference.ravel()) ** 2, np.finfo(float).eps)
    )


def magnitude_psnr_ssim(reconstruction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    reference_magnitude = np.abs(reference)
    reconstruction_magnitude = np.abs(reconstruction)
    data_range = float(reference_magnitude.max())
    mse = float(np.mean((reconstruction_magnitude - reference_magnitude) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(data_range**2 / mse))
    temporal_mean = reference_magnitude.mean(axis=0)
    roi = temporal_mean >= 0.05 * float(temporal_mean.max())
    scores = []
    for frame in range(reference.shape[0]):
        _, score_map = structural_similarity(
            reference_magnitude[frame], reconstruction_magnitude[frame], data_range=data_range, full=True
        )
        scores.append(float(score_map[roi].mean()))
    return psnr, float(np.mean(scores))
