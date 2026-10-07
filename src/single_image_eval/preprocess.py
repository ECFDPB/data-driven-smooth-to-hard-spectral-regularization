"""OCMR geometry handling and frame-wise ESPIRiT coil combination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .espirit import combine_dynamic_frames
from .fft import fft2c, ifft2c


@dataclass(frozen=True)
class PartialFourierGeometry:
    encoded_x: int
    acquired_samples: int
    acquired_start: int
    acquired_stop: int
    center_index: int

    @property
    def is_partial(self) -> bool:
        return self.acquired_samples < self.encoded_x


def partial_fourier_geometry(encoded_x: int, acquired_samples: int, center_sample: int):
    start = encoded_x - acquired_samples if center_sample * 2 < encoded_x else 0
    stop = start + acquired_samples
    center = start + center_sample
    if not (0 < acquired_samples <= encoded_x and start <= center < stop):
        raise ValueError("invalid partial-Fourier geometry")
    return PartialFourierGeometry(encoded_x, acquired_samples, start, stop, center)


def pocs_fill_readout(kspace, geometry: PartialFourierGeometry, iterations: int, xp):
    if not geometry.is_partial:
        return kspace
    original = kspace.copy()
    radius = min(
        geometry.center_index - geometry.acquired_start,
        geometry.acquired_stop - 1 - geometry.center_index,
    )
    if radius < 2:
        raise ValueError("insufficient symmetric readout center for POCS")
    low = xp.zeros_like(kspace)
    low[..., geometry.center_index - radius : geometry.center_index + radius + 1, :] = original[
        ..., geometry.center_index - radius : geometry.center_index + radius + 1, :
    ]
    phase = xp.exp(1j * xp.angle(ifft2c(low, xp)))
    filled = original.copy()
    for _ in range(iterations):
        filled = fft2c(xp.abs(ifft2c(filled, xp)) * phase, xp)
        filled[..., geometry.acquired_start : geometry.acquired_stop, :] = original[
            ..., geometry.acquired_start : geometry.acquired_stop, :
        ]
    return filled


def center_pad_kspace(kspace, target_x: int, target_y: int, xp):
    current_x, current_y = kspace.shape[-2:]
    target_x, target_y = max(current_x, target_x), max(current_y, target_y)
    before_x = (target_x - current_x) // 2
    before_y = (target_y - current_y) // 2
    after_x = target_x - current_x - before_x
    after_y = target_y - current_y - before_y
    return xp.pad(
        kspace,
        [(0, 0)] * (kspace.ndim - 2) + [(before_x, after_x), (before_y, after_y)],
    )


def center_crop_images(images, target_x: int, target_y: int):
    current_x, current_y = images.shape[-2:]
    start_x = (current_x - target_x) // 2
    start_y = (current_y - target_y) // 2
    return images[..., start_x : start_x + target_x, start_y : start_y + target_y]


def coil_compress(coil_images: np.ndarray, target_coils: int):
    if coil_images.ndim != 4:
        raise ValueError("coil_images must be [time,coil,x,y]")
    coils = coil_images.shape[1]
    target = min(int(target_coils), coils)
    matrix = np.moveaxis(coil_images, 1, 0).reshape(coils, -1)
    covariance = matrix @ matrix.conj().T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order].real, 0.0)
    basis = eigenvectors[:, order[:target]].conj().T.astype(np.complex64)
    compressed = np.einsum("vc,tcxy->tvxy", basis, coil_images, optimize=True).astype(np.complex64)
    retained = float(eigenvalues[:target].sum() / max(eigenvalues.sum(), np.finfo(float).eps))
    return compressed, basis, retained


def combine_ocmr_coils(
    coil_images: np.ndarray,
    target_coils: int,
    calibration_ky: int,
    kernel_shape: tuple[int, int],
    calibration_threshold: float,
    eigenvalue_threshold: float,
    progress=None,
):
    compressed, basis, retained = coil_compress(coil_images, target_coils)
    combined, diagnostics, first_csm = combine_dynamic_frames(
        compressed,
        calibration_ky=calibration_ky,
        kernel_shape=kernel_shape,
        calibration_threshold=calibration_threshold,
        eigenvalue_threshold=eigenvalue_threshold,
        progress=progress,
    )
    scale = float(np.linalg.norm(combined.ravel()) / np.sqrt(combined.size))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid combined-image RMS")
    return (
        (combined / scale).astype(np.complex64),
        first_csm,
        basis,
        retained,
        scale,
        diagnostics,
    )
