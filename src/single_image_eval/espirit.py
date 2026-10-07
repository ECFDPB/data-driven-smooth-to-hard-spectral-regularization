"""NumPy port of the supplied ``espirit_csm_wangzi`` MATLAB tutorial.

The public functions use Python layouts:

* one frame of coil k-space: ``[x, y, coil]``;
* a dynamic series of coil images: ``[time, coil, x, y]``;
* sensitivity maps: ``[x, y, coil]``.

The implementation follows the tutorial's 6x6 calibration-kernel SVD,
0.02 relative calibration singular-value threshold, image-space eigenvector,
and sensitivity-weighted sum ``sum(conj(CSM) * coil_image, coil)``.  It forms
the image-space Gram matrix incrementally so that the complete kernel image
bank does not need to be kept in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class EspiritDiagnostics:
    calibration_shape: tuple[int, int]
    kernel_shape: tuple[int, int]
    calibration_rank: int
    calibration_columns: int
    largest_calibration_singular_value: float
    retained_smallest_ratio: float
    map_norm_min: float
    map_norm_max: float
    support_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def _center_slices(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[slice, ...]:
    if len(source) != len(target):
        raise ValueError("source and target dimensions differ")
    slices = []
    for current, wanted in zip(source, target, strict=True):
        if wanted > current or wanted < 1:
            raise ValueError(f"invalid center crop/pad target {target} for source {source}")
        start = current // 2 - wanted // 2
        slices.append(slice(start, start + wanted))
    return tuple(slices)


def center_crop(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    slices = _center_slices(array.shape[:2], shape)
    return array[slices[0], slices[1], ...]


def center_pad(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if shape[0] < array.shape[0] or shape[1] < array.shape[1]:
        raise ValueError("target shape must not be smaller than input")
    output = np.zeros((*shape, *array.shape[2:]), dtype=array.dtype)
    target_slices = _center_slices(shape, array.shape[:2])
    output[target_slices[0], target_slices[1], ...] = array
    return output


def _fft2c_xy(array: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(array, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )


def _ifft2c_xy(array: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(array, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )


def _calibration_matrix(calibration: np.ndarray, kernel_shape: tuple[int, int]) -> np.ndarray:
    """Match MATLAB ``im2row`` followed by its column-major reshape."""

    sx, sy, coils = calibration.shape
    kx, ky = kernel_shape
    patch_x = sx - kx + 1
    patch_y = sy - ky + 1
    if patch_x < 1 or patch_y < 1:
        raise ValueError("calibration region is smaller than the ESPIRiT kernel")
    columns = []
    for coil in range(coils):
        for offset_y in range(ky):
            for offset_x in range(kx):
                patch = calibration[
                    offset_x : offset_x + patch_x,
                    offset_y : offset_y + patch_y,
                    coil,
                ]
                columns.append(patch.reshape(-1, order="F"))
    return np.stack(columns, axis=1)


def estimate_csm(
    kspace: np.ndarray,
    calibration_shape: tuple[int, int],
    kernel_shape: tuple[int, int] = (6, 6),
    calibration_threshold: float = 0.02,
    eigenvalue_threshold: float = 0.0,
    eigen_chunk_size: int = 4096,
) -> tuple[np.ndarray, EspiritDiagnostics]:
    """Estimate one ESPIRiT map set for one dynamic frame.

    ``calibration_shape`` should normally contain the full readout extent and
    only a central band in the phase-encoding direction, mirroring the supplied
    tutorial (for example, 224 x 22 in ``Brain_8ch.mat``).
    """

    data = np.asarray(kspace)
    if data.ndim != 3:
        raise ValueError("kspace must have layout [x, y, coil]")
    if not np.iscomplexobj(data):
        data = data.astype(np.complex64)
    sx, sy, coils = data.shape
    calib_x = min(int(calibration_shape[0]), sx)
    calib_y = min(int(calibration_shape[1]), sy)
    calibration = center_crop(data, (calib_x, calib_y))
    matrix = _calibration_matrix(calibration, kernel_shape)
    _, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0:
        raise ValueError("empty ESPIRiT calibration spectrum")
    retained = int(np.count_nonzero(singular_values >= singular_values[0] * calibration_threshold))
    retained = max(retained, 1)

    # MATLAB: V -> reshape([kx,ky,Nc,Nv]) using column-major order.
    right_vectors = right_h.conj().T[:, :retained]
    kernels = right_vectors.reshape(
        (*kernel_shape, coils, retained), order="F"
    )

    # MATLAB kernelEig rotates coil coordinates before the image-space SVD.
    rotated_matrix = np.transpose(kernels, (0, 1, 3, 2)).reshape(
        (-1, coils), order="F"
    )
    _, _, coil_rotation_h = np.linalg.svd(rotated_matrix, full_matrices=False)
    coil_rotation = coil_rotation_h.conj().T
    rotated_matrix = rotated_matrix @ coil_rotation
    rotated_kernels = np.transpose(
        rotated_matrix.reshape((*kernel_shape, retained, coils), order="F"),
        (0, 1, 3, 2),
    )

    # Accumulate K K^H at every pixel.  This is equivalent to the tutorial's
    # per-pixel SVD but avoids storing [x,y,coil,retained] simultaneously.
    gram = np.zeros((sx, sy, coils, coils), dtype=np.complex64)
    image_scale = np.sqrt(float(sx * sy)) / np.sqrt(float(np.prod(kernel_shape)))
    for kernel_index in range(retained):
        kernel = np.conj(rotated_kernels[::-1, ::-1, :, kernel_index])
        padded = center_pad(kernel, (sx, sy))
        image_kernel = (_fft2c_xy(padded) * image_scale).astype(np.complex64, copy=False)
        gram += np.einsum(
            "xyc,xyd->xycd", image_kernel, np.conj(image_kernel), optimize=True
        )

    flat_gram = gram.reshape((-1, coils, coils))
    maps_flat = np.empty((flat_gram.shape[0], coils), dtype=np.complex64)
    largest_flat = np.empty(flat_gram.shape[0], dtype=np.float32)
    for start in range(0, flat_gram.shape[0], eigen_chunk_size):
        stop = min(start + eigen_chunk_size, flat_gram.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(flat_gram[start:stop])
        top = eigenvectors[:, :, -1]
        # Match kernelEig: phase the first rotated component, then undo the
        # coil-coordinate rotation.
        phase = np.exp(-1j * np.angle(top[:, 0]))
        top = top * phase[:, None]
        maps_flat[start:stop] = np.einsum(
            "cd,nd->nc", coil_rotation, top, optimize=True
        ).astype(np.complex64, copy=False)
        largest_flat[start:stop] = np.sqrt(
            np.maximum(eigenvalues[:, -1].real, 0.0)
        ).astype(np.float32, copy=False)

    support = largest_flat > float(eigenvalue_threshold)
    maps_flat *= support[:, None]
    maps = maps_flat.reshape((sx, sy, coils))
    map_norm = np.sqrt(np.sum(np.abs(maps) ** 2, axis=2))
    supported_norm = map_norm[support.reshape(sx, sy)]
    diagnostics = EspiritDiagnostics(
        calibration_shape=(calib_x, calib_y),
        kernel_shape=tuple(map(int, kernel_shape)),
        calibration_rank=retained,
        calibration_columns=matrix.shape[1],
        largest_calibration_singular_value=float(singular_values[0]),
        retained_smallest_ratio=float(singular_values[retained - 1] / singular_values[0]),
        map_norm_min=float(supported_norm.min()) if supported_norm.size else 0.0,
        map_norm_max=float(map_norm.max(initial=0.0)),
        support_fraction=float(np.mean(support)),
    )
    return maps, diagnostics


def combine_frame(label_kspace: np.ndarray, csm: np.ndarray) -> np.ndarray:
    """Apply the exact sensitivity-weighted sum shown in the supplied demo."""

    if label_kspace.shape != csm.shape:
        raise ValueError("label k-space and CSM shapes differ")
    coil_images = _ifft2c_xy(np.asarray(label_kspace))
    return np.sum(np.conj(csm) * coil_images, axis=2).astype(np.complex64)


def combine_dynamic_frames(
    coil_images: np.ndarray,
    calibration_ky: int,
    kernel_shape: tuple[int, int] = (6, 6),
    calibration_threshold: float = 0.02,
    eigenvalue_threshold: float = 0.0,
    progress=None,
) -> tuple[np.ndarray, list[EspiritDiagnostics], np.ndarray]:
    """Estimate and combine every cardiac frame independently.

    Returns the combined cine, per-frame diagnostics, and the first-frame CSM
    for quality-control visualization.
    """

    images = np.asarray(coil_images)
    if images.ndim != 4:
        raise ValueError("coil_images must have layout [time, coil, x, y]")
    frames, _, sx, sy = images.shape
    combined = np.empty((frames, sx, sy), dtype=np.complex64)
    diagnostics: list[EspiritDiagnostics] = []
    first_csm = None
    for frame in range(frames):
        label_kspace = _fft2c_xy(np.transpose(images[frame], (1, 2, 0)))
        csm, info = estimate_csm(
            label_kspace,
            calibration_shape=(sx, min(int(calibration_ky), sy)),
            kernel_shape=kernel_shape,
            calibration_threshold=calibration_threshold,
            eigenvalue_threshold=eigenvalue_threshold,
        )
        combined[frame] = combine_frame(label_kspace, csm)
        diagnostics.append(info)
        if first_csm is None:
            first_csm = csm
        if progress is not None:
            progress(frame + 1, frames, info)
    assert first_csm is not None
    return combined, diagnostics, first_csm
