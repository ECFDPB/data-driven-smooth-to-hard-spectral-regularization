"""Dependency-light checks for the rewritten single-image evaluation core."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from single_image_eval.candidates import parameter_token, pilot_candidates  # noqa: E402
from single_image_eval.espirit import combine_frame, estimate_csm  # noqa: E402
from single_image_eval.noise import scale_noise_to_nominal_snr, standard_complex_noise  # noqa: E402
from single_image_eval.operator import adjoint, forward  # noqa: E402


def _load_grid_selector():
    path = ROOT / "scripts" / "select_condition_grids.py"
    spec = importlib.util.spec_from_file_location("condition_grid_selector", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fft2c_xy(array: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(array, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )


def test_operator_adjoint() -> None:
    rng = np.random.default_rng(7)
    image = (rng.standard_normal((3, 12, 14)) + 1j * rng.standard_normal((3, 12, 14))).astype(np.complex64)
    data = (rng.standard_normal(image.shape) + 1j * rng.standard_normal(image.shape)).astype(np.complex64)
    mask = rng.random((3, 14)) > 0.45
    left = np.vdot(forward(image, mask, np), data)
    right = np.vdot(image, adjoint(data, mask, np))
    if not np.allclose(left, right, rtol=2e-6, atol=2e-6):
        raise AssertionError(f"adjoint check failed: {left} != {right}")


def test_noise_snr() -> None:
    rng = np.random.default_rng(9)
    clean = (rng.standard_normal((12, 32, 28)) + 1j * rng.standard_normal((12, 32, 28))).astype(np.complex64)
    mask = rng.random((12, 28)) > 0.6
    clean *= mask[:, None, :]
    standard = standard_complex_noise(clean.shape, 101, 102)
    _, _, realized, _ = scale_noise_to_nominal_snr(clean, standard, mask, 20.0)
    if abs(realized - 20.0) > 0.15:
        raise AssertionError(f"realized SNR {realized} is too far from 20 dB")


def test_condition_specific_candidates() -> None:
    methods = ("proposed", "soft", "hard", "l_plus_s")
    for method in methods:
        grids = []
        for acceleration in (4, 6, 8):
            for snr in (0, 10, 20, 30):
                values = pilot_candidates(method, acceleration, snr)
                tokens = frozenset(parameter_token(item) for item in values)
                if len(values) < 4 or len(tokens) != len(values):
                    raise AssertionError(f"invalid pool for {method}, R={acceleration}, SNR={snr}")
                grids.append(tokens)
        if len(set(grids)) != 12:
            raise AssertionError(f"{method} does not have 12 distinct R-by-SNR pools")


def test_espirit_shape_and_combination() -> None:
    nx, ny, coils = 24, 24, 4
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    image = np.exp(-4.0 * (x**2 + y**2)) * np.exp(0.3j * (x - y))
    sensitivities = []
    for coil in range(coils):
        angle = 2.0 * np.pi * coil / coils
        magnitude = 0.7 + 0.25 * (x * np.cos(angle) + y * np.sin(angle))
        phase = np.exp(1j * (0.2 * coil + 0.25 * x * np.sin(angle)))
        sensitivities.append(magnitude * phase)
    sensitivities = np.stack(sensitivities, axis=2)
    sensitivities /= np.sqrt(np.sum(np.abs(sensitivities) ** 2, axis=2, keepdims=True))
    coil_images = sensitivities * image[:, :, None]
    kspace = fft2c_xy(coil_images)
    maps, diagnostics = estimate_csm(
        kspace,
        calibration_shape=(nx, 18),
        kernel_shape=(4, 4),
        calibration_threshold=0.02,
        eigenvalue_threshold=0.0,
        eigen_chunk_size=256,
    )
    combined = combine_frame(kspace, maps)
    if maps.shape != kspace.shape or combined.shape != image.shape:
        raise AssertionError("ESPIRiT output shape mismatch")
    if not np.isfinite(combined).all() or diagnostics.calibration_rank < 1:
        raise AssertionError("invalid ESPIRiT result")
    correlation = np.corrcoef(np.abs(combined).ravel(), np.abs(image).ravel())[0, 1]
    if correlation < 0.90:
        raise AssertionError(f"unexpectedly weak magnitude agreement: {correlation}")


def test_adaptive_formal_grid_size() -> None:
    selector = _load_grid_selector()
    cases = {"p1", "p2", "p3", "p4"}
    records = []
    for candidate in range(8):
        for case_index, case in enumerate(sorted(cases)):
            # Four different case winners plus a gradually worsening tail.
            score = 1.0 + 0.01 * abs(candidate - case_index) + 0.03 * max(candidate - 4, 0)
            records.append(
                {
                    "case": case,
                    "status": "ok",
                    "parameter": {"q": candidate / 100.0},
                    "validation_score": score,
                }
            )
    selected, _ = selector.select_group(
        records, cases, minimum_count=4, maximum_count=7, mean_regret_band=0.025
    )
    if not 4 <= len(selected) <= 7:
        raise AssertionError("adaptive selector ignored its condition-level bounds")


def main() -> None:
    checks = (
        test_operator_adjoint,
        test_noise_snr,
        test_condition_specific_candidates,
        test_adaptive_formal_grid_size,
        test_espirit_shape_and_combination,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    main()
