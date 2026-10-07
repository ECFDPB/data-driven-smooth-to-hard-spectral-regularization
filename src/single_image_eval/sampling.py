"""Deterministic time-varying Cartesian masks and validation splits."""

from __future__ import annotations

import hashlib

import numpy as np


def child_seed(base: int, case_id: str, acceleration: int, purpose: str, repeat: int = 0) -> int:
    key = f"{base}|{case_id}|R={acceleration}|repeat={repeat}|{purpose}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def acs_indices(ny: int, acs_lines: int) -> np.ndarray:
    if not 1 <= acs_lines < ny:
        raise ValueError("ACS line count must lie in [1, ny)")
    start = (ny - acs_lines) // 2
    return np.arange(start, start + acs_lines)


def variable_density_masks(
    ny: int, frames: int, acceleration: int, acs_lines: int, seed: int
) -> np.ndarray:
    lines_per_frame = int(round(ny / acceleration))
    if lines_per_frame <= acs_lines:
        raise ValueError("ACS leaves no peripheral samples")
    acs = acs_indices(ny, acs_lines)
    peripheral = np.setdiff1d(np.arange(ny), acs, assume_unique=True)
    center = (ny - 1) / 2.0
    radius = np.abs(peripheral - center) / max(center, 1.0)
    base_weight = 0.05 + (1.0 - radius) ** 4
    rng = np.random.default_rng(seed)
    usage = np.zeros(ny, dtype=np.int64)
    masks = np.zeros((frames, ny), dtype=bool)
    masks[:, acs] = True
    need = lines_per_frame - acs_lines
    for frame in range(frames):
        weight = base_weight / (1.0 + usage[peripheral])
        weight /= weight.sum()
        chosen = rng.choice(peripheral, size=need, replace=False, p=weight)
        masks[frame, chosen] = True
        usage[chosen] += 1
    return masks


def validation_split(
    acquired: np.ndarray, acs_lines: int, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if acquired.ndim != 2:
        raise ValueError("acquired mask must have layout [time, ky]")
    if not 0 < fraction < 0.5:
        raise ValueError("validation fraction must lie in (0, 0.5)")
    frames, ny = acquired.shape
    acs = set(acs_indices(ny, acs_lines).tolist())
    rng = np.random.default_rng(seed)
    validation = np.zeros_like(acquired)
    for frame in range(frames):
        candidates = np.array([line for line in np.flatnonzero(acquired[frame]) if line not in acs])
        count = max(1, int(round(fraction * candidates.size)))
        if count >= candidates.size:
            raise ValueError("validation removes every peripheral training line")
        order = candidates[np.argsort(np.abs(candidates - (ny - 1) / 2.0))]
        rotated = np.roll(order, int(rng.integers(0, len(order))))
        validation[frame, rotated[np.linspace(0, len(rotated) - 1, count, dtype=int)]] = True
    training = acquired & ~validation
    if np.any(training[:, list(acs)] != acquired[:, list(acs)]):
        raise AssertionError("ACS was removed from training")
    return training, validation
