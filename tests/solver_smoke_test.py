"""Small end-to-end solver checks; requires the full requirements file."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from single_image_eval.operator import forward  # noqa: E402
from single_image_eval.solvers import (  # noqa: E402
    solve_l_plus_s,
    solve_proposed,
    solve_spectral_baseline,
)


def main() -> None:
    rng = np.random.default_rng(41)
    frames, nx, ny = 5, 16, 14
    spatial = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    temporal = np.exp(1j * np.linspace(0, 0.4, frames))
    reference = (temporal[:, None, None] * spatial[None, ...]).astype(np.complex64)
    mask = rng.random((frames, ny)) > 0.45
    mask[:, ny // 2 - 2 : ny // 2 + 2] = True
    measured = forward(reference, mask, np)
    runs = {
        "proposed": lambda: solve_proposed(
            measured, mask, a=4.0, c=0.55, alpha=0.03, xp=np, maximum_iterations=3
        ),
        "soft": lambda: solve_spectral_baseline(
            measured, mask, normalized_threshold=0.02, kind="soft", xp=np, maximum_iterations=3
        ),
        "hard": lambda: solve_spectral_baseline(
            measured, mask, normalized_threshold=0.10, kind="hard", xp=np, maximum_iterations=3
        ),
        "l_plus_s": lambda: solve_l_plus_s(
            measured, mask, normalized_low_rank_threshold=0.01,
            normalized_sparse_threshold=0.01, xp=np, maximum_iterations=3
        ),
    }
    for name, run in runs.items():
        reconstruction, log = run()
        if reconstruction.shape != reference.shape or not np.isfinite(reconstruction).all():
            raise AssertionError(f"{name} returned an invalid reconstruction")
        if not 1 <= log["iterations"] <= 3:
            raise AssertionError(f"{name} reported an invalid iteration count")
        print(f"PASS {name} solver")


if __name__ == "__main__":
    main()
