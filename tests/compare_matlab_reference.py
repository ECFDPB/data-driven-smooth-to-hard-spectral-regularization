"""Compare the NumPy ESPIRiT port with binary arrays exported by MATLAB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from single_image_eval.espirit import combine_frame, estimate_csm  # noqa: E402


def read_complex(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    pairs = np.fromfile(path, dtype=np.float64).reshape(-1, 2)
    values = pairs[:, 0] + 1j * pairs[:, 1]
    return values.reshape(shape, order="F")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    args = parser.parse_args()
    sx, sy, coils, calib_x, calib_y = map(
        int, (args.reference_dir / "meta.txt").read_text(encoding="utf-8").split()
    )
    shape = (sx, sy, coils)
    k_input = read_complex(args.reference_dir / "k_input.bin", shape)
    k_label = read_complex(args.reference_dir / "k_label.bin", shape)
    matlab_csm = read_complex(args.reference_dir / "csm.bin", shape)
    matlab_combined = read_complex(args.reference_dir / "combined.bin", (sx, sy))
    python_csm, diagnostics = estimate_csm(k_input, (calib_x, calib_y))
    python_combined = combine_frame(k_label, python_csm)

    # Eigenvectors may differ by a unit-magnitude pixelwise phase.  Compare
    # their coil-vector subspaces and the final magnitude image.
    matlab_norm = np.linalg.norm(matlab_csm, axis=2)
    python_norm = np.linalg.norm(python_csm, axis=2)
    denominator = np.maximum(matlab_norm * python_norm, np.finfo(float).eps)
    alignment = np.abs(np.sum(np.conj(matlab_csm) * python_csm, axis=2)) / denominator
    magnitude_nrmse = float(
        np.linalg.norm(np.abs(python_combined) - np.abs(matlab_combined))
        / np.linalg.norm(np.abs(matlab_combined))
    )
    print(f"calibration_rank={diagnostics.calibration_rank}")
    print(f"median_CSM_subspace_alignment={np.median(alignment):.9f}")
    print(f"combined_magnitude_NRMSE={magnitude_nrmse:.9g}")
    if np.median(alignment) < 0.995 or magnitude_nrmse > 0.01:
        raise SystemExit("Python port does not agree sufficiently with the MATLAB tutorial")


if __name__ == "__main__":
    main()
