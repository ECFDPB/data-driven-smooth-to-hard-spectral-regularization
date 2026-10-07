# Single-Image Dynamic MRI Reconstruction

Code repository for the numerical experiments comparing the proposed method with Soft, Hard, and L+S reconstruction on retrospectively undersampled OCMR cardiac cine data.

## Repository Structure

```text
├── src/single_image_eval/        # Reconstruction library
│   ├── candidates.py             # Condition-specific parameter grids
│   ├── espirit.py                # ESPIRiT sensitivity estimation and coil combination
│   ├── fft.py                    # Centred two-dimensional Fourier transforms
│   ├── metrics.py                # PSNR, SSIM, and NMSE
│   ├── noise.py                  # Complex Gaussian noise at a nominal SNR
│   ├── operator.py               # Subsampled Fourier forward and adjoint operators
│   ├── preprocess.py             # Coil compression and OCMR preprocessing
│   ├── sampling.py               # Cartesian masks and validation splits
│   ├── solvers.py                # Proposed, Soft, Hard, and L+S solvers
│   └── spectral.py               # Spectral proximal computations
├── scripts/
│   ├── preprocess_combined.py    # Convert multi-coil OCMR data to combined cine images
│   ├── run_evaluation.py         # Run pilot or formal reconstruction experiments
│   └── select_condition_grids.py # Freeze condition-specific formal parameter grids
├── config/
│   ├── protocol.json             # Experimental settings and deterministic seeds
│   └── formal_candidates.json    # Frozen parameter candidates for every method and condition
├── manifests/
│   ├── pilot_cases.csv           # Four pilot cases used for parameter selection
│   └── formal_cases.csv          # Twenty held-out formal cases
├── vendor/ocmr/read_ocmr.py      # OCMR ISMRMRD reader
└── outputs/formal/
    └── final_results.jsonl       # Final per-case reconstruction results
```

Raw OCMR files, processed arrays, reconstructed volumes, candidate-search logs, figures, tables, and local runtime files are not included.

## Requirements

- Python 3.10+
- NumPy, SciPy, h5py, ISMRMRD, and scikit-image
- CuPy with CUDA 12 for optional GPU execution

Install the dependencies in an isolated environment:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CPU-only execution, omit `cupy-cuda12x` and select the NumPy backend.

## Data Preparation

Download the OCMR cases listed in `manifests/pilot_cases.csv` and `manifests/formal_cases.csv`, then place the `.h5` files under:

```text
data/raw/ocmr/
```

The preprocessing follows the same procedure for every cardiac frame:

1. compress to 12 virtual coils when necessary;
2. estimate ESPIRiT coil-sensitivity maps using a 6 x 6 kernel and 24 calibration lines;
3. combine the coil images as `sum(conj(CSM) * IFFT2c(k-space), coil)`;
4. normalize the resulting complex cine by its RMS amplitude.

Run preprocessing with:

```bash
python scripts/preprocess_combined.py --cohort pilot
python scripts/preprocess_combined.py --cohort formal
```

Use `--backend numpy` for CPU execution or `--backend cupy` for GPU execution.

## Running the Experiments

### Pilot Parameter Selection

```bash
python scripts/run_evaluation.py --stage pilot
python scripts/select_condition_grids.py
```

The second command writes the condition-specific candidate sets to `config/formal_candidates.json`. The committed file is the frozen grid used for the reported formal experiment.

### Formal Reconstruction

Inspect the planned number of candidate fits before running:

```bash
python scripts/run_evaluation.py --stage formal --dry-run
```

Run the complete formal experiment:

```bash
python scripts/run_evaluation.py --stage formal
```

The runner is resumable. Candidate records are written during execution, while the selected parameters and final reconstruction metrics are recorded in `outputs/formal/final_results.jsonl`.

## Experimental Protocol

- Acceleration factors: R = 4, 6, and 8
- Nominal SNR values: q = 0, 10, 20, and 30 dB; larger q means less added noise
- Autocalibration region: 8 phase-encoding lines
- Validation fraction: 15% of the acquired non-ACS samples
- Maximum iterations: 300
- Relative iterate tolerance: 1e-5
- Compared methods: Proposed, Soft, Hard, and L+S
- Metrics: magnitude PSNR, magnitude SSIM, and complex NMSE

All masks, validation splits, and complex-noise realizations use deterministic seeds recorded in `config/protocol.json`.

## Reproducibility

The four pilot cases are disjoint from the twenty formal cases. Parameter candidates are selected separately for every method and every R-by-SNR condition using pilot validation data only. The formal cases are evaluated with the frozen grids in `config/formal_candidates.json`; formal test metrics do not alter the grids.

The single committed result file, `outputs/formal/final_results.jsonl`, contains 960 records: 20 formal cases x 3 acceleration factors x 4 nominal SNR levels x 4 methods.

## Citation

Citation information will be added with the accompanying manuscript.
