# Single-image dynamic MRI evaluation

Reproducible code for a retrospective dynamic cardiac MRI comparison of the proposed method, Soft, Hard, and L+S. The repository contains the frame-wise ESPIRiT preprocessing pipeline, condition-specific parameter selection, reconstruction solvers, formal evaluation scripts, frozen experimental settings, and one formal result file.

## Experimental protocol

- Data source: fully sampled OCMR cardiac cine acquisitions.
- Coil processing: optional global semi-unitary compression to 12 virtual coils, followed by frame-wise ESPIRiT sensitivity estimation with a 6 x 6 kernel and 24 calibration lines.
- Single-image construction: `sum(conj(CSM) * IFFT2c(k-space), coil)`, followed by complex-cine RMS normalization.
- Retrospective sampling: Cartesian masks with 8 autocalibration lines at acceleration factors R = 4, 6, and 8.
- Noise: circular complex Gaussian noise in the sampled single-image k-space at nominal SNR q = 0, 10, 20, and 30 dB. Larger q means less added noise.
- Evaluation: PSNR and SSIM on magnitude images and NMSE on complex-valued reconstructions.
- Optimisation: at most 300 iterations with relative iterate tolerance 1e-5. A selected diagnostic subset was also run to 600 iterations.

The complete machine-readable protocol, seeds, stopping rules, and selection policy are recorded in [`config/protocol.json`](config/protocol.json). The formal, condition-specific candidate grids are frozen in [`config/formal_candidates.json`](config/formal_candidates.json).

## Repository layout

```text
config/                 Frozen protocol and formal candidate grids
manifests/              Disjoint pilot and formal case lists
src/single_image_eval/  Fourier operators, sampling, noise, metrics, and solvers
scripts/                Preprocessing, selection, evaluation, and figure scripts
vendor/ocmr/            OCMR ISMRMRD reader used by preprocessing
outputs/formal/         One newline-delimited JSON result file
```

Raw OCMR `.h5` files, processed arrays, reconstructed volumes, local environments, caches, candidate-search logs, smoke tests, derived tables, and paper figures are intentionally excluded. This keeps the repository focused on the minimum code and data needed to inspect the method and its reported numerical results.

## Setup

Python 3.10 or newer is recommended. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`cupy-cuda12x` enables GPU execution and requires a compatible NVIDIA driver. For a CPU-only environment, omit that package and run the scripts with the NumPy backend.

Place the required OCMR files listed in `manifests/pilot_cases.csv` and `manifests/formal_cases.csv` under:

```text
data/raw/ocmr/
```

## Reproduce the evaluation

Run commands from the repository root. First preprocess the disjoint pilot and formal cohorts:

```bash
python scripts/preprocess_combined.py --cohort pilot
python scripts/preprocess_combined.py --cohort formal
```

To repeat parameter selection from the pilot cohort:

```bash
python scripts/run_evaluation.py --stage pilot
python scripts/select_condition_grids.py
```

The committed `config/formal_candidates.json` is the frozen grid used for the reported formal experiment. Run the formal evaluation and regenerate summaries with:

```bash
python scripts/run_evaluation.py --stage formal
python scripts/summarize_formal_results.py
python scripts/generate_psnr_distribution_unified.py
python scripts/generate_secondary_metric_intervals.py
python scripts/generate_convergence_sensitivity_figure.py
```

Use `--help` on any script for available paths and backend options. Long evaluations are resumable because completed records are retained in JSONL outputs.

## Included results

- `outputs/formal/final_results.jsonl`: the only committed output; it contains the final per-case, per-condition records as newline-delimited JSON.

No software license is granted by this repository unless a license file is added later. The vendored OCMR reader retains its original attribution in the source file.
