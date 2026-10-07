"""Run pilot or formal single-image reconstruction experiments resumably."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parents[1]
LOCAL_RUNTIME = EVAL_ROOT / ".runtime"
if LOCAL_RUNTIME.is_dir():
    sys.path.insert(0, str(LOCAL_RUNTIME))
sys.path.insert(0, str(EVAL_ROOT / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from single_image_eval.candidates import (  # noqa: E402
    complexity_key,
    formal_candidates,
    parameter_token,
    pilot_candidates,
)
from single_image_eval.metrics import complex_nmse, magnitude_psnr_ssim  # noqa: E402
from single_image_eval.noise import scale_noise_to_nominal_snr, standard_complex_noise  # noqa: E402
from single_image_eval.operator import adjoint, forward  # noqa: E402
from single_image_eval.sampling import child_seed, validation_split, variable_density_masks  # noqa: E402
from single_image_eval.solvers import (  # noqa: E402
    solve_l_plus_s,
    solve_proposed,
    solve_spectral_baseline,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()


def load_manifest(stage: str) -> list[dict[str, str]]:
    path = (
        EVAL_ROOT / "manifests" / "pilot_cases.csv"
        if stage == "pilot"
        else EVAL_ROOT / "manifests" / "formal_cases.csv"
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("include", "true").lower() == "true"]


def choose_backend(name: str):
    if name == "numpy":
        return np, lambda value: value, lambda: None
    try:
        cuda_runtime = LOCAL_RUNTIME / "nvidia" / "cuda_runtime"
        if cuda_runtime.is_dir():
            os.environ.setdefault("CUDA_PATH", str(cuda_runtime))
            binary_paths = [
                str(path / "bin")
                for path in (LOCAL_RUNTIME / "nvidia").iterdir()
                if path.is_dir() and (path / "bin").is_dir()
            ]
            if binary_paths:
                os.environ["PATH"] = os.pathsep.join(binary_paths + [os.environ.get("PATH", "")])
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            os.environ.setdefault("CUPY_CACHE_DIR", str(EVAL_ROOT / "cache" / "cupy"))
            return cp, cp.asnumpy, cp.cuda.Stream.null.synchronize
    except Exception:
        if name == "cupy":
            raise
    return np, lambda value: value, lambda: None


def candidate_key(record: dict) -> tuple:
    return (
        record["case"], int(record["R"]), float(record["snr_db"]), record["method"],
        parameter_token(record["parameter"]),
    )


def final_key(record: dict) -> tuple:
    return record["case"], int(record["R"]), float(record["snr_db"]), record["method"]


def choose_candidate(records: list[dict], method: str, tolerance: dict) -> dict:
    valid = [record for record in records if record["status"] == "ok"]
    if not valid:
        raise RuntimeError(f"no successful {method} candidate")
    best_score = min(float(record["validation_score"]) for record in valid)
    threshold = float(tolerance["absolute"]) + float(tolerance["relative"]) * abs(best_score)
    tied = [record for record in valid if float(record["validation_score"]) <= best_score + threshold]
    return min(
        tied,
        key=lambda record: (
            complexity_key(method, record["parameter"]), parameter_token(record["parameter"])
        ),
    )


def candidates_for(stage: str, grid: dict | None, method: str, acceleration: int, snr_db: float):
    if grid is not None:
        return formal_candidates(grid, method, acceleration, snr_db)
    if stage == "pilot":
        return pilot_candidates(method, acceleration, snr_db)
    raise RuntimeError("formal evaluation requires a candidate grid")


def solve(method: str, parameter: dict, y, mask, initial, xp, protocol: dict):
    settings = protocol["optimization"]
    common = {
        "xp": xp,
        "relative_tolerance": float(settings["relative_iterate_tolerance"]),
        "maximum_iterations": int(settings["maximum_iterations"]),
        "initial_image": initial,
    }
    if method == "proposed":
        return solve_proposed(
            y, mask,
            a=float(parameter["a"]), c=float(parameter["c"]), alpha=float(parameter["alpha"]),
            root_tolerance=float(settings["root_tolerance"]), **common,
        )
    if method in {"soft", "hard"}:
        return solve_spectral_baseline(
            y, mask, normalized_threshold=float(parameter["q"]), kind=method, **common
        )
    if method == "l_plus_s":
        return solve_l_plus_s(
            y, mask,
            normalized_low_rank_threshold=float(parameter["q_L"]),
            normalized_sparse_threshold=float(parameter["q_S"]),
            **common,
        )
    raise ValueError(method)


def validation_snr(clean: np.ndarray, noise: np.ndarray, validation: np.ndarray) -> float:
    expanded = validation[:, None, :]
    signal_norm = float(np.linalg.norm((clean * expanded).ravel()))
    noise_norm = max(float(np.linalg.norm((noise * expanded).ravel())), np.finfo(float).eps)
    return float(20.0 * np.log10(signal_norm / noise_norm))


def save_reconstruction(
    output_root: Path, case: str, acceleration: int, snr_db: float, method: str,
    reconstruction: np.ndarray, parameter: dict,
) -> Path:
    path = output_root / "reconstructions" / case / f"R{acceleration}_snr{snr_db:g}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        if method in handle:
            del handle[method]
        dataset = handle.create_dataset(method, data=reconstruction, compression="gzip", compression_opts=1)
        dataset.attrs["parameter_json"] = json.dumps(parameter, sort_keys=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--R", type=int, nargs="*")
    parser.add_argument("--snr", type=float, nargs="*")
    parser.add_argument(
        "--methods", nargs="*", choices=("proposed", "soft", "hard", "l_plus_s")
    )
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy"), default="auto")
    parser.add_argument(
        "--candidate-grid", type=Path,
        help="Optional explicit candidate grid; useful for pilot-only boundary extensions.",
    )
    parser.add_argument(
        "--output-subdir",
        help="Optional simple name under outputs/ so diagnostics do not overwrite the main pilot.",
    )
    parser.add_argument(
        "--candidates-only", action="store_true",
        help="Evaluate validation candidates without selecting/refitting a final reconstruction.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = load_json(EVAL_ROOT / "config" / "protocol.json")
    accelerations = args.R or [int(item) for item in protocol["scope"]["accelerations"]]
    snr_levels = args.snr or [float(item) for item in protocol["scope"]["nominal_snr_db"]]
    methods = args.methods or list(protocol["scope"]["methods"])
    requested = set(args.cases or [])
    rows = [row for row in load_manifest(args.stage) if not requested or row["case_id"] in requested]
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    if requested and {row["case_id"] for row in rows} != requested:
        raise SystemExit("one or more requested cases are not in the selected manifest")
    if args.output_subdir is not None:
        if Path(args.output_subdir).name != args.output_subdir or args.output_subdir in {"", ".", ".."}:
            raise SystemExit("--output-subdir must be a simple directory name")
    formal_grid = None
    if args.candidate_grid is not None:
        if not args.candidate_grid.is_file():
            raise SystemExit(f"candidate grid is absent: {args.candidate_grid}")
        formal_grid = load_json(args.candidate_grid)
    elif args.stage == "formal":
        grid_path = EVAL_ROOT / "config" / "formal_candidates.json"
        if not grid_path.is_file():
            raise SystemExit("formal_candidates.json is absent; finish the pilot and run select_condition_grids.py")
        formal_grid = load_json(grid_path)

    if args.dry_run:
        total = 0
        for acceleration in accelerations:
            for snr_db in snr_levels:
                summary = {
                    method: len(candidates_for(args.stage, formal_grid, method, acceleration, snr_db))
                    for method in methods
                }
                total += len(rows) * sum(summary.values())
                print(f"R={acceleration} SNR={snr_db:g}: {summary}")
        refit_note = "no refits" if args.candidates_only else "plus selected refits"
        print(f"cases={len(rows)} candidate_fits={total} {refit_note}")
        return

    xp, to_cpu, synchronize = choose_backend(args.backend)
    output_root = EVAL_ROOT / "outputs" / (args.output_subdir or args.stage)
    candidate_path = output_root / "candidate_results.jsonl"
    final_path = output_root / "final_results.jsonl"
    candidate_records = load_jsonl(candidate_path)
    candidate_by_key = {candidate_key(record): record for record in candidate_records}
    finished = (
        set() if args.candidates_only
        else {final_key(record) for record in load_jsonl(final_path)}
    )
    sampling = protocol["sampling"]
    noise_settings = protocol["noise"]

    for row in rows:
        case = row["case_id"]
        processed = EVAL_ROOT / "data" / "processed" / args.stage / f"{case}.h5"
        if not processed.is_file():
            raise FileNotFoundError(f"{processed}; run preprocess_combined.py first")
        with h5py.File(processed, "r") as handle:
            reference = handle["reference"][...]
            preprocessing_metadata = json.loads(handle.attrs["metadata_json"])

        for acceleration in accelerations:
            acquired = variable_density_masks(
                reference.shape[-1], reference.shape[0], acceleration,
                int(sampling["acs_lines"]),
                child_seed(int(sampling["mask_seed_base"]), case, acceleration, "mask"),
            )
            training, validation = validation_split(
                acquired, int(sampling["acs_lines"]), float(sampling["validation_fraction"]),
                child_seed(int(sampling["validation_seed_base"]), case, acceleration, "split"),
            )
            clean = forward(reference, acquired, np)
            standard = standard_complex_noise(
                clean.shape,
                child_seed(int(noise_settings["noise_seed_base"]), case, acceleration, "noise_real"),
                child_seed(int(noise_settings["noise_seed_base"]), case, acceleration, "noise_imag"),
            )
            training_gpu = xp.asarray(training)
            acquired_gpu = xp.asarray(acquired)
            validation_count = int(validation.sum() * reference.shape[1])

            for snr_db in snr_levels:
                noise, sigma, realized_snr, measurement_count = scale_noise_to_nominal_snr(
                    clean, standard, acquired, snr_db
                )
                measured = clean + noise
                training_data = xp.asarray(measured * training[:, None, :])
                full_data = xp.asarray(measured)
                training_initial = adjoint(training_data, training_gpu, xp)
                full_initial = adjoint(full_data, acquired_gpu, xp)
                realized_validation_snr = validation_snr(clean, noise, validation)

                for method in methods:
                    key_final = (case, acceleration, float(snr_db), method)
                    if key_final in finished:
                        continue
                    method_records = []
                    condition_candidates = candidates_for(
                        args.stage, formal_grid, method, acceleration, snr_db
                    )
                    for parameter in condition_candidates:
                        prototype = {
                            "case": case, "R": acceleration, "snr_db": float(snr_db),
                            "method": method, "parameter": parameter,
                        }
                        key = candidate_key(prototype)
                        if key in candidate_by_key:
                            method_records.append(candidate_by_key[key])
                            continue
                        started = time.perf_counter()
                        try:
                            synchronize()
                            reconstruction_device, solver_log = solve(
                                method, parameter, training_data, training_gpu,
                                training_initial, xp, protocol,
                            )
                            synchronize()
                            reconstruction = to_cpu(reconstruction_device)
                            if not np.all(np.isfinite(reconstruction)):
                                raise FloatingPointError("solver produced a non-finite reconstruction")
                            residual = forward(reconstruction, validation, np) - measured * validation[:, None, :]
                            validation_score = float(
                                np.linalg.norm(residual.ravel()) ** 2 / validation_count
                            )
                            if not np.isfinite(validation_score):
                                raise FloatingPointError("validation score is non-finite")
                            record = {
                                **prototype,
                                "status": "ok",
                                "validation_score": validation_score,
                                "wall_seconds": time.perf_counter() - started,
                                "solver": solver_log,
                            }
                        except Exception as exc:
                            record = {
                                **prototype, "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "wall_seconds": time.perf_counter() - started,
                            }
                        append_jsonl(candidate_path, record)
                        candidate_by_key[key] = record
                        method_records.append(record)
                        print(
                            f"CANDIDATE {case} R={acceleration} SNR={snr_db:g} {method} "
                            f"{parameter_token(parameter)} {record['status']} "
                            f"score={record.get('validation_score')}", flush=True,
                        )

                    if args.candidates_only:
                        if hasattr(xp, "get_default_memory_pool"):
                            xp.get_default_memory_pool().free_all_blocks()
                        continue

                    selected = choose_candidate(
                        method_records, method, protocol["selection"]["tie_tolerance"]
                    )
                    parameter = selected["parameter"]
                    started = time.perf_counter()
                    synchronize()
                    reconstruction_device, solver_log = solve(
                        method, parameter, full_data, acquired_gpu, full_initial, xp, protocol
                    )
                    synchronize()
                    reconstruction = to_cpu(reconstruction_device)
                    refit_seconds = time.perf_counter() - started
                    psnr, ssim = magnitude_psnr_ssim(reconstruction, reference)
                    recon_path = save_reconstruction(
                        output_root, case, acceleration, snr_db, method, reconstruction, parameter
                    )
                    successful = [record for record in method_records if record["status"] == "ok"]
                    final_record = {
                        "case": case,
                        "patient_id": row.get("patient_id", "UNAVAILABLE"),
                        "view": row["view"],
                        "field_strength_t": float(row["field_strength_t"]),
                        "R": acceleration,
                        "snr_db": float(snr_db),
                        "method": method,
                        "selected_parameter": parameter,
                        "selected_validation_score": selected["validation_score"],
                        "realized_snr_db": realized_snr,
                        "realized_validation_snr_db": realized_validation_snr,
                        "sigma": sigma,
                        "measurement_count": measurement_count,
                        "validation_measurement_count": validation_count,
                        "complex_nmse": complex_nmse(reconstruction, reference),
                        "magnitude_psnr": psnr,
                        "magnitude_ssim_roi": ssim,
                        "search_candidate_count": len(method_records),
                        "search_success_count": len(successful),
                        "search_total_seconds": float(sum(record["wall_seconds"] for record in method_records)),
                        "refit_seconds": refit_seconds,
                        "refit_solver": solver_log,
                        "preprocessing": preprocessing_metadata,
                        "reconstruction_file": str(recon_path.relative_to(EVAL_ROOT)),
                    }
                    append_jsonl(final_path, final_record)
                    finished.add(key_final)
                    print(
                        f"FINAL {case} R={acceleration} SNR={snr_db:g} {method} "
                        f"NMSE={final_record['complex_nmse']:.6g} PSNR={psnr:.4f} SSIM={ssim:.5f}",
                        flush=True,
                    )
                    if hasattr(xp, "get_default_memory_pool"):
                        xp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
