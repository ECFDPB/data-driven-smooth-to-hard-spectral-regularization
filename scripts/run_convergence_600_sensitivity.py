"""Run a small fixed-parameter 600-iteration sensitivity diagnostic."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import run_evaluation as pipeline


EVAL_ROOT = pipeline.EVAL_ROOT
METHODS = ("proposed", "soft", "hard", "l_plus_s")
DEFAULT_COMPARISONS = (
    (4, 0.0, "l_plus_s"),
    (4, 30.0, "l_plus_s"),
    (8, 0.0, "l_plus_s"),
    (8, 30.0, "l_plus_s"),
)
ANOMALY_COMPARISONS = (
    (4, 10.0, "hard"),
    (4, 0.0, "l_plus_s"),
    (6, 0.0, "l_plus_s"),
)


def result_lookup(rows: list[dict]) -> dict[tuple, dict]:
    return {
        (row["case"], int(row["R"]), float(row["snr_db"]), row["method"]): row
        for row in rows
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def select_examples(
    rows: list[dict], comparisons: tuple[tuple[int, float, str], ...]
) -> list[dict]:
    lookup = result_lookup(rows)
    selections = []
    for acceleration, snr_db, baseline in comparisons:
        cases = sorted(
            row["case"]
            for row in rows
            if int(row["R"]) == acceleration
            and float(row["snr_db"]) == snr_db
            and row["method"] == "proposed"
        )
        deltas = []
        for case in cases:
            proposed = lookup[(case, acceleration, snr_db, "proposed")]
            comparator = lookup[(case, acceleration, snr_db, baseline)]
            deltas.append(
                (
                    case,
                    float(proposed["magnitude_psnr"])
                    - float(comparator["magnitude_psnr"]),
                )
            )
        condition_median = median([value for _, value in deltas])
        case, value = min(
            deltas,
            key=lambda item: (abs(item[1] - condition_median), item[0]),
        )
        selections.append(
            {
                "case": case,
                "R": acceleration,
                "snr_db": snr_db,
                "baseline": baseline,
                "condition_median_proposed_minus_baseline_psnr": condition_median,
                "selected_case_proposed_minus_baseline_psnr": value,
            }
        )
    return selections


def key(record: dict) -> tuple:
    return (
        record["case"],
        int(record["R"]),
        float(record["snr_db"]),
        record["method"],
        int(record["iteration_cap"]),
    )


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "case", "R", "snr_db", "method", "iteration_cap",
        "iterations", "converged", "final_relative_change",
        "psnr_300", "psnr_600", "delta_psnr_db",
        "ssim_300", "ssim_600", "delta_ssim",
        "nmse_300", "nmse_600", "relative_nmse_change",
        "runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("status") != "ok":
                continue
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy"), default="auto")
    parser.add_argument("--iteration-cap", type=int, default=600)
    parser.add_argument(
        "--preset", choices=("paper", "anomalies"), default="paper",
        help="paper reproduces the original four-condition check; anomalies runs the targeted diagnostic",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.iteration_cap <= 300:
        raise SystemExit("--iteration-cap must be greater than 300")

    formal_path = EVAL_ROOT / "outputs" / "formal" / "final_results.jsonl"
    formal_rows = pipeline.load_jsonl(formal_path)
    lookup = result_lookup(formal_rows)
    comparisons = (
        DEFAULT_COMPARISONS if args.preset == "paper" else ANOMALY_COMPARISONS
    )
    selections = select_examples(formal_rows, comparisons)
    suffix = "" if args.preset == "paper" else "_anomalies"
    output_root = EVAL_ROOT / "outputs" / f"formal_convergence_{args.iteration_cap}{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selected_cases.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selections, indent=2), flush=True)
    if args.dry_run:
        count = len(selections) * len(METHODS) if args.preset == "paper" else 2 * len(selections)
        print(f"fixed-parameter reconstructions={count}")
        return

    protocol = copy.deepcopy(
        pipeline.load_json(EVAL_ROOT / "config" / "protocol.json")
    )
    protocol["optimization"]["maximum_iterations"] = args.iteration_cap
    xp, to_cpu, synchronize = pipeline.choose_backend(args.backend)
    result_path = output_root / "results.jsonl"
    existing = pipeline.load_jsonl(result_path)
    by_key = {key(record): record for record in existing}
    sampling = protocol["sampling"]
    noise_settings = protocol["noise"]

    for selection in selections:
        case = selection["case"]
        acceleration = int(selection["R"])
        snr_db = float(selection["snr_db"])
        processed = EVAL_ROOT / "data" / "processed" / "formal" / f"{case}.h5"
        with pipeline.h5py.File(processed, "r") as handle:
            reference = handle["reference"][...]

        acquired = pipeline.variable_density_masks(
            reference.shape[-1],
            reference.shape[0],
            acceleration,
            int(sampling["acs_lines"]),
            pipeline.child_seed(
                int(sampling["mask_seed_base"]), case, acceleration, "mask"
            ),
        )
        clean = pipeline.forward(reference, acquired, pipeline.np)
        standard = pipeline.standard_complex_noise(
            clean.shape,
            pipeline.child_seed(
                int(noise_settings["noise_seed_base"]), case, acceleration, "noise_real"
            ),
            pipeline.child_seed(
                int(noise_settings["noise_seed_base"]), case, acceleration, "noise_imag"
            ),
        )
        noise, sigma, realized_snr, measurement_count = (
            pipeline.scale_noise_to_nominal_snr(
                clean, standard, acquired, snr_db
            )
        )
        measured = clean + noise
        acquired_gpu = xp.asarray(acquired)
        full_data = xp.asarray(measured)
        full_initial = pipeline.adjoint(full_data, acquired_gpu, xp)

        methods = METHODS if args.preset == "paper" else ("proposed", selection["baseline"])
        for method in methods:
            result_key = (case, acceleration, snr_db, method, args.iteration_cap)
            if result_key in by_key:
                continue
            original = lookup[(case, acceleration, snr_db, method)]
            parameter = original["selected_parameter"]
            prototype = {
                **selection,
                "method": method,
                "iteration_cap": args.iteration_cap,
                "selected_parameter": parameter,
                "sigma": sigma,
                "realized_snr_db": realized_snr,
                "measurement_count": measurement_count,
            }
            started = time.perf_counter()
            try:
                synchronize()
                reconstruction_device, solver = pipeline.solve(
                    method,
                    parameter,
                    full_data,
                    acquired_gpu,
                    full_initial,
                    xp,
                    protocol,
                )
                synchronize()
                reconstruction = to_cpu(reconstruction_device)
                if not pipeline.np.all(pipeline.np.isfinite(reconstruction)):
                    raise FloatingPointError("solver produced a non-finite reconstruction")
                psnr, ssim = pipeline.magnitude_psnr_ssim(reconstruction, reference)
                nmse = pipeline.complex_nmse(reconstruction, reference)
                old_nmse = float(original["complex_nmse"])
                recon_path = pipeline.save_reconstruction(
                    output_root,
                    case,
                    acceleration,
                    snr_db,
                    method,
                    reconstruction,
                    parameter,
                )
                record = {
                    **prototype,
                    "status": "ok",
                    "iterations": int(solver["iterations"]),
                    "converged": bool(solver["converged"]),
                    "final_relative_change": float(solver["final_relative_change"]),
                    "psnr_300": float(original["magnitude_psnr"]),
                    "psnr_600": float(psnr),
                    "delta_psnr_db": float(psnr - float(original["magnitude_psnr"])),
                    "ssim_300": float(original["magnitude_ssim_roi"]),
                    "ssim_600": float(ssim),
                    "delta_ssim": float(ssim - float(original["magnitude_ssim_roi"])),
                    "nmse_300": old_nmse,
                    "nmse_600": float(nmse),
                    "relative_nmse_change": float((nmse - old_nmse) / max(abs(old_nmse), 1e-30)),
                    "runtime_seconds": time.perf_counter() - started,
                    "solver": solver,
                    "reconstruction_file": str(recon_path.relative_to(EVAL_ROOT)),
                }
                if not all(
                    math.isfinite(float(record[name]))
                    for name in ("psnr_600", "ssim_600", "nmse_600")
                ):
                    raise FloatingPointError("a reconstruction metric is non-finite")
            except Exception as exc:
                record = {
                    **prototype,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_seconds": time.perf_counter() - started,
                }
            pipeline.append_jsonl(result_path, record)
            by_key[result_key] = record
            print(
                f"SENSITIVITY {case} R={acceleration} SNR={snr_db:g} {method} "
                f"{record['status']} delta_psnr={record.get('delta_psnr_db')}",
                flush=True,
            )
            if hasattr(xp, "get_default_memory_pool"):
                xp.get_default_memory_pool().free_all_blocks()

    completed = pipeline.load_jsonl(result_path)
    write_summary(output_root / "comparison_300_vs_600.csv", completed)
    failures = sum(record.get("status") != "ok" for record in completed)
    print(f"completed={len(completed)} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
