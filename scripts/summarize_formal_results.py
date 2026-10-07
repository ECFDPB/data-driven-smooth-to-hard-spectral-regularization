"""Summarize the frozen formal cohort in paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parents[1]

METHODS = ("proposed", "soft", "hard", "l_plus_s")
BASELINES = ("hard", "soft", "l_plus_s")
BASELINE_LABELS = {"hard": "Hard", "soft": "Soft", "l_plus_s": "L+S"}
METRICS = {
    "complex_nmse": -1.0,
    "magnitude_psnr": 1.0,
    "magnitude_ssim_roi": 1.0,
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def validate(rows: list[dict]) -> tuple[list[str], list[int], list[float]]:
    cases = sorted({str(row["case"]) for row in rows})
    accelerations = sorted({int(row["R"]) for row in rows})
    snr_levels = sorted({float(row["snr_db"]) for row in rows})
    if len(cases) != 20:
        raise ValueError(f"expected 20 formal cases, found {len(cases)}")
    expected = {
        (case, acceleration, snr_db, method)
        for case in cases
        for acceleration in accelerations
        for snr_db in snr_levels
        for method in METHODS
    }
    observed = {
        (str(row["case"]), int(row["R"]), float(row["snr_db"]), row["method"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        missing = sorted(expected - observed)[:10]
        duplicate_count = len(rows) - len(observed)
        raise ValueError(
            f"formal results are incomplete or duplicated: missing={missing}, "
            f"duplicate_count={duplicate_count}"
        )
    for row in rows:
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"non-finite {metric} in {row['case']}")
    return cases, accelerations, snr_levels


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=EVAL_ROOT / "outputs" / "formal" / "final_results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_ROOT / "outputs" / "formal" / "paper_summary",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    cases, accelerations, snr_levels = validate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lookup = {
        (str(row["case"]), int(row["R"]), float(row["snr_db"]), row["method"]): row
        for row in rows
    }

    condition_means = []
    convergence_rows = []
    grouped: dict[tuple[int, float, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["R"]), float(row["snr_db"]), row["method"])].append(row)
    for acceleration in accelerations:
        for snr_db in snr_levels:
            for method in METHODS:
                group = grouped[(acceleration, snr_db, method)]
                solver = [record["refit_solver"] for record in group]
                condition_means.append(
                    {
                        "R": acceleration,
                        "snr_db": snr_db,
                        "method": method,
                        "n_cases": len(group),
                        "mean_complex_nmse": statistics.fmean(float(x["complex_nmse"]) for x in group),
                        "mean_magnitude_psnr": statistics.fmean(float(x["magnitude_psnr"]) for x in group),
                        "mean_magnitude_ssim_roi": statistics.fmean(float(x["magnitude_ssim_roi"]) for x in group),
                        "median_refit_seconds": statistics.median(float(x["refit_seconds"]) for x in group),
                        "median_search_seconds": statistics.median(float(x["search_total_seconds"]) for x in group),
                    }
                )
                convergence_rows.append(
                    {
                        "R": acceleration,
                        "snr_db": snr_db,
                        "method": method,
                        "n_cases": len(group),
                        "converged_count": sum(bool(x.get("converged", False)) for x in solver),
                        "iteration_cap_count": sum(
                            int(x.get("iterations", 0)) >= 300 and not bool(x.get("converged", False))
                            for x in solver
                        ),
                        "median_iterations": statistics.median(int(x.get("iterations", 0)) for x in solver),
                        "median_final_relative_change": statistics.median(
                            float(x.get("final_relative_change", math.nan)) for x in solver
                        ),
                    }
                )

    paired_rows = []
    psnr_cells: dict[tuple[int, float, str], tuple[float, int]] = {}
    for acceleration in accelerations:
        for snr_db in snr_levels:
            for baseline in BASELINES:
                for metric, orientation in METRICS.items():
                    differences = []
                    for case in cases:
                        proposed = float(lookup[(case, acceleration, snr_db, "proposed")][metric])
                        comparator = float(lookup[(case, acceleration, snr_db, baseline)][metric])
                        differences.append(orientation * (proposed - comparator))
                    record = {
                        "R": acceleration,
                        "snr_db": snr_db,
                        "baseline": baseline,
                        "metric": metric,
                        "difference_definition": (
                            "baseline_minus_proposed" if metric == "complex_nmse"
                            else "proposed_minus_baseline"
                        ),
                        "n_cases": len(differences),
                        "mean_difference": statistics.fmean(differences),
                        "sample_sd": statistics.stdev(differences),
                        "median_difference": statistics.median(differences),
                        "proposed_wins": sum(value > 0 for value in differences),
                        "ties": sum(value == 0 for value in differences),
                        "proposed_losses": sum(value < 0 for value in differences),
                        "minimum_difference": min(differences),
                        "maximum_difference": max(differences),
                    }
                    paired_rows.append(record)
                    if metric == "magnitude_psnr":
                        psnr_cells[(acceleration, snr_db, baseline)] = (
                            record["mean_difference"], record["proposed_wins"]
                        )

    write_csv(args.output_dir / "condition_method_means.csv", condition_means)
    write_csv(args.output_dir / "paired_statistics.csv", paired_rows)
    write_csv(args.output_dir / "convergence_at_300.csv", convergence_rows)

    caption = (
        "Mean paired magnitude-PSNR difference in dB (Proposed minus baseline), "
        "with the number of Proposed wins in parentheses. Positive values favour "
        "Proposed; each cell contains 20 paired OCMR series."
    )
    markdown = [
        f"**Table. {caption}**",
        "",
        "| R | SNR (dB) | versus Hard | versus Soft | versus L+S |",
        "|---:|---:|---:|---:|---:|",
    ]
    for acceleration in accelerations:
        for snr_db in snr_levels:
            cells = []
            for baseline in BASELINES:
                mean, wins = psnr_cells[(acceleration, snr_db, baseline)]
                cells.append(f"{signed(mean)} ({wins}/20)")
            markdown.append(
                f"| {acceleration} | {snr_db:g} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )
    (args.output_dir / "paper_table_psnr.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + caption.replace("%", "\\%") + "}",
        "\\label{tab:single_image_psnr}",
        "\\begin{tabular}{ccccc}",
        "\\toprule",
        "$R$ & SNR (dB) & versus Hard & versus Soft & versus L+S \\\\",
        "\\midrule",
    ]
    for acceleration in accelerations:
        for snr_db in snr_levels:
            cells = []
            for baseline in BASELINES:
                mean, wins = psnr_cells[(acceleration, snr_db, baseline)]
                ascii_signed = f"{mean:+.3f}"
                cells.append(f"{ascii_signed} ({wins}/20)")
            latex.append(
                f"{acceleration} & {snr_db:g} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (args.output_dir / "paper_table_psnr.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )

    print(f"validated formal records: {len(rows)}")
    print(f"wrote {args.output_dir / 'paper_table_psnr.md'}")
    print(f"wrote {args.output_dir / 'paper_table_psnr.tex'}")
    print(f"wrote {args.output_dir / 'paired_statistics.csv'}")
    print(f"wrote {args.output_dir / 'condition_method_means.csv'}")
    print(f"wrote {args.output_dir / 'convergence_at_300.csv'}")


if __name__ == "__main__":
    main()
