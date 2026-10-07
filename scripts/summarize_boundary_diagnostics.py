"""Create a reproducible summary of the iterative pilot boundary audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parents[1]


def candidate_count(path: Path) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unique = 0
    active_groups = 0
    for condition in payload["conditions"].values():
        for candidates in condition.values():
            count = len(candidates)
            unique += count
            active_groups += int(count > 0)
    return unique, active_groups


def result_counts(path: Path) -> tuple[int, int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return len(rows), sum(row.get("status") != "ok" for row in rows)


def load_results(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def main() -> None:
    before = [
        EVAL_ROOT / "config" / "pilot_boundary_candidates_round1.json",
        EVAL_ROOT / "config" / "pilot_boundary_candidates_round2.json",
        EVAL_ROOT / "outputs" / "pilot" / "round3_boundary_audit.json",
        EVAL_ROOT / "outputs" / "pilot" / "round4_boundary_audit.json",
        EVAL_ROOT / "outputs" / "pilot" / "post_round4_boundary_audit.json",
        EVAL_ROOT / "outputs" / "pilot" / "post_round5_boundary_audit.json",
    ]
    after = [
        before[1],
        before[2],
        before[3],
        before[4],
        before[5],
        EVAL_ROOT / "outputs" / "pilot" / "post_round6_boundary_audit.json",
    ]
    result_dirs = [
        "pilot_boundary_round1",
        "pilot_boundary_round2",
        "pilot_boundary_round3",
        "pilot_boundary_round4",
        "pilot_boundary_round5_retry",
        "pilot_boundary_round6",
    ]

    rows = []
    for index, (grid, audit, result_dir) in enumerate(zip(before, after, result_dirs), start=1):
        tested, tested_groups = candidate_count(grid)
        pending, pending_groups = candidate_count(audit)
        fits, failures = result_counts(
            EVAL_ROOT / "outputs" / result_dir / "candidate_results.jsonl"
        )
        rows.append(
            {
                "round": index,
                "outward_parameter_settings_tested": tested,
                "active_condition_method_groups_tested": tested_groups,
                "pilot_reconstructions": fits,
                "failures": failures,
                "outward_settings_remaining_after_round": pending,
                "active_groups_remaining_after_round": pending_groups,
            }
        )

    output_dir = EVAL_ROOT / "outputs" / "pilot" / "boundary_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "boundary_round_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_settings = sum(row["outward_parameter_settings_tested"] for row in rows)
    total_fits = sum(row["pilot_reconstructions"] for row in rows)
    total_failures = sum(row["failures"] for row in rows)
    markdown = [
        "**Pilot-only iterative parameter-boundary audit. Each round tests only "
        "coordinate-wise outward extensions triggered by a boundary winner whose "
        "validation-loss advantage exceeds the prespecified numerical tolerance.**",
        "",
        "| Round | Outward settings tested | Active condition-method groups | Pilot reconstructions | Failures | Settings remaining after round |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {round} | {outward_parameter_settings_tested} | "
            "{active_condition_method_groups_tested} | {pilot_reconstructions} | "
            "{failures} | {outward_settings_remaining_after_round} |".format(**row)
        )
    markdown.extend(
        [
            "",
            f"Across all rounds, {total_settings} outward parameter settings were tested "
            f"in {total_fits} pilot reconstructions, with {total_failures} recorded failures. "
            "The post-round-6 audit produced no further outward candidate under the "
            "prespecified tolerance.",
        ]
    )
    md_path = output_dir / "boundary_round_summary.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    base_path = EVAL_ROOT / "outputs" / "pilot" / "candidate_results.jsonl"
    extension_paths = [
        EVAL_ROOT / "outputs" / result_dir / "candidate_results.jsonl"
        for result_dir in result_dirs
    ]
    base_rows = load_results([base_path])
    all_rows = base_rows + load_results(extension_paths)
    base_scores: dict[tuple, list[float]] = defaultdict(list)
    all_scores: dict[tuple, list[float]] = defaultdict(list)
    for record in base_rows:
        if record.get("status") == "ok":
            key = (record["case"], int(record["R"]), float(record["snr_db"]), record["method"])
            base_scores[key].append(float(record["validation_score"]))
    for record in all_rows:
        if record.get("status") == "ok":
            key = (record["case"], int(record["R"]), float(record["snr_db"]), record["method"])
            all_scores[key].append(float(record["validation_score"]))

    settings_by_method = defaultdict(int)
    for path in before:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for condition in payload["conditions"].values():
            for method, candidates in condition.items():
                settings_by_method[method] += len(candidates)

    method_rows = []
    for method in ("proposed", "soft", "hard", "l_plus_s"):
        keys = sorted(key for key in base_scores if key[-1] == method)
        improvements = []
        meaningful = 0
        for key in keys:
            original = min(base_scores[key])
            expanded = min(all_scores[key])
            reduction = max(0.0, original - expanded)
            relative = 100.0 * reduction / max(abs(original), 1e-30)
            improvements.append(relative)
            tolerance = 1e-8 + 1e-4 * abs(expanded)
            meaningful += int(reduction > tolerance)
        method_rows.append(
            {
                "method": method,
                "pilot_case_condition_combinations": len(keys),
                "combinations_improved_vs_initial_grid": meaningful,
                "maximum_relative_validation_loss_reduction_percent": max(improvements),
                "outward_parameter_settings_tested": settings_by_method[method],
                "pilot_reconstructions": 4 * settings_by_method[method],
                "final_outward_settings_pending": 0,
            }
        )
    method_csv = output_dir / "boundary_method_summary.csv"
    with method_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)
    method_md = [
        "**Cumulative pilot-only boundary sensitivity relative to the initial grid. "
        "An improvement is counted only when the reduction in validation loss exceeds "
        "the prespecified numerical tolerance.**",
        "",
        "| Method | Improved combinations | Maximum reduction | Outward settings tested | Pilot reconstructions | Final pending |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"proposed": "Proposed", "soft": "Soft", "hard": "Hard", "l_plus_s": "L+S"}
    for row in method_rows:
        method_md.append(
            "| {label} | {improved}/{total} | {maximum:.3f}% | {settings} | {fits} | 0 |".format(
                label=labels[row["method"]],
                improved=row["combinations_improved_vs_initial_grid"],
                total=row["pilot_case_condition_combinations"],
                maximum=row["maximum_relative_validation_loss_reduction_percent"],
                settings=row["outward_parameter_settings_tested"],
                fits=row["pilot_reconstructions"],
            )
        )
    method_md_path = output_dir / "boundary_method_summary.md"
    method_md_path.write_text("\n".join(method_md) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {method_csv}")
    print(f"wrote {method_md_path}")


if __name__ == "__main__":
    main()
