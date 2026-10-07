"""Generate one-step outward candidates wherever a pilot winner touches a boundary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parents[1]
sys.path.insert(0, str(EVAL_ROOT / "src"))

from single_image_eval.candidates import (  # noqa: E402
    complexity_key,
    condition_key,
    parameter_token,
)


PARAMETER_NAMES = {
    "proposed": ("alpha", "a", "c"),
    "soft": ("q",),
    "hard": ("q",),
    "l_plus_s": ("q_L", "q_S"),
}


def load_jsonl(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def expected_cases() -> set[str]:
    path = EVAL_ROOT / "manifests" / "pilot_cases.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["case_id"] for row in csv.DictReader(handle)
            if row.get("include", "true").lower() == "true"
        }


def rounded(value: float) -> float:
    return float(f"{value:.8g}")


def outward_level(levels: list[float], name: str, direction: str) -> float:
    unique = sorted(set(float(value) for value in levels))
    if len(unique) < 2:
        raise ValueError(f"need at least two levels for {name}")
    if direction == "lower":
        edge, neighbour = unique[0], unique[1]
        if name == "c":
            value = edge - (neighbour - edge)
        else:
            value = edge * edge / neighbour
        value = max(value, edge * 0.1, 1e-12)
    elif direction == "upper":
        neighbour, edge = unique[-2], unique[-1]
        if name == "c":
            value = edge + (edge - neighbour)
        else:
            value = edge * edge / neighbour
    else:
        raise ValueError(direction)
    return rounded(value)


def choose_winner(records: list[dict], method: str, tolerance: dict) -> dict:
    best_score = min(float(record["validation_score"]) for record in records)
    threshold = float(tolerance["absolute"]) + float(tolerance["relative"]) * abs(best_score)
    tied = [
        record for record in records
        if float(record["validation_score"]) <= best_score + threshold
    ]
    return min(
        tied,
        key=lambda record: (
            complexity_key(method, record["parameter"]),
            parameter_token(record["parameter"]),
        ),
    )


def expansion_group(
    records: list[dict], method: str, cases: set[str], tolerance: dict,
) -> tuple[list[dict], list[dict]]:
    by_token: dict[str, dict[str, dict]] = defaultdict(dict)
    parameters = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        token = parameter_token(record["parameter"])
        by_token[token][record["case"]] = record
        parameters[token] = record["parameter"]
    complete = {token: values for token, values in by_token.items() if set(values) == cases}
    if not complete:
        raise RuntimeError(f"no complete pilot candidates for {method}")

    names = PARAMETER_NAMES[method]
    levels = {
        name: sorted({float(parameter[name]) for parameter in parameters.values()})
        for name in names
    }
    lower = {name: values[0] for name, values in levels.items()}
    upper = {name: values[-1] for name, values in levels.items()}
    outward = {
        (name, direction): outward_level(values, name, direction)
        for name, values in levels.items()
        for direction in ("lower", "upper")
    }

    generated: dict[str, dict] = {}
    diagnostics = []
    for case in sorted(cases):
        case_records = [values[case] for values in complete.values()]
        winner = choose_winner(case_records, method, tolerance)
        source = dict(winner["parameter"])
        winner_score = float(winner["validation_score"])
        score_tolerance = (
            float(tolerance["absolute"])
            + float(tolerance["relative"]) * abs(winner_score)
        )
        for name in names:
            value = float(source[name])
            directions = []
            if math.isclose(value, lower[name], rel_tol=1e-10, abs_tol=1e-12):
                directions.append("lower")
            if math.isclose(value, upper[name], rel_tol=1e-10, abs_tol=1e-12):
                directions.append("upper")
            for direction in directions:
                # A flat validation plateau is not evidence that the range is
                # truncated. Expand only if the boundary winner improves on
                # the best candidate away from that boundary beyond the same
                # numerical tolerance used for candidate ties.
                interior_records = [
                    record for record in case_records
                    if not math.isclose(
                        float(record["parameter"][name]), value,
                        rel_tol=1e-10, abs_tol=1e-12,
                    )
                ]
                if not interior_records:
                    continue
                interior_best = min(
                    float(record["validation_score"]) for record in interior_records
                )
                if winner_score >= interior_best - score_tolerance:
                    continue
                candidate = dict(source)
                candidate[name] = outward[(name, direction)]
                token = parameter_token(candidate)
                if token in complete:
                    continue
                generated[token] = candidate
                diagnostics.append(
                    {
                        "case": case,
                        "source_parameter": source,
                        "source_validation_score": winner_score,
                        "best_interior_validation_score": interior_best,
                        "boundary_improvement": interior_best - winner_score,
                        "required_improvement": score_tolerance,
                        "axis": name,
                        "direction": direction,
                        "old_boundary": value,
                        "expanded_value": candidate[name],
                        "expanded_parameter": candidate,
                    }
                )
    return list(generated.values()), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path,
        default=EVAL_ROOT / "config" / "pilot_boundary_candidates.json",
    )
    args = parser.parse_args()
    for path in args.input:
        if not path.is_file():
            raise SystemExit(f"candidate result file not found: {path}")

    protocol = json.loads((EVAL_ROOT / "config" / "protocol.json").read_text(encoding="utf-8"))
    cases = expected_cases()
    records = load_jsonl(args.input)
    grouped: dict[tuple[int, float, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(int(record["R"]), float(record["snr_db"]), record["method"])].append(record)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": [str(path) for path in args.input],
        "selection_rule": (
            "one-step coordinate-wise outward expansion: when a per-case pilot winner "
            "touches an observed lower or upper axis boundary, move only that coordinate "
            "one level outward and keep the other coordinates fixed"
        ),
        "conditions": {},
        "diagnostics": {},
    }
    total = 0
    for acceleration in protocol["scope"]["accelerations"]:
        for snr_db in protocol["scope"]["nominal_snr_db"]:
            key = condition_key(acceleration, snr_db)
            output["conditions"][key] = {}
            output["diagnostics"][key] = {}
            for method in protocol["scope"]["methods"]:
                group_key = (int(acceleration), float(snr_db), method)
                if group_key not in grouped:
                    raise RuntimeError(f"pilot results missing {key}/{method}")
                candidates, diagnostics = expansion_group(
                    grouped[group_key], method, cases, protocol["selection"]["tie_tolerance"]
                )
                output["conditions"][key][method] = candidates
                output["diagnostics"][key][method] = {
                    "candidate_count": len(candidates),
                    "triggers": diagnostics,
                }
                total += len(candidates)
                print(f"{key} {method}: {len(candidates)} outward candidates")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}")
    print(f"UNIQUE_OUTWARD_CANDIDATES={total} PILOT_FITS={total * len(cases)}")


if __name__ == "__main__":
    main()
