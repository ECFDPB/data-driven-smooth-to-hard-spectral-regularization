"""Select a separate adaptive formal grid for every method and R x SNR."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parents[1]
sys.path.insert(0, str(EVAL_ROOT / "src"))

from single_image_eval.candidates import condition_key, parameter_token  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_cases() -> set[str]:
    with (EVAL_ROOT / "manifests" / "pilot_cases.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return {
            row["case_id"] for row in csv.DictReader(handle)
            if row.get("include", "true").lower() == "true"
        }


def select_group(
    records: list[dict],
    cases: set[str],
    minimum_count: int,
    maximum_count: int,
    mean_regret_band: float,
    fixed_count: int | None = None,
) -> tuple[list[dict], list[dict]]:
    by_candidate: dict[str, dict[str, dict]] = defaultdict(dict)
    parameter_by_token = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        token = parameter_token(record["parameter"])
        by_candidate[token][record["case"]] = record
        parameter_by_token[token] = record["parameter"]
    complete = {token: values for token, values in by_candidate.items() if set(values) == cases}
    required = fixed_count if fixed_count is not None else minimum_count
    if len(complete) < required:
        raise RuntimeError(
            f"only {len(complete)} candidates completed every pilot case; need {required}"
        )

    best_by_case = {
        case: min(float(values[case]["validation_score"]) for values in complete.values())
        for case in sorted(cases)
    }
    summaries = []
    for token, values in complete.items():
        regrets = [
            float(values[case]["validation_score"]) / max(best_by_case[case], np.finfo(float).eps) - 1.0
            for case in sorted(cases)
        ]
        wins = sum(
            np.isclose(
                float(values[case]["validation_score"]), best_by_case[case],
                rtol=1e-10, atol=1e-12,
            )
            for case in sorted(cases)
        )
        summaries.append(
            {
                "token": token,
                "parameter": parameter_by_token[token],
                "pilot_wins": int(wins),
                "mean_normalized_regret": float(np.mean(regrets)),
                "max_normalized_regret": float(np.max(regrets)),
            }
        )
    summaries.sort(
        key=lambda item: (
            -item["pilot_wins"], item["mean_normalized_regret"],
            item["max_normalized_regret"], item["token"],
        )
    )

    # Guarantee coverage of every case winner, then fill by mean pilot regret.
    winners = []
    for case in sorted(cases):
        best_token = min(
            complete,
            key=lambda token: (
                float(complete[token][case]["validation_score"]), token
            ),
        )
        if best_token not in winners:
            winners.append(best_token)
    ordered = winners + [item["token"] for item in summaries if item["token"] not in winners]
    if fixed_count is not None:
        selected_tokens = ordered[:fixed_count]
    else:
        best_mean_regret = min(item["mean_normalized_regret"] for item in summaries)
        near_best = {
            item["token"]
            for item in summaries
            if item["mean_normalized_regret"] <= best_mean_regret + mean_regret_band
        }
        selected_tokens = [token for token in ordered if token in set(winners) | near_best]
        for token in ordered:
            if len(selected_tokens) >= minimum_count:
                break
            if token not in selected_tokens:
                selected_tokens.append(token)
        # The maximum controls computation only; all individual pilot-case
        # winners remain protected even in the unlikely event that they exceed it.
        protected = set(winners)
        if len(selected_tokens) > maximum_count:
            selected_tokens = winners + [
                token for token in selected_tokens if token not in protected
            ][: max(0, maximum_count - len(winners))]
    selected = [parameter_by_token[token] for token in selected_tokens]
    selected_summaries = [
        next(item for item in summaries if item["token"] == token) for token in selected_tokens
    ]
    return selected, selected_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int,
        help="Optional explicit count for an audit; omit for adaptive per-condition counts.",
    )
    parser.add_argument("--minimum-count", type=int)
    parser.add_argument("--maximum-count", type=int)
    parser.add_argument("--mean-regret-band", type=float)
    parser.add_argument(
        "--input", type=Path, action="append",
        help="Candidate JSONL input; repeat to combine the base pilot and boundary rounds.",
    )
    parser.add_argument("--proposed-count", type=int)
    parser.add_argument("--soft-count", type=int)
    parser.add_argument("--hard-count", type=int)
    parser.add_argument("--l-plus-s-count", type=int)
    parser.add_argument("--output", type=Path, default=EVAL_ROOT / "config" / "formal_candidates.json")
    args = parser.parse_args()
    input_paths = args.input or [EVAL_ROOT / "outputs" / "pilot" / "candidate_results.jsonl"]
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"pilot result file not found: {path}")
    records = [record for path in input_paths for record in load_jsonl(path)]
    cases = expected_cases()
    grouped: dict[tuple[int, float, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(int(record["R"]), float(record["snr_db"]), record["method"])].append(record)

    protocol = json.loads((EVAL_ROOT / "config" / "protocol.json").read_text(encoding="utf-8"))
    adaptive = protocol["selection"]["adaptive_candidate_count"]
    minimum_count = int(args.minimum_count or adaptive["minimum"])
    maximum_count = int(args.maximum_count or adaptive["maximum"])
    mean_regret_band = float(
        adaptive["mean_normalized_regret_band"]
        if args.mean_regret_band is None else args.mean_regret_band
    )
    if minimum_count < 1 or maximum_count < minimum_count:
        raise ValueError("invalid adaptive candidate-count bounds")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": [
            str(path.relative_to(EVAL_ROOT)) if path.is_relative_to(EVAL_ROOT) else str(path)
            for path in input_paths
        ],
        "pilot_cases": sorted(cases),
        "candidate_count_policy": {
            "fixed_count": args.count,
            "method_fixed_counts": {
                "proposed": args.proposed_count,
                "soft": args.soft_count,
                "hard": args.hard_count,
                "l_plus_s": args.l_plus_s_count,
            },
            "minimum": minimum_count,
            "maximum": maximum_count,
            "mean_normalized_regret_band": mean_regret_band,
        },
        "selection_rule": "separate per-condition grids: protect per-case winners, retain near-best mean normalized validation regret, then enforce small adaptive bounds",
        "conditions": {},
        "diagnostics": {},
    }
    for acceleration in protocol["scope"]["accelerations"]:
        for snr_db in protocol["scope"]["nominal_snr_db"]:
            key = condition_key(acceleration, snr_db)
            output["conditions"][key] = {}
            output["diagnostics"][key] = {}
            for method in protocol["scope"]["methods"]:
                group_key = (int(acceleration), float(snr_db), method)
                if group_key not in grouped:
                    raise RuntimeError(f"pilot results missing {key}/{method}")
                method_count = {
                    "proposed": args.proposed_count,
                    "soft": args.soft_count,
                    "hard": args.hard_count,
                    "l_plus_s": args.l_plus_s_count,
                }[method]
                effective_fixed_count = method_count if method_count is not None else args.count
                selected, summaries = select_group(
                    grouped[group_key], cases,
                    minimum_count=minimum_count,
                    maximum_count=maximum_count,
                    mean_regret_band=mean_regret_band,
                    fixed_count=effective_fixed_count,
                )
                output["conditions"][key][method] = selected
                output["diagnostics"][key][method] = {
                    "selected_count": len(selected),
                    "selected_candidates": summaries,
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
