"""Condition-specific pilot pools and pilot-to-formal grid selection."""

from __future__ import annotations

import json


def condition_key(acceleration: int, snr_db: float) -> str:
    return f"R{int(acceleration)}_snr{float(snr_db):g}"


def parameter_token(parameter: dict) -> str:
    return json.dumps(parameter, sort_keys=True, separators=(",", ":"))


def _round(value: float) -> float:
    return float(f"{value:.8g}")


def pilot_candidates(method: str, acceleration: int, snr_db: float) -> list[dict]:
    """Return a sparse pilot pool tailored to one R x SNR condition.

    These are starting ranges, not final formal grids. The formal grid for each
    condition is chosen exclusively from disjoint pilot-series validation losses,
    and its candidate count is allowed to differ from other conditions.
    """

    r_scale = {4: 1.0, 6: 1.75, 8: 3.0}[int(acceleration)]
    snr = int(round(float(snr_db)))
    if snr not in {0, 10, 20, 30}:
        raise ValueError(f"unsupported SNR condition {snr_db}")

    if method == "proposed":
        center_alpha = {0: 0.30, 10: 0.10, 20: 0.03, 30: 0.01}[snr] * r_scale
        hardness = {
            0: (1.0, 2.0, 4.0),
            10: (2.0, 4.0, 8.0),
            20: (4.0, 8.0, 16.0),
            30: (8.0, 16.0, 32.0),
        }[snr]
        locations = (0.35, 0.55, 0.80)
        pattern = (
            (0.25, 0, 0), (0.25, 1, 1),
            (0.50, 0, 1), (0.50, 2, 2),
            (1.00, 0, 2), (1.00, 1, 0), (1.00, 2, 1),
            (2.00, 0, 0), (2.00, 1, 2), (2.00, 2, 1),
            (4.00, 1, 1), (4.00, 2, 2),
        )
        return [
            {"alpha": _round(center_alpha * multiplier), "a": hardness[a_index], "c": locations[c_index]}
            for multiplier, a_index, c_index in pattern
        ]

    if method == "soft":
        center = {0: 0.080, 10: 0.050, 20: 0.025, 30: 0.010}[snr] * r_scale**0.5
        return [{"q": _round(center * item)} for item in (0.2, 0.4, 0.7, 1.0, 1.5, 2.3, 3.4, 5.0)]

    if method == "hard":
        center = {0: 0.30, 10: 0.22, 20: 0.15, 30: 0.10}[snr] * r_scale**0.35
        return [{"q": _round(center * item)} for item in (0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4)]

    if method == "l_plus_s":
        low = {0: 0.020, 10: 0.012, 20: 0.007, 30: 0.004}[snr] * r_scale**0.5
        sparse = {0: 0.040, 10: 0.025, 20: 0.012, 30: 0.006}[snr] * r_scale**0.35
        pattern = ((0.4, 1.8), (0.6, 1.2), (0.8, 0.8), (1.0, 1.0), (1.3, 0.7), (1.7, 0.5), (2.2, 0.35), (3.0, 0.25), (0.35, 3.0))
        return [{"q_L": _round(low * a), "q_S": _round(sparse * b)} for a, b in pattern]

    raise ValueError(method)


def formal_candidates(grid: dict, method: str, acceleration: int, snr_db: float) -> list[dict]:
    key = condition_key(acceleration, snr_db)
    try:
        values = grid["conditions"][key][method]
    except KeyError as exc:
        raise KeyError(f"formal grid is missing {key}/{method}; run select_condition_grids.py") from exc
    return [dict(item) for item in values]


def complexity_key(method: str, parameter: dict) -> tuple:
    if method in {"soft", "hard"}:
        return (-parameter["q"],)
    if method == "l_plus_s":
        return (-(parameter["q_L"] + parameter["q_S"]), -parameter["q_L"], -parameter["q_S"])
    return (-parameter["alpha"], parameter["a"], -parameter["c"])
