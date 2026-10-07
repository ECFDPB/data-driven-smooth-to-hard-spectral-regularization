"""Generate the paper convergence and iteration-cap sensitivity figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


EVAL_ROOT = Path(__file__).resolve().parents[1]
GENERAL_RESULTS = EVAL_ROOT / "outputs" / "formal_convergence_600" / "results.jsonl"
ANOMALY_RESULTS = (
    EVAL_ROOT / "outputs" / "formal_convergence_600_anomalies" / "results.jsonl"
)
OUTPUT_ROOT = EVAL_ROOT / "outputs" / "formal" / "paper_summary"

CORNER_CONDITIONS = ((4, 0.0), (4, 30.0), (8, 0.0), (8, 30.0))
TARGETED_COMPARISONS = (
    (4, 10.0, "hard", "R=4, 10 dB\nvs Hard"),
    (4, 0.0, "l_plus_s", "R=4, 0 dB\nvs L+S"),
    (6, 0.0, "l_plus_s", "R=6, 0 dB\nvs L+S"),
)
METHOD_LABELS = {"proposed": "Proposed", "l_plus_s": "L+S"}
METHOD_COLORS = {"proposed": "#0072B2", "l_plus_s": "#D55E00"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lookup(rows: list[dict]) -> dict[tuple, dict]:
    return {
        (int(row["R"]), float(row["snr_db"]), row["method"]): row
        for row in rows
        if row.get("status") == "ok"
    }


def main() -> None:
    general = lookup(load_jsonl(GENERAL_RESULTS))
    targeted = lookup(load_jsonl(ANOMALY_RESULTS))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11.2, 6.8), constrained_layout=True)
    grid = GridSpec(2, 4, figure=fig, height_ratios=(1.15, 1.0))

    handles = []
    labels = []
    for panel, (acceleration, snr_db) in enumerate(CORNER_CONDITIONS):
        axis = fig.add_subplot(grid[0, panel])
        stops = []
        for method in ("proposed", "l_plus_s"):
            row = general[(acceleration, snr_db, method)]
            history = row["solver"]["history"]
            iterations = np.asarray([item["iteration"] for item in history], dtype=float)
            changes = np.maximum(
                np.asarray([item["relative_change"] for item in history], dtype=float),
                1e-8,
            )
            (line,) = axis.plot(
                iterations,
                changes,
                color=METHOD_COLORS[method],
                linewidth=1.7,
                label=METHOD_LABELS[method],
            )
            if panel == 0:
                handles.append(line)
                labels.append(METHOD_LABELS[method])
            stops.append(f"{METHOD_LABELS[method]}: {int(row['iterations'])}")

        axis.axvline(300, color="#777777", linestyle=":", linewidth=1.1)
        axis.axhline(1e-5, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xlim(0, 610)
        axis.set_ylim(5e-6, 1.0)
        axis.set_xticks((0, 150, 300, 450, 600))
        axis.grid(True, which="major", color="#DDDDDD", linewidth=0.6)
        axis.set_title(
            rf"$R={acceleration}$, $\mathrm{{SNR}}_{{\mathrm{{nom}}}}={snr_db:g}$ dB"
        )
        axis.set_xlabel("Iteration")
        if panel == 0:
            axis.set_ylabel("Relative iterate change")
            axis.text(
                -0.20,
                1.08,
                "(a)",
                transform=axis.transAxes,
                fontsize=9,
                fontweight="bold",
            )
        axis.text(
            0.04,
            0.05,
            "Stop iterations\n" + ", ".join(stops),
            transform=axis.transAxes,
            fontsize=6.8,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    axis = fig.add_subplot(grid[1, :])
    gaps_300 = []
    gaps_600 = []
    xlabels = []
    for acceleration, snr_db, baseline, label in TARGETED_COMPARISONS:
        proposed = targeted[(acceleration, snr_db, "proposed")]
        comparator = targeted[(acceleration, snr_db, baseline)]
        gaps_300.append(float(proposed["psnr_300"]) - float(comparator["psnr_300"]))
        gaps_600.append(float(proposed["psnr_600"]) - float(comparator["psnr_600"]))
        xlabels.append(label)

    positions = np.arange(len(xlabels), dtype=float)
    width = 0.32
    bars_300 = axis.bar(
        positions - width / 2,
        gaps_300,
        width,
        color="#999999",
        label="300 iterations",
    )
    bars_600 = axis.bar(
        positions + width / 2,
        gaps_600,
        width,
        color="#0072B2",
        label="600 iterations",
    )
    axis.axhline(0, color="#222222", linewidth=0.9)
    axis.set_ylabel(r"PSNR difference, Proposed $-$ baseline (dB)")
    axis.set_xticks(positions, xlabels)
    axis.set_ylim(-1.05, 0.55)
    axis.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.set_title("Fixed-parameter sensitivity to increasing the iteration cap")
    axis.text(
        -0.04,
        1.05,
        "(b)",
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    for bars in (bars_300, bars_600):
        for bar in bars:
            value = bar.get_height()
            offset = 0.035 if value >= 0 else -0.055
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.2,
            )

    fig.suptitle(
        "Convergence trajectories and 300-to-600 iteration-cap sensitivity",
        fontsize=11,
        y=1.075,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_ROOT / "fig3_convergence_sensitivity.png"
    pdf_path = OUTPUT_ROOT / "fig3_convergence_sensitivity.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
