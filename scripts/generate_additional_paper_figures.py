"""Generate distribution, qualitative, and full-metric paper figures."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT / "src"))

from single_image_eval.metrics import magnitude_psnr_ssim
from single_image_eval.noise import scale_noise_to_nominal_snr, standard_complex_noise
from single_image_eval.operator import adjoint, forward
from single_image_eval.sampling import child_seed, variable_density_masks

FINAL_RESULTS = EVAL_ROOT / "outputs" / "formal" / "final_results.jsonl"
OUTPUT_ROOT = EVAL_ROOT / "outputs" / "formal" / "paper_summary"
ACCELERATIONS = (4, 6, 8)
SNR_LEVELS = (0.0, 10.0, 20.0, 30.0)
BASELINES = ("hard", "soft", "l_plus_s")
RECONSTRUCTION_METHODS = ("proposed", "soft", "hard", "l_plus_s")
DISPLAY_ITEMS = (
    "reference",
    "sampling_mask",
    "zero_filled",
    "proposed",
    "soft",
    "hard",
    "l_plus_s",
)
METHOD_LABELS = {
    "reference": "Reference",
    "sampling_mask": "Sampling mask",
    "zero_filled": "Zero-filled",
    "proposed": "Proposed",
    "soft": "Soft",
    "hard": "Hard",
    "l_plus_s": "L+S",
}
REPRESENTATIVE_CONDITIONS = (
    ("fs_0020_3T", 4, 0.0, "Severe noise; median vs Soft"),
    ("fs_0010_1_5T", 8, 30.0, "Clean, high acceleration"),
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def result_lookup(rows: list[dict]) -> dict[tuple, dict]:
    return {
        (row["case"], int(row["R"]), float(row["snr_db"]), row["method"]): row
        for row in rows
        if row.get("status", "ok") == "ok"
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(draws, dtype=float)
    batch = 1_000
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def paired_values(
    rows: list[dict], baseline: str, acceleration: int, snr_db: float, metric: str
) -> np.ndarray:
    lookup = result_lookup(rows)
    cases = sorted(
        row["case"]
        for row in rows
        if int(row["R"]) == acceleration
        and float(row["snr_db"]) == snr_db
        and row["method"] == "proposed"
    )
    values = []
    for case in cases:
        proposed = float(lookup[(case, acceleration, snr_db, "proposed")][metric])
        comparator = float(lookup[(case, acceleration, snr_db, baseline)][metric])
        if metric == "complex_nmse":
            values.append(comparator - proposed)
        else:
            values.append(proposed - comparator)
    return np.asarray(values, dtype=float)


def generate_distribution_figure(rows: list[dict]) -> None:
    # Keep Figure 4 synchronized with the restrained blue-grey style used by
    # Figure 5.  The dedicated renderer also avoids style drift when this
    # multi-figure script is rerun later.
    from generate_psnr_distribution_unified import main as generate_unified_psnr_figure

    generate_unified_psnr_figure()
    return

    conditions = [(r, snr) for r in ACCELERATIONS for snr in SNR_LEVELS]
    labels = [f"R{r}\n{snr:g}" for r, snr in conditions]
    rng = np.random.default_rng(2026100305)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), sharey=True, constrained_layout=True)
    ci_rows = []

    for panel, (axis, baseline) in enumerate(zip(axes, BASELINES)):
        datasets = []
        means = []
        lows = []
        highs = []
        for condition_index, (acceleration, snr_db) in enumerate(conditions):
            values = paired_values(
                rows, baseline, acceleration, snr_db, "magnitude_psnr"
            )
            mean = float(values.mean())
            low, high = bootstrap_mean_ci(
                values, 2026100400 + 100 * panel + condition_index
            )
            datasets.append(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
            ci_rows.append(
                {
                    "R": acceleration,
                    "snr_nom_db": snr_db,
                    "baseline": baseline,
                    "n": values.size,
                    "mean_delta_psnr_db": mean,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "median_delta_psnr_db": float(np.median(values)),
                    "wins": int(np.sum(values > 0)),
                }
            )

        positions = np.arange(1, len(conditions) + 1)
        box = axis.boxplot(
            datasets,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#666666", "linewidth": 0.8},
            capprops={"color": "#666666", "linewidth": 0.8},
            boxprops={"edgecolor": "#666666", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#D9EAF7")
            patch.set_alpha(0.75)

        for position, values in zip(positions, datasets):
            jitter = rng.uniform(-0.19, 0.19, size=values.size)
            axis.scatter(
                position + jitter,
                values,
                s=9,
                color="#4C4C4C",
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )

        means_array = np.asarray(means)
        axis.errorbar(
            positions,
            means_array,
            yerr=np.vstack((means_array - np.asarray(lows), np.asarray(highs) - means_array)),
            fmt="D",
            markersize=4.5,
            color="#B2182B",
            ecolor="#B2182B",
            elinewidth=1.4,
            capsize=2.5,
            zorder=4,
        )
        axis.axhline(0, color="#111111", linewidth=0.9)
        axis.set_title(f"Proposed $-$ {METHOD_LABELS[baseline]}")
        axis.set_xticks(positions, labels)
        axis.set_xlabel(r"Condition: $R$ / $\mathrm{SNR}_{\mathrm{nom}}$ (dB)")
        axis.grid(True, axis="y", color="#E1E1E1", linewidth=0.6)
        if panel == 0:
            axis.set_ylabel("Paired PSNR difference (dB)")

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C4C4C", alpha=0.65, markersize=5, label="Individual series"),
        Patch(facecolor="#D9EAF7", edgecolor="#666666", label="Median and interquartile range"),
        Line2D([0], [0], marker="D", color="#B2182B", markersize=5, label="Mean and bootstrap 95% CI"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Distribution of paired magnitude-PSNR differences", fontsize=12, y=1.14)

    png = OUTPUT_ROOT / "fig4_paired_psnr_distributions.png"
    pdf = OUTPUT_ROOT / "fig4_paired_psnr_distributions.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    with (OUTPUT_ROOT / "paired_psnr_bootstrap_ci.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ci_rows[0]))
        writer.writeheader()
        writer.writerows(ci_rows)


def generate_metric_heatmap(rows: list[dict]) -> None:
    metric_specs = (
        ("complex_nmse", "Complex NMSE", 3),
        ("magnitude_psnr", "Magnitude PSNR (dB)", 2),
        ("magnitude_ssim_roi", "Magnitude SSIM", 3),
    )
    arrays: dict[tuple[str, str], np.ndarray] = {}
    limits = {}
    for metric, _, _ in metric_specs:
        all_values = []
        for baseline in BASELINES:
            matrix = np.zeros((len(ACCELERATIONS), len(SNR_LEVELS)), dtype=float)
            for i, acceleration in enumerate(ACCELERATIONS):
                for j, snr_db in enumerate(SNR_LEVELS):
                    values = paired_values(rows, baseline, acceleration, snr_db, metric)
                    matrix[i, j] = values.mean()
                    all_values.append(matrix[i, j])
            arrays[(metric, baseline)] = matrix
        limits[metric] = max(abs(float(np.min(all_values))), abs(float(np.max(all_values))), 1e-12)

    fig = plt.figure(figsize=(10.8, 8.2), constrained_layout=True)
    grid = GridSpec(3, 4, figure=fig, width_ratios=(1, 1, 1, 0.045))
    for row_index, (metric, metric_label, decimals) in enumerate(metric_specs):
        norm = TwoSlopeNorm(vmin=-limits[metric], vcenter=0.0, vmax=limits[metric])
        image = None
        for col_index, baseline in enumerate(BASELINES):
            axis = fig.add_subplot(grid[row_index, col_index])
            matrix = arrays[(metric, baseline)]
            image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
            axis.set_xticks(range(len(SNR_LEVELS)), [f"{value:g}" for value in SNR_LEVELS])
            axis.set_yticks(range(len(ACCELERATIONS)), [f"{value}" for value in ACCELERATIONS])
            if row_index == 0:
                axis.set_title(f"vs {METHOD_LABELS[baseline]}")
            if row_index == 2:
                axis.set_xlabel(r"$\mathrm{SNR}_{\mathrm{nom}}$ (dB)")
            if col_index == 0:
                axis.set_ylabel(f"{metric_label}\nAcceleration $R$")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    color = "white" if abs(value) > 0.55 * limits[metric] else "black"
                    axis.text(
                        j,
                        i,
                        f"{value:+.{decimals}f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )
        color_axis = fig.add_subplot(grid[row_index, 3])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Mean paired metric differences (positive values favour Proposed)",
        fontsize=12,
    )
    png = OUTPUT_ROOT / "fig5_full_metric_heatmap.png"
    pdf = OUTPUT_ROOT / "fig5_full_metric_heatmap.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def spatial_crop(reference: np.ndarray) -> tuple[slice, slice]:
    temporal_mean = np.mean(np.abs(reference), axis=0)
    mask = temporal_mean > 0.05 * float(temporal_mean.max())
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return slice(0, reference.shape[1]), slice(0, reference.shape[2])
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    margin_y = max(6, int(0.05 * (y1 - y0)))
    margin_x = max(6, int(0.05 * (x1 - x0)))
    y0 = max(0, int(y0) - margin_y)
    y1 = min(reference.shape[1], int(y1) + margin_y)
    x0 = max(0, int(x0) - margin_x)
    x1 = min(reference.shape[2], int(x1) + margin_x)
    return slice(y0, y1), slice(x0, x1)


def generate_qualitative_figure(rows: list[dict]) -> None:
    lookup = result_lookup(rows)
    protocol = json.loads(
        (EVAL_ROOT / "config" / "protocol.json").read_text(encoding="utf-8")
    )
    sampling = protocol["sampling"]
    noise_settings = protocol["noise"]
    fig = plt.figure(figsize=(16.0, 7.7), constrained_layout=True)
    grid = GridSpec(
        4,
        8,
        figure=fig,
        width_ratios=(1, 1, 1, 1, 1, 1, 1, 0.045),
        height_ratios=(1, 0.9, 1, 0.9),
    )

    for block, (case, acceleration, snr_db, description) in enumerate(REPRESENTATIVE_CONDITIONS):
        processed = EVAL_ROOT / "data" / "processed" / "formal" / f"{case}.h5"
        reconstruction_file = (
            EVAL_ROOT
            / "outputs"
            / "formal"
            / "reconstructions"
            / case
            / f"R{acceleration}_snr{snr_db:g}.h5"
        )
        with h5py.File(processed, "r") as handle:
            reference = handle["reference"][...]
        reconstructions = {"reference": reference}
        with h5py.File(reconstruction_file, "r") as handle:
            for method in RECONSTRUCTION_METHODS:
                reconstructions[method] = handle[method][...]

        acquired = variable_density_masks(
            reference.shape[-1],
            reference.shape[0],
            acceleration,
            int(sampling["acs_lines"]),
            child_seed(
                int(sampling["mask_seed_base"]), case, acceleration, "mask"
            ),
        )
        clean = forward(reference, acquired, np)
        standard = standard_complex_noise(
            clean.shape,
            child_seed(
                int(noise_settings["noise_seed_base"]),
                case,
                acceleration,
                "noise_real",
            ),
            child_seed(
                int(noise_settings["noise_seed_base"]),
                case,
                acceleration,
                "noise_imag",
            ),
        )
        noise, _, _, _ = scale_noise_to_nominal_snr(
            clean, standard, acquired, snr_db
        )
        measured = clean + noise
        zero_filled = adjoint(measured, acquired, np)
        reconstructions["zero_filled"] = zero_filled
        zero_filled_psnr, zero_filled_ssim = magnitude_psnr_ssim(
            zero_filled, reference
        )

        y_slice, x_slice = spatial_crop(reference)
        frame = reference.shape[0] // 2
        reference_crop = np.abs(reference[frame, y_slice, x_slice])
        magnitude_vmax = float(np.percentile(np.abs(reference[:, y_slice, x_slice]), 99.5))
        errors = [
            np.abs(np.abs(reconstructions[method]) - np.abs(reference))[
                :, y_slice, x_slice
            ]
            for method in ("zero_filled",) + RECONSTRUCTION_METHODS
        ]
        error_vmax = float(np.percentile(np.concatenate([item.ravel() for item in errors]), 99.0))
        image_row = 2 * block
        error_row = image_row + 1

        mask_frame = np.tile(
            acquired[frame][None, :], (reference.shape[1], 1)
        )[y_slice, x_slice]
        for col, method in enumerate(DISPLAY_ITEMS):
            image_axis = fig.add_subplot(grid[image_row, col])
            if method == "sampling_mask":
                image_axis.imshow(
                    mask_frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
                )
            else:
                image = np.abs(reconstructions[method][frame, y_slice, x_slice])
                image_axis.imshow(image, cmap="gray", vmin=0, vmax=magnitude_vmax)
            if method == "reference":
                title = "Reference"
            elif method == "sampling_mask":
                title = "Sampling mask\n(sampled = white)"
            elif method == "zero_filled":
                title = (
                    "Zero-filled\n"
                    f"PSNR {zero_filled_psnr:.2f} dB, SSIM {zero_filled_ssim:.3f}"
                )
            else:
                result = lookup[(case, acceleration, snr_db, method)]
                title = (
                    f"{METHOD_LABELS[method]}\n"
                    f"PSNR {float(result['magnitude_psnr']):.2f} dB, "
                    f"SSIM {float(result['magnitude_ssim_roi']):.3f}"
                )
            image_axis.set_title(title, fontsize=8)
            image_axis.set_xticks([])
            image_axis.set_yticks([])
            if col == 0:
                image_axis.set_ylabel(
                    f"({chr(97 + block)}) {description}\n"
                    f"$R={acceleration}$, $\\mathrm{{SNR}}_{{\\mathrm{{nom}}}}={snr_db:g}$ dB\n"
                    f"Frame {frame + 1}",
                    fontsize=8,
                )

            error_axis = fig.add_subplot(grid[error_row, col])
            if method in {"reference", "sampling_mask"}:
                error_axis.set_xticks([])
                error_axis.set_yticks([])
                error_axis.patch.set_visible(False)
                for spine in error_axis.spines.values():
                    spine.set_visible(False)
                if col == 0:
                    error_axis.set_ylabel("Absolute error", fontsize=8)
            else:
                error = np.abs(
                    np.abs(reconstructions[method][frame, y_slice, x_slice])
                    - reference_crop
                )
                error_image = error_axis.imshow(
                    error, cmap="jet", vmin=0, vmax=error_vmax
                )
                error_axis.set_xticks([])
                error_axis.set_yticks([])

        color_axis = fig.add_subplot(grid[error_row, 7])
        colorbar = fig.colorbar(error_image, cax=color_axis)
        colorbar.set_label("|error|", fontsize=7)
        colorbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Acquisition pattern, zero-filled input, reconstructions, and absolute errors",
        fontsize=12,
    )
    png = OUTPUT_ROOT / "fig6_qualitative_temporal_comparison.png"
    pdf = OUTPUT_ROOT / "fig6_qualitative_temporal_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = load_jsonl(FINAL_RESULTS)
    generate_distribution_figure(rows)
    generate_metric_heatmap(rows)
    generate_qualitative_figure(rows)
    for stem in (
        "fig4_paired_psnr_distributions",
        "fig5_full_metric_heatmap",
        "fig6_qualitative_temporal_comparison",
    ):
        print(OUTPUT_ROOT / f"{stem}.png")
        print(OUTPUT_ROOT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
