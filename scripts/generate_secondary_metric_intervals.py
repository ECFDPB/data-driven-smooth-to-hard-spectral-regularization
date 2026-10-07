"""Generate a restrained paper-style interval plot for NMSE and SSIM."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parents[1]
LOCAL_RUNTIME = EVAL_ROOT / ".runtime"
if LOCAL_RUNTIME.is_dir():
    sys.path.insert(0, str(LOCAL_RUNTIME))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from reportlab.pdfgen import canvas as pdfcanvas  # noqa: E402

RESULTS = EVAL_ROOT / "outputs" / "formal" / "final_results.jsonl"
OUTPUT = EVAL_ROOT / "outputs" / "formal" / "paper_summary"
ACCELERATIONS = (4, 6, 8)
SNR_LEVELS = (0.0, 10.0, 20.0, 30.0)
BASELINES = ("hard", "soft", "l_plus_s")
BASELINE_LABELS = {"hard": "Hard", "soft": "Soft", "l_plus_s": "L+S"}
METRICS = (
    ("complex_nmse", "Complex NMSE", "baseline - Proposed"),
    ("magnitude_ssim_roi", "Magnitude SSIM", "Proposed - baseline"),
)


def load_rows() -> list[dict]:
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line]


def paired_values(rows: list[dict], baseline: str, acceleration: int, snr: float, metric: str):
    lookup = {
        (row["case"], int(row["R"]), float(row["snr_db"]), row["method"]): row
        for row in rows
    }
    cases = sorted(
        row["case"]
        for row in rows
        if row["method"] == "proposed"
        and int(row["R"]) == acceleration
        and float(row["snr_db"]) == snr
    )
    values = []
    for case in cases:
        proposed = float(lookup[(case, acceleration, snr, "proposed")][metric])
        comparator = float(lookup[(case, acceleration, snr, baseline)][metric])
        values.append(comparator - proposed if metric == "complex_nmse" else proposed - comparator)
    return np.asarray(values, dtype=float)


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 20_000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return np.percentile(means, (2.5, 97.5))


def calculate():
    rows = load_rows()
    conditions = [(r, snr) for r in ACCELERATIONS for snr in SNR_LEVELS]
    result = {}
    for row_index, (metric, _, _) in enumerate(METRICS):
        for column_index, baseline in enumerate(BASELINES):
            points = []
            for condition_index, (acceleration, snr) in enumerate(conditions):
                values = paired_values(rows, baseline, acceleration, snr, metric)
                low, high = bootstrap_interval(
                    values,
                    seed=2026101000 + 100 * row_index + 20 * column_index + condition_index,
                )
                points.append((float(values.mean()), float(low), float(high)))
            result[(row_index, column_index)] = points
    return conditions, result


def ranges(result):
    output = []
    for row_index in range(2):
        vals = [v for col in range(3) for point in result[(row_index, col)] for v in point]
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = max((hi - lo) * 0.12, 1e-4)
        output.append((lo - pad, hi + pad))
    return output


def nice_ticks(lo: float, hi: float, count: int = 5):
    span = max(hi - lo, 1e-9)
    raw = span / count
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda x: abs(x * power - raw)) * power
    value = math.ceil(lo / step) * step
    ticks = []
    while value <= hi + step * 0.1:
        ticks.append(0.0 if abs(value) < step * 1e-8 else value)
        value += step
    return ticks


def draw_figure(width: float, height: float, text, line, rect, circle):
    conditions, data = calculate()
    row_ranges = ranges(data)
    margin_left, margin_right = 92, 18
    margin_top, margin_bottom = 50, 46
    gap_x, gap_y = 15, 30
    panel_w = (width - margin_left - margin_right - 2 * gap_x) / 3
    panel_h = (height - margin_top - margin_bottom - gap_y) / 2
    accent, grey, light, grid = "#3F5D73", "#767676", "#F2F2F2", "#D5D5D5"
    text(width / 2, height - 17, "Secondary fidelity measures: paired mean differences and 95% bootstrap intervals", 11, "center", "bold")

    for row_index, (_, ylabel, orientation) in enumerate(METRICS):
        y0 = height - margin_top - (row_index + 1) * panel_h - row_index * gap_y
        lo, hi = row_ranges[row_index]
        ticks = nice_ticks(lo, hi)
        for column_index, baseline in enumerate(BASELINES):
            x0 = margin_left + column_index * (panel_w + gap_x)
            px0, px1 = x0 + 34, x0 + panel_w - 5
            py0, py1 = y0 + 25, y0 + panel_h - 18
            for start in (0, 8):
                left = px0 + (start - 0.45) / 11.1 * (px1 - px0)
                right = px0 + (start + 3.45) / 11.1 * (px1 - px0)
                rect(left, py0, right, py1, light, None)
            for tick in ticks:
                yy = py0 + (tick - lo) / (hi - lo) * (py1 - py0)
                line(px0, yy, px1, yy, grid, 0.5)
                if column_index == 0:
                    text(px0 - 4, yy - 2.5, f"{tick:.3g}", 6.8, "right")
            zero_y = py0 + (0 - lo) / (hi - lo) * (py1 - py0)
            line(px0, zero_y, px1, zero_y, "#202020", 0.8)
            for boundary in (3.5, 7.5):
                xx = px0 + boundary / 11 * (px1 - px0)
                line(xx, py0, xx, py1, "#A8A8A8", 0.5, dash=True)
            for index, (mean, low, high) in enumerate(data[(row_index, column_index)]):
                xx = px0 + index / 11 * (px1 - px0)
                ym = py0 + (mean - lo) / (hi - lo) * (py1 - py0)
                yl = py0 + (low - lo) / (hi - lo) * (py1 - py0)
                yh = py0 + (high - lo) / (hi - lo) * (py1 - py0)
                line(xx, yl, xx, yh, grey, 0.8)
                line(xx - 2, yl, xx + 2, yl, grey, 0.8)
                line(xx - 2, yh, xx + 2, yh, grey, 0.8)
                circle(xx, ym, 2.15, accent, "#FFFFFF")
            line(px0, py0, px1, py0, "#444444", 0.6)
            line(px0, py0, px0, py1, "#444444", 0.6)
            text((px0 + px1) / 2, py1 + 21, f"Proposed vs {BASELINE_LABELS[baseline]}", 8.4, "center", "bold")
            for index, (_, snr) in enumerate(conditions):
                xx = px0 + index / 11 * (px1 - px0)
                text(xx, py0 - 10, f"{snr:g}", 6.5, "center")
            if row_index == 1:
                text((px0 + px1) / 2, y0 + 3, "Nominal SNR (dB)", 7.4, "center")
            for center, label in zip((1.5, 5.5, 9.5), ("R=4", "R=6", "R=8")):
                xx = px0 + center / 11 * (px1 - px0)
                text(xx, py1 + 7, label, 6.8, "center")
        text(13, y0 + panel_h / 2 + 7, ylabel, 7.8, "left", "bold")
        text(13, y0 + panel_h / 2 - 4, f"({orientation})", 7.0, "left")


def make_pdf(path: Path):
    width, height = 10.7 * 72, 5.8 * 72
    canvas = pdfcanvas.Canvas(str(path), pagesize=(width, height))

    def text(x, y, value, size, align="left", weight="normal"):
        canvas.setFont("Helvetica-Bold" if weight == "bold" else "Helvetica", size)
        if align == "center":
            canvas.drawCentredString(x, y, value)
        elif align == "right":
            canvas.drawRightString(x, y, value)
        else:
            canvas.drawString(x, y, value)

    def line(x1, y1, x2, y2, colour, width, dash=False):
        canvas.setStrokeColor(colour)
        canvas.setLineWidth(width)
        canvas.setDash(2, 2) if dash else canvas.setDash()
        canvas.line(x1, y1, x2, y2)

    def rect(x1, y1, x2, y2, fill, stroke):
        canvas.setFillColor(fill)
        canvas.setStrokeColor(stroke or fill)
        canvas.rect(x1, y1, x2 - x1, y2 - y1, fill=1, stroke=1 if stroke else 0)

    def circle(x, y, radius, fill, stroke):
        canvas.setFillColor(fill)
        canvas.setStrokeColor(stroke)
        canvas.setLineWidth(0.4)
        canvas.circle(x, y, radius, fill=1, stroke=1)

    draw_figure(width, height, text, line, rect, circle)
    canvas.showPage()
    canvas.save()


def make_png(path: Path):
    scale = 3
    width, height = int(10.7 * 72 * scale), int(5.8 * 72 * scale)
    image = Image.new("RGB", (width, height), "white")
    painter = ImageDraw.Draw(image)
    font_regular = "C:/Windows/Fonts/arial.ttf"
    font_bold = "C:/Windows/Fonts/arialbd.ttf"

    def font(size, weight):
        return ImageFont.truetype(font_bold if weight == "bold" else font_regular, max(8, int(size * scale)))

    def text(x, y, value, size, align="left", weight="normal"):
        fnt = font(size, weight)
        box = painter.textbbox((0, 0), value, font=fnt)
        tw = box[2] - box[0]
        if align == "center":
            x -= tw / (2 * scale)
        elif align == "right":
            x -= tw / scale
        painter.text((x * scale, height - y * scale), value, fill="#1F1F1F", font=fnt, anchor="ls")

    def line(x1, y1, x2, y2, colour, width, dash=False):
        a = (x1 * scale, height - y1 * scale)
        b = (x2 * scale, height - y2 * scale)
        if dash:
            segments = 14
            for i in range(0, segments, 2):
                t1, t2 = i / segments, min((i + 1) / segments, 1)
                painter.line((a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1, a[0] + (b[0] - a[0]) * t2, a[1] + (b[1] - a[1]) * t2), fill=colour, width=max(1, int(width * scale)))
        else:
            painter.line((*a, *b), fill=colour, width=max(1, int(width * scale)))

    def rect(x1, y1, x2, y2, fill, stroke):
        painter.rectangle((x1 * scale, height - y2 * scale, x2 * scale, height - y1 * scale), fill=fill, outline=stroke)

    def circle(x, y, radius, fill, stroke):
        painter.ellipse(((x - radius) * scale, height - (y + radius) * scale, (x + radius) * scale, height - (y - radius) * scale), fill=fill, outline=stroke, width=1)

    draw_figure(width / scale, height / scale, text, line, rect, circle)
    image.save(path, dpi=(300, 300))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    make_pdf(OUTPUT / "fig5_secondary_metric_intervals.pdf")
    make_png(OUTPUT / "fig5_secondary_metric_intervals.png")


if __name__ == "__main__":
    main()
