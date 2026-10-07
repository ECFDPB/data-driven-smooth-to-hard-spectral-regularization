"""Regenerate the PSNR distribution figure in the shared blue-grey paper style."""

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


def load_rows() -> list[dict]:
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line]


def paired_values(rows: list[dict], baseline: str, acceleration: int, snr: float):
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
    return np.asarray(
        [
            float(lookup[(case, acceleration, snr, "proposed")]["magnitude_psnr"])
            - float(lookup[(case, acceleration, snr, baseline)]["magnitude_psnr"])
            for case in cases
        ],
        dtype=float,
    )


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 20_000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return np.percentile(means, (2.5, 97.5))


def calculate():
    rows = load_rows()
    conditions = [(r, snr) for r in ACCELERATIONS for snr in SNR_LEVELS]
    datasets, summaries = {}, {}
    all_values = []
    for panel, baseline in enumerate(BASELINES):
        for index, (acceleration, snr) in enumerate(conditions):
            values = paired_values(rows, baseline, acceleration, snr)
            low, high = bootstrap_interval(values, 2026100400 + 100 * panel + index)
            datasets[(panel, index)] = values
            summaries[(panel, index)] = (float(values.mean()), float(low), float(high))
            all_values.extend(values.tolist())
    lo, hi = min(all_values + [0.0]), max(all_values + [0.0])
    pad = 0.08 * (hi - lo)
    return conditions, datasets, summaries, lo - pad, hi + pad


def nice_ticks(lo: float, hi: float, count: int = 7):
    raw = (hi - lo) / count
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda x: abs(x * power - raw)) * power
    value = math.ceil(lo / step) * step
    ticks = []
    while value <= hi + step * 0.1:
        ticks.append(0.0 if abs(value) < step * 1e-8 else value)
        value += step
    return ticks


def draw_figure(width: float, height: float, text, line, rect, circle, diamond):
    conditions, datasets, summaries, lo, hi = calculate()
    left, right, top, bottom, gap = 48, 14, 85, 43, 14
    panel_w = (width - left - right - 2 * gap) / 3
    panel_h = height - top - bottom
    accent = "#3F5D73"
    points = "#8A8A8A"
    interval = "#767676"
    box_fill = "#E8EFF3"
    box_edge = "#6F7E87"
    grid = "#D5D5D5"
    shade = "#F2F2F2"

    text(width / 2, height - 16, "Paired magnitude-PSNR differences", 11, "center", "bold")
    # Compact shared legend.
    ly = height - 36
    circle(width / 2 - 175, ly + 1, 1.8, points, points)
    text(width / 2 - 168, ly - 2, "Individual series", 7.1)
    rect(width / 2 - 78, ly - 3, width / 2 - 66, ly + 5, box_fill, box_edge)
    text(width / 2 - 60, ly - 2, "Median and IQR", 7.1)
    line(width / 2 + 37, ly - 4, width / 2 + 37, ly + 6, interval, 0.8)
    diamond(width / 2 + 37, ly + 1, 2.4, accent, accent)
    text(width / 2 + 45, ly - 2, "Mean and bootstrap 95% CI", 7.1)

    ticks = nice_ticks(lo, hi)
    rng = np.random.default_rng(2026100305)
    for panel, baseline in enumerate(BASELINES):
        x0 = left + panel * (panel_w + gap)
        px0, px1 = x0 + 30, x0 + panel_w - 4
        py0, py1 = bottom, bottom + panel_h
        for start in (0, 8):
            sx0 = px0 + (start - 0.45) / 11.1 * (px1 - px0)
            sx1 = px0 + (start + 3.45) / 11.1 * (px1 - px0)
            rect(sx0, py0, sx1, py1, shade, None)
        for tick in ticks:
            yy = py0 + (tick - lo) / (hi - lo) * (py1 - py0)
            line(px0, yy, px1, yy, grid, 0.5)
            if panel == 0:
                text(px0 - 4, yy - 2.5, f"{tick:g}", 6.8, "right")
        zero_y = py0 + (0.0 - lo) / (hi - lo) * (py1 - py0)
        line(px0, zero_y, px1, zero_y, "#202020", 0.8)
        for boundary in (3.5, 7.5):
            xx = px0 + boundary / 11 * (px1 - px0)
            line(xx, py0, xx, py1, "#A8A8A8", 0.5, dash=True)

        for index, (_, snr) in enumerate(conditions):
            values = datasets[(panel, index)]
            xx = px0 + index / 11 * (px1 - px0)
            q1, median, q3 = np.percentile(values, (25, 50, 75))
            iqr = q3 - q1
            valid = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
            whisker_low, whisker_high = float(valid.min()), float(valid.max())
            yq1 = py0 + (q1 - lo) / (hi - lo) * (py1 - py0)
            yq3 = py0 + (q3 - lo) / (hi - lo) * (py1 - py0)
            ymed = py0 + (median - lo) / (hi - lo) * (py1 - py0)
            ywl = py0 + (whisker_low - lo) / (hi - lo) * (py1 - py0)
            ywh = py0 + (whisker_high - lo) / (hi - lo) * (py1 - py0)
            half = 0.22 / 11 * (px1 - px0)
            line(xx, ywl, xx, ywh, box_edge, 0.7)
            line(xx - half * 0.55, ywl, xx + half * 0.55, ywl, box_edge, 0.7)
            line(xx - half * 0.55, ywh, xx + half * 0.55, ywh, box_edge, 0.7)
            rect(xx - half, yq1, xx + half, yq3, box_fill, box_edge)
            line(xx - half, ymed, xx + half, ymed, "#303030", 0.9)
            for value in values:
                jitter = rng.uniform(-0.18, 0.18) / 11 * (px1 - px0)
                yy = py0 + (float(value) - lo) / (hi - lo) * (py1 - py0)
                circle(xx + jitter, yy, 1.05, points, points)
            mean, low_ci, high_ci = summaries[(panel, index)]
            ym = py0 + (mean - lo) / (hi - lo) * (py1 - py0)
            yl = py0 + (low_ci - lo) / (hi - lo) * (py1 - py0)
            yh = py0 + (high_ci - lo) / (hi - lo) * (py1 - py0)
            line(xx, yl, xx, yh, interval, 1.0)
            line(xx - 2.2, yl, xx + 2.2, yl, interval, 1.0)
            line(xx - 2.2, yh, xx + 2.2, yh, interval, 1.0)
            diamond(xx, ym, 2.5, accent, accent)
            text(xx, py0 - 11, f"{snr:g}", 6.7, "center")

        line(px0, py0, px1, py0, "#444444", 0.6)
        line(px0, py0, px0, py1, "#444444", 0.6)
        text((px0 + px1) / 2, py1 + 26, f"Proposed vs {BASELINE_LABELS[baseline]}", 8.6, "center", "bold")
        for center, label in zip((1.5, 5.5, 9.5), ("R=4", "R=6", "R=8")):
            xx = px0 + center / 11 * (px1 - px0)
            text(xx, py1 + 9, label, 6.9, "center")
        text((px0 + px1) / 2, 11, "Nominal SNR (dB)", 7.5, "center")
        if panel == 0:
            text(7, (py0 + py1) / 2 + 3, "Paired PSNR", 7.5, "left", "bold")
            text(7, (py0 + py1) / 2 - 8, "difference (dB)", 7.1, "left")


def make_pdf(path: Path):
    width, height = 10.7 * 72, 4.45 * 72
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
        canvas.setLineWidth(0.6)
        canvas.rect(x1, y1, x2 - x1, y2 - y1, fill=1, stroke=1 if stroke else 0)

    def circle(x, y, radius, fill, stroke):
        canvas.setFillColor(fill)
        canvas.setStrokeColor(stroke)
        canvas.circle(x, y, radius, fill=1, stroke=0)

    def diamond(x, y, radius, fill, stroke):
        path_obj = canvas.beginPath()
        path_obj.moveTo(x, y + radius)
        path_obj.lineTo(x + radius, y)
        path_obj.lineTo(x, y - radius)
        path_obj.lineTo(x - radius, y)
        path_obj.close()
        canvas.setFillColor(fill)
        canvas.setStrokeColor(stroke)
        canvas.drawPath(path_obj, fill=1, stroke=0)

    draw_figure(width, height, text, line, rect, circle, diamond)
    canvas.showPage()
    canvas.save()


def make_png(path: Path):
    scale = 3
    width, height = int(10.7 * 72 * scale), int(4.45 * 72 * scale)
    image = Image.new("RGB", (width, height), "white")
    painter = ImageDraw.Draw(image)
    regular, bold = "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"

    def font(size, weight):
        return ImageFont.truetype(bold if weight == "bold" else regular, max(8, int(size * scale)))

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
            for i in range(0, 14, 2):
                t1, t2 = i / 14, (i + 1) / 14
                painter.line((a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1, a[0] + (b[0] - a[0]) * t2, a[1] + (b[1] - a[1]) * t2), fill=colour, width=max(1, int(width * scale)))
        else:
            painter.line((*a, *b), fill=colour, width=max(1, int(width * scale)))

    def rect(x1, y1, x2, y2, fill, stroke):
        painter.rectangle((x1 * scale, height - y2 * scale, x2 * scale, height - y1 * scale), fill=fill, outline=stroke, width=max(1, int(0.6 * scale)))

    def circle(x, y, radius, fill, stroke):
        painter.ellipse(((x - radius) * scale, height - (y + radius) * scale, (x + radius) * scale, height - (y - radius) * scale), fill=fill, outline=stroke)

    def diamond(x, y, radius, fill, stroke):
        painter.polygon(((x * scale, height - (y + radius) * scale), ((x + radius) * scale, height - y * scale), (x * scale, height - (y - radius) * scale), ((x - radius) * scale, height - y * scale)), fill=fill, outline=stroke)

    draw_figure(width / scale, height / scale, text, line, rect, circle, diamond)
    image.save(path, dpi=(300, 300))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    make_pdf(OUTPUT / "fig4_paired_psnr_distributions.pdf")
    make_png(OUTPUT / "fig4_paired_psnr_distributions.png")


if __name__ == "__main__":
    main()
