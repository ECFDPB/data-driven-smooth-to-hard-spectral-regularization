"""Create a compact visual QC panel for one preprocessed combined case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = ROOT / ".runtime"
if LOCAL_RUNTIME.is_dir():
    sys.path.insert(0, str(LOCAL_RUNTIME))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def grayscale(array: np.ndarray, percentile: float = 99.5) -> Image.Image:
    values = np.asarray(array, dtype=float)
    scale = float(np.percentile(values[np.isfinite(values)], percentile))
    scaled = np.clip(values / max(scale, np.finfo(float).eps), 0.0, 1.0)
    return Image.fromarray(np.round(255.0 * scaled).astype(np.uint8), mode="L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.input, "r") as handle:
        reference = handle["reference"][0]
        rss = handle["rss_frame0"][...]
        csm = handle["csm_frame0"][...]
    combined = np.abs(reference)
    rss_scaled = rss * (np.linalg.norm(combined) / max(np.linalg.norm(rss), np.finfo(float).eps))
    difference = np.abs(combined - rss_scaled)
    csm_norm = np.sqrt(np.sum(np.abs(csm) ** 2, axis=2))
    panels = [
        ("ESPIRiT combined |X|", combined),
        ("Scaled RSS", rss_scaled),
        ("Absolute magnitude difference", difference),
        ("CSM root-sum-of-squares", csm_norm),
    ]
    images = [grayscale(array) for _, array in panels]
    width, height = images[0].size
    title_height = 28
    canvas = Image.new("RGB", (2 * width, 2 * (height + title_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, ((title, _), image) in enumerate(zip(panels, images, strict=True)):
        col, row = index % 2, index // 2
        x, y = col * width, row * (height + title_height)
        draw.text((x + 6, y + 6), title, fill="black")
        canvas.paste(image, (x, y + title_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
