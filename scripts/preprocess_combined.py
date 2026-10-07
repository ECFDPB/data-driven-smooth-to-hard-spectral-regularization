"""Create frame-wise ESPIRiT-combined single-image OCMR cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parents[1]
LOCAL_RUNTIME = EVAL_ROOT / ".runtime"
if LOCAL_RUNTIME.is_dir():
    sys.path.insert(0, str(LOCAL_RUNTIME))
sys.path.insert(0, str(EVAL_ROOT / "src"))
sys.path.insert(0, str(EVAL_ROOT / "vendor" / "ocmr"))

import h5py  # noqa: E402
import ismrmrd  # noqa: E402
import ismrmrd.xsd  # noqa: E402
import numpy as np  # noqa: E402

from read_ocmr import read_ocmr  # noqa: E402
from single_image_eval.fft import ifft2c  # noqa: E402
from single_image_eval.preprocess import (  # noqa: E402
    center_crop_images,
    center_pad_kspace,
    combine_ocmr_coils,
    partial_fourier_geometry,
    pocs_fill_readout,
)


RAW_ROOT = EVAL_ROOT / "data" / "raw" / "ocmr"


def load_protocol() -> dict:
    return json.loads((EVAL_ROOT / "config" / "protocol.json").read_text(encoding="utf-8"))


def load_manifest(cohort: str) -> list[dict[str, str]]:
    path = (
        EVAL_ROOT / "manifests" / "pilot_cases.csv"
        if cohort == "pilot"
        else EVAL_ROOT / "manifests" / "formal_cases.csv"
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("include", "true").lower() == "true"]


def acquisition_geometry(path: Path) -> dict:
    dataset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    encoding = header.encoding[0]
    first = None
    noise_count = 0
    for index in range(dataset.number_of_acquisitions()):
        acquisition = dataset.read_acquisition(index)
        if acquisition.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            noise_count += 1
            continue
        first = acquisition
        break
    if first is None:
        raise RuntimeError(f"no imaging acquisitions in {path}")
    return {
        "encoded_x": int(encoding.encodedSpace.matrixSize.x),
        "encoded_y": int(encoding.encodingLimits.kspace_encoding_step_1.maximum + 1),
        "recon_x": int(encoding.reconSpace.matrixSize.x),
        "recon_y": int(encoding.reconSpace.matrixSize.y),
        "acquired_samples": int(first.number_of_samples),
        "center_sample": int(first.center_sample),
        "noise_acquisitions": noise_count,
    }


def choose_backend(name: str):
    if name == "numpy":
        return np, lambda value: value
    if name in {"auto", "cupy"}:
        try:
            cuda_runtime = LOCAL_RUNTIME / "nvidia" / "cuda_runtime"
            if cuda_runtime.is_dir():
                os.environ.setdefault("CUDA_PATH", str(cuda_runtime))
                binary_paths = [
                    str(path / "bin")
                    for path in (LOCAL_RUNTIME / "nvidia").iterdir()
                    if path.is_dir() and (path / "bin").is_dir()
                ]
                if binary_paths:
                    os.environ["PATH"] = os.pathsep.join(
                        binary_paths + [os.environ.get("PATH", "")]
                    )
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                os.environ.setdefault("CUPY_CACHE_DIR", str(EVAL_ROOT / "cache" / "cupy"))
                return cp, cp.asnumpy
        except Exception:
            if name == "cupy":
                raise
    return np, lambda value: value


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def preprocess_case(row: dict[str, str], cohort: str, backend: str, overwrite: bool) -> Path:
    protocol = load_protocol()
    settings = protocol["preprocessing"]
    output = EVAL_ROOT / "data" / "processed" / cohort / f"{row['case_id']}.h5"
    if output.is_file() and not overwrite:
        print(f"SKIP existing {output.relative_to(EVAL_ROOT)}", flush=True)
        return output
    raw_path = RAW_ROOT / row["relative_file"]
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    geometry_data = acquisition_geometry(raw_path)
    all_data, scan_parameters = read_ocmr(str(raw_path))
    slice_index = int(row["slice_index"])
    if slice_index >= all_data.shape[6]:
        raise IndexError(f"slice {slice_index} unavailable in {raw_path.name}")
    # read_ocmr layout -> [time,coil,readout,phase]
    kspace = all_data[:, :, 0, :, :, 0, slice_index, 0, 0]
    kspace = np.transpose(kspace, (3, 2, 0, 1)).astype(np.complex64, copy=False)
    del all_data

    geometry = partial_fourier_geometry(
        geometry_data["encoded_x"],
        geometry_data["acquired_samples"],
        geometry_data["center_sample"],
    )
    xp, to_numpy = choose_backend(backend)
    filled = pocs_fill_readout(
        xp.asarray(kspace), geometry, int(settings["pocs_readout_iterations"]), xp
    )
    filled = center_pad_kspace(
        filled, geometry_data["recon_x"], geometry_data["recon_y"], xp
    )
    images = center_crop_images(
        ifft2c(filled, xp), geometry_data["recon_x"], geometry_data["recon_y"]
    )
    coil_images = to_numpy(images).astype(np.complex64, copy=False)
    del filled, images, kspace

    print(
        f"COMBINE {row['case_id']} frames={coil_images.shape[0]} "
        f"coils={coil_images.shape[1]} matrix={coil_images.shape[2:]}",
        flush=True,
    )
    reference, first_csm, basis, retained, rms_scale, diagnostics = combine_ocmr_coils(
        coil_images,
        target_coils=int(settings["target_virtual_coils"]),
        calibration_ky=int(settings["calibration_ky"]),
        kernel_shape=tuple(settings["espirit_kernel"]),
        calibration_threshold=float(settings["calibration_singular_value_threshold"]),
        eigenvalue_threshold=float(settings["image_eigenvalue_threshold"]),
        progress=lambda completed, total, info: print(
            f"  ESPIRiT frame {completed}/{total} rank={info.calibration_rank}", flush=True
        ),
    )
    rss_first = np.sqrt(np.sum(np.abs(coil_images[0]) ** 2, axis=0)).astype(np.float32)
    metadata = {
        "case_id": row["case_id"],
        "source_file": row["relative_file"],
        "slice_index": slice_index,
        "view": row["view"],
        "field_strength_t": row["field_strength_t"],
        "frames": int(reference.shape[0]),
        "original_coils": int(coil_images.shape[1]),
        "virtual_coils": int(first_csm.shape[2]),
        "coil_energy_retained": retained,
        "combination_order": "frame-wise ESPIRiT before retrospective sampling/noise",
        "reference_rms_scale": rms_scale,
        "partial_fourier": geometry.is_partial,
        "noise_acquisitions": geometry_data["noise_acquisitions"],
        "espirit": [item.to_dict() for item in diagnostics],
        "scan_parameters": json_safe(scan_parameters),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".h5.part")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("reference", data=reference, compression="gzip", compression_opts=1)
        handle.create_dataset("csm_frame0", data=first_csm, compression="gzip", compression_opts=1)
        handle.create_dataset("rss_frame0", data=rss_first, compression="gzip", compression_opts=1)
        handle.create_dataset("coil_compression_basis", data=basis)
        handle.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
    with h5py.File(temporary, "r") as handle:
        if handle["reference"].shape != reference.shape:
            raise RuntimeError("written reference verification failed")
        json.loads(handle.attrs["metadata_json"])
    os.replace(temporary, output)
    print(f"WROTE {output.relative_to(EVAL_ROOT)}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("pilot", "formal"), required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    requested = set(args.cases or [])
    rows = [row for row in load_manifest(args.cohort) if not requested or row["case_id"] in requested]
    if requested and {row["case_id"] for row in rows} != requested:
        missing = requested - {row["case_id"] for row in rows}
        raise SystemExit(f"unknown case IDs: {sorted(missing)}")
    for row in rows:
        preprocess_case(row, args.cohort, args.backend, args.overwrite)


if __name__ == "__main__":
    main()
