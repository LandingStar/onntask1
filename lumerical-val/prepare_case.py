from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    RunArtifacts,
    build_imported_source,
    load_run_artifacts,
    load_sample,
    make_plane_wave_sample,
    run_python_reference,
    safe_stem,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_ROOT = SCRIPT_DIR / "cases"


def build_case_name(artifacts: RunArtifacts, source_path: str, sample_index: int | None) -> str:
    run_name = artifacts.run_dir.name
    source_name = safe_stem(Path(source_path).stem)
    if sample_index is None:
        return f"{run_name}__{source_name}"
    return f"{run_name}__idx{sample_index:04d}__{source_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Lumerical validation case for task1 scheme-B workflow."
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Training result directory containing best_model.pth and config.json. Empty means latest under lumerical-val/models, then falls back to main/results.",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Optional explicit image path. If empty, read from dataset split/sample index.",
    )
    parser.add_argument(
        "--dataset-split",
        default="val",
        choices=("train", "val"),
        help="Dataset split used when --image is empty.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index within the selected dataset split.",
    )
    parser.add_argument(
        "--case-dir",
        default="",
        help="Output directory for the generated case. Empty means lumerical-val/cases/<auto_name>/.",
    )
    parser.add_argument(
        "--case-name",
        default="",
        help="Optional case folder name when --case-dir is empty.",
    )
    parser.add_argument(
        "--sample-mode",
        default="auto",
        choices=("auto", "dataset", "plane-wave"),
        help="How to construct the validation input field. 'auto' tries dataset first and falls back to plane-wave.",
    )
    return parser.parse_args()


def prepare_case(
    run_dir: str | Path | None = None,
    image_path: str | Path | None = None,
    dataset_split: str = "val",
    sample_index: int = 0,
    case_dir: str | Path | None = None,
    case_name: str = "",
    sample_mode: str = "auto",
) -> dict[str, object]:
    artifacts = load_run_artifacts(run_dir)
    if sample_mode == "plane-wave":
        sample = make_plane_wave_sample(artifacts.config)
    elif sample_mode == "dataset":
        sample = load_sample(
            artifacts.config,
            image_path=image_path,
            dataset_split=dataset_split,
            sample_index=sample_index,
        )
    else:
        try:
            sample = load_sample(
                artifacts.config,
                image_path=image_path,
                dataset_split=dataset_split,
                sample_index=sample_index,
            )
        except Exception:
            sample = make_plane_wave_sample(artifacts.config)

    sample_index_for_name = None if image_path else sample_index
    case_name = case_name.strip() or build_case_name(artifacts, sample.source_path, sample_index_for_name)
    if case_dir:
        case_dir_path = Path(case_dir).expanduser().resolve()
    else:
        case_dir_path = (DEFAULT_CASE_ROOT / safe_stem(case_name)).resolve()
    case_dir_path.mkdir(parents=True, exist_ok=True)

    pixel_size = float(artifacts.config.get("pixel_size", 8e-6))
    x, y, z, ex, ey, ez = build_imported_source(sample.amplitude, pixel_size)
    phase_masks_rad = (2.0 * np.pi * artifacts.phase_masks).astype(np.float32, copy=False)
    reference = run_python_reference(artifacts, sample.amplitude)

    case_npz_path = case_dir_path / "case_data.npz"
    np.savez_compressed(
        case_npz_path,
        amplitude_hw=sample.amplitude.astype(np.float32, copy=False),
        x=x,
        y=y,
        z=z,
        Ex=ex,
        Ey=ey,
        Ez=ez,
        phase_masks=artifacts.phase_masks.astype(np.float32, copy=False),
        phase_masks_rad=phase_masks_rad,
        detector_pos=artifacts.detector_pos.astype(np.float32, copy=False),
        detector_mask=artifacts.detector_mask.astype(np.float32, copy=False),
        detector_minus=artifacts.detector_minus.astype(np.float32, copy=False),
        reference_output_vec=reference["output_vec"].astype(np.float32, copy=False),
        reference_intensity=reference["intensity"].astype(np.float32, copy=False),
        reference_raw_detectors=reference["raw_detectors"].astype(np.float32, copy=False),
        reference_predicted_detector=reference["predicted_detector"],
    )

    manifest = {
        "scheme": "B",
        "description": "System-level consistency case for Lumerical imported-source + detector-monitor workflow.",
        "run_dir": str(artifacts.run_dir),
        "case_dir": str(case_dir_path),
        "case_data_npz": str(case_npz_path),
        "source_path": sample.source_path,
        "label": sample.label,
        "class_name": sample.class_name,
        "expected_detector_from_label": None if sample.label is None else int(sample.label // 4),
        "reference_predicted_detector": int(reference["predicted_detector"][0]),
        "config": {
            "wavelength": float(artifacts.config.get("wavelength", 532e-9)),
            "pixel_size": pixel_size,
            "img_size": artifacts.config.get("img_size", [1000, 1000]),
            "phase_mask_size": artifacts.config.get("phase_mask_size", [1200, 1200]),
            "num_layers": int(artifacts.config.get("num_layers", artifacts.phase_masks.shape[0])),
            "distance_between_layers": float(artifacts.config.get("distance_between_layers", 0.15)),
            "distance_to_detectors": float(artifacts.config.get("distance_to_detectors", 0.2)),
            "detector_shape": artifacts.config.get("detector_shape", "circle"),
            "detector_size": float(artifacts.config.get("detector_size", 60)),
            "num_classes": int(artifacts.config.get("num_classes", len(artifacts.detector_pos))),
            "label_num": int(artifacts.config.get("label_num", 20)),
        },
        "array_shapes": {
            "amplitude_hw": list(sample.amplitude.shape),
            "Ex": list(ex.shape),
            "phase_masks": list(artifacts.phase_masks.shape),
            "reference_output_vec": list(reference["output_vec"].shape),
            "reference_intensity": list(reference["intensity"].shape),
        },
        "notes": [
            "This case stores imported-source arrays and Python-side reference outputs.",
            "Scheme-B does not enforce a specific in-Lumerical realization of the phase layers.",
            "After Lumerical produces detector-plane fields, use compare_lumerical_output.py for detector integration and comparison.",
        ],
    }
    manifest["sample_mode"] = sample_mode
    manifest_path = case_dir_path / "case_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
        f.write("\n")

    result = {
        "case_dir": case_dir_path,
        "case_data_npz": case_npz_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "source_path": sample.source_path,
        "reference_predicted_detector": int(reference["predicted_detector"][0]),
    }
    return result


def main() -> None:
    args = parse_args()
    result = prepare_case(
        run_dir=args.run_dir or None,
        image_path=args.image or None,
        dataset_split=args.dataset_split,
        sample_index=args.sample_index,
        case_dir=args.case_dir or None,
        case_name=args.case_name,
        sample_mode=args.sample_mode,
    )

    print(f"Prepared case: {result['case_dir']}")
    print(f"  source image : {result['source_path']}")
    print(f"  case data    : {result['case_data_npz']}")
    print(f"  manifest     : {result['manifest_path']}")
    print(f"  ref detector : {result['reference_predicted_detector']}")


if __name__ == "__main__":
    main()
