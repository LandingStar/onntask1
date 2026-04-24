from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_MODELS_DIR, list_run_dirs, safe_stem
from compare_lumerical_output import compare_case_monitor
from prepare_case import prepare_case
from run_fdtd_b import DEFAULT_LUMAPI_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_ROOT = SCRIPT_DIR / "batch_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete scheme-B validation chain for all trained runs under lumerical-val/models."
    )
    parser.add_argument(
        "--models-root",
        default=str(DEFAULT_MODELS_DIR),
        help="Root directory containing model run folders.",
    )
    parser.add_argument(
        "--dataset-split",
        default="val",
        choices=("train", "val"),
        help="Dataset split used when sample-mode uses dataset samples.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index used for every run when sample-mode uses dataset samples.",
    )
    parser.add_argument(
        "--sample-mode",
        default="auto",
        choices=("auto", "dataset", "plane-wave"),
        help="How to construct each validation input field.",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Optional explicit image path applied to every run.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_BATCH_ROOT),
        help="Directory for generated cases and batch summaries.",
    )
    parser.add_argument(
        "--backend",
        default="farfield",
        choices=("farfield", "lumerical"),
        help="Validation backend. 'farfield' is the default system-level path for long-distance propagation.",
    )
    parser.add_argument(
        "--lumapi-path",
        default=str(DEFAULT_LUMAPI_PATH),
        help="Directory containing lumapi Python package.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Lumerical worker thread count.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the Lumerical GUI instead of running offscreen.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only prepare cases and save .fsp skeletons, without calling fdtd.run().",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare cases and Python reference outputs. Do not call Lumerical at all.",
    )
    parser.add_argument(
        "--qt-debug-plugins",
        action="store_true",
        help="Enable QT_DEBUG_PLUGINS in the Lumerical subprocess for plugin-load diagnostics.",
    )
    return parser.parse_args()


def _write_summary_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_dir",
        "backend",
        "status",
        "case_dir",
        "reference_predicted_detector",
        "lumerical_predicted_detector",
        "detector_argmax_match",
        "normalized_output_l2_error",
        "raw_detector_l2_error",
        "intensity_nrmse",
        "intensity_cosine_similarity",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    models_root = Path(args.models_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = list_run_dirs(models_root)
    print(f"Found {len(run_dirs)} run(s) under {models_root}")

    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        print(f"\n=== Validating {run_dir.name} ===")
        run_output_dir = output_root / safe_stem(run_dir.name)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        row: dict[str, object] = {
            "run_dir": str(run_dir),
            "backend": args.backend,
            "status": "pending",
            "case_dir": "",
            "error": "",
        }
        try:
            prepared = prepare_case(
                run_dir=run_dir,
                image_path=args.image or None,
                dataset_split=args.dataset_split,
                sample_index=args.sample_index,
                case_dir=run_output_dir / "case",
                sample_mode=args.sample_mode,
            )
            row["case_dir"] = str(prepared["case_dir"])
            row["reference_predicted_detector"] = prepared["reference_predicted_detector"]

            if args.prepare_only:
                row["status"] = "prepared_only"
                rows.append(row)
                continue

            if args.backend == "farfield":
                monitor_path = run_output_dir / "farfield_monitor_output.npz"
                backend_log_path = run_output_dir / "farfield_stdout_stderr.log"
                backend_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "run_farfield_b.py"),
                    "--case",
                    str(prepared["case_data_npz"]),
                    "--output",
                    str(monitor_path),
                ]
            else:
                monitor_path = run_output_dir / "lumerical_monitor_output.npz"
                fsp_path = run_output_dir / "scheme_b_skeleton.fsp"
                backend_log_path = run_output_dir / "fdtd_stdout_stderr.log"
                backend_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "run_fdtd_b.py"),
                    "--case",
                    str(prepared["case_data_npz"]),
                    "--output",
                    str(monitor_path),
                    "--fsp",
                    str(fsp_path),
                    "--lumapi-path",
                    args.lumapi_path,
                    "--threads",
                    str(args.threads),
                ]
                if args.show:
                    backend_cmd.append("--show")
                if args.skip_run:
                    backend_cmd.append("--skip-run")
                if args.qt_debug_plugins:
                    backend_cmd.append("--qt-debug-plugins")

            with open(backend_log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    backend_cmd,
                    cwd=SCRIPT_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            row["backend_log"] = str(backend_log_path)
            if proc.returncode != 0:
                row["status"] = "failed"
                row["error"] = (
                    f"{Path(backend_cmd[1]).name} exited with code {proc.returncode}"
                )
                rows.append(row)
                continue

            if args.backend == "lumerical" and args.skip_run:
                row["status"] = "prepared_only"
            else:
                summary = compare_case_monitor(
                    case_path=prepared["case_data_npz"],
                    monitor_path=monitor_path,
                    save_json=run_output_dir / "comparison_summary.json",
                    save_png=run_output_dir / "validation_samples.png",
                )
                row.update(
                    {
                        "status": "completed",
                        "lumerical_predicted_detector": summary["lumerical_predicted_detector"],
                        "detector_argmax_match": summary["detector_argmax_match"],
                        "normalized_output_l2_error": summary["normalized_output_l2_error"],
                        "raw_detector_l2_error": summary["raw_detector_l2_error"],
                        "intensity_nrmse": summary["intensity_nrmse"],
                        "intensity_cosine_similarity": summary["intensity_cosine_similarity"],
                    }
                )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"Failed: {row['error']}")

        rows.append(row)

    summary_json = output_root / "batch_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)
        f.write("\n")

    summary_csv = output_root / "batch_summary.csv"
    _write_summary_csv(summary_csv, rows)

    print(f"\nSaved batch summary JSON: {summary_json}")
    print(f"Saved batch summary CSV : {summary_csv}")


if __name__ == "__main__":
    main()
