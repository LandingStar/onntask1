# Lumerical Validation

This folder implements the `ref.py` scheme-B direction for `task1`:

- prepare an imported-source validation case from a trained run
- build a minimal Lumerical FDTD skeleton with source + detector monitor
- compare detector-plane fields against the Python reference model

## Workflow

1. Prepare a case from a trained run:

```bash
python prepare_case.py --run-dir "path/to/run_dir" --dataset-split val --sample-index 0
```

If `--run-dir` is omitted, the script now prefers the newest run under `lumerical-val/models/` and only falls back to `main/results/`.

This writes a case folder under `lumerical-val/cases/` with:

- `case_data.npz`
- `case_manifest.json`

2. Build and run the Lumerical scheme-B skeleton:

```bash
python run_fdtd_b.py --case "path/to/case_data.npz"
```

This writes:

- `scheme_b_skeleton.fsp`
- `lumerical_monitor_output.npz`

3. Compare Lumerical output with the Python reference:

```bash
python compare_lumerical_output.py ^
  --case "path/to/case_data.npz" ^
  --monitor "path/to/lumerical_monitor_output.npz"
```

## Notes

- Scheme-B is a system-level consistency workflow. It does not impose a specific in-Lumerical implementation of the trained phase layers.
- `prepare_case.py` saves the Python-side detector outputs and detector-plane intensity from the trained ONN, so later comparisons use the exact same sample.
- `run_fdtd_b.py` expects the `lumapi` Python package to be available from the configured Lumerical installation path.
- `validate_models.py` scans every run folder under `lumerical-val/models/` and executes the whole chain in batch. Use `--prepare-only` when Lumerical is not installed on the current machine.
- `run_fdtd_b.py` now clears common Qt environment variables before importing `lumapi`. Use `--qt-debug-plugins` if you need detailed Qt plugin search logs.
- `run_farfield_b.py` provides a system-level long-distance propagation approximation and is the default backend in `validate_models.py`.
