from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


TASK_DIR = Path(__file__).resolve().parents[1]
MAIN_DIR = TASK_DIR / "main"
LUMERICAL_VAL_DIR = TASK_DIR / "lumerical-val"
DEFAULT_MODELS_DIR = LUMERICAL_VAL_DIR / "models"
DEFAULT_RESULTS_DIR = MAIN_DIR / "results"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SampleInfo:
    amplitude: np.ndarray
    source_path: str
    label: int | None
    class_name: str | None


@dataclass
class RunArtifacts:
    run_dir: Path
    config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    phase_masks: np.ndarray
    detector_pos: np.ndarray
    detector_mask: np.ndarray
    detector_minus: np.ndarray


def _to_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    return Path(path_like).expanduser().resolve()


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module."):]: value for key, value in state_dict.items()}


def load_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for candidate_key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(candidate_key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint structure: {type(checkpoint)!r}")

    return _strip_module_prefix(checkpoint)


def _extract_phase_masks(state_dict: dict[str, Any]) -> np.ndarray:
    explicit = state_dict.get("phase_mask")
    if torch.is_tensor(explicit):
        phase_masks = explicit.detach().cpu().numpy()
        if phase_masks.ndim == 2:
            phase_masks = phase_masks[None, ...]
        return phase_masks.astype(np.float32, copy=False)

    if isinstance(explicit, (list, tuple)) and explicit:
        stacked = []
        for item in explicit:
            if not torch.is_tensor(item):
                raise TypeError("phase_mask list contains non-tensor entries")
            stacked.append(item.detach().cpu().numpy())
        return np.stack(stacked, axis=0).astype(np.float32, copy=False)

    pattern = re.compile(r"^phase_mask\.(\d+)$")
    indexed: list[tuple[int, np.ndarray]] = []
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match and torch.is_tensor(value):
            indexed.append((int(match.group(1)), value.detach().cpu().numpy()))

    if not indexed:
        raise KeyError("Could not find phase mask tensors in checkpoint")

    indexed.sort(key=lambda item: item[0])
    return np.stack([value for _, value in indexed], axis=0).astype(np.float32, copy=False)


def _extract_tensor_1d_or_2d(
    state_dict: dict[str, Any],
    key: str,
    default_shape: tuple[int, ...],
    default_value: float,
) -> np.ndarray:
    value = state_dict.get(key)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)

    return np.full(default_shape, default_value, dtype=np.float32)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_run_dir(run_dir: str | Path | None = None) -> Path:
    explicit = _to_path(run_dir)
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Run directory not found: {explicit}")
        return explicit

    search_roots = [DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(
            path.parent
            for path in root.rglob("best_model.pth")
            if (path.parent / "config.json").exists()
        )
    if not candidates:
        raise FileNotFoundError(
            "No run directories with best_model.pth found under "
            f"{DEFAULT_MODELS_DIR} or {DEFAULT_RESULTS_DIR}"
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def load_run_artifacts(run_dir: str | Path | None = None) -> RunArtifacts:
    resolved_run_dir = find_run_dir(run_dir)
    config = load_config(resolved_run_dir / "config.json")
    state_dict = load_state_dict(resolved_run_dir / "best_model.pth")
    phase_masks = _extract_phase_masks(state_dict)

    num_classes = int(config.get("num_classes", phase_masks.shape[0]))
    detector_pos = _extract_tensor_1d_or_2d(
        state_dict,
        "detector_pos",
        (num_classes, 2),
        0.0,
    )
    if not detector_pos.any():
        detector_pos = np.asarray(config.get("detector_pos", []), dtype=np.float32)
    detector_mask = _extract_tensor_1d_or_2d(state_dict, "detector_mask", (num_classes,), 1.0)
    detector_minus = _extract_tensor_1d_or_2d(state_dict, "detector_minus", (num_classes,), 0.0)

    return RunArtifacts(
        run_dir=resolved_run_dir,
        config=config,
        state_dict=state_dict,
        phase_masks=phase_masks,
        detector_pos=detector_pos,
        detector_mask=detector_mask,
        detector_minus=detector_minus,
    )


def list_run_dirs(models_root: str | Path | None = None) -> list[Path]:
    root = _to_path(models_root) or DEFAULT_MODELS_DIR
    if not root.exists():
        raise FileNotFoundError(f"Models directory not found: {root}")

    run_dirs = sorted(
        {
            path.parent
            for path in root.rglob("best_model.pth")
            if (path.parent / "config.json").exists()
        }
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with best_model.pth found under {root}")
    return run_dirs


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    dataset_path_config = str(config.get("dataset_path", "./dataset"))
    dataset_path = Path(dataset_path_config)
    if dataset_path.is_absolute():
        return dataset_path

    current_search_dir = MAIN_DIR
    for _ in range(3):
        possible_path = (current_search_dir / dataset_path).resolve()
        if possible_path.exists():
            return possible_path
        parent = current_search_dir.parent
        if parent == current_search_dir:
            break
        current_search_dir = parent

    return (MAIN_DIR / dataset_path).resolve()


def _scan_imagefolder(split_dir: Path) -> list[tuple[Path, int, str]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    classes = sorted([path for path in split_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    samples: list[tuple[Path, int, str]] = []
    for label, class_dir in enumerate(classes):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTS:
                samples.append((image_path, label, class_dir.name))

    if not samples:
        raise FileNotFoundError(f"No image samples found under {split_dir}")
    return samples


def _resize_to_img_size(image: Image.Image, img_size: list[int] | tuple[int, int]) -> Image.Image:
    height, width = int(img_size[0]), int(img_size[1])
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _pad_to_phase_mask(amplitude: np.ndarray, phase_mask_size: list[int] | tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(phase_mask_size[0]), int(phase_mask_size[1])
    src_h, src_w = amplitude.shape
    if src_h > target_h or src_w > target_w:
        raise ValueError(
            f"Input image size {amplitude.shape} exceeds phase mask size {(target_h, target_w)}"
        )

    pad_x = (target_h - src_h) // 2
    pad_y = (target_w - src_w) // 2
    return np.pad(
        amplitude,
        ((pad_x, target_h - src_h - pad_x), (pad_y, target_w - src_w - pad_y)),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32, copy=False)


def image_to_amplitude(image_path: str | Path, config: dict[str, Any]) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    image = _resize_to_img_size(image, config.get("img_size", [1000, 1000]))
    amplitude = np.asarray(image, dtype=np.float32) / 255.0
    return _pad_to_phase_mask(amplitude, config.get("phase_mask_size", [1200, 1200]))


def make_plane_wave_sample(config: dict[str, Any]) -> SampleInfo:
    phase_mask_size = config.get("phase_mask_size", [1200, 1200])
    amplitude = np.ones((int(phase_mask_size[0]), int(phase_mask_size[1])), dtype=np.float32)
    return SampleInfo(
        amplitude=amplitude,
        source_path="<plane_wave>",
        label=None,
        class_name=None,
    )


def load_sample(
    config: dict[str, Any],
    image_path: str | Path | None = None,
    dataset_split: str = "val",
    sample_index: int = 0,
) -> SampleInfo:
    if image_path is not None:
        resolved_image_path = _to_path(image_path)
        if resolved_image_path is None or not resolved_image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        amplitude = image_to_amplitude(resolved_image_path, config)
        return SampleInfo(
            amplitude=amplitude,
            source_path=str(resolved_image_path),
            label=None,
            class_name=None,
        )

    dataset_root = resolve_dataset_root(config)
    split_dir = dataset_root / dataset_split
    samples = _scan_imagefolder(split_dir)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"Sample index {sample_index} out of range for {split_dir} ({len(samples)} samples)")

    sample_path, label, class_name = samples[sample_index]
    amplitude = image_to_amplitude(sample_path, config)
    return SampleInfo(
        amplitude=amplitude,
        source_path=str(sample_path),
        label=label,
        class_name=class_name,
    )


def build_imported_source(amplitude_hw: np.ndarray, pixel_size: float) -> tuple[np.ndarray, ...]:
    height, width = amplitude_hw.shape
    x = (np.arange(width, dtype=np.float64) - width / 2.0) * pixel_size
    y = (np.arange(height, dtype=np.float64) - height / 2.0) * pixel_size
    z = np.array([0.0], dtype=np.float64)

    ex = amplitude_hw.T[:, :, None].astype(np.complex128)
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)
    return x, y, z, ex, ey, ez


def coerce_field_to_hw(field: np.ndarray, expected_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(field)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D field after squeeze, got shape {arr.shape}")

    if arr.shape == expected_hw:
        return arr
    if arr.shape == (expected_hw[1], expected_hw[0]):
        return arr.T
    raise ValueError(f"Field shape {arr.shape} does not match expected {expected_hw} or its transpose")


def intensity_from_components(
    expected_hw: tuple[int, int],
    ex: np.ndarray | None,
    ey: np.ndarray | None = None,
    ez: np.ndarray | None = None,
) -> np.ndarray:
    intensity = np.zeros(expected_hw, dtype=np.float64)
    for component in (ex, ey, ez):
        if component is None:
            continue
        plane = coerce_field_to_hw(component, expected_hw)
        intensity += np.abs(plane) ** 2
    return intensity


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def detector_readout(
    intensity_hw: np.ndarray,
    detector_pos: np.ndarray,
    detector_size: float,
    detector_shape: str,
    detector_mask: np.ndarray | None = None,
    detector_minus: np.ndarray | None = None,
) -> dict[str, Any]:
    height, width = intensity_hw.shape
    raw_vals: list[float] = []

    for x_center, y_center in detector_pos:
        if detector_shape == "square":
            x0 = max(int(np.floor(x_center - detector_size / 2.0)), 0)
            x1 = min(int(np.floor(x_center + detector_size / 2.0)), height)
            y0 = max(int(np.floor(y_center - detector_size / 2.0)), 0)
            y1 = min(int(np.floor(y_center + detector_size / 2.0)), width)
            raw_val = float(intensity_hw[x0:x1, y0:y1].sum())
        else:
            crop_s = int(detector_size) + 4
            cx = int(np.floor(x_center))
            cy = int(np.floor(y_center))
            x0 = max(0, cx - crop_s)
            x1 = min(height, cx + crop_s)
            y0 = max(0, cy - crop_s)
            y1 = min(width, cy + crop_s)

            grid_x, grid_y = np.meshgrid(
                np.arange(x0, x1, dtype=np.float64),
                np.arange(y0, y1, dtype=np.float64),
                indexing="ij",
            )
            dist_sq = (grid_x - x_center) ** 2 + (grid_y - y_center) ** 2
            radius_sq = (detector_size / 2.0) ** 2
            soft_mask = _sigmoid(radius_sq - dist_sq)
            raw_val = float((intensity_hw[x0:x1, y0:y1] * soft_mask).sum())

        raw_vals.append(raw_val)

    raw = np.asarray(raw_vals, dtype=np.float64)
    scaled = raw.copy()
    if detector_mask is not None:
        scaled = scaled * np.asarray(detector_mask, dtype=np.float64)
    if detector_minus is not None:
        scaled = scaled - np.asarray(detector_minus, dtype=np.float64)

    normalized = scaled / (scaled.sum() + 1e-8)
    return {
        "raw": raw,
        "scaled": scaled,
        "normalized": normalized,
        "predicted_detector": int(np.argmax(normalized)),
    }


def load_train_module():
    train_path = MAIN_DIR / "train.py"
    module_name = "task1_main_train_lumerical_val"
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load train module from {train_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_python_reference(artifacts: RunArtifacts, amplitude_hw: np.ndarray) -> dict[str, np.ndarray]:
    train_module = load_train_module()
    config = dict(artifacts.config)

    train_module.config = config
    train_module.device = torch.device("cpu")
    train_module.IMG_SIZE = config.get("img_size", [1000, 1000])
    train_module.PhaseMask = config.get("phase_mask_size", [1200, 1200])
    train_module.PIXEL_SIZE = float(config.get("pixel_size", 8e-6))
    train_module.wl = float(config.get("wavelength", 532e-9))
    train_module.PADDINGx = 0
    train_module.PADDINGy = 0
    train_module.detector_pos_xy = [tuple(map(float, pos)) for pos in artifacts.detector_pos.tolist()]

    model = train_module.DNN(
        num_layers=int(config.get("num_layers", artifacts.phase_masks.shape[0])),
        wl_param=float(config.get("wavelength", 532e-9)),
        PhaseMask_param=config.get("phase_mask_size", [1200, 1200]),
        pixel_size_param=float(config.get("pixel_size", 8e-6)),
        distance_between_layers=float(config.get("distance_between_layers", 0.15)),
        distance_to_detectors=float(config.get("distance_to_detectors", 0.2)),
        num_classes=int(config.get("num_classes", len(artifacts.detector_pos))),
        train_detector_pos=bool(config.get("train_detector_pos", False)),
    )
    model.load_state_dict(artifacts.state_dict, strict=True)
    model.eval()

    input_tensor = torch.from_numpy(amplitude_hw.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.no_grad():
        output_vec, intensity, _, raw_detectors = model(input_tensor)

    return {
        "output_vec": output_vec.squeeze(0).detach().cpu().numpy(),
        "intensity": intensity.squeeze(0).detach().cpu().numpy(),
        "raw_detectors": raw_detectors.squeeze(0).detach().cpu().numpy(),
        "predicted_detector": np.asarray([int(output_vec.argmax(dim=1).item())], dtype=np.int64),
    }


def run_python_exit_field(artifacts: RunArtifacts, amplitude_hw: np.ndarray) -> dict[str, np.ndarray]:
    train_module = load_train_module()
    config = dict(artifacts.config)

    train_module.config = config
    train_module.device = torch.device("cpu")
    train_module.IMG_SIZE = config.get("img_size", [1000, 1000])
    train_module.PhaseMask = config.get("phase_mask_size", [1200, 1200])
    train_module.PIXEL_SIZE = float(config.get("pixel_size", 8e-6))
    train_module.wl = float(config.get("wavelength", 532e-9))
    train_module.PADDINGx = 0
    train_module.PADDINGy = 0
    train_module.detector_pos_xy = [tuple(map(float, pos)) for pos in artifacts.detector_pos.tolist()]

    model = train_module.DNN(
        num_layers=int(config.get("num_layers", artifacts.phase_masks.shape[0])),
        wl_param=float(config.get("wavelength", 532e-9)),
        PhaseMask_param=config.get("phase_mask_size", [1200, 1200]),
        pixel_size_param=float(config.get("pixel_size", 8e-6)),
        distance_between_layers=float(config.get("distance_between_layers", 0.15)),
        distance_to_detectors=float(config.get("distance_to_detectors", 0.2)),
        num_classes=int(config.get("num_classes", len(artifacts.detector_pos))),
        train_detector_pos=bool(config.get("train_detector_pos", False)),
    )
    model.load_state_dict(artifacts.state_dict, strict=True)
    model.eval()

    input_tensor = torch.from_numpy(amplitude_hw.astype(np.float32, copy=False)).unsqueeze(0).to(torch.cfloat)
    with torch.no_grad():
        exit_field = input_tensor
        for index, layer in enumerate(model.diffractive_layers):
            propagated = layer(exit_field)
            phase_values = 2 * torch.pi * model.phase_mask[index]
            modulation = torch.exp(1j * phase_values)
            exit_field = propagated * modulation

    exit_field_np = exit_field.squeeze(0).detach().cpu().numpy()
    return {
        "exit_field": exit_field_np,
        "exit_intensity": (np.abs(exit_field_np) ** 2).astype(np.float32, copy=False),
    }


def safe_stem(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._") or "case"
