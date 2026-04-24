import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


TASK1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.join(TASK1_DIR, "main")
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)

import train  # noqa: E402
from train import DNN, build_gpu_transforms, cpu_transform  # noqa: E402


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_results_dir(main_config: dict) -> str:
    results_dir_config = main_config.get("results_dir", "results")
    if os.path.isabs(results_dir_config):
        return results_dir_config
    return os.path.join(MAIN_DIR, results_dir_config)


def find_result_dir(result_dir_arg: Optional[str]) -> str:
    main_config_path = os.path.join(MAIN_DIR, "config.json")
    main_config = load_json(main_config_path) if os.path.exists(main_config_path) else {}
    results_dir = resolve_results_dir(main_config)

    if result_dir_arg:
        return result_dir_arg if os.path.isabs(result_dir_arg) else os.path.join(results_dir, result_dir_arg)

    import glob

    model_paths = glob.glob(os.path.join(results_dir, "**", "best_model.pth"), recursive=True)
    if not model_paths:
        raise FileNotFoundError(f"No run directory containing best_model.pth found under {results_dir}")

    model_paths.sort(key=os.path.getmtime, reverse=True)
    return os.path.dirname(model_paths[0])


def find_all_result_dirs() -> List[str]:
    main_config_path = os.path.join(MAIN_DIR, "config.json")
    main_config = load_json(main_config_path) if os.path.exists(main_config_path) else {}
    results_dir = resolve_results_dir(main_config)
    if not os.path.isdir(results_dir):
        return []

    timestamp_suffix = re.compile(r".*_\d{8}_\d{4}$")
    matched_dirs = []
    for entry in os.scandir(results_dir):
        if not entry.is_dir():
            continue
        if not timestamp_suffix.fullmatch(entry.name):
            continue
        if not os.path.exists(os.path.join(entry.path, "config.json")):
            continue
        if not os.path.exists(os.path.join(entry.path, "best_model.pth")):
            continue
        matched_dirs.append(entry.path)

    matched_dirs.sort()
    return matched_dirs


def resolve_dataset_path(config: dict) -> str:
    dataset_path_config = config.get("dataset_path", "./dataset")
    if os.path.isabs(dataset_path_config):
        return dataset_path_config

    current_search_dir = MAIN_DIR
    for _ in range(3):
        candidate = os.path.join(current_search_dir, dataset_path_config)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current_search_dir)
        if parent == current_search_dir:
            break
        current_search_dir = parent

    return os.path.join(MAIN_DIR, dataset_path_config)


def patch_train_module(config: dict) -> None:
    detector_pos_init_config = config.get("detector_pos", None)
    detector_pos_xy = []
    if detector_pos_init_config is not None:
        for x, y in detector_pos_init_config:
            detector_pos_xy.append((x, y))

    train.config = config
    train.detector_pos_xy = detector_pos_xy


def create_val_dataloader(config: dict, batch_size: int) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    dataset_root = resolve_dataset_path(config)
    val_root = os.path.join(dataset_root, "val")
    use_in_memory = config.get("in_memory_dataset", True)

    if use_in_memory:
        val_dataset = train.InMemoryImageFolder(val_root, transform=cpu_transform)
    else:
        val_dataset = torchvision.datasets.ImageFolder(val_root, transform=cpu_transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return val_dataset, val_loader


def select_target_indices(classes: List[str], max_samples: int) -> List[int]:
    try:
        float_classes = []
        for c in classes:
            match = re.search(r"[-+]?\d*\.\d+|\d+", c)
            if not match:
                raise ValueError(f"No numeric token in class name: {c}")
            float_classes.append(float(match.group()))

        targets = [0.1, 0.5, 0.9, 1.3, 1.7, 2.0][:max_samples]
        selected = []
        for target in targets:
            closest_idx = int(np.argmin([abs(fc - target) for fc in float_classes]))
            selected.append(closest_idx)
        return sorted(set(selected))
    except Exception:
        if not classes:
            return []
        num_to_select = min(max_samples, len(classes))
        return sorted(set(np.linspace(0, len(classes) - 1, num_to_select, dtype=int).tolist()))


def collect_plot_samples(
    val_dataset,
    val_loader: DataLoader,
    device: torch.device,
    max_samples: int,
    max_batches_to_scan: int,
) -> Tuple[List[torch.Tensor], List[int]]:
    classes = getattr(val_dataset, "classes", [str(i) for i in range(max_samples)])
    target_indices = select_target_indices(classes, max_samples)
    target_labels_set = set(target_indices)
    target_labels_found = {}

    batch_iterator = tqdm(
        val_loader,
        total=max_batches_to_scan,
        desc="Scanning validation samples",
        leave=False,
    )
    for batch_count, (images, labels) in enumerate(batch_iterator):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        for i in range(len(labels)):
            label = labels[i].item()
            if label in target_labels_set and label not in target_labels_found:
                target_labels_found[label] = images[i]

        batch_iterator.set_postfix(found=f"{len(target_labels_found)}/{len(target_labels_set) or max_samples}")
        if len(target_labels_found) == len(target_labels_set):
            break

        if batch_count + 1 >= max_batches_to_scan:
            break

    sorted_labels = sorted(target_labels_found.keys())
    plot_images = [target_labels_found[label] for label in sorted_labels]

    if not plot_images:
        data_iter = iter(val_loader)
        try:
            images, labels = next(data_iter)
            plot_images = [images[i].to(device).float() for i in range(min(max_samples, len(labels)))]
            sorted_labels = [labels[i].item() for i in range(min(max_samples, len(labels)))]
        except StopIteration:
            return [], []

    return plot_images, sorted_labels


def prepare_model_inputs(config: dict, images_to_process: torch.Tensor, device: torch.device) -> torch.Tensor:
    img_size = config.get("img_size", [1000, 1000])
    phase_mask = config.get("phase_mask_size", [1200, 1200])
    padding_x = (phase_mask[0] - img_size[0]) // 2
    padding_y = (phase_mask[1] - img_size[1]) // 2

    _, gpu_transform_val, _, _, _ = build_gpu_transforms(config, img_size, padding_x, padding_y)
    images_aug = gpu_transform_val(images_to_process.to(device).float())
    images_squeezed = images_aug.squeeze(1) if images_aug.shape[1] == 1 else images_aug

    if images_squeezed.shape[-1] == phase_mask[0]:
        return images_squeezed
    return F.pad(images_squeezed, pad=(padding_y, padding_y, padding_x, padding_x))


def normalize_intensity(intensity: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    upper = np.percentile(intensity, percentile)
    clipped = np.clip(intensity, 0, upper)
    return (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)


def get_detector_index(label: int, out_label: np.ndarray, mode: str) -> int:
    if mode == "target":
        return int(label // 4)
    if mode == "predicted":
        return int(np.argmax(out_label))
    raise ValueError(f"Unknown detector mode: {mode}")


def crop_detector_region(
    intensity_norm: np.ndarray,
    detector_pos: np.ndarray,
    detector_index: int,
    detector_size: float,
    crop_scale: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    det_pos = detector_pos[detector_index]
    cx, cy = int(det_pos[0]), int(det_pos[1])
    crop_half = max(1, int(detector_size * crop_scale))

    x0 = max(0, cx - crop_half)
    x1 = min(intensity_norm.shape[0], cx + crop_half)
    y0 = max(0, cy - crop_half)
    y1 = min(intensity_norm.shape[1], cy + crop_half)
    return intensity_norm[x0:x1, y0:y1], (x0, x1, y0, y1)


def scale_surface_height(surface_data: np.ndarray, z_scale: float) -> np.ndarray:
    return surface_data * z_scale


def generate_smoothed_surfaces(
    surface_data: np.ndarray,
    smoothing_orders: List[int],
    device: Optional[torch.device] = None,
) -> dict:
    normalized_orders = sorted(set(max(0, int(order)) for order in smoothing_orders))
    if not normalized_orders:
        normalized_orders = [0]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kernel = torch.tensor(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)

    current = torch.as_tensor(surface_data, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    results = {0: current.squeeze(0).squeeze(0).detach().cpu().numpy()}
    max_order = normalized_orders[-1]

    for order in range(1, max_order + 1):
        current = F.pad(current, (1, 1, 1, 1), mode="replicate")
        current = F.conv2d(current, kernel)
        if order in normalized_orders:
            results[order] = current.squeeze(0).squeeze(0).detach().cpu().numpy()

    return results


def build_detector_outline_xy(
    center_x: float,
    center_y: float,
    detector_size: float,
    detector_shape: str,
) -> Tuple[np.ndarray, np.ndarray]:
    half = detector_size / 2.0
    if detector_shape == "square":
        xs = np.array([center_y - half, center_y + half, center_y + half, center_y - half, center_y - half], dtype=float)
        ys = np.array([center_x - half, center_x - half, center_x + half, center_x + half, center_x - half], dtype=float)
        return xs, ys

    theta = np.linspace(0, 2 * np.pi, 80)
    xs = center_y + half * np.cos(theta)
    ys = center_x + half * np.sin(theta)
    return xs, ys


def build_detector_projection_traces(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_top: float,
    color: str,
    name: str,
    showlegend: bool,
    wall_opacity: float,
):
    if go is None:
        return []

    if len(x_coords) > 1 and np.isclose(x_coords[0], x_coords[-1]) and np.isclose(y_coords[0], y_coords[-1]):
        x_loop = x_coords[:-1]
        y_loop = y_coords[:-1]
    else:
        x_loop = x_coords
        y_loop = y_coords

    top_outline = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=np.full_like(x_coords, z_top),
        mode="lines",
        line=dict(color=color, width=7),
        name=name,
        showlegend=showlegend,
    )
    bottom_outline = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=np.zeros_like(x_coords),
        mode="lines",
        line=dict(color=color, width=4, dash="dot"),
        name=f"{name} footprint",
        showlegend=False,
    )

    n = len(x_loop)
    top_z = np.full(n, z_top, dtype=float)
    bottom_z = np.zeros(n, dtype=float)
    mesh_x = np.concatenate([x_loop, x_loop])
    mesh_y = np.concatenate([y_loop, y_loop])
    mesh_z = np.concatenate([top_z, bottom_z])

    tri_i = []
    tri_j = []
    tri_k = []
    for idx in range(n):
        nxt = (idx + 1) % n
        top_idx = idx
        top_nxt = nxt
        bottom_idx = idx + n
        bottom_nxt = nxt + n
        tri_i.extend([top_idx, top_nxt])
        tri_j.extend([bottom_idx, bottom_idx])
        tri_k.extend([top_nxt, bottom_nxt])

    wall_mesh = go.Mesh3d(
        x=mesh_x,
        y=mesh_y,
        z=mesh_z,
        i=tri_i,
        j=tri_j,
        k=tri_k,
        color=color,
        opacity=wall_opacity,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
        name=f"{name} wall",
    )

    return [wall_mesh, top_outline, bottom_outline]


def build_global_detector_traces(
    detector_pos: np.ndarray,
    detector_size: float,
    detector_shape: str,
    active_detector_index: Optional[int],
    z_level: float,
):
    if go is None:
        return []

    traces = []
    for idx, pos in enumerate(detector_pos):
        x_coords, y_coords = build_detector_outline_xy(pos[0], pos[1], detector_size, detector_shape)
        traces.extend(
            build_detector_projection_traces(
                x_coords=x_coords,
                y_coords=y_coords,
                z_top=z_level,
                color="cyan" if idx == active_detector_index else "magenta",
                name=f"Detector {idx}",
                showlegend=(idx == active_detector_index),
                wall_opacity=0.18 if idx == active_detector_index else 0.10,
            )
        )
    return traces


def build_local_detector_trace(
    detector_pos: np.ndarray,
    detector_index: int,
    detector_size: float,
    detector_shape: str,
    crop_bounds: Tuple[int, int, int, int],
    z_level: float,
):
    if go is None:
        return []

    x0, x1, y0, y1 = crop_bounds
    center_x, center_y = detector_pos[detector_index]
    local_center_x = center_x - x0
    local_center_y = center_y - y0
    x_coords, y_coords = build_detector_outline_xy(local_center_x, local_center_y, detector_size, detector_shape)

    x_coords = np.clip(x_coords, 0, max(0, y1 - y0 - 1))
    y_coords = np.clip(y_coords, 0, max(0, x1 - x0 - 1))

    return build_detector_projection_traces(
        x_coords=x_coords,
        y_coords=y_coords,
        z_top=z_level,
        color="cyan",
        name=f"Detector {detector_index}",
        showlegend=True,
        wall_opacity=0.18,
    )


def draw_detector_overlay(
    ax,
    detector_pos: np.ndarray,
    detector_size: float,
    detector_shape: str,
    active_detector_index: Optional[int] = None,
) -> None:
    for idx, pos in enumerate(detector_pos):
        x, y = pos
        is_active = idx == active_detector_index
        edgecolor = "cyan" if is_active else "magenta"
        linewidth = 2 if is_active else 1
        alpha = 0.85 if is_active else 0.5

        if detector_shape == "square":
            rect = Rectangle(
                (y - detector_size / 2 - 0.5, x - detector_size / 2 - 0.5),
                detector_size,
                detector_size,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor="none",
                alpha=alpha,
            )
            ax.add_patch(rect)
        else:
            circle = plt.Circle(
                (y, x),
                detector_size / 2,
                color=edgecolor,
                fill=False,
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.add_patch(circle)


def save_input_distribution_figure(
    save_path: str,
    input_image: np.ndarray,
    intensity_norm: np.ndarray,
    class_name: str,
    detector_index: int,
    detector_pos: np.ndarray,
    detector_size: float,
    detector_shape: str,
    crop_bounds: Tuple[int, int, int, int],
) -> None:
    x0, x1, y0, y1 = crop_bounds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(input_image, cmap="gray")
    axes[0].set_title(f"Input ({class_name})")
    axes[0].axis("off")

    axes[1].imshow(intensity_norm, cmap="hot")
    axes[1].set_title(f"Detector Plane 2D (D{detector_index})")
    draw_detector_overlay(
        axes[1],
        detector_pos=detector_pos,
        detector_size=detector_size,
        detector_shape=detector_shape,
        active_detector_index=detector_index,
    )
    axes[1].add_patch(
        Rectangle((y0, x0), y1 - y0, x1 - x0, linewidth=1.5, edgecolor="white", facecolor="none")
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_surface_figure(
    save_path: str,
    surface_data: np.ndarray,
    title: str,
    x_label: str = "Y",
    y_label: str = "X",
    z_label: str = "Norm Intensity",
    elev: float = 40.0,
    azim: float = -120.0,
    z_scale: float = 0.2,
) -> None:
    scaled_surface = scale_surface_height(surface_data, z_scale)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    yy, xx = np.mgrid[0:scaled_surface.shape[0], 0:scaled_surface.shape[1]]
    ax.plot_surface(xx, yy, scaled_surface, cmap="hot", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(f"{z_label} x {z_scale:.2f}")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_plotly_surface_html(
    save_path: str,
    surface_data: np.ndarray,
    title: str,
    x_label: str = "Y",
    y_label: str = "X",
    z_label: str = "Norm Intensity",
    detector_traces: Optional[list] = None,
    z_scale: float = 0.2,
    smoothing_orders: Optional[List[int]] = None,
) -> bool:
    if go is None:
        return False

    if smoothing_orders is None:
        smoothing_orders = [0, 5, 9, 15, 25]
    smoothing_orders = sorted(set(max(0, int(order)) for order in smoothing_orders)) or [0]
    smoothed_surfaces = generate_smoothed_surfaces(surface_data, smoothing_orders)

    traces = []
    detector_trace_count = len(detector_traces) if detector_traces else 0

    for idx, smoothing_order in enumerate(smoothing_orders):
        smoothed_surface = smoothed_surfaces[smoothing_order]
        scaled_surface = scale_surface_height(smoothed_surface, z_scale)
        traces.append(
            go.Surface(
                z=scaled_surface,
                colorscale="Hot",
                showscale=True,
                visible=(idx == 0),
                name=f"Intensity Surface (smooth={smoothing_order})",
            )
        )
    if detector_traces:
        traces.extend(detector_traces)

    fig = go.Figure(data=traces)
    buttons = []
    total_traces = len(traces)
    for idx, smoothing_order in enumerate(smoothing_orders):
        visible = [False] * total_traces
        visible[idx] = True
        for trace_idx in range(len(smoothing_orders), len(smoothing_orders) + detector_trace_count):
            visible[trace_idx] = True
        buttons.append(
            dict(
                label=f"Smooth {smoothing_order}",
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"{title} [Smooth {smoothing_order}]"},
                ],
            )
        )

    fig.update_layout(
        title=f"{title} [Smooth {smoothing_orders[0]}]",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=f"{z_label} x {z_scale:.2f}",
            aspectratio=dict(x=1, y=1, z=z_scale),
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )
        ],
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(save_path, include_plotlyjs="cdn")
    return True


def render_individual_figures(
    output_dir: str,
    intensities: np.ndarray,
    out_labels: np.ndarray,
    input_images: np.ndarray,
    labels: List[int],
    class_names: List[str],
    detector_pos: np.ndarray,
    detector_size: float,
    detector_shape: str,
    detector_mode: str,
    crop_scale: float,
    write_interactive_html: bool,
    z_scale: float,
    html_smoothing_orders: List[int],
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    interactive_count = 0

    sample_iterator = tqdm(
        list(enumerate(labels)),
        total=len(labels),
        desc="Rendering detector surfaces",
        leave=True,
    )
    for i, label in sample_iterator:
        intensity_norm = normalize_intensity(intensities[i])
        detector_index = get_detector_index(label, out_labels[i], detector_mode)
        crop_img, (x0, x1, y0, y1) = crop_detector_region(
            intensity_norm,
            detector_pos,
            detector_index,
            detector_size,
            crop_scale,
        )
        class_name = class_names[label] if label < len(class_names) else str(label)
        sample_dir = os.path.join(output_dir, f"sample_{i+1:02d}_label_{label}_detector_{detector_index}")
        os.makedirs(sample_dir, exist_ok=True)

        save_input_distribution_figure(
            save_path=os.path.join(sample_dir, "input_and_distribution.png"),
            input_image=input_images[i],
            intensity_norm=intensity_norm,
            class_name=class_name,
            detector_index=detector_index,
            detector_pos=detector_pos,
            detector_size=detector_size,
            detector_shape=detector_shape,
            crop_bounds=(x0, x1, y0, y1),
        )

        save_surface_figure(
            save_path=os.path.join(sample_dir, "surface_global.png"),
            surface_data=intensity_norm,
            title=f"Global Detector Plane 3D ({class_name}, D{detector_index})",
            z_scale=z_scale,
        )
        save_surface_figure(
            save_path=os.path.join(sample_dir, "surface_detector_local.png"),
            surface_data=crop_img,
            title=f"Local Detector Plane 3D ({class_name}, D{detector_index})",
            z_scale=z_scale,
        )

        if write_interactive_html:
            scaled_global = scale_surface_height(intensity_norm, z_scale)
            scaled_local = scale_surface_height(crop_img, z_scale)
            global_z_level = float(scaled_global.max() + max(0.01, 0.02 * z_scale))
            local_z_level = float(scaled_local.max() + max(0.01, 0.02 * z_scale))
            wrote_global = save_plotly_surface_html(
                save_path=os.path.join(sample_dir, "surface_global.html"),
                surface_data=intensity_norm,
                title=f"Global Detector Plane 3D ({class_name}, D{detector_index})",
                detector_traces=build_global_detector_traces(
                    detector_pos=detector_pos,
                    detector_size=detector_size,
                    detector_shape=detector_shape,
                    active_detector_index=detector_index,
                    z_level=global_z_level,
                ),
                z_scale=z_scale,
                smoothing_orders=html_smoothing_orders,
            )
            wrote_local = save_plotly_surface_html(
                save_path=os.path.join(sample_dir, "surface_detector_local.html"),
                surface_data=crop_img,
                title=f"Local Detector Plane 3D ({class_name}, D{detector_index})",
                detector_traces=build_local_detector_trace(
                    detector_pos=detector_pos,
                    detector_index=detector_index,
                    detector_size=detector_size,
                    detector_shape=detector_shape,
                    crop_bounds=(x0, x1, y0, y1),
                    z_level=local_z_level,
                ),
                z_scale=z_scale,
                smoothing_orders=html_smoothing_orders,
            )
            interactive_count += int(wrote_global) + int(wrote_local)
        sample_iterator.set_postfix(sample=i + 1, html=interactive_count)

    return interactive_count


def process_result_dir(result_dir: str, args) -> int:
    result_config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(result_config_path):
        raise FileNotFoundError(f"config.json not found in {result_dir}")

    config = load_json(result_config_path)
    patch_train_module(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = min(config.get("batch_size", 16), 64)
    val_dataset, val_loader = create_val_dataloader(config, batch_size)

    model = DNN(
        num_layers=config.get("num_layers", 1),
        num_classes=config.get("num_classes", 5),
        train_detector_pos=config.get("train_detector_pos", True),
    ).to(device)

    model_path = os.path.join(result_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pth not found in {result_dir}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    plot_images, plot_labels = collect_plot_samples(
        val_dataset,
        val_loader,
        device,
        max_samples=args.num_samples,
        max_batches_to_scan=args.max_batches_scan,
    )
    if not plot_images:
        raise RuntimeError("No samples found for visualization.")

    images_to_process = torch.stack(plot_images).to(device).float()
    model_inputs = prepare_model_inputs(config, images_to_process, device)

    with torch.no_grad():
        out_label, out_img, _, _ = model(model_inputs)

    input_images = images_to_process.squeeze(1).detach().cpu().numpy()
    intensities = out_img.detach().cpu().numpy()
    out_labels = out_label.detach().cpu().numpy()
    detector_pos = model.detector_pos.detach().cpu().numpy()
    detector_size = float(config.get("detector_size", 60))
    detector_shape = config.get("detector_shape", "square")
    class_names = getattr(val_dataset, "classes", [str(i) for i in range(len(plot_labels))])
    html_smoothing_orders = [
        max(0, int(token.strip()))
        for token in args.html_smoothing_orders.split(",")
        if token.strip()
    ]
    if not html_smoothing_orders:
        html_smoothing_orders = [0, 5, 9, 15, 25]

    individual_dir = os.path.join(result_dir, "detector_surface_samples")
    interactive_count = render_individual_figures(
        output_dir=individual_dir,
        intensities=intensities,
        out_labels=out_labels,
        input_images=input_images,
        labels=plot_labels,
        class_names=class_names,
        detector_pos=detector_pos,
        detector_size=detector_size,
        detector_shape=detector_shape,
        detector_mode=args.detector_mode,
        crop_scale=args.crop_scale,
        write_interactive_html=args.interactive_html,
        z_scale=args.z_scale,
        html_smoothing_orders=html_smoothing_orders,
    )

    print(f"Saved per-sample detector surface figures to: {individual_dir}")
    if args.interactive_html:
        if go is None:
            print("Interactive HTML export requested, but plotly is not installed. Install plotly for draggable 3D views.")
        else:
            print(f"Saved {interactive_count} interactive HTML surface files.")
    return interactive_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize detector-plane intensity as 3D height maps.")
    parser.add_argument("--result-dir", type=str, default=None, help="Absolute path or run folder name under results.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all result directories under results/ whose names end with a YYYYMMDD_HHMM timestamp.",
    )
    parser.add_argument("--num-samples", type=int, default=6, help="Number of validation samples to visualize.")
    parser.add_argument("--max-batches-scan", type=int, default=20, help="How many shuffled batches to scan for target labels.")
    parser.add_argument("--crop-scale", type=float, default=1.0, help="Crop half-size relative to detector_size.")
    parser.add_argument("--z-scale", type=float, default=0.2, help="Scale factor applied to 3D surface height.")
    parser.add_argument(
        "--html-smoothing-orders",
        type=str,
        default="0,5,9,15,25",
        help="Comma-separated smoothing orders to include in HTML, e.g. 0,5,9,15,25.",
    )
    parser.add_argument(
        "--detector-mode",
        type=str,
        choices=["target", "predicted"],
        default="target",
        help="Use target detector or predicted detector as the crop center.",
    )
    parser.add_argument(
        "--interactive-html",
        action="store_true",
        default=True,
        help="Also export draggable 3D HTML surfaces when plotly is available. Enabled by default.",
    )
    parser.add_argument(
        "--no-interactive-html",
        dest="interactive_html",
        action="store_false",
        help="Disable HTML export.",
    )
    args = parser.parse_args()
    if args.all:
        result_dirs = find_all_result_dirs()
        if not result_dirs:
            raise FileNotFoundError("No result directories matched the expected YYYYMMDD_HHMM suffix pattern.")
        run_iterator = tqdm(result_dirs, desc="Processing result directories", leave=True)
        for result_dir in run_iterator:
            run_iterator.set_postfix(run=os.path.basename(result_dir))
            print(f"\n=== Processing {result_dir} ===")
            process_result_dir(result_dir, args)
    else:
        result_dir = find_result_dir(args.result_dir)
        process_result_dir(result_dir, args)


if __name__ == "__main__":
    main()
