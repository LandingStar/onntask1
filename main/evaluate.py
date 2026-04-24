
import sys
import os
import json
import glob

# 1. Numpy & Matplotlib first
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 2. Torch & Torchvision
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as v2

# 3. Others

# 4. Local imports
from train import DNN, cpu_transform


def _discover_model_paths(run_dir):
    model_paths = sorted(glob.glob(os.path.join(run_dir, "*.pth")))
    if not model_paths:
        return []

    def sort_key(path):
        basename = os.path.basename(path)
        return (0 if basename == "best_model.pth" else 1, basename)

    return sorted(model_paths, key=sort_key)


def _checkpoint_output_dir(run_dir, model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(run_dir, "eval_by_checkpoint", model_name)


def _save_sample_visualizations(
    model,
    config,
    device,
    val_dataset,
    val_dataloader,
    classes,
    PhaseMask,
    PADDINGx,
    PADDINGy,
    output_dir,
):
    print(f"Generating sample visualizations -> {output_dir}")

    target_labels_found = {}
    target_indices = []

    try:
        import re
        float_classes = []
        for c in classes:
            match = re.search(r"[-+]?\d*\.\d+|\d+", c)
            if match:
                float_classes.append(float(match.group()))
            else:
                raise ValueError(f"No number in {c}")
        is_numeric = True
    except Exception:
        is_numeric = False

    if is_numeric and len(classes) >= 6:
        targets = [0.1, 0.5, 0.9, 1.3, 1.7, 2.0]
        for t in targets:
            closest_idx = np.argmin([abs(fc - t) for fc in float_classes])
            target_indices.append(closest_idx)
        target_indices = sorted(list(set(target_indices)))
    else:
        num_to_select = min(6, len(classes))
        if num_to_select > 0:
            target_indices = np.linspace(0, len(classes) - 1, num_to_select, dtype=int).tolist()
            target_indices = sorted(list(set(target_indices)))

    target_labels_set = set(target_indices)
    max_batches_to_scan = 20
    batch_count = 0

    for images, labels in val_dataloader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        for i in range(len(labels)):
            lbl = labels[i].item()
            if lbl in target_labels_set and lbl not in target_labels_found:
                target_labels_found[lbl] = images[i]

        if len(target_labels_found) == len(target_labels_set):
            break

        batch_count += 1
        if batch_count >= max_batches_to_scan:
            break

    sorted_found_labels = sorted(target_labels_found.keys())
    plot_images = []
    plot_labels = []

    for lbl in sorted_found_labels:
        plot_images.append(target_labels_found[lbl])
        plot_labels.append(lbl)

    if not plot_images:
        print("Warning: No target samples found for visualization.")
        data_iter = iter(val_dataloader)
        try:
            images, labels = next(data_iter)
            plot_images = [images[i] for i in range(min(6, len(labels)))]
            plot_labels = [labels[i].item() for i in range(min(6, len(labels)))]
        except StopIteration:
            pass

    num_samples = len(plot_images)
    if num_samples == 0:
        print("No samples to plot.")
        return

    images_to_process = torch.stack(plot_images).to(device).float()

    from train import gpu_transform_val
    images_aug = gpu_transform_val(images_to_process)

    if images_aug.shape[1] == 1:
        images_squeezed = images_aug.squeeze(1)
    else:
        images_squeezed = images_aug

    if images_squeezed.shape[-1] == PhaseMask[0]:
        images_input = images_squeezed
    else:
        images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))

    with torch.no_grad():
        out_label, out_img, _, _ = model(images_input)

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples), squeeze=False)
    detector_pos = model.detector_pos

    for i in range(num_samples):
        ax_in = axes[i, 0]
        ax_in.imshow(images_squeezed[i].cpu().numpy(), cmap='gray')
        try:
            class_name = val_dataset.classes[plot_labels[i]]
        except Exception:
            class_name = str(plot_labels[i])
        ax_in.set_title(f"Input (Label: {class_name})")
        ax_in.axis('off')

        ax_out = axes[i, 1]
        intensity = out_img[i].cpu().numpy()
        p99 = np.percentile(intensity, 99.9)
        intensity_norm = np.clip(intensity, 0, p99)
        intensity_norm = (intensity_norm - intensity_norm.min()) / (intensity_norm.max() - intensity_norm.min() + 1e-8)
        ax_out.imshow(intensity_norm, cmap='hot')
        ax_out.set_title("Output Intensity")

        for pos in detector_pos:
            x, y = pos.detach().cpu().numpy()
            det_shape = config.get('detector_shape', 'circle')
            det_size = config.get('detector_size', 60)

            if det_shape == 'square':
                from matplotlib.patches import Rectangle
                rect_det = Rectangle(
                    (y - det_size / 2 - 0.5, x - det_size / 2 - 0.5),
                    det_size,
                    det_size,
                    linewidth=2,
                    edgecolor='magenta',
                    facecolor='none',
                    alpha=0.5,
                )
                ax_out.add_patch(rect_det)
            else:
                circle = plt.Circle((y, x), det_size / 2, color='magenta', fill=False, linewidth=2, alpha=0.5)
                ax_out.add_patch(circle)

        ax_out.axis('off')

        max_det_idx = out_label[i].argmax().item()
        det_pos = detector_pos[max_det_idx].detach().cpu().numpy()
        dx, dy = int(det_pos[0]), int(det_pos[1])

        det_shape = config.get('detector_shape', 'circle')
        det_size = config.get('detector_size', 60)
        crop_size = det_size

        dy_center = dy
        x0, x1 = max(0, dx - crop_size), min(intensity_norm.shape[0], dx + crop_size)
        y0, y1 = max(0, dy_center - crop_size), min(intensity_norm.shape[1], dy_center + crop_size)

        if x1 > x0 and y1 > y0:
            crop_img = intensity_norm[x0:x1, y0:y1]

            from matplotlib.patches import Rectangle
            rect = Rectangle((y0, x0), y1 - y0, x1 - x0, linewidth=1, edgecolor='white', facecolor='none')
            ax_out.add_patch(rect)

            ax_zoom = axes[i, 2]
            ax_zoom.imshow(crop_img, cmap='hot')
            ax_zoom.set_title(f"Zoom: Detector {max_det_idx}")
            ax_zoom.axis('off')

            from matplotlib.patches import ConnectionPatch
            con1 = ConnectionPatch(
                xyA=(y1, x0), xyB=(0, 0), coordsA="data", coordsB="data",
                axesA=ax_out, axesB=ax_zoom, color="cyan", alpha=0.5, linewidth=2, linestyle="-"
            )
            con2 = ConnectionPatch(
                xyA=(y1, x1), xyB=(0, crop_img.shape[0]), coordsA="data", coordsB="data",
                axesA=ax_out, axesB=ax_zoom, color="cyan", alpha=0.5, linewidth=2, linestyle="-"
            )
            ax_out.add_artist(con1)
            ax_out.add_artist(con2)

            zoom_cx = dy - y0
            zoom_cy = dx - x0
            if det_shape == 'square':
                rect_zoom = Rectangle(
                    (zoom_cx - det_size / 2 - 0.5, zoom_cy - det_size / 2 - 0.5),
                    det_size,
                    det_size,
                    linewidth=2,
                    edgecolor='magenta',
                    facecolor='none',
                    alpha=0.5,
                )
                ax_zoom.add_patch(rect_zoom)
            else:
                circle_zoom = plt.Circle((zoom_cx, zoom_cy), det_size / 2, color='magenta', fill=False, linewidth=2, alpha=0.5)
                ax_zoom.add_patch(circle_zoom)
        else:
            axes[i, 2].axis('off')

        ax_bar = axes[i, 3]
        vals = out_label[i].cpu().numpy()
        ax_bar.bar(range(len(vals)), vals, color='skyblue')
        ax_bar.set_title("Detector Values")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_samples.png"))
    plt.close(fig)


def _save_confusion_matrix(
    model,
    device,
    val_dataloader,
    PhaseMask,
    PADDINGx,
    PADDINGy,
    output_dir,
):
    print(f"Computing confusion matrix -> {output_dir}")
    all_preds = []
    all_labels = []

    from train import gpu_transform_val

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            images_aug = gpu_transform_val(images)

            if images_aug.shape[1] == 1:
                images_squeezed = images_aug.squeeze(1)
            else:
                images_squeezed = images_aug

            if images_squeezed.shape[-1] == PhaseMask[0]:
                images_input = images_squeezed
            else:
                images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))

            out_label, _, _, _ = model(images_input)
            _, preds = torch.max(out_label, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    import pandas as pd
    df_cm = pd.crosstab(
        pd.Series(all_labels, name='True Label'),
        pd.Series(all_preds, name='Predicted Bin'),
        normalize='index'
    )

    plt.figure(figsize=(10, 12))
    cm_data = df_cm.values
    plt.imshow(cm_data, interpolation='nearest', cmap='Blues', aspect='auto')
    plt.colorbar()

    thresh = cm_data.max() / 2.
    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            plt.text(
                j, i, format(cm_data[i, j], '.2f'),
                ha="center", va="center",
                color="white" if cm_data[i, j] > thresh else "black"
            )

    plt.xticks(np.arange(cm_data.shape[1]), df_cm.columns)
    plt.yticks(np.arange(cm_data.shape[0]), df_cm.index)
    plt.xlabel('Predicted Bin (0-4)')
    plt.ylabel('True Label (0-19)')
    plt.title('Confusion Matrix (Row Normalized)')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def _save_parameter_visualizations(model, latest_subdir, PhaseMask, output_dir):
    print(f"Visualizing trainable parameters -> {output_dir}")
    param_dir = os.path.join(output_dir, "params_vis")
    os.makedirs(param_dir, exist_ok=True)

    for i, mask in enumerate(model.phase_mask):
        plt.figure(figsize=(10, 10))
        plt.imshow(mask.detach().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Phase Mask Layer {i+1}")
        plt.savefig(os.path.join(param_dir, f"phase_mask_layer_{i+1}.png"))
        plt.close()

    if hasattr(model, 'detector_mask') and hasattr(model, 'detector_minus'):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        scale = model.detector_mask.detach().cpu().numpy()
        plt.bar(range(len(scale)), scale, color='green')
        plt.title("Detector Scale (Mask)")
        plt.xlabel("Detector Index")
        plt.ylabel("Value")

        plt.subplot(1, 2, 2)
        bias = model.detector_minus.detach().cpu().numpy()
        plt.bar(range(len(bias)), bias, color='red')
        plt.title("Detector Bias (Minus)")
        plt.xlabel("Detector Index")
        plt.ylabel("Value")

        plt.tight_layout()
        plt.savefig(os.path.join(param_dir, "detector_params.png"))
        plt.close()

    if hasattr(model, 'detector_pos'):
        plt.figure(figsize=(8, 8))
        plt.xlim(0, PhaseMask[1])
        plt.ylim(PhaseMask[0], 0)

        positions = model.detector_pos.detach().cpu().numpy()
        for i, pos in enumerate(positions):
            plt.scatter(pos[1], pos[0], c='blue', s=100, label='Detector' if i == 0 else "")
            plt.text(pos[1] + 10, pos[0], f"D{i}", fontsize=12)

        plt.title("Detector Positions")
        plt.grid(True)
        plt.savefig(os.path.join(param_dir, "detector_positions.png"))
        plt.close()

def evaluate(custom_val_dataset=None, result_dir_override=None):
    # 1. Resolve target result directory first
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Temporarily load main config just to find results_dir base path
    main_config_path = os.path.join(BASE_DIR, 'config.json')
    main_config = {}
    if os.path.exists(main_config_path):
        with open(main_config_path, 'r') as f:
            main_config = json.load(f)
            
    results_dir_config = main_config.get('results_dir', 'results')
    if not os.path.isabs(results_dir_config):
        results_dir = os.path.join(BASE_DIR, results_dir_config)
    else:
        results_dir = results_dir_config
        
    latest_subdir = None
    if result_dir_override is not None:
        latest_subdir = result_dir_override
        print(f"Using explicit result directory: {latest_subdir}")
    else:
        if os.path.exists(results_dir):
            import glob
            # We look for all subdirectories containing best_model.pth
            model_paths = glob.glob(os.path.join(results_dir, "**", "best_model.pth"), recursive=True)
            if model_paths:
                # Sort by modification time to find the latest
                model_paths.sort(key=os.path.getmtime, reverse=True)
                latest_subdir = os.path.dirname(model_paths[0])
                
        if not latest_subdir:
            print(f"No valid result subdirectories (containing best_model.pth) found in {results_dir}.")
            return
        print(f"Latest result directory found: {latest_subdir}")

    # 2. Load the config specific to THIS result directory
    result_config_path = os.path.join(latest_subdir, 'config.json')
    if os.path.exists(result_config_path):
        with open(result_config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {result_config_path}")
    else:
        print(f"Warning: config.json not found in {latest_subdir}. Using main config.")
        config = main_config
        
    exp_name = config.get('exp_name', 'floating_5det')
    print(f"Starting evaluation for Task 1: {exp_name}...")
    
    # Set constants from the loaded config
    BATCH_SIZE = config.get('batch_size', 16)
    
    # Auto-adjust BATCH_SIZE for evaluation based on VRAM to prevent OOM
    if torch.cuda.is_available():
        # Get total VRAM in GB and SM count
        device_props = torch.cuda.get_device_properties(0)
        total_vram_gb = device_props.total_memory / (1024**3)
        sm_count = device_props.multi_processor_count
        
        # Calculate optimal async batch size based on SM count
        # (Same logic as in train.py to maximize overlap and parallel efficiency)
        optimal_async_batch = max(16, int(sm_count * 0.75))
        
        if total_vram_gb < 6.0:
            BATCH_SIZE = min(BATCH_SIZE, 16)
        elif total_vram_gb < 10.0:
            BATCH_SIZE = min(BATCH_SIZE, 32)
        elif total_vram_gb < 16.0:
            BATCH_SIZE = min(BATCH_SIZE, max(48, optimal_async_batch))
        else:
            # If VRAM is abundant, cap at the SM-optimized async batch size rather than a hard 256
            # We can afford to be slightly more aggressive in eval than train, so we use * 3
            BATCH_SIZE = min(BATCH_SIZE, optimal_async_batch * 3)
            
        print(f"Auto-adjusted BATCH_SIZE to {BATCH_SIZE} based on {total_vram_gb:.1f}GB VRAM and {sm_count} SMs.")
    
    IMG_SIZE = config.get('img_size', [1000, 1000])
    PhaseMask = config.get('phase_mask_size', [1200, 1200])
    PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 2
    PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Detectors config overriding
    # Override global detector_pos with the one from result config if it exists
    detector_pos_init_config = config.get('detector_pos', None)
    detector_pos_xy = []
    if detector_pos_init_config is not None:
        for x, y in detector_pos_init_config:
            detector_pos_xy.append((x, y))
    else:
        detector_pos_init = [
            (803, 843, 273, 313),
            (941, 981, 463, 503),
            (941, 981, 697, 737),
            (580, 620, 960, 1000),
            (219, 259, 697, 737),
        ]
        for x0, x1, y0, y1 in detector_pos_init:
            detector_pos_xy.append(((x0+x1)/2, (y0+y1)/2))
    
    # We must patch the DNN class's global detector_pos_xy since it's initialized during class definition in train.py
    import train
    train.detector_pos_xy = detector_pos_xy
    train.config = config # also inject config into train.py's scope so DNN init uses the right parameters
    
    # Dataset
    dataset_path_config = config.get('dataset_path', './dataset')
    
    # Resolve dataset path relative to BASE_DIR if it's not absolute
    if not os.path.isabs(dataset_path_config):
        current_search_dir = BASE_DIR
        found = False
        # Look up to 2 levels up (current, parent, grandparent)
        for _ in range(3):
            possible_path = os.path.join(current_search_dir, dataset_path_config)
            if os.path.exists(possible_path):
                dataset_name = possible_path
                found = True
                break
            parent = os.path.dirname(current_search_dir)
            if parent == current_search_dir: # Reached root
                break
            current_search_dir = parent
        
        if not found:
            # Fallback to default behavior (relative to CWD or BASE_DIR directly)
            dataset_name = os.path.join(BASE_DIR, dataset_path_config)
            print(f"Warning: Dataset path '{dataset_path_config}' not found in parent directories. Using: {dataset_name}")
    else:
        dataset_name = dataset_path_config
    
    try:
        if custom_val_dataset is not None:
            print("Evaluation: Using custom validation dataset passed from memory.")
            val_dataset = custom_val_dataset
        else:
            # Check if in-memory dataset was used
            use_in_memory = config.get('in_memory_dataset', True)
            if use_in_memory:
                from train import InMemoryImageFolder
                print("Evaluation: Using InMemoryImageFolder...")
                val_dataset = InMemoryImageFolder(f"{dataset_name}/val", transform=cpu_transform)
            else:
                val_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/val", transform=cpu_transform)
        # Shuffle=True for better visualization coverage
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Model
    num_layers = config.get('num_layers', 1)
    num_classes = config.get('num_classes', 5)
    train_detector_pos = config.get('train_detector_pos', True)
    
    # Determine target labels (indices)
    if hasattr(val_dataset, 'classes'):
        classes = val_dataset.classes
    else:
        classes = [str(i) for i in range(num_classes)]
    model_paths = _discover_model_paths(latest_subdir)
    if not model_paths:
        print(f"No model checkpoints (*.pth) found in {latest_subdir}.")
        return

    print(f"Found {len(model_paths)} checkpoint(s) to evaluate.")
    for model_path in model_paths:
        checkpoint_name = os.path.basename(model_path)
        output_dir = _checkpoint_output_dir(latest_subdir, model_path)
        os.makedirs(output_dir, exist_ok=True)

        model = DNN(num_layers=num_layers, num_classes=num_classes, train_detector_pos=train_detector_pos).to(device)
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        _save_sample_visualizations(
            model=model,
            config=config,
            device=device,
            val_dataset=val_dataset,
            val_dataloader=val_dataloader,
            classes=classes,
            PhaseMask=PhaseMask,
            PADDINGx=PADDINGx,
            PADDINGy=PADDINGy,
            output_dir=output_dir,
        )
        _save_confusion_matrix(
            model=model,
            device=device,
            val_dataloader=val_dataloader,
            PhaseMask=PhaseMask,
            PADDINGx=PADDINGx,
            PADDINGy=PADDINGy,
            output_dir=output_dir,
        )
        _save_parameter_visualizations(model, latest_subdir, PhaseMask, output_dir)

        if checkpoint_name == "best_model.pth":
            for filename in ("evaluation_samples.png", "confusion_matrix.png"):
                src = os.path.join(output_dir, filename)
                dst = os.path.join(latest_subdir, filename)
                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, dst)
            legacy_param_dir = os.path.join(latest_subdir, "params_vis")
            checkpoint_param_dir = os.path.join(output_dir, "params_vis")
            if os.path.exists(checkpoint_param_dir):
                import shutil
                if os.path.exists(legacy_param_dir):
                    shutil.rmtree(legacy_param_dir)
                shutil.copytree(checkpoint_param_dir, legacy_param_dir)
            print(f"Updated legacy root outputs from {checkpoint_name}.")

    print("Evaluation finished for all checkpoints.")

if __name__ == "__main__":
    evaluate()
