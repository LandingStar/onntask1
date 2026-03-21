
import sys
import os

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
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 4. Local imports
from train import DNN, cpu_transform
import json
import os

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path) as f:
    config = json.load(f)

# Detectors
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

import json

# Load Config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
else:
    print(f"Config not found at {config_path}, using defaults")

def evaluate():
    # 1. Find the latest result directory first
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
    if os.path.exists(results_dir):
        import re
        from datetime import datetime
        
        # Regex to match default naming format: name_YYYYMMDD_HHMM
        name_pattern = re.compile(r'^.+_(\d{8}_\d{4})$')
        
        valid_subdirs = []
        for d in os.listdir(results_dir):
            dir_path = os.path.join(results_dir, d)
            if not os.path.isdir(dir_path):
                continue
                
            # Determine effective path
            target_path = None
            target_name = d
            
            if os.path.exists(os.path.join(dir_path, "best_model.pth")):
                target_path = dir_path
            elif os.path.isdir(os.path.join(dir_path, d)) and os.path.exists(os.path.join(dir_path, d, "best_model.pth")):
                target_path = os.path.join(dir_path, d)
                
            if target_path:
                match = name_pattern.match(target_name)
                if match:
                    try:
                        timestamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M")
                        valid_subdirs.append({'path': target_path, 'time': timestamp})
                    except ValueError:
                        pass
                
        if valid_subdirs:
            latest_item = max(valid_subdirs, key=lambda x: x['time'])
            latest_subdir = latest_item['path']
            
    if not latest_subdir:
        print(f"No valid result subdirectories (containing best_model.pth and matching date format) found in {results_dir}.")
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
        # Get total VRAM in GB
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Empirical scaling: 128 batch size takes ~4GB, 256 takes ~8GB for this model
        if total_vram_gb < 6.0:
            BATCH_SIZE = min(BATCH_SIZE, 32)
        elif total_vram_gb < 10.0:
            BATCH_SIZE = min(BATCH_SIZE, 64)
        elif total_vram_gb < 16.0:
            BATCH_SIZE = min(BATCH_SIZE, 128)
        else:
            BATCH_SIZE = min(BATCH_SIZE, 256)
        print(f"Auto-adjusted BATCH_SIZE to {BATCH_SIZE} based on {total_vram_gb:.1f}GB VRAM.")
    
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
        # Use Standard ImageFolder (all 20 classes)
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
    
    model = DNN(num_layers=num_layers, num_classes=num_classes, train_detector_pos=train_detector_pos).to(device)
    
    # Load Best Model
    model_path = os.path.join(latest_subdir, "best_model.pth")
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Model file not found: {model_path}")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 1. Visualization
    print("Generating sample visualizations...")
    
    # Robust sample selection for coverage
    target_labels_found = {}
    
    # Determine target labels (indices)
    if hasattr(val_dataset, 'classes'):
        classes = val_dataset.classes
    else:
        classes = [str(i) for i in range(num_classes)]
        
    target_indices = []
    
    # Check if classes are numeric (float-like), handling "N" suffix
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
    except:
        is_numeric = False
        
    if is_numeric and len(classes) >= 6:
        # Try to find specific targets: 0.1, 0.5, 0.9, 1.3, 1.7, 2.0
        targets = [0.1, 0.5, 0.9, 1.3, 1.7, 2.0]
        for t in targets:
            # Find closest class
            closest_idx = np.argmin([abs(fc - t) for fc in float_classes])
            target_indices.append(closest_idx)
        # Remove duplicates and sort
        target_indices = sorted(list(set(target_indices)))
    else:
        # Fallback: Equidistant samples
        # Select up to 6 samples
        num_to_select = min(6, len(classes))
        if num_to_select > 0:
            target_indices = np.linspace(0, len(classes)-1, num_to_select, dtype=int).tolist()
            target_indices = sorted(list(set(target_indices)))
        else:
            target_indices = []

    target_labels_set = set(target_indices)
    
    # Scan dataset for these labels
    max_batches_to_scan = 20
    batch_count = 0
    
    for images, labels in val_dataloader:
        for i in range(len(labels)):
            lbl = labels[i].item()
            if lbl in target_labels_set and lbl not in target_labels_found:
                target_labels_found[lbl] = images[i]
        
        if len(target_labels_found) == len(target_labels_set):
            break
            
        batch_count += 1
        if batch_count >= max_batches_to_scan:
            break
            
    # Sort samples by label
    sorted_found_labels = sorted(target_labels_found.keys())
    
    plot_images = []
    plot_labels = []
    
    for lbl in sorted_found_labels:
        plot_images.append(target_labels_found[lbl])
        plot_labels.append(lbl)
        
    # If we found nothing
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
    
    if images_to_process.shape[1] == 1:
        images_squeezed = images_to_process.squeeze(1)
    else:
        images_squeezed = images_to_process
    images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))
    
    with torch.no_grad():
        out_label, out_img, _ = model(images_input)
        
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples), squeeze=False)
    
    detector_pos = model.detector_pos
    
    for i in range(num_samples):
        # Input
        ax_in = axes[i, 0]
        ax_in.imshow(images_squeezed[i].cpu().numpy(), cmap='gray')
        try:
            class_name = val_dataset.classes[plot_labels[i]]
        except:
            class_name = str(plot_labels[i])
        ax_in.set_title(f"Input (Label: {class_name})")
        ax_in.axis('off')
        
        # Output
        ax_out = axes[i, 1]
        intensity = out_img[i].cpu().numpy()
        # Robust normalization: Linear Min-Max scaling with saturation (percentile clipping)
        # Clip top 1% to handle hot spots, then normalize linearly
        p99 = np.percentile(intensity, 99.9)
        intensity_norm = np.clip(intensity, 0, p99)
        intensity_norm = (intensity_norm - intensity_norm.min()) / (intensity_norm.max() - intensity_norm.min() + 1e-8)
        ax_out.imshow(intensity_norm, cmap='hot')
        ax_out.set_title("Output Intensity")
        
        # Overlay Detectors
        for pos in detector_pos:
            x, y = pos.detach().cpu().numpy()
            
            det_shape = config.get('detector_shape', 'circle')
            det_size = config.get('detector_size', 60)
            
            if det_shape == 'square':
                from matplotlib.patches import Rectangle
                # Subtract 0.5 for pixel boundary alignment
                rect_det = Rectangle((y - det_size/2 - 0.5, x - det_size/2 - 0.5), det_size, det_size, 
                                     linewidth=2, edgecolor='magenta', facecolor='none', alpha=0.5)
                ax_out.add_patch(rect_det)
            else:
                circle = plt.Circle((y, x), det_size/2, color='magenta', fill=False, linewidth=2, alpha=0.5)
                ax_out.add_patch(circle)
                
        ax_out.axis('off')
        
        # Zoomed Detector Region
        max_det_idx = out_label[i].argmax().item()
        det_pos = detector_pos[max_det_idx].detach().cpu().numpy()
        dx, dy = int(det_pos[0]), int(det_pos[1])
        
        det_shape = config.get('detector_shape', 'circle')
        det_size = config.get('detector_size', 60)
        crop_size = det_size
        
        if det_shape == 'square':
            dy_center = dy
        else:
            dy_center = dy
        
        x0, x1 = max(0, dx - crop_size), min(intensity_norm.shape[0], dx + crop_size)
        y0, y1 = max(0, dy_center - crop_size), min(intensity_norm.shape[1], dy_center + crop_size)
        
        if x1 > x0 and y1 > y0:
            crop_img = intensity_norm[x0:x1, y0:y1]
            
            # 1. Draw the box on ax_out (Output Intensity)
            from matplotlib.patches import Rectangle
            rect = Rectangle((y0, x0), y1-y0, x1-x0, linewidth=1, edgecolor='white', facecolor='none')
            ax_out.add_patch(rect)
            
            # 2. Show the zoomed image in the next column
            ax_zoom = axes[i, 2]
            ax_zoom.imshow(crop_img, cmap='hot')
            ax_zoom.set_title(f"Zoom: Detector {max_det_idx}")
            ax_zoom.axis('off')
            
            # 3. Add connecting lines (ConnectionPatch)
            from matplotlib.patches import ConnectionPatch
            con1 = ConnectionPatch(xyA=(y1, x0), xyB=(0, 0), coordsA="data", coordsB="data",
                                   axesA=ax_out, axesB=ax_zoom, color="cyan", alpha=0.5, linewidth=2, linestyle="-")
            con2 = ConnectionPatch(xyA=(y1, x1), xyB=(0, crop_img.shape[0]), coordsA="data", coordsB="data",
                                   axesA=ax_out, axesB=ax_zoom, color="cyan", alpha=0.5, linewidth=2, linestyle="-")
            ax_out.add_artist(con1)
            ax_out.add_artist(con2)
            
            if det_shape == 'square':
                zoom_cx = dy - y0
                zoom_cy = dx - x0
                rect_zoom = Rectangle((zoom_cx - det_size/2 - 0.5, zoom_cy - det_size/2 - 0.5), det_size, det_size,
                                      linewidth=2, edgecolor='magenta', facecolor='none', alpha=0.5)
                ax_zoom.add_patch(rect_zoom)
            else:
                zoom_cx = dy - y0
                zoom_cy = dx - x0
                circle_zoom = plt.Circle((zoom_cx, zoom_cy), det_size/2, color='magenta', fill=False, linewidth=2, alpha=0.5)
                ax_zoom.add_patch(circle_zoom)
        else:
            axes[i, 2].axis('off')
        
        # Values
        ax_bar = axes[i, 3]
        vals = out_label[i].cpu().numpy()
        ax_bar.bar(range(len(vals)), vals, color='skyblue')
        ax_bar.set_title("Detector Values")
        
    plt.tight_layout()
    plt.savefig(os.path.join(latest_subdir, "evaluation_samples.png"))
    
    # 2. Confusion Matrix
    print("Computing confusion matrix...")
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device).float()
            if images.shape[1] == 1:
                images_squeezed = images.squeeze(1)
            else:
                images_squeezed = images
            images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))
            
            out_label, _, _ = model(images_input)
            _, preds = torch.max(out_label, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Confusion Matrix:
    # Rows: True Labels (0-19)
    # Cols: Predicted Bins (0-4)
    # We want to see how the 20 true labels map to the 6 predicted bins.
    
    # confusion_matrix(y_true, y_pred) usually expects same set of classes.
    # But here classes differ. sklearn handles this by union of labels found.
    # However, to get a nice rectangular matrix (20x6), we should manually construct it or pad.
    # Or just use the raw values, as long as we interpret the axes correctly.
    
    # Let's use crosstab for better flexibility with mismatched classes
    import pandas as pd
    df_cm = pd.crosstab(pd.Series(all_labels, name='True Label'), 
                        pd.Series(all_preds, name='Predicted Bin'), 
                        normalize='index') # Normalize by row (True Label)
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted Bin (0-4)')
    plt.ylabel('True Label (0-19)')
    plt.title('Confusion Matrix (Row Normalized)')
    plt.savefig(os.path.join(latest_subdir, "confusion_matrix.png"))
    print("Confusion matrix saved.")
    
    # 3. Save Trainable Parameters Visualization
    print("Visualizing trainable parameters...")
    
    # Create param_vis directory
    param_dir = os.path.join(latest_subdir, "params_vis")
    os.makedirs(param_dir, exist_ok=True)
    
    # Phase Mask
    for i, mask in enumerate(model.phase_mask):
        plt.figure(figsize=(10, 10))
        plt.imshow(mask.detach().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Phase Mask Layer {i+1}")
        plt.savefig(os.path.join(param_dir, f"phase_mask_layer_{i+1}.png"))
        plt.close()
        
    # Detector Scale & Bias
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
        
    # Detector Positions
    if hasattr(model, 'detector_pos'):
        plt.figure(figsize=(8, 8))
        plt.xlim(0, PhaseMask[1])
        plt.ylim(PhaseMask[0], 0)
        
        positions = model.detector_pos.detach().cpu().numpy()
        for i, pos in enumerate(positions):
            plt.scatter(pos[1], pos[0], c='blue', s=100, label='Detector' if i == 0 else "")
            plt.text(pos[1]+10, pos[0], f"D{i}", fontsize=12)
            
        plt.title("Detector Positions")
        plt.grid(True)
        plt.savefig(os.path.join(param_dir, "detector_positions.png"))
        plt.close()
        
    print("Parameter visualizations saved.")

if __name__ == "__main__":
    evaluate()
