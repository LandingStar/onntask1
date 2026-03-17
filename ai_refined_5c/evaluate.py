
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
import torchvision.transforms.v2 as v2

# 3. Others
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 4. Local imports
from train import DNN, cpu_transform, detector_pos_xy, SelectedClassesDataset, detector_region

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

# Constants (Must match train.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = config.get('batch_size', 16)
IMG_SIZE = config.get('img_size', [1000, 1000])
PhaseMask = config.get('phase_mask_size', [1200, 1200])
PIXEL_SIZE = config.get('pixel_size', 8e-6)
wl = config.get('wavelength', 532e-9)
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 4
PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys

# Logging setup
log_file_path = os.path.join(BASE_DIR, "eval_status.log")
def log(msg):
    with open(log_file_path, "a") as f:
        f.write(msg + "\n")
    print(msg)

def evaluate():
    exp_name = config.get('exp_name', 'default_run_5c')
    log(f"Starting evaluation for Task 1: {exp_name}...")
    
    # Dataset
    dataset_path_config = config.get('dataset_path', 'task1/continue5')
    
    # Resolve dataset path relative to BASE_DIR if it's not absolute
    if not os.path.isabs(dataset_path_config):
        dataset_name = os.path.join(BASE_DIR, dataset_path_config)
        if not os.path.exists(dataset_name):
            if os.path.exists(dataset_path_config):
                 dataset_name = dataset_path_config
            elif os.path.exists(os.path.join(BASE_DIR, "../..", dataset_path_config)):
                 dataset_name = os.path.join(BASE_DIR, "../..", dataset_path_config)
            else:
                 dataset_name = dataset_path_config
    else:
        dataset_name = dataset_path_config
    
    log(f"Dataset path: {dataset_name}")
    selected_indices = config.get('selected_indices', [0, 4, 9, 14, 19])
    val_dataset = SelectedClassesDataset(f"{dataset_name}/val", transform=cpu_transform, selected_indices=selected_indices)
    # Shuffle=True for visualization
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    log("Dataset loaded.")
    
    # Model
    num_layers = config.get('num_layers', 1)
    num_classes = config.get('num_classes', 5)
    
    model = DNN(num_layers=num_layers, num_classes=num_classes).to(device)
    log("Model initialized.")
    
    # Load Best Model
    results_dir_config = config.get('results_dir', 'results')
    if not os.path.isabs(results_dir_config):
        results_dir = os.path.join(BASE_DIR, results_dir_config)
    else:
        results_dir = results_dir_config
        
    # Search for the latest results directory
    if os.path.exists(results_dir):
        # Look for experiments matching exp_name
        subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and exp_name in d]
        # Fallback
        if not subdirs:
             subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
             
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            model_path = os.path.join(latest_subdir, "best_model.pth")
            log(f"Loading model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            log(f"No result subdirectories found matching '{exp_name}'.")
            return
    else:
        log(f"Results directory not found: {results_dir}")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 1. Visualization of Samples
    log("Generating sample visualizations...")
    
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
        log("Warning: No target samples found for visualization.")
        data_iter = iter(val_dataloader)
        try:
            images, labels = next(data_iter)
            plot_images = [images[i] for i in range(min(6, len(labels)))]
            plot_labels = [labels[i].item() for i in range(min(6, len(labels)))]
        except StopIteration:
            pass

    num_samples = len(plot_images)
    if num_samples == 0:
        log("No samples to plot.")
        return

    images_to_process = torch.stack(plot_images).to(device).float()
    
    # Preprocessing (Padding)
    if images_to_process.shape[1] == 1:
        images_squeezed = images_to_process.squeeze(1)
    else:
        images_squeezed = images_to_process
    images_input = F.pad(images_squeezed, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
    
    with torch.no_grad():
        out_label, out_img, _ = model(images_input)
        
    # Plot samples
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    # Detector info
    detector_pos = model.detector_pos
    if isinstance(detector_pos, list):
        detector_pos_tensor = torch.tensor(detector_pos, device=device)
    else:
        detector_pos_tensor = detector_pos
        
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
        
        # Output Intensity
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
        for pos in detector_pos_tensor:
            x, y = pos.cpu().numpy()
            
            det_shape = config.get('detector_shape', 'circle')
            det_size = config.get('detector_size', 60)
            
            if det_shape == 'square':
                from matplotlib.patches import Rectangle
                # Subtract 0.5 for pixel boundary alignment
                rect_det = Rectangle((y - det_size/2 - 0.5, x - det_size/2 - 0.5), det_size, det_size, 
                                     linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.5)
                ax_out.add_patch(rect_det)
            else:
                circle = plt.Circle((y, x), det_size/2, color='cyan', fill=False, linewidth=2, alpha=0.5) # Note: x, y might be swapped in imshow
                ax_out.add_patch(circle)
                
        ax_out.axis('off')
        
        # Zoomed Detector Region
        max_det_idx = out_label[i].argmax().item()
        det_pos = detector_pos_tensor[max_det_idx].cpu().numpy()
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
                zoom_cx = dy_center - y0
                zoom_cy = dx - x0
                rect_zoom = Rectangle((zoom_cx - det_size/2 - 0.5, zoom_cy - det_size/2 - 0.5), det_size, det_size,
                                      linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.5)
                ax_zoom.add_patch(rect_zoom)
            else:
                zoom_cx = dy - y0
                zoom_cy = dx - x0
                circle_zoom = plt.Circle((zoom_cx, zoom_cy), det_size/2, color='cyan', fill=False, linewidth=2, alpha=0.5)
                ax_zoom.add_patch(circle_zoom)
        else:
            axes[i, 2].axis('off')
        
        # Detector Values
        ax_bar = axes[i, 3]
        vals = out_label[i].cpu().numpy()
        bars = ax_bar.bar(range(len(vals)), vals, color='skyblue')
        ax_bar.set_xticks(range(len(vals)))
        ax_bar.set_xticklabels(val_dataset.classes)
        ax_bar.set_title("Detector Values")
        ax_bar.set_ylim(0, max(vals.max()*1.2, 1e-6))
        
        # Highlight prediction
        pred_idx = vals.argmax()
        bars[pred_idx].set_color('orange')
        
    plt.tight_layout()
    plt.savefig(os.path.join(latest_subdir, "evaluation_samples.png"))
    log("Sample visualization saved.")
    
    # 2. Confusion Matrix
    log("Computing confusion matrix...")
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device).float()
            
            if images.shape[1] == 1:
                images_squeezed = images.squeeze(1)
            else:
                images_squeezed = images
            images_input = F.pad(images_squeezed, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
            
            out_label, _, _ = model(images_input)
            _, preds = torch.max(out_label, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(latest_subdir, "confusion_matrix.png"))
    log("Confusion matrix saved.")
    
    # 3. Misclassified Sample Analysis
    log("Analyzing misclassified samples...")
    misclassified_indices = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p != t]
    
    if misclassified_indices:
        num_mis = min(5, len(misclassified_indices))
        fig, axes = plt.subplots(num_mis, 3, figsize=(15, 5*num_mis))
        if num_mis == 1: axes = axes.reshape(1, -1)
        
        # We need random access to dataset for visualization
        # val_dataset is accessible
        
        for idx, mis_idx in enumerate(misclassified_indices[:num_mis]):
            # Retrieve original image (re-read or from dataloader sequence if possible)
            # Since we iterated dataloader before, we can't easily jump.
            # Better to just grab the batch again or store them.
            # But storing all images might be heavy.
            # Let's re-fetch specific samples from dataset
            
            img_tensor, true_label = val_dataset[mis_idx] # This returns cpu_transform'd image
            pred_label = all_preds[mis_idx]
            
            # Re-run forward pass to get intensity map
            img_in = img_tensor.unsqueeze(0).to(device).float()
            img_in_padded = F.pad(img_in, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
            with torch.no_grad():
                out_l, out_i, _ = model(img_in_padded)
            
            # Plot
            # Input
            axes[idx, 0].imshow(img_tensor.squeeze().numpy(), cmap='gray')
            axes[idx, 0].set_title(f"True: {val_dataset.classes[true_label]} | Pred: {val_dataset.classes[pred_label]}")
            axes[idx, 0].axis('off')
            
            # Intensity
            axes[idx, 1].imshow(out_i[0].cpu().numpy(), cmap='hot')
            # Normalize intensity for better visualization
            intensity_mis = out_i[0].cpu().numpy()
            # Linear saturation normalization
            p99 = np.percentile(intensity_mis, 99.9)
            intensity_mis = np.clip(intensity_mis, 0, p99)
            intensity_mis = (intensity_mis - intensity_mis.min()) / (intensity_mis.max() - intensity_mis.min() + 1e-8)
            axes[idx, 1].imshow(intensity_mis, cmap='hot')
            axes[idx, 1].set_title("Output Intensity")
            for pos in detector_pos_tensor:
                x, y = pos.cpu().numpy()
                
                det_shape = config.get('detector_shape', 'circle')
                det_size = config.get('detector_size', 60)
                
                if det_shape == 'square':
                    from matplotlib.patches import Rectangle
                    # Subtract 0.5 for pixel boundary alignment
                    # User reported 1/2 size shift to left. Adjusting X coordinate.
                    rect_det = Rectangle((y - det_size, x - det_size/2 - 0.5), det_size, det_size, 
                                         linewidth=2, edgecolor='red', facecolor='none', alpha=0.5)
                    axes[idx, 1].add_patch(rect_det)
                else:
                    circle = plt.Circle((y, x), det_size/2, color='red', fill=False, linewidth=2, alpha=0.5)
                    axes[idx, 1].add_patch(circle)
            axes[idx, 1].axis('off')
            
            # Bars
            vals = out_l[0].cpu().numpy()
            bars = axes[idx, 2].bar(range(len(vals)), vals, color='skyblue')
            axes[idx, 2].set_xticks(range(len(vals)))
            axes[idx, 2].set_xticklabels(val_dataset.classes)
            bars[pred_label].set_color('red') # Wrong prediction
            bars[true_label].set_color('green') # True label
            axes[idx, 2].set_title("Detector Values (Green=True, Red=Pred)")
            
        plt.tight_layout()
        plt.savefig(os.path.join(latest_subdir, "misclassified_samples.png"))
        log("Misclassified samples saved.")
    else:
        log("No misclassified samples found (Accuracy 100%!).")
    
    log("Evaluation completed successfully.")
    
    # 4. Save Trainable Parameters Visualization
    log("Visualizing trainable parameters...")
    
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
    # Need to check if model has these attributes (it should)
    if hasattr(model, 'detector_mask') and hasattr(model, 'detector_minus'):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        # Handle parameterlist or parameter
        if isinstance(model.detector_mask, torch.nn.Parameter):
            scale = model.detector_mask.detach().cpu().numpy()
        else:
            scale = model.detector_mask # Fallback
            
        plt.bar(range(len(scale)), scale, color='green')
        plt.title("Detector Scale (Mask)")
        plt.xlabel("Detector Index")
        plt.ylabel("Value")
        
        plt.subplot(1, 2, 2)
        if isinstance(model.detector_minus, torch.nn.Parameter):
            bias = model.detector_minus.detach().cpu().numpy()
        else:
            bias = model.detector_minus
            
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
        
        # Check if it's tensor or list
        if isinstance(model.detector_pos, torch.Tensor):
            positions = model.detector_pos.detach().cpu().numpy()
        elif isinstance(model.detector_pos, list):
            positions = np.array(model.detector_pos)
        else:
            positions = []
            
        for i, pos in enumerate(positions):
            plt.scatter(pos[1], pos[0], c='blue', s=100, label='Detector' if i == 0 else "")
            plt.text(pos[1]+10, pos[0], f"D{i}", fontsize=12)
            
        plt.title("Detector Positions")
        plt.grid(True)
        plt.savefig(os.path.join(param_dir, "detector_positions.png"))
        plt.close()
        
    log("Parameter visualizations saved.")

if __name__ == "__main__":
    evaluate()
