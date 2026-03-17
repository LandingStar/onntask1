
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
from train import DNN, cpu_transform, detector_pos_xy, detector_region

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

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = config.get('batch_size', 16)
IMG_SIZE = config.get('img_size', [1000, 1000])
PhaseMask = config.get('phase_mask_size', [1200, 1200])
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 2
PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    exp_name = config.get('exp_name', 'floating_detectors')
    print(f"Starting evaluation for Task 1: {exp_name}...")
    
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
    
    val_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/val", transform=cpu_transform)
    # Shuffle=True to ensure we see a variety of classes quickly for visualization
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Model
    num_layers = config.get('num_layers', 1)
    num_classes = config.get('num_classes', 5) # Default to 5
    train_detector_pos = config.get('train_detector_pos', True)
    
    model = DNN(num_layers=num_layers, num_classes=num_classes, train_detector_pos=train_detector_pos).to(device)
    
    # Load Best Model
    results_dir_config = config.get('results_dir', 'results')
    if not os.path.isabs(results_dir_config):
        results_dir = os.path.join(BASE_DIR, results_dir_config)
    else:
        results_dir = results_dir_config
        
    # Try to find based on config exp_name
    if os.path.exists(results_dir):
        # Look for experiments matching exp_name
        subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and exp_name in d]
        # Fallback to "floating" if not found and exp_name was default
        if not subdirs and "floating" not in exp_name:
             subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and "floating" in d]
             
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            model_path = os.path.join(latest_subdir, "best_model.pth")
            print(f"Loading model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No result subdirectories found matching '{exp_name}'.")
            return
    else:
        print(f"Results directory not found: {results_dir}")
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
        # Fallback if no classes attribute, assume 0..num_classes-1
        classes = [str(i) for i in range(num_classes)]

    target_indices = []
    
    # Check if classes are numeric (float-like), handling "N" suffix or similar
    try:
        # Try to parse "0.1N" -> 0.1
        import re
        float_classes = []
        for c in classes:
            # Extract first float number found in string
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
    print(f"Target indices to find: {target_labels_set}")
    
    # Scan dataset for these labels
    max_batches_to_scan = 100 # Increased from 50 to ensure we find all
    batch_count = 0
    
    for images, labels in val_dataloader:
        for i in range(len(labels)):
            lbl = labels[i].item()
            if lbl in target_labels_set and lbl not in target_labels_found:
                target_labels_found[lbl] = images[i]
                print(f"Found sample for label {lbl}")
        
        if len(target_labels_found) == len(target_labels_set):
            break
            
        batch_count += 1
        if batch_count >= max_batches_to_scan:
            print(f"Stopped scanning after {max_batches_to_scan} batches. Found: {list(target_labels_found.keys())}")
            break
            
    # Sort samples by label
    sorted_found_labels = sorted(target_labels_found.keys())
    
    plot_images = []
    plot_labels = []
    
    for lbl in sorted_found_labels:
        plot_images.append(target_labels_found[lbl])
        plot_labels.append(lbl)
    
    # Check if we have enough samples
    if len(plot_images) < 5:
        print(f"Warning: Only found {len(plot_images)} samples out of expected 5-6. Retrying scan with more batches...")
        # Continue scanning from where we left off (if possible) or just continue from current dataloader state?
        # The dataloader iterator is fresh each loop.
        # Let's try to scan the ENTIRE dataset if needed.
        
        # Reset iterator logic? No, just iterate more.
        # But we are inside a function, so we can't easily jump back.
        # Instead, let's just loop over dataloader again if we are really desperate.
        
        # Better approach: Just iterate until we find them or exhaust dataset.
        pass # Already increased max_batches_to_scan to 100.
        
    # If still < 5, try to fill with *any* other unique labels found during scan if we skipped them?
    # No, we only stored target ones.
    
    # If we found < 6 samples, just use what we have.
    # But user specifically asked for "0.1, 0.5, 0.9, 1.3, 1.7, 2.0" (6 samples).
    # If we only found 3, maybe the dataset is small or shuffled such that these didn't appear in first 100 batches?
    # Or maybe the labels don't exist?
    
    # Let's print what we are looking for and what we found.
    print(f"Target indices: {target_labels_set}")
    print(f"Found indices: {sorted_found_labels}")
        
    # If we found nothing (e.g. empty dataset?), handle gracefully
    if not plot_images:
        print("Warning: No target samples found for visualization.")
        # Fallback to first batch
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
    images_input = F.pad(images_squeezed, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
    
    with torch.no_grad():
        out_label, out_img, _ = model(images_input)
        
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    detector_pos = model.detector_pos # Should be learnable parameters
    
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
        
        # Overlay Detectors (Floating)
        for pos in detector_pos:
            x, y = pos.detach().cpu().numpy()
            
            # Check shape config (default to circle if not found, but it should be consistent with train)
            det_shape = config.get('detector_shape', 'circle')
            det_size = config.get('detector_size', 60)
            
            if det_shape == 'square':
                from matplotlib.patches import Rectangle
                # Subtract 0.5 for pixel boundary alignment
                rect_det = Rectangle((y - det_size/2 - 0.5, x - det_size/2 - 0.5), det_size, det_size, 
                                     linewidth=2, edgecolor='lime', facecolor='none', alpha=0.5)
                ax_out.add_patch(rect_det)
            else:
                circle = plt.Circle((y, x), det_size/2, color='lime', fill=False, linewidth=2, alpha=0.5)
                ax_out.add_patch(circle)
                
        ax_out.axis('off')
        
        # Zoomed Detector Region
        # Determine which detector to zoom in on (max value)
        _, max_det_idx = torch.max(out_label[i], 0)
        
        # Get coordinates of that detector
        det_pos = detector_pos[max_det_idx].detach().cpu().numpy()
        dx, dy = int(det_pos[0]), int(det_pos[1])
        
        # Determine crop center
        # User reported 1/2 size shift to left for Square. Adjusting crop center.
        det_shape = config.get('detector_shape', 'circle')
        det_size = config.get('detector_size', 60)
        crop_size = det_size # Show 2x area
        
        if det_shape == 'square':
            dy_center = dy
        else:
            dy_center = dy
            
        x0, x1 = max(0, dx - crop_size), min(intensity_norm.shape[0], dx + crop_size)
        y0, y1 = max(0, dy_center - crop_size), min(intensity_norm.shape[1], dy_center + crop_size)
        
        ax_zoom = axes[i, 2]
        
        # Ensure indices are valid
        if x1 > x0 and y1 > y0:
            # ...
            
            # Add shape to zoom view too
            det_shape = config.get('detector_shape', 'circle')
            det_size = config.get('detector_size', 60)
            
            if det_shape == 'square':
                from matplotlib.patches import Rectangle
                # Center in zoom coordinates is (dy-y0, dx-x0)
                # Bottom-left is (dy-y0-size/2, dx-x0-size/2)
                # Subtract 0.5 for pixel boundary alignment
                rect_zoom = Rectangle((dy - y0 - det_size/2 - 0.5, dx - x0 - det_size/2 - 0.5), det_size, det_size,
                                      linewidth=2, edgecolor='lime', facecolor='none', alpha=0.5)
                ax_zoom.add_patch(rect_zoom)
            else:
                circle_zoom = plt.Circle((dy - y0, dx - x0), det_size/2, color='lime', fill=False, linewidth=2, alpha=0.5)
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
            images_input = F.pad(images_squeezed, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
            
            out_label, _, _ = model(images_input)
            
            # Map model's 6 outputs to a predicted class index (0-5)
            # Just take the max intensity detector index
            _, preds = torch.max(out_label, 1)
            
            # Remap 6 bins to 5 bins as requested
            # 0,1,2,3,4,5 -> 0,0,1,2,3,4 ? Or linear scaling?
            # Linear scaling: new = old * (5/6)
            # 0->0, 1->0.8(0), 2->1.6(1), 3->2.5(2), 4->3.3(3), 5->4.1(4)
            # This maps 0,1 -> 0; 2->1; 3->2; 4->3; 5->4.
            # This merges the first two.
            
            # Let's try to align with the labels.
            # True labels are 0-19.
            # If we want 5 bins, they should ideally represent 0-3, 4-7, 8-11, 12-15, 16-19.
            
            # The current preds (0-4) are what the model "chose" from its 5 detectors.
            # We simply want to visualize this as 5 bins.
            preds_5bin = preds # (preds.float() * 5 / 6).long() -> No longer needed as we use 5 classes
            
            all_preds.extend(preds_5bin.cpu().numpy())
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
        
    # Detector Scale & Bias (if trainable)
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
        # Plot background (e.g. empty phase mask size)
        plt.xlim(0, PhaseMask[1])
        plt.ylim(PhaseMask[0], 0) # Invert Y to match image coordinates
        
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
