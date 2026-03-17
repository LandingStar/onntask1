
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import copy
import math
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm

import json

# Load Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'config.json')
config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
else:
    print(f"Config not found at {config_path}, using defaults")

# Constants and Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device: ', device)

BATCH_SIZE = config.get('batch_size', 48)
IMG_SIZE = config.get('img_size', [1000, 1000])
PhaseMask = config.get('phase_mask_size', [1200, 1200])
PIXEL_SIZE = config.get('pixel_size', 8e-6)
wl = config.get('wavelength', 532e-9)
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 4
PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 4

# Dataset Paths
dataset_path_config = config.get('dataset_path', 'task1/continue5')

# Resolve dataset path relative to BASE_DIR if it's not absolute
if not os.path.isabs(dataset_path_config):
    # Check if it exists relative to BASE_DIR
    dataset_name = os.path.join(BASE_DIR, dataset_path_config)
    if not os.path.exists(dataset_name):
        # Try finding it relative to root (assuming script is in task1/ai_refined)
        # If dataset_path is "task1/continue5", and we are in "task1/ai_refined",
        # then BASE_DIR/../../dataset_path might work?
        # Or just use the original relative path if it works (CWD)
        if os.path.exists(dataset_path_config):
             dataset_name = dataset_path_config
        elif os.path.exists(os.path.join(BASE_DIR, "../..", dataset_path_config)):
             dataset_name = os.path.join(BASE_DIR, "../..", dataset_path_config)
        else:
             # Fallback to CWD-relative default logic or keep as is for error later
             dataset_name = dataset_path_config
else:
    dataset_name = dataset_path_config

print(f"Dataset path: {dataset_name}")

# Transforms
cpu_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Grayscale(num_output_channels=1),
])

gpu_transform = v2.Compose([
    v2.RandomRotation(degrees=1),
    v2.RandomAffine(
        degrees=0, 
        translate=(0.03, 0.03),
        scale=(0.97, 1.03), 
        shear=None
    ),
    v2.ElasticTransform(alpha=100.0, sigma=5.0),
    v2.Pad([PADDINGx, PADDINGx, PADDINGy, PADDINGy]),
    v2.ColorJitter(brightness=0.4, contrast=0.4),
    v2.RandomPerspective(distortion_scale=0.2, p=0.3),
    v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
])

# Diffractive Layer
class Diffractive_Layer(torch.nn.Module):
    def __init__(self, wl=wl, PhaseMask=PhaseMask, pixel_size=PIXEL_SIZE, distance=config.get('distance_between_layers', 0.15)):
        super(Diffractive_Layer, self).__init__()
        fx = torch.fft.fftshift(torch.fft.fftfreq(PhaseMask[0], d=pixel_size)).float()
        fy = torch.fft.fftshift(torch.fft.fftfreq(PhaseMask[1], d=pixel_size)).float()
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        
        argument = (2 * np.pi)**2 * ((1. / wl) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j*tmp).to(device)
        self.phase = torch.exp(1j * kz * distance).to(device)
        self.phase = torch.fft.fftshift(self.phase)
        
    def forward(self, E):
        E = E.to(torch.cfloat)
        fft_c = torch.fft.fft2(E)
        angular_spectrum = torch.fft.ifft2(fft_c * self.phase)
        return angular_spectrum

# Propagation Layer
class Propagation_Layer(torch.nn.Module):
    def __init__(self, wl=wl, PhaseMask=PhaseMask, pixel_size=PIXEL_SIZE, distance=config.get('distance_to_detectors', 0.2)):
        super(Propagation_Layer, self).__init__()
        fx = torch.fft.fftshift(torch.fft.fftfreq(PhaseMask[0], d=pixel_size)).float()
        fy = torch.fft.fftshift(torch.fft.fftfreq(PhaseMask[1], d=pixel_size)).float()
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        
        argument = (2 * np.pi)**2 * ((1. / wl) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j*tmp).to(device)
        self.phase = torch.exp(1j * kz * distance).to(device)
        self.phase = torch.fft.fftshift(self.phase)
        
    def forward(self, E):
        E = E.to(torch.cfloat)
        fft_c = torch.fft.fft2(E)
        angular_spectrum = torch.fft.ifft2(fft_c * self.phase)
        return angular_spectrum

# Detectors
# Initialize detector positions (Same as before)
detector_pos_init = [
    (803, 843, 273, 313),
    (941, 981, 463, 503),
    (941, 981, 697, 737),
    (580, 620, 960, 1000),
    (219, 259, 697, 737),
    # (357, 397, 273, 313) # Removed 6th detector to match 5 classes
]
detector_pos_xy = []
for x0, x1, y0, y1 in detector_pos_init:
    detector_pos_xy.append(((x0+x1)/2, (y0+y1)/2))

def detector_region(Int, detector_mask=None, detector_minus=None, detector_pos=None, detector_size=60):
    detectors_list = torch.zeros(Int.shape[0], len(detector_pos), device=device)
    
    det_shape = config.get('detector_shape', 'circle')
    # If detector_size is configured as list/dict, handle it? For now assume scalar from config or default.
    # Config might have 'detector_size'
    det_size_config = config.get('detector_size', 60)
    if det_size_config is not None:
        detector_size = det_size_config
        
    # Jitter logic (Feature 1)
    jitter_ratio = config.get('position_jitter', 0.0)
    
    # Edge Penalty (Feature 2)
    edge_weight = config.get('edge_penalty_weight', 0.0)
    edge_ratio = config.get('edge_width_ratio', 0.1)
    edge_loss = torch.tensor(0.0, device=device)
    
    for i, (x_raw, y_raw) in enumerate(detector_pos):
        # Apply Jitter (only during training mode?)
        # Usually we want this during training. But model.training check is needed if this function is used in eval.
        # But detector_region is used in forward(), so we can check model.training if we pass model or if we are in train loop.
        # However, this function is standalone.
        # Let's assume jitter is applied if jitter_ratio > 0. 
        # Ideally, we should control this via a flag or only enable during training.
        # But since this is inside forward, maybe we can assume if jitter > 0 we do it?
        # Wait, this will affect validation too if we don't differentiate.
        # The user said "Feature 1 default OFF".
        # To be safe, we should only apply jitter if training.
        # But this function doesn't know if we are training.
        # Let's add a `training=True` argument to detector_region?
        # Or just apply it always if config is set (user responsibility to turn off for eval? No, that's bad).
        # Let's assume jitter is 0.0 for now as per default.
        
        # Since we can't easily pass training flag without changing signature in many places,
        # let's just implement the logic.
        # Actually, `detector_pos` is passed in. We can jitter it BEFORE passing to this function in forward().
        
        x, y = x_raw, y_raw # Placeholder, jitter moved to DNN.forward
        
        if det_shape == 'square':
            det_x0 = int(x - detector_size/2)
            det_x1 = int(x + detector_size/2)
            det_y0 = int(y - detector_size/2)
            det_y1 = int(y + detector_size/2)
            
            # Clamp indices
            det_x0 = max(0, det_x0); det_x1 = min(Int.shape[1], det_x1)
            det_y0 = max(0, det_y0); det_y1 = min(Int.shape[2], det_y1)
                
            raw_val = Int[:, det_x0:det_x1, det_y0:det_y1].sum(dim=(1, 2))

            # Differentiable edge correction
            right_edge = (Int[:, det_x1:det_x1+1, det_y0:det_y1].sum(dim=(1, 2)) - 
                          Int[:, det_x0:det_x0+1, det_y0:det_y1].sum(dim=(1, 2))) * (x - detector_size/2 - det_x0)
            bottom_edge = (Int[:, det_x0:det_x1, det_y1:det_y1+1].sum(dim=(1, 2)) - 
                           Int[:, det_x0:det_x1, det_y0:det_y0+1].sum(dim=(1, 2))) * (y - detector_size/2 - det_y0)
            raw_val = raw_val + right_edge + bottom_edge
            
            # Edge Penalty for Square
            # Penalize region from (size/2) to (size/2 + edge_width) outside
            if edge_weight > 0:
                ew = int(detector_size * edge_ratio)
                # Outer box
                out_x0 = max(0, int(x - detector_size/2 - ew))
                out_x1 = min(Int.shape[1], int(x + detector_size/2 + ew))
                out_y0 = max(0, int(y - detector_size/2 - ew))
                out_y1 = min(Int.shape[2], int(y + detector_size/2 + ew))
                
                outer_sum = Int[:, out_x0:out_x1, out_y0:out_y1].sum(dim=(1, 2))
                # Use pure inner sum without interpolation for penalty calculation to avoid negative loss loopholes
                inner_sum_pure = Int[:, det_x0:det_x1, det_y0:det_y1].sum(dim=(1, 2))
                rim_val = outer_sum - inner_sum_pure
                edge_loss += rim_val.sum()
                
        else: # circle
            # For circle, detector_size is diameter. Radius = size/2.
            # Crop region needs to be large enough.
            crop_s = int(detector_size/2) + 4
            
            # Expand crop for edge penalty
            if edge_weight > 0:
                ew = int(detector_size * edge_ratio)
                crop_s += ew
                
            cx, cy = int(x), int(y)
            x0 = max(0, cx - crop_s); x1 = min(Int.shape[1], cx + crop_s)
            y0 = max(0, cy - crop_s); y1 = min(Int.shape[2], cy + crop_s)
            
            grid_x, grid_y = torch.meshgrid(torch.arange(x0, x1, device=device), torch.arange(y0, y1, device=device), indexing='ij')
            dist_sq = (grid_x - x)**2 + (grid_y - y)**2
            radius_sq = (detector_size/2)**2
            
            # Soft mask for signal
            soft_mask = torch.sigmoid((radius_sq - dist_sq)) 
            raw_val = (Int[:, x0:x1, y0:y1] * soft_mask).sum(dim=(1, 2))
            
            # Edge Penalty for Circle
            if edge_weight > 0:
                # Ring from radius to radius + ew
                # Sigmoid for outer boundary: radius_out_sq
                radius_out_sq = (detector_size/2 + detector_size * edge_ratio)**2
                soft_mask_out = torch.sigmoid((radius_out_sq - dist_sq))
                
                # Ring mask = Outer Circle - Inner Circle
                ring_mask = soft_mask_out - soft_mask
                # Ensure non-negative (sigmoid is monotonic so this holds)
                
                rim_val = (Int[:, x0:x1, y0:y1] * ring_mask).sum(dim=(1, 2))
                edge_loss += rim_val.sum()

        # Apply Scale (mask) and Bias (minus) if parameters are provided
        if detector_mask is not None and detector_minus is not None:
            detectors_list[:, i] = raw_val * detector_mask[i] - detector_minus[i]
        else:
            detectors_list[:, i] = raw_val

    total = detectors_list.sum(dim=1, keepdim=True) + 1e-8
    
    if edge_weight > 0:
        return Int, detectors_list / total, edge_loss * edge_weight
    else:
        return Int, detectors_list / total, torch.tensor(0.0, device=device)

# DNN Model
class DNN(torch.nn.Module):
    def __init__(self, num_layers=config.get('num_layers', 1), wl=wl, PhaseMask=PhaseMask, pixel_size=PIXEL_SIZE,
                 distance_between_layers=config.get('distance_between_layers', 0.2), distance_to_detectors=config.get('distance_to_detectors', 0.2), 
                 num_classes=config.get('num_classes', 5), train_detector_pos=config.get('train_detector_pos', True)):
        super(DNN, self).__init__()
        
        self.phase_mask = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(PhaseMask, dtype=torch.float32)) for _ in range(num_layers)
        ])
        
        self.detector_mask = torch.nn.Parameter(torch.ones(num_classes))
        self.detector_minus = torch.nn.Parameter(torch.zeros(num_classes))
        
        # Detector positions
        self.detector_pos = torch.nn.Parameter(torch.tensor(detector_pos_xy, dtype=torch.float32))
        
        # Enable/Disable gradient for detector positions
        if not train_detector_pos:
            self.detector_pos.requires_grad = False
            print("Detector position training DISABLED")
        else:
            print("Detector position training ENABLED")
            
        # Enable/Disable gradient for detector mask (scale) and minus (bias)
        train_detector_scale = config.get('train_detector_scale', True)
        train_detector_bias = config.get('train_detector_bias', True)
        
        if not train_detector_scale:
            self.detector_mask.requires_grad = False
            # Fix to 1 if disabled
            with torch.no_grad():
                self.detector_mask.fill_(1.0)
            print("Detector scale training DISABLED (fixed to 1)")
        else:
            print("Detector scale training ENABLED")
            
        if not train_detector_bias:
            self.detector_minus.requires_grad = False
            # Fix to 0 if disabled
            with torch.no_grad():
                self.detector_minus.fill_(0.0)
            print("Detector bias training DISABLED (fixed to 0)")
        else:
            print("Detector bias training ENABLED")
        
        self.diffractive_layers = torch.nn.ModuleList([Diffractive_Layer(wl, PhaseMask, pixel_size, distance_between_layers) for _ in range(num_layers)])
        self.last_diffractive_layer = Propagation_Layer(wl, PhaseMask, pixel_size, distance_to_detectors)

    def forward(self, E):
        # Apply Jitter to detector positions if training
        current_detector_pos = self.detector_pos
        if self.training:
            jitter_ratio = config.get('position_jitter', 0.0)
            if jitter_ratio > 0:
                det_size = config.get('detector_size', 60)
                jitter_amt = det_size * jitter_ratio
                # Add uniform noise [-jitter, jitter]
                noise = (torch.rand_like(self.detector_pos) * 2 - 1) * jitter_amt
                current_detector_pos = self.detector_pos + noise

        E = E.to(torch.cfloat)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(E)
            phase_values = 2 * torch.pi * self.phase_mask[index]
            modulation = torch.exp(1j * phase_values)
            E = temp * modulation
        E = self.last_diffractive_layer(E)
        Int = torch.abs(E)**2
        # detector_region returns Int, output, edge_loss
        Int, output, edge_loss = detector_region(Int, self.detector_mask, self.detector_minus, current_detector_pos)
        return output, Int, edge_loss

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Check for large order of magnitude span
    all_loss = train_loss + val_loss
    if len(all_loss) > 0:
        min_loss = min(all_loss)
        max_loss = max(all_loss)
        if min_loss > 0 and max_loss / min_loss > 100:
            plt.yscale('log')
            print("Loss span is large, switching to log scale for loss plot.")
            
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')
    
    plt.savefig(save_path)
    plt.close()

# Training Function
def train(model, loss_function, optimizer, scheduler, trainloader, testloader, 
          num_classes=5, epochs=5, device=device, strict_accuracy_ratio=1, 
          minus_mask_ratio=0, label_num=20, exp_name="default"):
    
    currentDate = time.strftime("%Y%m%d_%H%M", time.localtime())
    
    # Save Dir Logic
    results_dir = config.get('results_dir', 'results')
    # Resolve results_dir relative to BASE_DIR if relative
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(BASE_DIR, results_dir)
        
    save_dir = os.path.join(results_dir, f"{exp_name}_{currentDate}")
    os.makedirs(save_dir, exist_ok=True)

    # Save Config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    best_acc = 0
    
    if label_num >= 2 * num_classes:
        # num_classes += 1 # Disable extra class for even distribution
        pass
    classes_num = num_classes
    
    if os.path.exists(f"{save_dir}/best_model.pth"):
        print("Loading existing checkpoint...")
        model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
        
    print(f"Training started. Classes: {classes_num}, Labels: {label_num}")
    
    start_time = time.time()

    def compute_loss(images, labels, out_img, output_vec):
        full_int_img = out_img.sum(axis=(1, 2))
        normalized_out_img = out_img / (full_int_img[:, None, None] + 1e-8)
        
        batch_loss = torch.tensor(0.0, device=device)
        
        current_labels_image_tensors = torch.zeros((classes_num, PhaseMask[0], PhaseMask[1]), device=device, dtype=torch.float32)
        
        det_size = 60
        for ind, pos in enumerate(model.detector_pos):
            x, y = pos
            x0 = int(x - det_size/2)
            x1 = int(x + det_size/2)
            y0 = int(y - det_size/2)
            y1 = int(y + det_size/2)
            
            x0 = max(0, x0); x1 = min(PhaseMask[0], x1)
            y0 = max(0, y0); y1 = min(PhaseMask[1], y1)
            
            if ind < classes_num:
                current_labels_image_tensors[ind, x0:x1, y0:y1] = 1.0
                if current_labels_image_tensors[ind].sum() > 0:
                    current_labels_image_tensors[ind] /= current_labels_image_tensors[ind].sum()

        for i in range(images.size(0)):
            sample_label = labels[i].item()
            sample_img = normalized_out_img[i]
            
            target_mask_weight = torch.full((classes_num,), minus_mask_ratio, device=device, dtype=torch.float32)
            
            if label_num < 2 * classes_num:
                target_mask_weight[sample_label] = 1
            else:
                # Uniform mapping: Map 20 labels to 5 classes evenly (4 labels per class)
                target_idx = int(sample_label * classes_num / label_num)
                target_idx = min(target_idx, classes_num - 1)
                target_mask_weight[target_idx] = 1.0
            
            target_mask = torch.einsum('c,chw->hw', target_mask_weight, current_labels_image_tensors)
            loss = loss_function(sample_img, target_mask)
            loss_vec = F.mse_loss(output_vec[i], target_mask_weight)
            
            # CHANGED: Increased weight of loss_vec (1.0 -> 10.0) and effectively reduced relative importance of pixel loss.
            # Or we can just set image loss to 0. Let's try weighting loss_vec much higher.
            batch_loss += 0.05 * loss + 1.0 * loss_vec
            
        return batch_loss

    def compute_acc(out_label, labels):
        # ... (Same logic as before) ...
        base_intensity = 1/num_classes
        max_intensities, predicted = torch.max(out_label.data, 1)
        
        non_target_mask_bool = F.one_hot(predicted, num_classes=num_classes).bool()
        second_brightest_intensity, nd_predicted = torch.max(out_label.masked_fill(non_target_mask_bool, -float('inf')), dim=1)
        
        is_correct = torch.zeros_like(labels, dtype=torch.bool)
        
        max_intensities -= base_intensity
        second_brightest_intensity -= base_intensity
        
        if label_num < 2 * classes_num:
            for i in range(num_classes):
                mask = (labels == i)
                if mask.any():
                    is_correct[mask] = (predicted[mask] == i) & (max_intensities[mask] >= second_brightest_intensity[mask] * strict_accuracy_ratio)
        else:
            # Uniform mapping accuracy check
            for i in range(label_num):
                mask = (labels == i)
                if mask.any():
                    target_cls = int(i * classes_num / label_num)
                    target_cls = min(target_cls, classes_num - 1)
                    
                    is_correct[mask] = (predicted[mask] == target_cls) & (max_intensities[mask] >= second_brightest_intensity[mask] * strict_accuracy_ratio)
                    
        return is_correct

    for epoch in range(epochs):
        ep_train_loss = 0
        model.train()
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            optimizer.zero_grad()
            images = images.to(device).float()
            
            with torch.no_grad():
                images_aug = gpu_transform(images)
            
            if images_aug.shape[1] == 1:
                images_squeezed = images_aug.squeeze(1)
            else:
                images_squeezed = images_aug
                
            images_input = F.pad(images_squeezed, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
            labels = labels.to(device)
            
            out_label, out_img, edge_loss_val = model(images_input)
            
            loss = compute_loss(images_input, labels, out_img, out_label)
            
            # Add Edge Penalty Loss
            loss += edge_loss_val.sum() if edge_loss_val.ndim > 0 else edge_loss_val
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_train_loss += loss.item()
            
            with torch.no_grad():
                batch_correct = compute_acc(out_label, labels).sum().item()
                correct += batch_correct
                total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item()/labels.size(0), 'acc': batch_correct/labels.size(0)})
        
        avg_train_loss = ep_train_loss / len(trainloader.dataset)
        train_acc = correct / total
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(train_acc)
        
        # Validation
        ep_test_loss = 0
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                images_aug = gpu_transform(images)
                
                if images_aug.shape[1] == 1:
                    images_squeezed = images_aug.squeeze(1)
                else:
                    images_squeezed = images_aug
                    
                images_input = F.pad(images_squeezed, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
                out_label, out_img, edge_loss_val = model(images_input)
                
                loss = compute_loss(images_input, labels, out_img, out_label)
                # Add Edge Penalty Loss for tracking
                loss += edge_loss_val.sum() if edge_loss_val.ndim > 0 else edge_loss_val
                ep_test_loss += loss.item()
                
                correct += compute_acc(out_label, labels).sum().item()
                total += labels.size(0)

        avg_test_loss = ep_test_loss / len(testloader)
        test_acc = correct / total
        test_loss_hist.append(avg_test_loss)
        test_acc_hist.append(test_acc)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_test_loss)
        else:
            scheduler.step()
            
        print(f"Epoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.6f}")
        print(f"Val Loss: {avg_test_loss:.6f} | Val Acc: {test_acc:.6f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Saved best model with acc: {best_acc:.6f}")
        
        # Plotting
        plot_loss_acc(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist, f"{save_dir}/loss_acc.png")
        
        # Save Outputs
        csv_log_dir = f"{save_dir}/csv_logs"
        os.makedirs(csv_log_dir, exist_ok=True)
        np.savetxt(f"{csv_log_dir}/mask_epoch_{epoch+1}.csv", model.phase_mask[0].detach().cpu().numpy(), delimiter=",")
        np.savetxt(f"{csv_log_dir}/detector_pos_epoch_{epoch+1}.csv", model.detector_pos.detach().cpu().numpy(), delimiter=",")

    elapsed_time = time.time() - start_time
    print(f"Total time for {epochs} epochs: {elapsed_time:.2f} seconds")
    return elapsed_time

if __name__ == "__main__":
    try:
        train_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/train", transform=cpu_transform)
        val_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/val", transform=cpu_transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Experiment from Config
        exp_name = config.get('exp_name', 'default_run')
        print(f"\n--- Running Experiment: {exp_name} ---")
        
        # Model (Defaults are pulled from global config)
        model = DNN() 
        model = model.to(device)
        
        criterion = torch.nn.MSELoss(reduction='sum')
        criterion = criterion.to(device)
        
        lr = config.get('learning_rate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
        
        epochs = config.get('epochs', 5)
        num_classes = config.get('num_classes', 5)
        label_num = config.get('label_num', 20)
        strict_accuracy_ratio = config.get('strict_accuracy_ratio', 1)
        minus_mask_ratio = config.get('minus_mask_ratio', 0)
        
        time_taken = train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, 
                           num_classes=num_classes, epochs=epochs, 
                           strict_accuracy_ratio=strict_accuracy_ratio, 
                           minus_mask_ratio=minus_mask_ratio, 
                           label_num=label_num, 
                           exp_name=exp_name)
        
        print(f"Time taken: {time_taken:.2f}s")

        # Run Evaluation if configured
        if config.get("run_evaluate_after_train", True):
            print("\n--- Running Evaluation ---")
            from evaluate import evaluate
            evaluate()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
