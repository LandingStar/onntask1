
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
import sys

# Setup global debug.log redirection if not in a terminal
# Wait, we want to write EVERYTHING to debug.log in the result directory.
# But the result directory is not created until `train()` is called.
# Let's redirect inside `train()` or after `save_dir` is created.

# Load Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, 'config.json')

# Check if a custom config path was passed as an argument
is_custom_config = False
if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
    custom_config = sys.argv[1]
    if os.path.isabs(custom_config):
        config_path = custom_config
    else:
        config_path = os.path.join(BASE_DIR, custom_config)
    is_custom_config = True

config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
else:
    print(f"Config not found at {config_path}, using defaults")

# Intercept for batch training if configured
# BUT ONLY IF we are reading the default config.json. 
# If we are reading a custom temp config from batch_train.py, we MUST NOT intercept again.
if config.get('batch_train', False) and not is_custom_config:
    print("\n[INFO] 'batch_train' flag is true in config.json. Redirecting to batch_train.py...")
    import subprocess
    try:
        # Run batch_train.py in the same directory
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'batch_train.py')], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Batch training failed with code {e.returncode}")
        sys.exit(e.returncode)
    sys.exit(0)  # Exit current train.py process since batch_train handles everything

# Constants and Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device: ', device)

BATCH_SIZE = config.get('batch_size', 48)

# Auto-adjust BATCH_SIZE for training based on VRAM to prevent OOM
if torch.cuda.is_available():
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_vram_gb < 6.0:
        BATCH_SIZE = min(BATCH_SIZE, 32)
    elif total_vram_gb < 10.0:
        BATCH_SIZE = min(BATCH_SIZE, 64)
    elif total_vram_gb < 16.0:
        BATCH_SIZE = min(BATCH_SIZE, 128)
    # else keep config's BATCH_SIZE (could be 256 or higher)
    print(f"Auto-adjusted training BATCH_SIZE to {BATCH_SIZE} based on {total_vram_gb:.1f}GB VRAM.")

IMG_SIZE = config.get('img_size', [1000, 1000])
PhaseMask = config.get('phase_mask_size', [1200, 1200])
PIXEL_SIZE = config.get('pixel_size', 8e-6)
wl = config.get('wavelength', 532e-9)
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 2
PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 2

# Dataset Paths
dataset_path_config = config.get('dataset_path', './dataset')

# Resolve dataset path relative to BASE_DIR if it's not absolute
if not os.path.isabs(dataset_path_config):
    # Search upwards for the dataset
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

print(f"Dataset path: {dataset_name}")

# Custom Dataset for 5-class selection (MODIFIED: Use all classes but map them)
# Actually, the user wants to use 20 input classes and map them to 5 detectors.
# So we should use the standard ImageFolder, not SelectedClassesDataset.
# But wait, ai_refined_5_detectors/train.py was using SelectedClassesDataset.
# I will revert this to use standard ImageFolder and implement the mapping logic.

# Transforms
cpu_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Grayscale(num_output_channels=1),
])

# Read transform intensity from config, default to 1.0
transform_intensity = config.get('transform_intensity', 1.0)

# Build transform list conditionally
gpu_transform_list = []

if transform_intensity > 0:
    # Scale parameters according to intensity
    # Base values
    base_rotation = 1
    base_translate = 0.03
    base_scale_range = 0.03
    base_jitter = 0.4
    base_perspective_scale = 0.2
    base_perspective_p = 0.3
    base_sharpness_factor = 2.0
    base_sharpness_p = 0.5
    
    gpu_transform_list.extend([
        v2.RandomRotation(degrees=base_rotation * transform_intensity),
        v2.RandomAffine(
            degrees=0, 
            translate=(base_translate * transform_intensity, base_translate * transform_intensity),
            scale=(1.0 - base_scale_range * transform_intensity, 1.0 + base_scale_range * transform_intensity), 
            shear=None
        ),
        v2.Pad([PADDINGx, PADDINGx, PADDINGy, PADDINGy]),
        v2.ColorJitter(brightness=base_jitter * transform_intensity, contrast=base_jitter * transform_intensity),
        v2.RandomPerspective(distortion_scale=min(1.0, base_perspective_scale * transform_intensity), p=min(1.0, base_perspective_p * transform_intensity)),
        # Sharpness factor 1.0 is no change, so we scale the difference from 1.0
        v2.RandomAdjustSharpness(sharpness_factor=1.0 + (base_sharpness_factor - 1.0) * transform_intensity, p=min(1.0, base_sharpness_p * transform_intensity)),
    ])
else:
    # If intensity is 0, only apply padding
    gpu_transform_list.append(v2.Pad([PADDINGx, PADDINGx, PADDINGy, PADDINGy]))

gpu_transform = v2.Compose(gpu_transform_list)

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

# Initialize detector positions
detector_pos_init_config = config.get('detector_pos', None)
detector_pos_xy = []

if detector_pos_init_config is not None:
    # Use config positions directly (assuming they are [x, y] center coordinates)
    for x, y in detector_pos_init_config:
        detector_pos_xy.append((x, y))
else:
    # Fallback to hardcoded detectors
    detector_pos_init = [
        (803, 843, 273, 313),
        (941, 981, 463, 503),
        (941, 981, 697, 737),
        (580, 620, 960, 1000),
        (219, 259, 697, 737),
    ]
    for x0, x1, y0, y1 in detector_pos_init:
        detector_pos_xy.append(((x0+x1)/2, (y0+y1)/2))

def detector_region(Int, detector_mask=None, detector_minus=None, detector_pos=None, detector_size=60):
    num_det = len(detector_pos)
    batch_size = Int.shape[0]
    detectors_list = torch.zeros(batch_size, num_det, device=device)
    # Per-sample per-detector penalty (caller decides which detectors to penalize)
    penalty_per_det = torch.zeros(batch_size, num_det, device=device)
    
    det_shape = config.get('detector_shape', 'circle')
    det_size_config = config.get('detector_size', 60)
    if det_size_config is not None:
        detector_size = det_size_config
        
    edge_weight = config.get('edge_penalty_weight', 0.0)
    edge_ratio = config.get('edge_width_ratio', 0.1)
    
    conc_weight = config.get('concentration_loss_weight', 0.0)
    
    for i, (x, y) in enumerate(detector_pos):
        if det_shape == 'square':
            det_x0 = torch.floor(x - detector_size/2).int()
            det_x1 = torch.floor(x + detector_size/2).int()
            det_y0 = torch.floor(y - detector_size/2).int()
            det_y1 = torch.floor(y + detector_size/2).int()
            
            det_x0 = torch.clamp(det_x0, 0, Int.shape[1]-2)
            det_x1 = torch.clamp(det_x1, det_x0+1, Int.shape[1]-1)
            det_y0 = torch.clamp(det_y0, 0, Int.shape[2]-2)
            det_y1 = torch.clamp(det_y1, det_y0+1, Int.shape[2]-1)

            raw_val = Int[:, det_x0:det_x1, det_y0:det_y1].sum(dim=(1, 2))
            
            if detector_pos.requires_grad:
                right_edge = (Int[:, det_x1:det_x1+1, det_y0:det_y1].sum(dim=(1, 2)) - 
                              Int[:, det_x0:det_x0+1, det_y0:det_y1].sum(dim=(1, 2))) * (x - detector_size/2 - det_x0)
                bottom_edge = (Int[:, det_x0:det_x1, det_y1:det_y1+1].sum(dim=(1, 2)) - 
                               Int[:, det_x0:det_x1, det_y0:det_y0+1].sum(dim=(1, 2))) * (y - detector_size/2 - det_y0)
                raw_val = raw_val + right_edge + bottom_edge

            if edge_weight > 0:
                ew = int(detector_size * edge_ratio)
                out_x0 = max(0, int(x - detector_size/2 - ew))
                out_x1 = min(Int.shape[1], int(x + detector_size/2 + ew))
                out_y0 = max(0, int(y - detector_size/2 - ew))
                out_y1 = min(Int.shape[2], int(y + detector_size/2 + ew))
                
                outer_sum = Int[:, out_x0:out_x1, out_y0:out_y1].sum(dim=(1, 2))
                inner_sum_pure = Int[:, det_x0:det_x1, det_y0:det_y1].sum(dim=(1, 2))
                rim_val = outer_sum - inner_sum_pure
                penalty_per_det[:, i] += rim_val * edge_weight

            if conc_weight > 0:
                grid_x = torch.arange(det_x0, det_x1, device=device).float()
                grid_y = torch.arange(det_y0, det_y1, device=device).float()
                gx, gy = torch.meshgrid(grid_x, grid_y, indexing='ij')
                dist_sq = (gx - x)**2 + (gy - y)**2
                radius_sq = (detector_size / 2) ** 2
                region_int = Int[:, det_x0:det_x1, det_y0:det_y1]
                signal_sum = region_int.sum(dim=(1, 2)) + 1e-8
                moment = (region_int * dist_sq).sum(dim=(1, 2)) / (signal_sum * radius_sq)
                penalty_per_det[:, i] += moment * conc_weight

        else: # circle
            crop_s = int(detector_size) + 4
            cx, cy = torch.floor(x).int(), torch.floor(y).int()
            x0 = max(0, cx - crop_s); x1 = min(Int.shape[1], cx + crop_s)
            y0 = max(0, cy - crop_s); y1 = min(Int.shape[2], cy + crop_s)
            
            grid_x, grid_y = torch.meshgrid(torch.arange(x0, x1, device=device), torch.arange(y0, y1, device=device), indexing='ij')
            dist_sq = (grid_x - x)**2 + (grid_y - y)**2
            radius_sq = (detector_size/2)**2
            soft_mask = torch.sigmoid((radius_sq - dist_sq)) 
            raw_val = (Int[:, x0:x1, y0:y1] * soft_mask).sum(dim=(1, 2))

            if edge_weight > 0:
                radius_out_sq = (detector_size/2 + detector_size * edge_ratio)**2
                soft_mask_out = torch.sigmoid((radius_out_sq - dist_sq))
                ring_mask = soft_mask_out - soft_mask
                rim_val = (Int[:, x0:x1, y0:y1] * ring_mask).sum(dim=(1, 2))
                penalty_per_det[:, i] += rim_val * edge_weight

            if conc_weight > 0:
                region_int = Int[:, x0:x1, y0:y1]
                signal_int = region_int * soft_mask
                signal_sum = signal_int.sum(dim=(1, 2)) + 1e-8
                moment = (signal_int * dist_sq).sum(dim=(1, 2)) / (signal_sum * radius_sq)
                penalty_per_det[:, i] += moment * conc_weight

        # Apply Scale (mask) and Bias (minus)
        if detector_mask is not None and detector_minus is not None:
            detectors_list[:, i] = raw_val * detector_mask[i] - detector_minus[i]
        else:
            detectors_list[:, i] = raw_val

    total = detectors_list.sum(dim=1, keepdim=True) + 1e-8
        
    return Int, detectors_list / total, penalty_per_det

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
            with torch.no_grad():
                self.detector_mask.fill_(1.0)
            print("Detector scale training DISABLED (fixed to 1)")
        else:
            print("Detector scale training ENABLED")
            
        if not train_detector_bias:
            self.detector_minus.requires_grad = False
            with torch.no_grad():
                self.detector_minus.fill_(0.0)
            print("Detector bias training DISABLED (fixed to 0)")
        else:
            print("Detector bias training ENABLED")
        
        self.diffractive_layers = torch.nn.ModuleList([Diffractive_Layer(wl, PhaseMask, pixel_size, distance_between_layers) for _ in range(num_layers)])
        self.last_diffractive_layer = Propagation_Layer(wl, PhaseMask, pixel_size, distance_to_detectors)

    def forward(self, E):
        # Apply Jitter
        current_detector_pos = self.detector_pos
        if self.training:
            jitter_ratio = config.get('position_jitter', 0.0)
            if jitter_ratio > 0:
                det_size = config.get('detector_size', 60)
                jitter_amt = det_size * jitter_ratio
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
        # Don't save batch_train=False if it was originally true, but it's safe to just dump current
        json.dump(config, f, indent=4)
    
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    best_acc = 0
    
    if label_num >= 2 * num_classes:
        # num_classes += 1 # Disable for 5-detector specific task
        pass
    classes_num = num_classes
    
    # Inherit last best model logic
    if config.get('inherit_best_model', False):
        import glob
        model_paths = glob.glob(os.path.join(results_dir, "*", "best_model.pth"))
        if model_paths:
            # Sort by modification time, newest first
            model_paths.sort(key=os.path.getmtime, reverse=True)
            last_best_model_path = model_paths[0]
            print(f"Inheriting last best model from: {last_best_model_path}")
            try:
                model.load_state_dict(torch.load(last_best_model_path))
                print("Successfully loaded last best model.")
            except Exception as e:
                print(f"Failed to load last best model: {e}")
        else:
            print("No previous best_model.pth found to inherit.")
            
    if os.path.exists(f"{save_dir}/best_model.pth"):
        print("Loading existing checkpoint from current save_dir...")
        model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
        
    print(f"Training started. Classes: {classes_num}, Labels: {label_num}")
    
    # Setup training log file
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Date: {currentDate}\n")
        f.write("="*50 + "\n")
        f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Time(s) | Status\n")
        f.write("-" * 80 + "\n")
        
    start_time = time.time()

    def compute_loss(images, labels, out_img, output_vec):
        full_int_img = out_img.sum(axis=(1, 2))
        normalized_out_img = out_img / (full_int_img[:, None, None] + 1e-8)
        
        batch_loss = torch.tensor(0.0, device=device)
        
        current_labels_image_tensors = torch.zeros((classes_num, PhaseMask[0], PhaseMask[1]), device=device, dtype=torch.float32)
        
        det_size = config.get('detector_size', 60)
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
            
            # Mapping: 20 labels → 5 detectors (4-to-1)
            base_det = sample_label // 4
            
            soft_label_on = config.get('soft_label_enabled', False)
            if soft_label_on:
                # Soft label mode: boundary labels (rem 0, 3) split energy between adjacent detectors
                rem = sample_label % 4
                sl_offset = config.get('soft_label_offset', 0.25)
                if rem == 1 or rem == 2:
                    offset = 0.0
                elif rem == 0:
                    offset = -sl_offset
                else:  # rem == 3
                    offset = sl_offset
                    
                target_float = max(0.0, min(base_det + offset, 4.0))
                floor_idx = int(math.floor(target_float))
                ceil_idx = int(math.ceil(target_float))
                
                if floor_idx == ceil_idx:
                    target_mask_weight[floor_idx] = 1.0
                else:
                    weight_ceil = target_float - floor_idx
                    target_mask_weight[floor_idx] = 1.0 - weight_ceil
                    target_mask_weight[ceil_idx] = weight_ceil
                    # Temperature scaling: flatten distribution (e.g. [0.75, 0.25] → ~[0.63, 0.37])
                    temp = config.get('soft_label_temperature', 0.5)
                    target_mask_weight = torch.pow(target_mask_weight, temp)
                    target_mask_weight = target_mask_weight / target_mask_weight.sum()
            else:
                # Hard label mode: each label maps to exactly one detector
                target_mask_weight[base_det] = 1.0
            
            target_mask = torch.einsum('c,chw->hw', target_mask_weight, current_labels_image_tensors)
            
            # Optimization: 
            # Pixel-wise MSE (loss) enforces uniform distribution within the detector, which is too strict and unnecessary.
            # We only care about the total energy in the detector (loss_vec).
            # We reduce the weight of pixel-wise loss or remove it.
            # Here we keep it but with very small weight, or rely mainly on loss_vec.
            
            # loss = loss_function(sample_img, target_mask) # Original Pixel-wise MSE
            
            # New Strategy: Rely primarily on Vector Loss (loss_vec) which allows any distribution within the detector box.
            # To prevent energy from scattering everywhere, we can add a regularization term minimizing energy OUTSIDE all detectors.
            
            # But for now, let's just trust loss_vec which we fixed.
            # We will use a mixed approach: 0.0 * image_loss + 1.0 * vector_loss
            # Actually, let's keep a small image loss just to guide spatial locality, but rely on vec loss.
            
            loss = loss_function(sample_img, target_mask)
            loss_vec = F.mse_loss(output_vec[i], target_mask_weight)
            det_weights = config.get('detector_loss_weight', None)
            w = det_weights[base_det] if det_weights else 1.0
            batch_loss += w * (0.05 * loss + 1.0 * loss_vec)

        return batch_loss

    def compute_acc(out_label, labels):
        # Classification acc
        base_intensity = 1/num_classes
        max_intensities, predicted = torch.max(out_label.data, 1)
        
        non_target_mask_bool = F.one_hot(predicted, num_classes=num_classes).bool()
        second_brightest_intensity, nd_predicted = torch.max(out_label.masked_fill(non_target_mask_bool, -float('inf')), dim=1)
        
        is_correct = torch.zeros_like(labels, dtype=torch.bool)
        
        max_intensities -= base_intensity
        second_brightest_intensity -= base_intensity
        
        for i in range(label_num):
            mask = (labels == i)
            if mask.any():
                # For accuracy, check if prediction matches the nearest integer bin
                # Using the same uniform center-based mapping
                target_float = (i - 1.5) / 4.0
                target_float = max(0.0, min(target_float, 4.0))
                target_cls = int(round(target_float))
                
                is_correct[mask] = (predicted[mask] == target_cls) & (max_intensities[mask] >= second_brightest_intensity[mask] * strict_accuracy_ratio)
                    
        return is_correct

    for epoch in range(epochs):
        epoch_start_time = time.time()
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
                
            # Check if padding is needed. gpu_transform already pads!
            # PADDINGx is (1200-1000)//2 = 100.
            # gpu_transform adds 100 padding.
            # If input is 1000, output is 1200.
            if images_squeezed.shape[-1] == PhaseMask[0]:
                 images_input = images_squeezed
            else:
                 images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))
            
            labels = labels.to(device)
            
            out_label, out_img, penalty_per_det = model(images_input)
            
            loss = compute_loss(images_input, labels, out_img, out_label)
            # Only penalize target detectors (not all detectors)
            target_det = labels // 4
            target_penalty = penalty_per_det[torch.arange(len(labels), device=device), target_det]
            loss += target_penalty.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_train_loss += loss.item()
            
            with torch.no_grad():
                batch_correct = compute_acc(out_label, labels).sum().item()
                current_batch_acc = batch_correct / labels.size(0)
                
                # Debug: Analyze Low Accuracy Batches
                if 0 and current_batch_acc < 0.5 and epoch > 0:
                    print(f"\n[DEBUG] Low Acc Batch ({current_batch_acc:.2f}) at Epoch {epoch+1}")
                    # Analyze Label Distribution
                    unique_labels, counts = torch.unique(labels, return_counts=True)
                    print(f"  Labels: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}")
                    # Check if it's mostly edge labels (0, 3, 4, 7, 8, 11...)
                    edge_labels = [0, 3, 4, 7, 8, 11, 12, 15, 16, 19]
                    edge_count = sum([counts[i].item() for i, l in enumerate(unique_labels) if l.item() in edge_labels])
                    print(f"  Edge Label Count: {edge_count}/{labels.size(0)}")
                    
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
                
                if images.shape[1] == 1:
                    images_squeezed = images.squeeze(1)
                else:
                    images_squeezed = images
                    
                images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))
                out_label, out_img, penalty_per_det = model(images_input)
                
                loss = compute_loss(images_input, labels, out_img, out_label)
                target_det = labels // 4
                target_penalty = penalty_per_det[torch.arange(len(labels), device=device), target_det]
                loss += target_penalty.sum()
                ep_test_loss += loss.item()
                
                correct += compute_acc(out_label, labels).sum().item()
                total += labels.size(0)

        avg_test_loss = ep_test_loss / len(testloader.dataset)
        test_acc = correct / total
        test_loss_hist.append(avg_test_loss)
        test_acc_hist.append(test_acc)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_metric = config.get('scheduler_metric', 'acc')
            if scheduler_metric == 'acc':
                scheduler.step(test_acc)
            else:
                scheduler.step(avg_test_loss)
        else:
            scheduler.step()
            
        print(f"Epoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.6f}")
        print(f"Val Loss: {avg_test_loss:.6f} | Val Acc: {test_acc:.6f}")
        
        status_msg = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Saved best model with acc: {best_acc:.6f}")
            status_msg = "Best Model Saved"
            
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Write to log file
        with open(log_file_path, "a") as f:
            f.write(f"{epoch+1:5d} | {avg_train_loss:10.6f} | {train_acc:9.4f} | {avg_test_loss:8.6f} | {test_acc:7.4f} | {current_lr:.2e} | {epoch_time:7.2f} | {status_msg}\n")
        
        # Plotting
        plot_loss_acc(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist, f"{save_dir}/loss_acc.png")
        
        # Save Outputs (if configured)
        if config.get("save_csv_logs", False):
            csv_log_dir = f"{save_dir}/csv_logs"
            os.makedirs(csv_log_dir, exist_ok=True)
            np.savetxt(f"{csv_log_dir}/mask_epoch_{epoch+1}.csv", model.phase_mask[0].detach().cpu().numpy(), delimiter=",")
            np.savetxt(f"{csv_log_dir}/detector_pos_epoch_{epoch+1}.csv", model.detector_pos.detach().cpu().numpy(), delimiter=",")

    elapsed_time = time.time() - start_time
    print(f"Total time for {epochs} epochs: {elapsed_time:.2f} seconds")
    
    with open(log_file_path, "a") as f:
        f.write("-" * 80 + "\n")
        f.write(f"Total training time: {elapsed_time:.2f} seconds\n")
        f.write(f"Best Validation Accuracy: {best_acc:.6f}\n")
        
    return elapsed_time

if __name__ == "__main__":
    try:
        # Use Standard ImageFolder (all 20 classes)
        train_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/train", transform=cpu_transform)
        val_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/val", transform=cpu_transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Experiment from Config
        exp_name = config.get('exp_name', 'default_run_5det')
        print(f"\n--- Running Experiment: {exp_name} ---")
        
        # Model (Defaults from global config)
        model = DNN()
        model = model.to(device)
        
        # Use MSELoss as in ai_refined
        criterion = torch.nn.MSELoss(reduction='sum')
        criterion = criterion.to(device)
        
        lr = config.get('learning_rate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        scheduler_metric = config.get('scheduler_metric', 'acc')
        scheduler_mode = 'max' if scheduler_metric == 'acc' else 'min'
        scheduler_patience = config.get('scheduler_patience', 4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=scheduler_patience)
        
        epochs = config.get('epochs', 5)
        num_classes = config.get('num_classes', 5)
        label_num = config.get('label_num', 20)
        strict_accuracy_ratio = config.get('strict_accuracy_ratio', 1)
        minus_mask_ratio = config.get('minus_mask_ratio', 0)
        
        time_1 = train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, 
                       num_classes=num_classes, epochs=epochs, 
                       strict_accuracy_ratio=strict_accuracy_ratio,
                       minus_mask_ratio=minus_mask_ratio,
                       label_num=label_num,
                       exp_name=exp_name)
        
        print(f"Time taken (1 epoch): {time_1:.2f}s")

        # Use the global save_dir initialized in train function.
        # We need to construct it here since we are in __main__ and time_1 only returns elapsed time.
        # The best way is to fetch the latest created folder that matches exp_name
        results_dir_final = config.get('results_dir', 'results')
        if not os.path.isabs(results_dir_final):
            results_dir_final = os.path.join(BASE_DIR, results_dir_final)
            
        import glob
        # Find the specific experiment directory just created
        possible_dirs = glob.glob(os.path.join(results_dir_final, f"{exp_name}_*"))
        if possible_dirs:
            possible_dirs.sort(key=os.path.getmtime, reverse=True)
            target_save_dir = possible_dirs[0]
        else:
            # Fallback to results_dir root if we can't find the specific one
            target_save_dir = results_dir_final
            
        debug_log_path = os.path.join(target_save_dir, 'debug.log')
        
        # Setup global debug.log redirection if not in a terminal
        # This will capture ALL print statements from this point forward (including subprocess outputs if printed)
        class LoggerWriter:
            def __init__(self, filename, stream):
                self.filename = filename
                self.stream = stream
            def write(self, message):
                self.stream.write(message)
                self.stream.flush()
                with open(self.filename, 'a', encoding='utf-8') as log_file:
                    log_file.write(message)
            def flush(self):
                self.stream.flush()

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = LoggerWriter(debug_log_path, sys.stdout)
        sys.stderr = LoggerWriter(debug_log_path, sys.stderr)

        # Run Evaluation if configured
        if config.get("run_evaluate_after_train", True):
            print("\n--- Running Evaluation ---")
            # Run as a subprocess to avoid context/memory issues and ensure it works cross-platform
            import subprocess
            
            # Determine capture mode based on debug flag
            debug_eval = config.get("debug_eval_subprocess", False)
            
            try:
                if debug_eval:
                    log_msg = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running evaluate.py...\n"
                    # Print to terminal, which will be caught by LoggerWriter and written to debug.log
                    print(log_msg.strip())
                    
                    result = subprocess.run(
                        [sys.executable, os.path.join(BASE_DIR, 'evaluate.py')], 
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    
                    # Print captured output so LoggerWriter can write it to debug.log
                    print(f"Return code: {result.returncode}")
                    print(f"STDOUT:\n{result.stdout}")
                    print(f"STDERR:\n{result.stderr}")
                    print("-" * 50)
                        
                    if result.returncode != 0:
                        print(f"Evaluation script failed with code {result.returncode}. See {debug_log_path} for details.")
                else:
                    # Silent mode: redirect subprocess output directly to debug.log
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running evaluate.py (silent mode)...")
                    with open(debug_log_path, 'a', encoding='utf-8') as f_log:
                        try:
                            result = subprocess.run(
                                [sys.executable, os.path.join(BASE_DIR, 'evaluate.py')], 
                                check=True,
                                stdout=f_log,
                                stderr=subprocess.STDOUT
                            )
                            print("Evaluation completed successfully.")
                        except subprocess.CalledProcessError as e:
                            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Evaluation failed with code {e.returncode}")
                            raise e
            except subprocess.CalledProcessError as e:
                print(f"Evaluation script failed: {e}")
            except Exception as e:
                print(f"Evaluation script failed to start: {e}")
            
        # Archive Results
        print("\n--- Archiving Results ---")
        try:
            import subprocess
            
            debug_eval = config.get("debug_eval_subprocess", False)
            if debug_eval:
                log_msg = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running archive_results.py...\n"
                print(log_msg.strip())
                
                result = subprocess.run(
                    [sys.executable, os.path.join(BASE_DIR, 'archive_results.py')], 
                    check=False,
                    capture_output=True,
                    text=True
                )
                
                print(f"Return code: {result.returncode}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                print("-" * 50)
                    
                if result.returncode != 0:
                    print(f"Archiving script failed with code {result.returncode}. See {debug_log_path} for details.")
                        
            else:
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running archive_results.py (silent mode)...")
                with open(debug_log_path, 'a', encoding='utf-8') as f_log:
                    try:
                        result = subprocess.run(
                            [sys.executable, os.path.join(BASE_DIR, 'archive_results.py')], 
                            check=True,
                            stdout=f_log,
                            stderr=subprocess.STDOUT
                        )
                        print("Archiving completed successfully.")
                    except subprocess.CalledProcessError as e:
                        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Archiving failed with code {e.returncode}")
                        raise e
        except Exception as e:
            print(f"Archiving script failed: {e}")
            
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
