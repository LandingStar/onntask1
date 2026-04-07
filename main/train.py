
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import copy
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
is_subprocess = False
is_imported = __name__ != "__main__"

# Parse command line arguments
if not is_imported:
    print(f"\n[INIT] train.py launched with args: {sys.argv}")

for arg in sys.argv[1:]:
    if arg == "--is-subprocess":
        is_subprocess = True
    elif arg.endswith('.json'):
        custom_config = arg
        if os.path.isabs(custom_config):
            config_path = custom_config
        else:
            config_path = os.path.join(BASE_DIR, custom_config)
        is_custom_config = True

if not is_imported:
    print(f"[INIT] is_subprocess: {is_subprocess}, is_custom_config: {is_custom_config}")

config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    if not is_imported:
        print(f"[INIT] Loaded config from {config_path}")
else:
    if not is_imported:
        print(f"[INIT] Config not found at {config_path}, using defaults")

# Intercept for batch training if configured
# BUT ONLY IF we are reading the default config.json AND we are not a subprocess AND not imported.
# Also ensure batch_train doesn't trigger if launched via torchrun (LOCAL_RANK exists)
if config.get('batch_train', False) and not is_subprocess and "LOCAL_RANK" not in os.environ and not is_imported:
    print(f"\n[INFO] 'batch_train' flag is true in config.json. Redirecting to batch_train.py...")
    import subprocess
    try:
        # Run batch_train.py in the same directory
        result = subprocess.run([sys.executable, os.path.join(BASE_DIR, 'batch_train.py')], check=False)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Batch training execution failed: {e}")
        sys.exit(1)
    # The script should always exit here because batch_train handles all the runs

# DDP Initialization
is_ddp = "LOCAL_RANK" in os.environ and not is_imported
if is_ddp:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    if local_rank == 0:
        print(f'Using DDP with {world_size} GPUs. Current Device: {device}')
else:
    local_rank = 0
    world_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using Device: ', device)

BATCH_SIZE = config.get('batch_size', 48)

# Auto-adjust BATCH_SIZE for training based on VRAM to prevent OOM
# Note: Since we use non_blocking async data transfer to GPU, smaller batches are actually better
# because they allow fine-grained overlapping of PCIe transfer and GPU FFT computation.
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    total_vram_gb = device_props.total_memory / (1024**3)
    sm_count = device_props.multi_processor_count
    
    optimal_async_batch = max(16, int(sm_count * 0.75))
    
    if total_vram_gb < 6.0:
        BATCH_SIZE = min(BATCH_SIZE, 16)
    elif total_vram_gb < 10.0:
        BATCH_SIZE = min(BATCH_SIZE, 32)
    elif total_vram_gb < 16.0:
        BATCH_SIZE = min(BATCH_SIZE, max(48, optimal_async_batch))
    else:
        # If VRAM is abundant, cap at the SM-optimized async batch size rather than a hard 64
        BATCH_SIZE = min(BATCH_SIZE, optimal_async_batch * 2)
    # else keep config's BATCH_SIZE (could be 256 or higher)
    print(f"Auto-adjusted training BATCH_SIZE to {BATCH_SIZE} based on {total_vram_gb:.1f}GB VRAM and {sm_count} SMs.")

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
    v2.ToDtype(torch.float32, scale=True),
    v2.Grayscale(num_output_channels=1),
])

# Read transform intensity from config, default to 1.0
transform_intensity = config.get('transform_intensity', 1.0)

# Build transform list conditionally
gpu_transform_train_list = [
    v2.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True)
]

gpu_transform_val = v2.Compose([
    v2.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True)
])

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
    
    gpu_transform_train_list.extend([
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
    gpu_transform_train_list.append(v2.Pad([PADDINGx, PADDINGx, PADDINGy, PADDINGy]))

gpu_transform = v2.Compose(gpu_transform_train_list)

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
    def __init__(self, num_layers=None, wl_param=None, PhaseMask_param=None, pixel_size_param=None,
                 distance_between_layers=None, distance_to_detectors=None, 
                 num_classes=None, train_detector_pos=None):
        super(DNN, self).__init__()
        
        # Dynamically fetch from global scope/config if not explicitly provided
        # This prevents issues during inline batch execution where defaults are bound at module import time
        num_layers = num_layers if num_layers is not None else config.get('num_layers', 1)
        wl_param = wl_param if wl_param is not None else wl
        PhaseMask_param = PhaseMask_param if PhaseMask_param is not None else PhaseMask
        pixel_size_param = pixel_size_param if pixel_size_param is not None else PIXEL_SIZE
        distance_between_layers = distance_between_layers if distance_between_layers is not None else config.get('distance_between_layers', 0.2)
        distance_to_detectors = distance_to_detectors if distance_to_detectors is not None else config.get('distance_to_detectors', 0.2)
        num_classes = num_classes if num_classes is not None else config.get('num_classes', 5)
        train_detector_pos = train_detector_pos if train_detector_pos is not None else config.get('train_detector_pos', True)
        
        self.phase_mask = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(PhaseMask_param, dtype=torch.float32)) for _ in range(num_layers)
        ])
        
        self.detector_mask = torch.nn.Parameter(torch.ones(num_classes))
        self.detector_minus = torch.nn.Parameter(torch.zeros(num_classes))
        
        # Initialize detector positions
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
        
        self.diffractive_layers = torch.nn.ModuleList([Diffractive_Layer(wl_param, PhaseMask_param, pixel_size_param, distance_between_layers) for _ in range(num_layers)])
        self.last_diffractive_layer = Propagation_Layer(wl_param, PhaseMask_param, pixel_size_param, distance_to_detectors)

        # Misalignment Layer Simulation
        self.simulate_misalignment = config.get('simulate_misalignment', False)
        self.misalignment_translation_max = config.get('misalignment_translation_max_pixels', 2.0)
        self.misalignment_rotation_max = config.get('misalignment_rotation_max_degrees', 0.5)
        self.misalignment_tilt_max = config.get('misalignment_tilt_max_degrees', 0.1)
        
        # Pre-compute coordinate grid for tilt phase gradient (efficiency)
        # We only need this if tilt is enabled and > 0
        if self.simulate_misalignment and self.misalignment_tilt_max > 0:
            # Physical coordinates (x, y) based on pixel size
            # Center is at (0, 0)
            x_range = (torch.arange(PhaseMask_param[1], dtype=torch.float32, device=device) - PhaseMask_param[1] / 2) * pixel_size_param
            y_range = (torch.arange(PhaseMask_param[0], dtype=torch.float32, device=device) - PhaseMask_param[0] / 2) * pixel_size_param
            self.yy, self.xx = torch.meshgrid(y_range, x_range, indexing='ij')
            # Wavenumber k = 2pi / lambda
            self.k = 2 * np.pi / wl_param

    def apply_misalignment(self, E):
        if not self.training or not self.simulate_misalignment:
            return E
            
        batch_size = E.shape[0]
        
        # Handle shape differences: E could be [B, H, W] or [B, C, H, W]
        has_channel_dim = len(E.shape) == 4
        if has_channel_dim:
            h_idx, w_idx = 2, 3
        else:
            h_idx, w_idx = 1, 2
        
        # 1. Random translation (shift in x and y)
        shift_x = (torch.rand(batch_size, device=device) * 2 - 1) * self.misalignment_translation_max
        shift_y = (torch.rand(batch_size, device=device) * 2 - 1) * self.misalignment_translation_max
        
        # 2. In-plane rotation (around Z axis)
        angle_rad = (torch.rand(batch_size, device=device) * 2 - 1) * self.misalignment_rotation_max * np.pi / 180.0
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # Normalized translation for grid_sample (-1 to 1)
        tx = shift_x / (E.shape[w_idx] / 2)
        ty = shift_y / (E.shape[h_idx] / 2)
        
        affine_matrix = torch.zeros(batch_size, 2, 3, device=device)
        affine_matrix[:, 0, 0] = cos_a
        affine_matrix[:, 0, 1] = -sin_a
        affine_matrix[:, 0, 2] = tx
        affine_matrix[:, 1, 0] = sin_a
        affine_matrix[:, 1, 1] = cos_a
        affine_matrix[:, 1, 2] = ty
        
        # Grid sample requires real tensors (Channels=2 for complex)
        # We need to temporarily unsqueeze the channel dimension if it doesn't exist
        E_working = E if has_channel_dim else E.unsqueeze(1)
        
        E_real = E_working.real # [B, 1, H, W]
        E_imag = E_working.imag
        E_concat = torch.cat([E_real, E_imag], dim=1) # [B, 2, H, W]
        
        grid = F.affine_grid(affine_matrix, E_concat.size(), align_corners=False)
        E_warped = F.grid_sample(E_concat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        E_out = torch.complex(E_warped[:, 0:1], E_warped[:, 1:2]) # Keep the channel dim [B, 1, H, W]
        
        # Remove channel dim if it wasn't there originally
        if not has_channel_dim:
            E_out = E_out.squeeze(1)
        
        # 3. Out-of-plane tilt (around X and Y axis -> Phase gradient)
        if self.misalignment_tilt_max > 0:
            # Tilt angles in radians
            tilt_x = (torch.rand(batch_size, 1, 1, device=device) * 2 - 1) * self.misalignment_tilt_max * np.pi / 180.0
            tilt_y = (torch.rand(batch_size, 1, 1, device=device) * 2 - 1) * self.misalignment_tilt_max * np.pi / 180.0
            
            # Phase gradient = k * (x * sin(tilt_y) + y * sin(tilt_x))
            # For small angles, sin(theta) ≈ theta
            phase_gradient = self.k * (self.xx.unsqueeze(0) * tilt_y + self.yy.unsqueeze(0) * tilt_x)
            
            # Add channel dimension to phasor if E_out has it
            if has_channel_dim:
                phase_gradient = phase_gradient.unsqueeze(1)
                
            tilt_phasor = torch.exp(1j * phase_gradient)
            E_out = E_out * tilt_phasor
            
        return E_out

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
            if index > 0 and self.simulate_misalignment:
                E = self.apply_misalignment(E)
                
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
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    model_to_save = model.module if isinstance(model, DDP) else model
    
    # Save Dir Logic
    results_dir = config.get('results_dir', 'results')
    # Resolve results_dir relative to BASE_DIR if relative
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(BASE_DIR, results_dir)
        
    save_dir = os.path.join(results_dir, f"{exp_name}_{currentDate}")
    
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

        # Save Config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            # Don't save batch_train=False if it was originally true, but it's safe to just dump current
            json.dump(config, f, indent=4)
            
        # Setup CSV logger for metrics
        metrics_csv_path = os.path.join(save_dir, 'metrics.csv')
        with open(metrics_csv_path, 'w') as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc,learning_rate\n")
    
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    best_score = -float('inf')
    best_acc_for_log = 0.0
    
    # Dynamic Spatial Mask Loss Weight parameters
    spatial_mask_weight = float(config.get('spatial_mask_loss_weight', 0.05))
    target_ratio = config.get('auto_spatial_mask_target_ratio', 0.1)
    
    aggressive_intensity_optimization = config.get('aggressive_intensity_optimization', True)
    aggressive_acc_threshold = config.get('aggressive_acc_threshold', 0.99)
    aggressive_weight_multiplier = config.get('aggressive_weight_multiplier', 5.0)
    
    if label_num >= 2 * num_classes:
        # num_classes += 1 # Disable for 5-detector specific task
        pass
    classes_num = num_classes
    
    if config.get('inherit_best_model', False):
        import glob
        inherit_model_path = config.get('inherit_model_path', "")
        
        # 决定匹配的名称：如果有设置则用设置的，否则用当前的 exp_name
        target_name = inherit_model_path if inherit_model_path else exp_name
        
        # 尝试精确匹配前缀
        search_pattern = os.path.join(results_dir, f"{target_name}_*", "best_model.pth")
        model_paths = glob.glob(search_pattern)
        
        # 尝试更宽松的匹配
        if not model_paths:
            model_paths = glob.glob(os.path.join(results_dir, f"*{target_name}*", "best_model.pth"))
            
        # 如果依然没找到，回退到匹配所有最新的模型，并打印警告
        if not model_paths:
            fallback_paths = glob.glob(os.path.join(results_dir, "*", "best_model.pth"))
            if fallback_paths:
                if is_main_process:
                    print(f"WARNING: Could not find previous model for '{target_name}'. Falling back to the absolute newest model in results_dir.")
                model_paths = fallback_paths
            
        if model_paths:
            # Sort by modification time, newest first
            model_paths.sort(key=os.path.getmtime, reverse=True)
            last_best_model_path = model_paths[0]
            if is_main_process:
                print(f"Inheriting last best model from: {last_best_model_path}")
            try:
                # Need map_location to load onto correct device
                model_to_save.load_state_dict(torch.load(last_best_model_path, map_location=device))
                if is_main_process:
                    print("Successfully loaded last best model.")
            except Exception as e:
                if is_main_process:
                    print(f"Failed to load last best model: {e}")
        else:
            if is_main_process:
                print("No previous best_model.pth found to inherit.")
                
    if os.path.exists(f"{save_dir}/best_model.pth"):
        if is_main_process:
            print("Loading existing checkpoint from current save_dir...")
        model_to_save.load_state_dict(torch.load(f"{save_dir}/best_model.pth", map_location=device))
        
    if is_main_process:
        print(f"Training started. Classes: {classes_num}, Labels: {label_num}")
    
    # Setup training log file
    log_file_path = os.path.join(save_dir, "training_log.txt")
    if is_main_process:
        with open(log_file_path, "w") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Date: {currentDate}\n")
            f.write("="*50 + "\n")
            f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Int.Ratio | MaskWt | Time(s) | Status\n")
            f.write("-" * 80 + "\n")
        
    start_time = time.time()

    def compute_loss(images, labels, out_img, output_vec, current_spatial_weight, is_aggressive=False):
        det_shape = config.get('detector_shape', 'circle')
        det_size_config = config.get('detector_size', 60)
        
        if det_shape == 'square':
            det_area = det_size_config * det_size_config
        else: # circle
            det_area = np.pi * (det_size_config / 2)**2
            
        global_area = PhaseMask[0] * PhaseMask[1]
        area_ratio = global_area / det_area

        # Vectorized label mapping
        base_det = labels // 4
        
        soft_label_on = config.get('soft_label_enabled', False)
        batch_size = images.size(0)
        target_mask_weight = torch.full((batch_size, classes_num), minus_mask_ratio, device=device, dtype=torch.float32)
        
        if soft_label_on:
            # We process this row by row for now since soft labels have complex logic, 
            # but it's much faster than doing the whole loss loop row by row.
            sl_offset = config.get('soft_label_offset', 0.25)
            temp = config.get('soft_label_temperature', 0.5)
            for i in range(batch_size):
                rem = labels[i].item() % 4
                if rem == 1 or rem == 2:
                    offset = 0.0
                elif rem == 0:
                    offset = -sl_offset
                else:  # rem == 3
                    offset = sl_offset
                    
                target_float = max(0.0, min(base_det[i].item() + offset, 4.0))
                floor_idx = int(math.floor(target_float))
                ceil_idx = int(math.ceil(target_float))
                
                if floor_idx == ceil_idx:
                    target_mask_weight[i, floor_idx] = 1.0
                else:
                    weight_ceil = target_float - floor_idx
                    target_mask_weight[i, floor_idx] = 1.0 - weight_ceil
                    target_mask_weight[i, ceil_idx] = weight_ceil
                    
                    target_mask_weight[i] = torch.pow(target_mask_weight[i], temp)
                    target_mask_weight[i] = target_mask_weight[i] / target_mask_weight[i].sum()
        else:
            # Fast vectorized hard label mapping
            target_mask_weight.scatter_(1, base_det.unsqueeze(1), 1.0)
            
        # 1. Vector Loss (Batch-wise)
        # Using reduction='none' so we can apply individual detector weights
        loss_vec = F.mse_loss(output_vec, target_mask_weight, reduction='none').mean(dim=1)
        
        # 2. Target Energy Penalty (Batch-wise)
        current_target_intensity_ratio = (output_vec * target_mask_weight).sum(dim=1) * area_ratio
        
        if is_aggressive:
            # Switch to LeakyReLU (or softplus) to keep pushing intensity even after target_ratio is met.
            # Normal ReLU clips negative values to 0 (meaning zero loss/gradient once target is met).
            # LeakyReLU with slope 0.1 provides continuous gradient to keep improving intensity indefinitely.
            aggressive_slope = config.get('aggressive_leaky_slope', 0.1)
            loss_energy = F.leaky_relu(target_ratio - current_target_intensity_ratio, negative_slope=aggressive_slope)
        else:
            loss_energy = F.relu(target_ratio - current_target_intensity_ratio)
        
        # Apply detector weights
        det_weights_list = config.get('detector_loss_weight', None)
        if det_weights_list:
            det_weights_tensor = torch.tensor(det_weights_list, device=device, dtype=torch.float32)
            w = det_weights_tensor[base_det]
        else:
            w = torch.ones(batch_size, device=device, dtype=torch.float32)
            
        # Composite Loss
        batch_loss = (w * (1.0 * loss_vec + current_spatial_weight * loss_energy)).sum()

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
        
        # Determine current epoch's spatial mask weight
        current_epoch_spatial_weight = spatial_mask_weight
        is_aggressive = False
        if aggressive_intensity_optimization and epoch > 0:
            # Use previous epoch's test accuracy to decide
            if len(test_acc_hist) > 0 and test_acc_hist[-1] >= aggressive_acc_threshold:
                current_epoch_spatial_weight = spatial_mask_weight * aggressive_weight_multiplier
                is_aggressive = True
                
        for images, labels in pbar:
            optimizer.zero_grad()
            
            # Asynchronous data transfer to GPU (overlap compute & transfer)
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            
            with torch.no_grad():
                if gpu_transform:
                    images_aug = gpu_transform(images)
                else:
                    images_aug = images
            
            if images_aug.shape[1] == 1:
                images_squeezed = images_aug.squeeze(1)
            else:
                images_squeezed = images_aug
                
            # Check if padding is needed. gpu_transform already pads!
            # PADDINGx is (1200-1000)//2 = 100.
            # gpu_transform adds 100 padding.
            # If input is 1000, output is 1200.
            # Apply padding (can also be done async if using appropriate padding module)
            if images_squeezed.shape[-1] == PhaseMask[0]:
                 images_input = images_squeezed
            else:
                 images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))
            
            out_label, out_img, penalty_per_det = model(images_input)
            
            loss = compute_loss(images_input, labels, out_img, out_label, current_epoch_spatial_weight, is_aggressive)
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
        
        if dist.is_initialized():
            train_metrics = torch.tensor([ep_train_loss, correct, total], dtype=torch.float64, device=device)
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            ep_train_loss = train_metrics[0].item()
            correct = train_metrics[1].item()
            total = train_metrics[2].item()

        avg_train_loss = ep_train_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(train_acc)
        
        # Validation
        ep_test_loss = 0
        model.eval()
        correct = 0
        total = 0
        
        # To track average intensity ratio in target detector
        val_target_intensity_sum = 0.0
        
        with torch.no_grad():
            for images, labels in testloader:
                # Asynchronous transfer
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)
                
                # Apply validation transforms on GPU (Resize)
                if gpu_transform_val:
                    images_aug = gpu_transform_val(images)
                else:
                    images_aug = images
                
                if images_aug.shape[1] == 1:
                    images_squeezed = images_aug.squeeze(1)
                else:
                    images_squeezed = images_aug
                    
                if images_squeezed.shape[-1] == PhaseMask[0]:
                    images_input = images_squeezed
                else:
                    images_input = F.pad(images_squeezed, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))
                    
                out_label, out_img, penalty_per_det = model(images_input)
                
                loss = compute_loss(images_input, labels, out_img, out_label, current_epoch_spatial_weight, is_aggressive)
                target_det = labels // 4
                target_penalty = penalty_per_det[torch.arange(len(labels), device=device), target_det]
                loss += target_penalty.sum()
                ep_test_loss += loss.item()
                
                correct += compute_acc(out_label, labels).sum().item()
                total += labels.size(0)
                
                # Calculate target detector AVERAGE intensity ratio for this batch
                # out_label is the ratio of integral sum.
                # To get average intensity: (Detector_Sum / Detector_Area) / (Global_Sum / Global_Area)
                # Since out_label = Detector_Sum / Global_Sum
                # Avg_Intensity_Ratio = out_label * (Global_Area / Detector_Area)
                target_intensities = out_label[torch.arange(len(labels), device=device), target_det]
                
                det_shape = config.get('detector_shape', 'circle')
                det_size_config = config.get('detector_size', 60)
                if det_shape == 'square':
                    det_area = det_size_config * det_size_config
                else: # circle
                    det_area = np.pi * (det_size_config / 2)**2
                
                global_area = PhaseMask[0] * PhaseMask[1]
                area_ratio = global_area / det_area
                
                target_avg_intensity_ratio = target_intensities * area_ratio
                val_target_intensity_sum += target_avg_intensity_ratio.sum().item()

        if dist.is_initialized():
            val_metrics = torch.tensor([ep_test_loss, correct, total, val_target_intensity_sum], dtype=torch.float64, device=device)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
            ep_test_loss = val_metrics[0].item()
            correct = val_metrics[1].item()
            total = val_metrics[2].item()
            val_target_intensity_sum = val_metrics[3].item()

        avg_test_loss = ep_test_loss / total if total > 0 else 0
        test_acc = correct / total if total > 0 else 0
        
        # Calculate dataset-wide average intensity ratio
        # Instead of averaging the batch averages, we calculate the total sum of intensities 
        # divided by total samples to get the true dataset-wide average.
        avg_target_intensity_ratio = val_target_intensity_sum / total if total > 0 else 0
        
        test_loss_hist.append(avg_test_loss)
        test_acc_hist.append(test_acc)
        
        # Evaluate if this is the best model
        status_msg = ""
        # The score metric balances accuracy and spatial concentration
        # We value both. Give higher score for reaching intensity goals while maintaining acc.
        # Example: 0.90 acc + 0.15 ratio = 1.05. 0.88 acc + 0.20 ratio = 1.08. 
        # We weight acc more heavily to ensure it doesn't plummet just for energy.
        acc_importance = config.get('best_model_acc_weight', 1.0)
        int_importance = config.get('best_model_intensity_weight', 0.5)
        current_score = (test_acc * acc_importance) + (avg_target_intensity_ratio * int_importance)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_metric = config.get('scheduler_metric', 'acc')
            if scheduler_metric == 'acc':
                # Use current_score instead of pure acc, so that LR doesn't drop 
                # if intensity is improving while acc is plateaued.
                scheduler.step(current_score)
            else:
                scheduler.step(avg_test_loss)
        else:
            scheduler.step()
            
        if is_main_process:
            print(f"Epoch {epoch+1} Results:")
            if is_aggressive:
                print(f"** AGGRESSIVE OPTIMIZATION ACTIVE ** (Acc >= {aggressive_acc_threshold}, Weight: x{aggressive_weight_multiplier})")
            print(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.6f}")
            print(f"Val Loss: {avg_test_loss:.6f} | Val Acc: {test_acc:.6f}")
            print(f"Avg Target Intensity Ratio: {avg_target_intensity_ratio:.4f}")
            print(f"Composite Score: {current_score:.4f}")

        if current_score > best_score:
            best_score = current_score
            best_acc_for_log = test_acc
            if is_main_process:
                torch.save(model_to_save.state_dict(), f"{save_dir}/best_model.pth")
                print(f"Saved best model with acc: {test_acc:.6f}, intensity: {avg_target_intensity_ratio:.4f}, score: {best_score:.4f}")
            status_msg = "Best Model Saved"
            
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        if is_main_process:
            # Log metrics to CSV
            with open(metrics_csv_path, 'a') as f:
                f.write(f"{epoch + 1},{avg_train_loss:.6f},{avg_test_loss:.6f},{train_acc:.6f},{test_acc:.6f},{current_lr:.2e}\n")
            
            # Write to log file
            with open(log_file_path, "a") as f:
                f.write(f"{epoch+1:5d} | {avg_train_loss:10.6f} | {train_acc:9.4f} | {avg_test_loss:8.6f} | {test_acc:7.4f} | {current_lr:.2e} | {avg_target_intensity_ratio:7.4f} | {current_epoch_spatial_weight:7.4f} | {epoch_time:7.2f} | {status_msg}\n")
            
            # Plotting - move to a background thread to prevent blocking the next epoch
            import threading
            plot_thread = threading.Thread(
                target=plot_loss_acc, 
                args=(list(train_loss_hist), list(test_loss_hist), list(train_acc_hist), list(test_acc_hist), f"{save_dir}/loss_acc.png")
            )
            plot_thread.start()
            
            # Save Outputs (if configured)
            if config.get("save_csv_logs", False):
                # Only save CSV logs at the last epoch or if explicitly needed frequently, as np.savetxt is extremely slow
                if epoch == epochs - 1 or config.get("save_csv_logs_every_epoch", False):
                    csv_log_dir = f"{save_dir}/csv_logs"
                    os.makedirs(csv_log_dir, exist_ok=True)
                    np.savetxt(f"{csv_log_dir}/mask_epoch_{epoch+1}.csv", model_to_save.phase_mask[0].detach().cpu().numpy(), delimiter=",")
                    np.savetxt(f"{csv_log_dir}/detector_pos_epoch_{epoch+1}.csv", model_to_save.detector_pos.detach().cpu().numpy(), delimiter=",")

    elapsed_time = time.time() - start_time
    if is_main_process:
        print(f"Total time for {epochs} epochs: {elapsed_time:.2f} seconds")
        
        with open(log_file_path, "a") as f:
            f.write("-" * 80 + "\n")
            f.write(f"Total training time: {elapsed_time:.2f} seconds\n")
            f.write(f"Best Validation Accuracy (associated with best score): {best_acc_for_log:.6f}\n")
            f.write(f"Best Score: {best_score:.6f}\n")
        
    return elapsed_time

import io
from PIL import Image

class InMemoryImageFolder(torchvision.datasets.ImageFolder):
    # Class-level cache dictionary to share memory across instances and imports within the same process
    # Key: absolute path to the dataset directory (e.g., '/path/to/dataset/train')
    # Value: list of tuples (img_bytes, target)
    _SHARED_CACHE = {}

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        abs_root = os.path.abspath(root)
        
        if abs_root in self.__class__._SHARED_CACHE:
            print(f"Dataset already in RAM cache. Reusing {len(self.samples)} images from {root}...")
            self.samples_in_memory = self.__class__._SHARED_CACHE[abs_root]
        else:
            self.samples_in_memory = []
            print(f"Loading {len(self.samples)} images into memory from {root}...")
            for path, target in tqdm(self.samples, desc=f"Loading {os.path.basename(root)} to RAM"):
                with open(path, 'rb') as f:
                    # We store the raw bytes instead of decoded Tensors/PIL Images.
                    # This prevents RAM explosion (10GB disk space -> 400GB+ float32 RAM)
                    # while still completely bypassing remote disk I/O latency.
                    self.samples_in_memory.append((f.read(), target))
            print("Done loading into memory.")
            # Store in class-level cache for future instantiations in the same process
            self.__class__._SHARED_CACHE[abs_root] = self.samples_in_memory

    def __getitem__(self, index):
        img_bytes, target = self.samples_in_memory[index]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

def get_dir_size(path):
    total_size = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def get_available_memory():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        import platform
        if platform.system() == 'Windows':
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys
        elif platform.system() == 'Linux':
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemAvailable' in line:
                            return int(line.split()[1]) * 1024
            except:
                pass
    return None

def main():
    # 1. We need to clear/reset the state for consecutive inline runs
    global config, BATCH_SIZE, IMG_SIZE, PhaseMask, PIXEL_SIZE, wl, PADDINGx, PADDINGy, dataset_name, gpu_transform, gpu_transform_val, cpu_transform, device, is_ddp, local_rank, world_size, detector_pos_xy
    
    # Check if a custom config path was passed as an argument
    is_custom_config = False
    is_subprocess = False
    config_path = os.path.join(BASE_DIR, 'config.json')
    
    for arg in sys.argv[1:]:
        if arg == "--is-subprocess":
            is_subprocess = True
        elif arg.endswith('.json'):
            custom_config = arg
            if os.path.isabs(custom_config):
                config_path = custom_config
            else:
                config_path = os.path.join(BASE_DIR, custom_config)
            is_custom_config = True
            
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Intercept for batch training if configured
    # BUT ONLY IF we are reading the default config.json AND we are not a subprocess
    # Also ensure batch_train doesn't trigger if launched via torchrun (LOCAL_RANK exists)
    if config.get('batch_train', False) and not is_subprocess and "LOCAL_RANK" not in os.environ and not is_custom_config:
        print(f"\n[INFO] 'batch_train' flag is true in config.json. Redirecting to batch_train.py...")
        import subprocess
        try:
            # Run batch_train.py in the same directory
            result = subprocess.run([sys.executable, os.path.join(BASE_DIR, 'batch_train.py')], check=False)
            sys.exit(result.returncode)
        except Exception as e:
            print(f"Batch training execution failed: {e}")
            sys.exit(1)

    # DDP Initialization
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        # Prevent re-initializing process group if already initialized (inline batch runs)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = config.get('batch_size', 48)

    # Auto-adjust BATCH_SIZE for training based on VRAM to prevent OOM
    # Note: Since we use non_blocking async data transfer to GPU, smaller batches are actually better
    # because they allow fine-grained overlapping of PCIe transfer and GPU FFT computation.
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        total_vram_gb = device_props.total_memory / (1024**3)
        sm_count = device_props.multi_processor_count
        
        # Calculate optimal async batch size based on SM count (Streaming Multiprocessors)
        # To keep all SMs busy but maintain high-frequency PCIe transfers, 
        # a good rule of thumb is 0.5 to 1 batch item per SM.
        # For example, RTX 3090 has 82 SMs -> optimal async batch ~41-82
        # A100 has 108 SMs -> optimal async batch ~54-108
        optimal_async_batch = max(16, int(sm_count * 0.75))
        
        if total_vram_gb < 6.0:
            BATCH_SIZE = min(BATCH_SIZE, 16)
        elif total_vram_gb < 10.0:
            BATCH_SIZE = min(BATCH_SIZE, 32)
        elif total_vram_gb < 16.0:
            BATCH_SIZE = min(BATCH_SIZE, max(48, optimal_async_batch))
        else:
            # If VRAM is abundant, cap at the SM-optimized async batch size rather than a hard 64
            BATCH_SIZE = min(BATCH_SIZE, optimal_async_batch * 2)
            
        print(f"Auto-adjusted training BATCH_SIZE to {BATCH_SIZE} (VRAM: {total_vram_gb:.1f}GB, SMs: {sm_count}).")
            
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

    # Transforms
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=1),
    ])

    # Read transform intensity from config, default to 1.0
    transform_intensity = config.get('transform_intensity', 1.0)

    # Build transform list conditionally
    gpu_transform_train_list = [
        v2.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True)
    ]

    gpu_transform_val = v2.Compose([
        v2.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True)
    ])

    if transform_intensity > 0:
        base_rotation = 1
        base_translate = 0.03
        base_scale_range = 0.03
        base_jitter = 0.4
        base_perspective_scale = 0.2
        base_perspective_p = 0.3
        base_sharpness_factor = 2.0
        base_sharpness_p = 0.5
        
        gpu_transform_train_list.extend([
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
            v2.RandomAdjustSharpness(sharpness_factor=1.0 + (base_sharpness_factor - 1.0) * transform_intensity, p=min(1.0, base_sharpness_p * transform_intensity)),
        ])
    else:
        gpu_transform_train_list.append(v2.Pad([PADDINGx, PADDINGx, PADDINGy, PADDINGy]))

    gpu_transform = v2.Compose(gpu_transform_train_list)

    # Initialize detector positions
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
            
    try:
        # Use Standard ImageFolder or InMemoryImageFolder based on config
        use_in_memory = config.get('in_memory_dataset', True) # Enabled by default for faster I/O
        
        if use_in_memory:
            train_dir = f"{dataset_name}/train"
            val_dir = f"{dataset_name}/val"
            dataset_size = get_dir_size(train_dir) + get_dir_size(val_dir)
            avail_mem = get_available_memory()

            if avail_mem is not None:
                # Require dataset size + 20% overhead + 2GB safety margin
                required_mem = dataset_size * 1.2 + (2 * 1024**3)
                print(f"[Memory Check] Dataset Size: {dataset_size / 1024**3:.2f} GB, Available RAM: {avail_mem / 1024**3:.2f} GB")
                if required_mem > avail_mem:
                    print(f"[Memory Check] Insufficient RAM! Required ~{required_mem / 1024**3:.2f} GB. Falling back to disk I/O.")
                    use_in_memory = False
                else:
                    print("[Memory Check] Sufficient RAM available. Proceeding with in-memory dataset.")
            else:
                print("[Memory Check] Could not determine available RAM. Attempting in-memory load anyway...")

        if use_in_memory:
            print("Using InMemoryImageFolder to bypass I/O bottlenecks...")
            train_dataset = InMemoryImageFolder(f"{dataset_name}/train", transform=cpu_transform)
            val_dataset = InMemoryImageFolder(f"{dataset_name}/val", transform=cpu_transform)
        else:
            train_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/train", transform=cpu_transform)
            val_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/val", transform=cpu_transform)
        
        # DataLoader configurations for better performance
        # Detect available CPU cores
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
        except NotImplementedError:
            cpu_count = 4
            
        # Dynamically allocate workers if not explicitly set in config
        # Rule of thumb: leave 1-2 cores for OS, split the rest (more for train, fewer for val)
        # Cap train workers at 8-16 to avoid excessive memory overhead
        available_workers = max(1, cpu_count - 2)
        
        default_train_workers = min(16, max(2, int(available_workers * 0.75)))
        default_val_workers = max(1, int(available_workers * 0.25))
        
        train_num_workers = config.get('train_num_workers', default_train_workers)
        val_num_workers = config.get('val_num_workers', default_val_workers)
        
        print(f"CPU Cores detected: {cpu_count}. Using {train_num_workers} workers for Train, {val_num_workers} for Val.")

        train_prefetch = config.get('prefetch_factor', 2) if train_num_workers > 0 else None
        val_prefetch = config.get('prefetch_factor', 2) if val_num_workers > 0 else None
        
        # DDP Samplers
        train_sampler = DistributedSampler(train_dataset) if is_ddp else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=(train_sampler is None), # Shuffle must be False if using Sampler
            sampler=train_sampler,
            num_workers=train_num_workers, 
            pin_memory=True,
            prefetch_factor=train_prefetch,
            # Disable persistent_workers when using InMemoryImageFolder in inline batch_trains
            # because the shared memory file descriptors break across runs and cause ConnectionResetError
            persistent_workers=False,
            drop_last=True # Helps with memory alignment and batch processing
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            sampler=val_sampler,
            num_workers=val_num_workers, 
            pin_memory=True,
            prefetch_factor=val_prefetch,
            persistent_workers=False,
            drop_last=False
        )
        
        # Experiment from Config
        exp_name = config.get('exp_name', 'default_run_5det')
        if local_rank == 0:
            print(f"\n--- Running Experiment: {exp_name} ---")
        
        # Model (Defaults from global config)
        model = DNN()
        model = model.to(device)
        
        if is_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            # When using DDP, model parameters are accessed via model.module
            model_to_save = model.module
        else:
            model_to_save = model
        
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
        
        if local_rank == 0:
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
                # Import evaluate directly to share memory (like InMemoryImageFolder)
                try:
                    import evaluate as eval_module
                    
                    debug_eval = config.get("debug_eval_subprocess", False)
                    if debug_eval:
                        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running evaluation inline...")
                        # Pass the already loaded val_dataset to save RAM and time
                        eval_module.evaluate(custom_val_dataset=val_dataset)
                    else:
                        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running evaluation inline (silent mode)...")
                        # Temporarily redirect stdout to log file
                        with open(debug_log_path, 'a', encoding='utf-8') as f_log:
                            original_stdout = sys.stdout
                            sys.stdout = f_log
                            try:
                                eval_module.evaluate(custom_val_dataset=val_dataset)
                            finally:
                                sys.stdout = original_stdout
                        print("Evaluation completed successfully.")
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                
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
        
        if is_ddp:
            dist.destroy_process_group()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if 'is_ddp' in locals() and is_ddp and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
