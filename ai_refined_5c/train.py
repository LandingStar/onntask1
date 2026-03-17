
print("SCRIPT STARTING")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import copy
import math
import os
import time
print("Importing torch...")
import torch
print("Importing torchvision...")
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
print("Imports done.")
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm
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

# Constants and Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device: ', device)

BATCH_SIZE = config.get('batch_size', 48)
IMG_SIZE = config.get('img_size', [1000, 1000])
PhaseMask = config.get('phase_mask_size', [1200, 1200])
PIXEL_SIZE = config.get('pixel_size', 8e-6)
wl = config.get('wavelength', 532e-9)
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 4 # Note: This was // 4 in original code, but typical is // 2. Assuming // 4 is intentional for 5c?
# Wait, PhaseMask=1200, IMG_SIZE=1000. (1200-1000)/2 = 100.
# If // 4, it is 50. Total pad is 50+50 = 100. So 1000+100=1100 != 1200.
# The original code had:
# PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 4
# PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 4
# And in gpu_transform: v2.Pad([PADDINGx, PADDINGx, PADDINGy, PADDINGy])
# So it pads 50 on left, 50 on right. Total width = 1000 + 50 + 50 = 1100.
# But Diffractive Layer uses PhaseMask size (1200).
# This means there is a mismatch?
# Or maybe gpu_transform is applied, and then in training loop:
# images_input = F.pad(images_squeezed, pad=(PADDINGx, PADDINGx, PADDINGy, PADDINGy))
# If gpu_transform pads 50+50=100 (to 1100), and then F.pad pads 50+50=100 (to 1200).
# Yes! 1000 -> 1100 (gpu) -> 1200 (F.pad).
# So PADDINGx should be (1200-1000)/4 = 50.
# So // 4 is correct IF we pad twice.
# Let's keep the logic but use config values.
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 4
PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 4

# Dataset Paths
dataset = "continue5" # Default
dataset_path_config = config.get('dataset_path', None)
if dataset_path_config:
    if os.path.exists(dataset_path_config):
        dataset_name = dataset_path_config
    elif os.path.exists(os.path.join("task1", dataset_path_config)): # Handle relative path from root
        dataset_name = os.path.join("task1", dataset_path_config)
    else:
        dataset_name = dataset_path_config # Fallback
else:
    # Fallback to logic
    dataset_name = os.path.join("task1", dataset)
    if not os.path.exists(dataset_name):
        if os.path.exists(dataset):
            dataset_name = dataset
        elif os.path.exists(f"../{dataset}"):
            dataset_name = f"../{dataset}"

print(f"Dataset path: {dataset_name}")

# Custom Dataset for 5-class selection
class SelectedClassesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, selected_indices=[0, 4, 9, 14, 19]):
        self.full_dataset = torchvision.datasets.ImageFolder(root_dir, transform=transform)
        self.classes = [self.full_dataset.classes[i] for i in selected_indices]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.indices = []
        for idx, (_, label) in enumerate(self.full_dataset.samples):
            if label in selected_indices:
                self.indices.append(idx)
        
        self.label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.full_dataset[original_idx]
        return image, self.label_map[label]

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
def generate_det_row(det_size, start_pos_x, start_pos_y, det_step, N_det):
    p = []
    for i in range(N_det):
        left = start_pos_x+i*(int(det_step)+det_size)
        right = left + det_size
        up = start_pos_y
        down = start_pos_y + det_size
        p.append((up, down, left, right))
    return p

def set_det_pos(det_size=60, start_pos_x=36, start_pos_y=36,
                N_det_sets=[2, 0, 2], det_steps_x=[5, 11, 5], det_steps_y=5):
    p = []
    for i in range(len(N_det_sets)):
        p.append(generate_det_row(det_size, start_pos_x, start_pos_y+i*(det_steps_y+1)*det_size, det_steps_x[i]*det_size, N_det_sets[i]))
    return list(itertools.chain.from_iterable(p))

# Initialize detector positions - Fixed for 5-class classification
# Using standard positions or as defined in original code for 5-class?
# Assuming 5 detectors for 5 classes.
# The original code had complex layouts. I'll use a simple layout for 5 classes if not specified.
# But wait, the original code had `detector_pos` defined in multiple ways.
# For baseline 5-class, I'll pick 5 distinct locations.
# Let's use "Circle Five Size40 Far" or similar if available, or just 5 points.
# I'll use the first 5 points from the `detector_pos_init` I had before, or define a new simple one.
# Let's use a horizontal row for simplicity or 5 corners + center.
# Actually, let's use the layout from the previous code but take first 5.
detector_pos_init = [
    (803, 843, 273, 313),
    (941, 981, 463, 503),
    (941, 981, 697, 737),
    (580, 620, 960, 1000),
    (219, 259, 697, 737),
    #(357, 397, 273, 313) # 6th detector removed for 5-class
]
# Wait, 5 classes usually need 5 detectors. The previous list had 6.
# I will use 5 detectors.

detector_pos_xy = []
for x0, x1, y0, y1 in detector_pos_init:
    detector_pos_xy.append(((x0+x1)/2, (y0+y1)/2))

def detector_region(Int, detector_mask=None, detector_minus=None, detector_pos=None, detector_size=60):
    detectors_list = torch.zeros(Int.shape[0], len(detector_pos), device=device)
    det_shape = config.get('detector_shape', 'circle')
    det_size_config = config.get('detector_size', 60)
    if det_size_config is not None:
        detector_size = det_size_config
        
    # Edge Penalty
    edge_weight = config.get('edge_penalty_weight', 0.0)
    edge_ratio = config.get('edge_width_ratio', 0.1)
    edge_loss = torch.tensor(0.0, device=device)
        
    for i, (x, y) in enumerate(detector_pos):
        if det_shape == 'square':
            det_x0 = int(x - detector_size/2)
            det_x1 = int(x + detector_size/2)
            det_y0 = int(y - detector_size/2)
            det_y1 = int(y + detector_size/2)
            
            # Clamp indices
            det_x0 = max(0, det_x0); det_x1 = min(Int.shape[1], det_x1)
            det_y0 = max(0, det_y0); det_y1 = min(Int.shape[2], det_y1)
                
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
                # Use pure inner sum without interpolation for penalty calculation to avoid negative loss loopholes
                inner_sum_pure = Int[:, det_x0:det_x1, det_y0:det_y1].sum(dim=(1, 2))
                rim_val = outer_sum - inner_sum_pure
                edge_loss += rim_val.sum()
        else: # circle
            crop_s = int(detector_size/2) + 4
            if edge_weight > 0:
                ew = int(detector_size * edge_ratio)
                crop_s += ew
                
            cx, cy = int(x), int(y)
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
                edge_loss += rim_val.sum()

        # Apply Scale (mask) and Bias (minus)
        if detector_mask is not None and detector_minus is not None:
            detectors_list[:, i] = raw_val * detector_mask[i] - detector_minus[i]
        else:
            detectors_list[:, i] = raw_val

    if edge_weight > 0:
        return Int, detectors_list / total, edge_loss * edge_weight
    else:
        return Int, detectors_list / total, torch.tensor(0.0, device=device)

# DNN Model
class DNN(torch.nn.Module):
    def __init__(self, num_layers=config.get('num_layers', 1), wl=wl, PhaseMask=PhaseMask, pixel_size=PIXEL_SIZE,
                 distance_between_layers=config.get('distance_between_layers', 0.2), distance_to_detectors=config.get('distance_to_detectors', 0.2), 
                 num_classes=config.get('num_classes', 5)):
        super(DNN, self).__init__()
        
        self.phase_mask = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(PhaseMask, dtype=torch.float32)) for _ in range(num_layers)
        ])
        
        # Fixed detector positions for 5-class task
        self.detector_pos = detector_pos_xy # List of tuples
        
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
          num_classes=5, epochs=5, device=device, exp_name="default"):
    
    currentDate = time.strftime("%Y%m%d_%H%M", time.localtime())
    
    # Save Dir Logic
    results_dir = config.get('results_dir', 'results')
    # Resolve results_dir relative to BASE_DIR if relative
    if not os.path.isabs(results_dir):
        # We need BASE_DIR here. It is defined in global scope.
        # But wait, BASE_DIR was not defined in global scope in 5c/train.py (I need to check line 22-25 of 5c/train.py)
        # Let's check 5c/train.py again.
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_dir)

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
    
    if os.path.exists(f"{save_dir}/best_model.pth"):
        print("Loading existing checkpoint...")
        model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
        
    print(f"Training 5-class Classification started.")

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
            loss += edge_loss_val.sum() if edge_loss_val.ndim > 0 else edge_loss_val
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_train_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(out_label.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            # if total % 10 == 0:
            #     print(f"Batch loss: {loss.item():.4f}, Acc: {correct/total:.4f}")
        
        avg_train_loss = ep_train_loss / len(trainloader)
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
                out_label, out_img = model(images_input)
                
                loss = loss_function(out_label, labels)
                ep_test_loss += loss.item()
                
                _, predicted = torch.max(out_label.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

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
            
        # Plot every epoch
        plot_loss_acc(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist, f"{save_dir}/loss_acc.png")
        
        # Save mask
        csv_log_dir = f"{save_dir}/csv_logs"
        os.makedirs(csv_log_dir, exist_ok=True)
        np.savetxt(f"{csv_log_dir}/mask_epoch_{epoch+1}.csv", model.phase_mask[0].detach().cpu().numpy(), delimiter=",")

    return best_acc

if __name__ == "__main__":
    try:
        # Check if dataset path is correct, otherwise try standard path
        if not os.path.exists(dataset_name):
             dataset_name = "task1/continue5" # Fallback
             
        # Use SelectedClassesDataset
        selected_indices = config.get('selected_indices', [0, 4, 9, 14, 19])
        train_dataset = SelectedClassesDataset(f"{dataset_name}/train", transform=cpu_transform, selected_indices=selected_indices)
        val_dataset = SelectedClassesDataset(f"{dataset_name}/val", transform=cpu_transform, selected_indices=selected_indices)
        
        print(f"Selected classes: {train_dataset.classes}")
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Experiment from Config
        exp_name = config.get('exp_name', 'default_run_5c')
        
        # Model (Defaults from global config)
        model = DNN()
        model = model.to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        
        lr = config.get('learning_rate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
        
        epochs = config.get('epochs', 30)
        num_classes = config.get('num_classes', 5)
        
        # Run training
        best_acc = train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, num_classes=num_classes, epochs=epochs, exp_name=exp_name)
        
        print(f"Training completed. Best Acc: {best_acc:.4f}")

        # Run Evaluation if configured
        if config.get("run_evaluate_after_train", True):
            print("\n--- Running Evaluation ---")
            from evaluate import evaluate
            evaluate()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
