
print("Starting script...")
import numpy as np
print("Imports starting...")
import matplotlib.pyplot as plt
print("imported plt")
import itertools
import copy
import math
import os
import time
import torch
print("imported torch")
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
print("imported torchvision")
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm
print("imported tqdm")

# Constants and Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device: ', device)

BATCH_SIZE = 48
IMG_SIZE = [1000, 1000]
N_pixels = 1024
PhaseMask = [1200, 1200]
PIXEL_SIZE = 8e-6
wl = 532e-9
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 4
PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 4

# Dataset Paths
dataset = "continue5"
type = None # "task1"
if type:
    dataset_name = f"{type}/{dataset}"
else:
    dataset_name = dataset
    
# Check if data exists
if not os.path.exists(dataset_name):
    # Fallback for relative path
    dataset_name = os.path.join("task1", dataset)

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
    def __init__(self, wl=wl, PhaseMask=PhaseMask, pixel_size=PIXEL_SIZE, distance=0.15):
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
        c = fft_c
        angular_spectrum = torch.fft.ifft2(c * self.phase)
        return angular_spectrum

# Propagation Layer
class Propagation_Layer(torch.nn.Module):
    def __init__(self, wl=wl, PhaseMask=PhaseMask, pixel_size=PIXEL_SIZE, distance=0.2):
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
        c = fft_c
        angular_spectrum = torch.fft.ifft2(c * self.phase)
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

# Initialize detector positions
det_size = 60
det_pad_x = (PhaseMask[0] - 13*det_size)//2
det_pad_y = (PhaseMask[1] - 13*det_size)//2
detector_pos_init = set_det_pos(det_size, det_pad_y, det_pad_x)

# Circle Ten Size40 Near
detector_pos_init = [
    (803, 843, 273, 313),
    (941, 981, 463, 503),
    (941, 981, 697, 737),
    (580, 620, 960, 1000),
    (219, 259, 697, 737),
    (357, 397, 273, 313)
]

detector_pos_xy = []
for x0, x1, y0, y1 in detector_pos_init:
    detector_pos_xy.append(((x0+x1)/2, (y0+y1)/2))

# Global label image tensor (Initial)
labels_image_tensors = torch.zeros((len(detector_pos_init), PhaseMask[0], PhaseMask[1]), device=device, dtype=torch.float32)
for ind, pos in enumerate(detector_pos_init):
    pos_l, pos_r, pos_u, pos_d = pos
    labels_image_tensors[ind, pos_l:pos_r, pos_u:pos_d] = 1
    if labels_image_tensors[ind].sum() > 0:
        labels_image_tensors[ind] = labels_image_tensors[ind] / labels_image_tensors[ind].sum()

def detector_region(Int, detector_mask=None, detector_minus=None, detector_pos=None, detector_size=60):
    detectors_list = torch.zeros(Int.shape[0], len(detector_pos), device=device)
    
    # We need to compute gradients for detector_pos, so we use the differentiable approximation
    for i, (x, y) in enumerate(detector_pos):
        det_x0 = torch.floor(x - detector_size/2).int()
        det_x1 = torch.floor(x + detector_size/2).int()
        det_y0 = torch.floor(y - detector_size/2).int()
        det_y1 = torch.floor(y + detector_size/2).int()
        
        # Clamp indices to be safe
        det_x0 = torch.clamp(det_x0, 0, Int.shape[1]-2)
        det_x1 = torch.clamp(det_x1, det_x0+1, Int.shape[1]-1)
        det_y0 = torch.clamp(det_y0, 0, Int.shape[2]-2)
        det_y1 = torch.clamp(det_y1, det_y0+1, Int.shape[2]-1)

        # Apply mask
        if detector_mask is not None:
            # Note: This masking is not differentiable w.r.t position indices
            Int[:, det_x0:det_x1+1, det_y0:det_y1+1] *= detector_mask[i]
            
        if detector_minus is not None:
            # Placeholder for minus logic if needed
            pass
            
        # Differentiable intensity summation
        base_sum = Int[:, det_x0:det_x1, det_y0:det_y1].sum(dim=(1, 2))
        
        # Boundary terms for gradients
        # Right edge
        right_edge = (Int[:, det_x1:det_x1+1, det_y0:det_y1].sum(dim=(1, 2)) - 
                      Int[:, det_x0:det_x0+1, det_y0:det_y1].sum(dim=(1, 2))) * (x - detector_size/2 - det_x0)
        
        # Bottom edge
        bottom_edge = (Int[:, det_x0:det_x1, det_y1:det_y1+1].sum(dim=(1, 2)) - 
                       Int[:, det_x0:det_x1, det_y0:det_y0+1].sum(dim=(1, 2))) * (detector_size/2 - det_y0)
                       
        detectors_list[:, i] = base_sum + right_edge + bottom_edge

    total = detectors_list.sum(dim=1, keepdim=True) + 1e-8
    return Int, detectors_list / total

# DNN Model
class DNN(torch.nn.Module):
    def __init__(self, num_layers=1, wl=wl, PhaseMask=PhaseMask, pixel_size=PIXEL_SIZE,
                 distance_between_layers=0.2, distance_to_detectors=0.2, num_classes=5):
        super(DNN, self).__init__()
        
        self.phase_mask = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(PhaseMask, dtype=torch.float32)) for _ in range(num_layers)
        ])
        
        self.detector_mask = torch.nn.Parameter(torch.ones(num_classes))
        self.detector_minus = torch.nn.Parameter(torch.zeros(num_classes))
        # Initialize with center points
        self.detector_pos = torch.nn.Parameter(torch.tensor(detector_pos_xy, dtype=torch.float32))
        
        self.diffractive_layers = torch.nn.ModuleList([Diffractive_Layer(wl, PhaseMask, pixel_size, distance_between_layers) for _ in range(num_layers)])
        self.last_diffractive_layer = Propagation_Layer(wl, PhaseMask, pixel_size, distance_to_detectors)

    def forward(self, E):
        E = E.to(torch.cfloat)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(E)
            phase_values = 2 * torch.pi * self.phase_mask[index]
            modulation = torch.exp(1j * phase_values)
            E = temp * modulation
        E = self.last_diffractive_layer(E)
        Int = torch.abs(E)**2
        Int, output = detector_region(Int, self.detector_mask, self.detector_minus, self.detector_pos)
        return output, Int

# Training Function
def train(model, loss_function, optimizer, scheduler, trainloader, testloader, 
          num_classes=5, epochs=5, device=device, strict_accuracy_ratio=1, 
          minus_mask_ratio=0, label_num=20):
    
    currentDate = time.strftime("%Y%m%d_%H%M", time.localtime())
    Date = time.strftime("%Y%m%d", time.localtime())
    
    if not os.path.exists(f"log/{Date}"):
        os.makedirs(f"log/{Date}")
    
    filename = f"model_{currentDate}.pth"
    log_file = open(f"log/{Date}/log_{currentDate}.txt", "w")
    
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    best_acc = 0
    
    if label_num >= 2 * num_classes:
        num_classes += 1
    classes_num = num_classes
    
    print(f"Training started. Classes: {classes_num}, Labels: {label_num}")

    def compute_loss(images, labels, out_img, output_vec):
        full_int_img = out_img.sum(axis=(1, 2))
        normalized_out_img = out_img / (full_int_img[:, None, None] + 1e-8)
        
        batch_loss = torch.tensor(0.0, device=device)
        
        # Dynamic Target Mask Generation based on CURRENT detector_pos
        # We need to regenerate the target mask because detector_pos might have moved
        current_labels_image_tensors = torch.zeros((classes_num, PhaseMask[0], PhaseMask[1]), device=device, dtype=torch.float32)
        
        # Use current detector positions to create mask
        # Note: This step is non-differentiable w.r.t pos, but provides correct target for Image Loss
        det_size = 60
        for ind, pos in enumerate(model.detector_pos):
            x, y = pos
            x0 = int(x - det_size/2)
            x1 = int(x + det_size/2)
            y0 = int(y - det_size/2)
            y1 = int(y + det_size/2)
            
            # Clamp
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
                floor = ((sample_label + 1) * (classes_num - 1)) // label_num
                ceil = ((sample_label + 1) * (classes_num - 1) + label_num - 1) // label_num
                position = ((sample_label + 1) % (label_num / (classes_num - 1))) * ((classes_num - 1) / label_num)
                
                if position == 0:
                    target_mask_weight[floor] = 1
                else:
                    target_mask_weight[floor] = 1 - position
                    target_mask_weight[ceil] = position
            
            # Image Loss
            target_mask = torch.einsum('c,chw->hw', target_mask_weight, current_labels_image_tensors)
            loss = loss_function(sample_img, target_mask)
            
            # Vector Loss (To provide gradients to detector_pos)
            # We want the output vector to match target_mask_weight
            loss_vec = F.mse_loss(output_vec[i], target_mask_weight)
            
            # Combine losses
            batch_loss += loss + 1.0 * loss_vec # Add weight to vector loss if needed

        return batch_loss

    def compute_acc(out_label, labels):
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
            for i in range(label_num):
                mask = (labels == i)
                if mask.any():
                    floor = ((i + 1) * (classes_num - 1)) // label_num
                    ceil = ((i + 1) * (classes_num - 1) + label_num - 1) // label_num
                    
                    single_check = torch.logical_and(second_brightest_intensity[mask] <= 0, 
                                                     torch.abs(predicted[mask] - ((i + 1) * (classes_num - 1) / label_num)) < 0.5)
                    
                    if ceil == floor:
                        correct_pair = torch.logical_or((predicted[mask] == floor), single_check)
                        intensity_check = max_intensities[mask] + base_intensity >= (second_brightest_intensity[mask] + base_intensity) * strict_accuracy_ratio
                    else:
                        non_target_mask_bool = torch.logical_or(non_target_mask_bool, F.one_hot(nd_predicted, num_classes=num_classes))
                        third_brightest_intensity, rd_predicted = torch.max(out_label.masked_fill(non_target_mask_bool, -float('inf')), dim=1)
                        
                        correct_pair = torch.logical_or(
                            torch.logical_or(
                                ((predicted[mask] == floor) & (nd_predicted[mask] == ceil)),
                                ((predicted[mask] == ceil) & (nd_predicted[mask] == floor))
                            ), single_check)
                        
                        intensity_check = (max_intensities[mask] + second_brightest_intensity[mask] >= third_brightest_intensity[mask] * strict_accuracy_ratio)
                    
                    is_correct[mask] = correct_pair & intensity_check
                    
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
            
            out_label, out_img = model(images_input)
            
            loss = compute_loss(images_input, labels, out_img, out_label)
            
            loss.backward()
            
            # Gradient clipping (optional but recommended)
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
                out_label, out_img = model(images_input)
                
                loss = compute_loss(images_input, labels, out_img, out_label)
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
        print(f"Detector Pos Mean: {model.detector_pos.mean().item():.2f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"weight/{Date}/{filename}")
            print(f"Saved best model with acc: {best_acc:.6f}")

    return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, model, best_acc

# Main execution
if __name__ == "__main__":
    print("In main block...")
    train_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/train", transform=cpu_transform)
    val_dataset = torchvision.datasets.ImageFolder(f"{dataset_name}/val", transform=cpu_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = DNN(num_layers=1, num_classes=6).to(device)
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, num_classes=5, epochs=1)
