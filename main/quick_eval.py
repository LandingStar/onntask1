"""Quick per-class accuracy evaluation - prints text results only."""
import os, sys, torch, numpy as np, torch.nn.functional as F, torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path) as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = config.get('img_size', [1000, 1000])
PhaseMask = config.get('phase_mask_size', [1200, 1200])
PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 2
PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 2

cpu_transform = v2.Compose([
    v2.ToImage(), v2.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True),
    v2.ToDtype(torch.float32, scale=True), v2.Grayscale(num_output_channels=1)
])

from train import DNN

# Find model
result_dir = sys.argv[1] if len(sys.argv) > 1 else None
if result_dir is None:
    # 1. Load main config to find results base dir
    results_base = os.path.join(os.path.dirname(__file__), config.get('results_dir', 'results'))
    
    # 2. Find latest valid subdir by parsing folder name date
    if os.path.exists(results_base):
        import re
        from datetime import datetime
        
        name_pattern = re.compile(r'^.+_(\d{8}_\d{4})$')
        valid_subdirs = []
        
        for d in os.listdir(results_base):
            dir_path = os.path.join(results_base, d)
            if not os.path.isdir(dir_path):
                continue
                
            target_path = None
            target_name = d
            
            # Direct match
            if os.path.exists(os.path.join(dir_path, "best_model.pth")):
                target_path = dir_path
            # Nested match (e.g. unzipped folders that created a double directory)
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
            result_dir = latest_item['path']
        else:
            print("No valid subdirectories (containing best_model.pth and matching date format) found in results.")
            sys.exit(1)
    else:
        print("Results directory not found.")
        sys.exit(1)

# 3. Load the config from the latest result directory to override main config
result_config_path = os.path.join(result_dir, 'config.json')
if os.path.exists(result_config_path):
    with open(result_config_path, 'r') as f:
        config = json.load(f)
        
    # Update dependent variables
    IMG_SIZE = config.get('img_size', [1000, 1000])
    PhaseMask = config.get('phase_mask_size', [1200, 1200])
    PADDINGx = (PhaseMask[0] - IMG_SIZE[0]) // 2
    PADDINGy = (PhaseMask[1] - IMG_SIZE[1]) // 2
    
    # We must also inject this config into train module so DNN initializes correctly
    import train
    train.config = config
    
    # Also patch detector_pos_xy if needed
    detector_pos_init_config = config.get('detector_pos', None)
    if detector_pos_init_config is not None:
        detector_pos_xy = []
        for x, y in detector_pos_init_config:
            detector_pos_xy.append((x, y))
        train.detector_pos_xy = detector_pos_xy

model_path = os.path.join(result_dir, 'best_model.pth')
print(f"Model: {model_path}")

# Dataset
dataset_path = config.get('dataset_path', 'task1/continue5')
base = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    p = os.path.join(base, dataset_path)
    if os.path.exists(p):
        dataset_path = p
        break
    base = os.path.dirname(base)

val_ds = torchvision.datasets.ImageFolder(f"{dataset_path}/val", transform=cpu_transform)

# Auto-adjust BATCH_SIZE based on VRAM
eval_batch_size = config.get('batch_size', 16)
if torch.cuda.is_available():
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_vram_gb < 6.0:
        eval_batch_size = min(eval_batch_size, 32)
    elif total_vram_gb < 10.0:
        eval_batch_size = min(eval_batch_size, 64)
    elif total_vram_gb < 16.0:
        eval_batch_size = min(eval_batch_size, 128)
    else:
        eval_batch_size = min(eval_batch_size, 256)

val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
print(f"Val samples: {len(val_ds)}, classes: {val_ds.classes}")

model = DNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

label_num = config.get('label_num', 20)
correct = np.zeros(label_num)
total = np.zeros(label_num)

with torch.no_grad():
    for batch_idx, (imgs, lbls) in enumerate(val_dl):
        imgs = imgs.to(device)
        if imgs.shape[1] == 1:
            imgs = imgs.squeeze(1)
        inp = F.pad(imgs, pad=(PADDINGy, PADDINGy, PADDINGx, PADDINGx))
        out, _, _, _ = model(inp)
        _, preds = torch.max(out, 1)
        preds = preds.cpu()
        for l in range(label_num):
            m = (lbls == l)
            if m.any():
                total[l] += m.sum().item()
                correct[l] += (preds[m] == l // 4).sum().item()
        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx+1}/{len(val_dl)}", flush=True)

lines = []
lines.append("\n===== Per-Label Accuracy =====")
for i in range(label_num):
    acc = correct[i] / max(total[i], 1) * 100
    lines.append(f"  Label {i:2d} ({val_ds.classes[i]:>5s}) -> Det {i//4}: {acc:5.1f}%  ({int(correct[i])}/{int(total[i])})")

lines.append(f"\n===== Per-Detector Accuracy =====")
for d in range(5):
    c = sum(correct[d*4:(d+1)*4])
    t = sum(total[d*4:(d+1)*4])
    lines.append(f"  Det {d}: {c/t*100:5.1f}%  ({int(c)}/{int(t)})")

lines.append(f"\nOverall: {correct.sum()/total.sum()*100:.1f}%  ({int(correct.sum())}/{int(total.sum())})")

output = "\n".join(lines)
print(output)
with open(os.path.join(result_dir, 'eval_result.txt'), 'w') as f:
    f.write(output)
