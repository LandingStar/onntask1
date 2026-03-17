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
    results_base = os.path.join(os.path.dirname(__file__), config.get('results_dir', 'results'))
    subdirs = [os.path.join(results_base, d) for d in os.listdir(results_base) 
               if os.path.isdir(os.path.join(results_base, d))]
    result_dir = max(subdirs, key=os.path.getmtime)

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
val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
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
        out, _, _ = model(inp)
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
with open(os.path.join(os.path.dirname(__file__), 'eval_result.txt'), 'w') as f:
    f.write(output)
