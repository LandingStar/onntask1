# Task 1: Optical Neural Network (ONN) Training Framework

This directory (`task1`) contains a comprehensive PyTorch-based framework for simulating, training, and evaluating Optical Neural Networks (ONNs). It is designed to optimize physical phase masks, detector positions, and evaluate the robustness of optical models against physical manufacturing and assembly tolerances (misalignments).

## 📂 Directory Structure

```text
task1/
├── main/                       # Core source code directory
│   ├── train.py                # Main training script (Model definition, DDP, loss calculation)
│   ├── evaluate.py             # Evaluation script (Accuracy, energy concentration, confusion matrix)
│   ├── batch_train.py          # Batch training engine for running multiple experiments automatically
│   ├── archive_results.py      # Utility to organize and archive results after training
│   ├── quick_eval.py           # Fast evaluation script for checking multiple result directories
│   ├── vis_detectors.py        # Tool for visualizing detector energy distribution
│   ├── config.json             # Global configuration file for a single run
│   ├── config.example.json     # Example configuration file
│   ├── config_readme.md        # Detailed documentation for all configuration parameters
│   └── batch_config/           # Directory for batch training configurations
│       ├── overall_config.json # Shared global settings for batch runs
│       ├── config_one.json     # Specific experiment config 1
│       └── config_two_layer.json# Specific experiment config 2
├── report/                     # Archived results and reports
│   ├── one_layer/              # Results for single-layer ONN experiments
│   ├── two_layer/              # Results for multi-layer ONN experiments (with tilt/misalignment tests)
│   └── Task_Update_Report.md   # Progress and update tracking document
├── auto_run.sh                 # Shell script for automated execution
├── run_dcu_task1.slurm         # SLURM script for cluster/HPC deployment
└── requirements.txt            # Python dependencies
```

## ✨ Core Features

1. **Physical Optical Simulation**: 
   - Uses Angular Spectrum Method for accurate light propagation.
   - Simulates physical components: Phase Masks, Distances, Wavelengths, and Pixel Sizes.
   - Supports single-layer and multi-layer ONN architectures.
2. **Robustness & Misalignment Modeling**:
   - Built-in simulation for physical assembly tolerances (`simulate_misalignment`).
   - Supports modeling of in-plane translation (shift X/Y), in-plane rotation (Z-axis), and out-of-plane tilt (phase gradients).
3. **Advanced Loss Functions**:
   - Classification Vector Loss (MSE).
   - **Intensity/Energy Concentration Loss**: Ensures light energy is focused on the detectors, preventing energy dispersion. Includes aggressive optimization using LeakyReLU to endlessly push for higher optical efficiency.
   - Soft-label mapping for continuous detector distributions.
4. **Automated Batch Training Engine**:
   - `batch_train.py` allows sequential or parallel execution of multiple experiments using JSON configurations.
   - Features automatic config merging, dynamic port allocation for DDP, and GPU assignment.
5. **High-Performance Data Pipeline**:
   - Support for `InMemoryImageFolder` to completely eliminate disk I/O bottlenecks during fast forward passes, with automatic RAM capacity checks.
   - Auto-scaling of `BATCH_SIZE` based on available GPU VRAM and Streaming Multiprocessors (SMs) to maximize PCIe transfer overlap.

## 🚀 Getting Started

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### 1. Single Experiment Training
To run a single training experiment, configure your settings in `main/config.json` (refer to `main/config_readme.md` for parameter details) and run:
```bash
cd main
python train.py
```
*Note: If `batch_train` is set to `true` in `config.json`, `train.py` will automatically redirect to the batch training engine.*

### 2. Batch Training (Recommended for Hyperparameter Search)
Batch training allows you to queue multiple experiments.
1. Edit shared settings in `main/batch_config/overall_config.json` (e.g., `max_parallel`, `batch_size`).
2. Add individual experiment configurations as `.json` files in the `batch_config/` folder.
3. Run the batch engine:
```bash
cd main
python batch_train.py
```

### 3. Evaluation
By default, evaluation and archiving run automatically after training. To manually evaluate the latest results:
```bash
cd main
python evaluate.py
```
This will generate confusion matrices, performance metrics, and visualization of the learned phase masks and detector parameters in the `results/` directory.

## ⚙️ Configuration Highlight
The system is heavily configuration-driven. Key configuration areas include:
- **`inherit_best_model`**: Seamlessly resume training or inherit weights from a previous experiment's `best_model.pth` based on `inherit_model_path`.
- **`simulate_misalignment`**: Crucial for bridging the sim-to-real gap by introducing translation, rotation, and tilt errors during training.
- **`in_memory_dataset`**: Set to `true` to load raw dataset bytes into RAM, drastically speeding up training for lightweight ONNs.

*For a full list of parameters, see [config_readme.md](main/config_readme.md).*
