# 🤖 AI System Instructions & Architecture Overview (task1)

This document is specifically designed for AI assistants and LLM agents working on the `task1` codebase. It provides a deep architectural breakdown, physical rationales, and the underlying logic of the Optical Neural Network (ONN) framework.

## 1. Project Context & Purpose
`task1` is a PyTorch-based framework designed to simulate and optimize **diffractive Optical Neural Networks (ONNs)**. 
Unlike standard digital neural networks, ONNs compute using light propagation (diffraction) through physically fabricated phase masks. The primary goal is to train these phase masks (and optionally detector positions) to perform image classification purely in the optical domain at the speed of light.

### Key Challenges Solved:
1. **Sim-to-Real Gap**: Physical fabrication and assembly inevitably introduce misalignments (translation, rotation, tilt). The framework simulates these during training so the learned masks are robust.
2. **Energy Efficiency**: A standard MSE loss might classify correctly but scatter light everywhere. This framework introduces an **Intensity Ratio Loss** to force the light to concentrate intensely on the target detectors, increasing the physical signal-to-noise ratio (SNR).
3. **I/O Bottlenecks**: Optical forward passes are mathematically lightweight (just FFTs and element-wise multiplications). To prevent GPU starvation caused by disk I/O, the framework implements an in-memory dataset loader.

---

## 2. Core Architecture Breakdown

### 2.1 Optical Forward Pass (`train.py` -> `DNN`)
The network (`DNN` class) does not use standard linear/conv layers. Instead, it simulates physical light propagation.

- **`Diffractive_Layer` & `Propagation_Layer`**: These use the **Angular Spectrum Method (ASM)**.
  - The input Electric field $E$ is transformed to the frequency domain via 2D FFT (`torch.fft.fft2`).
  - It is multiplied by a free-space transfer function: $H(f_x, f_y) = \exp(j k_z z)$, where $k_z$ is the longitudinal wave vector.
  - Finally, it is transformed back via IFFT.
- **Phase Masks (`self.phase_mask`)**: These are the actual learnable parameters (`nn.ParameterList`). They represent the physical height/refractive index of the manufactured glass/polymer layers. The modulation is applied as $E_{out} = E_{in} \times \exp(j \cdot 2\pi \cdot \text{mask})$.
- **Misalignment Simulation (`apply_misalignment`)**: If enabled, random affine transformations (shift, rotate) and out-of-plane tilts (applied as a phase gradient phasor $e^{j\vec{k}\cdot\vec{r}}$) are dynamically injected between layers during training.

### 2.2 Detector & Loss Logic
The physical detectors integrate the light intensity $|E|^2$ over a specific area.

- **`detector_region()`**: Integrates energy over square or circular regions. It supports calculating edge penalties (light hitting the borders of the detector) and concentration moments (forcing light to the dead-center of the detector).
- **`compute_loss()`**:
  1. **Soft Label & Class Mapping**: The framework is specifically configured to map 20 input classes to 5 physical detectors (`base_det = labels // 4`). If `soft_label_enabled` is true, it spreads the target intensity distribution across adjacent detectors based on class offsets and a temperature scaling parameter, rather than a strict one-hot target.
  2. **Vector Loss (MSE)**: Computes the error between the normalized detector intensities and the target one-hot (or soft) labels.
  3. **Energy Penalty (`loss_energy`)**: Ensures the target detector receives a minimum ratio of the total global energy (`auto_spatial_mask_target_ratio`). 
  4. **Aggressive Optimization**: Once the model hits a high accuracy threshold (e.g., 99%), it switches to a `LeakyReLU` for the energy loss. This ensures the gradient never zeroes out, endlessly pushing the model to concentrate more light even after the target ratio is met.

### 2.3 The Batch Training Engine (`batch_train.py`)
Because ONN hyperparameter search (e.g., distances, layer counts, error tolerances) requires massive parallel runs, the system uses a custom batch orchestrator.

- **Configuration Merging**: `batch_train.py` reads `batch_config/overall_config.json` (global defaults like GPU settings) and merges it with individual `.json` files in the same directory (experiment-specific settings).
- **Dynamic Injection**: It generates unique temporary JSON files in the `.temp_batch_configs/` directory and passes their paths to `train.py` via `sys.argv` (inline) or subprocess arguments. It **does not** physically overwrite the main `config.json`, allowing multiple tasks to run safely in parallel without configuration collisions.
- **Execution Modes**:
  - **Inline Execution**: For single-GPU tasks, it imports `train.py` as a module and runs `train.main()`, modifying `sys.argv` in memory. This is extremely fast and shares RAM cache.
  - **Subprocess (DDP)**: For multi-GPU tasks, it uses `subprocess.run` to spawn `torchrun`/`torch.distributed.run` commands dynamically assigning free ports.

---

## 3. Important Design Patterns & "Gotchas"

### A. Memory Management & Data Pipeline Optimization
- **The RAM Cache Problem**: Image loading from SSDs is too slow for FFT-based models. Standard `DataLoader` with many workers causes RAM to explode if images are decoded to `float32` Tensors.
- **The Solution**: `InMemoryImageFolder` reads the entire dataset as **raw bytes** into RAM once. Decoding (`Image.open`) happens on-the-fly during `__getitem__`.
- **Gotcha**: When `batch_train.py` runs multiple experiments inline, `InMemoryImageFolder` uses a class-level `_SHARED_CACHE` dictionary so the 10GB dataset is only loaded *once* across all consecutive experiments. **Do not remove this cache logic.**

- **CPU/GPU Heterogeneous Pipeline**: The data augmentation pipeline is strictly split to prevent CPU bottlenecks.
  - **CPU Transform**: Only handles lightweight I/O and Grayscale conversion.
  - **GPU Transform**: Computationally intensive operations (`Resize`, `RandomRotation`, `RandomAffine`, `ColorJitter`) are executed asynchronously on the GPU. **Do not move these augmentations back to the CPU**, as it will cause severe GPU starvation.

### B. Configuration System Hierarchy
The configuration is deeply nested. If you need to modify a parameter, check the priority:
1. Hardcoded defaults in `train.py` (Lowest)
2. `config.json` (Normal single run)
3. `batch_config/overall_config.json` (Batch global base)
4. `batch_config/config_*.json` (Batch specific override - Highest)

**Rule of Thumb for AI edits**: If you add a new feature (e.g., a new loss type), you MUST add its toggle/weight to the `config.get()` calls in `train.py` and document it in `config_readme.md`.

### C. Model Inheritance (`inherit_best_model`)
- Used for curriculum learning (e.g., train with 0 error, inherit that model, then train with high error).
- Driven by `inherit_model_path`. If empty, it defaults to the current `exp_name`. If it can't find a match, it falls back to the absolute newest model in the `results/` directory and prints a warning.

### D. Device Placement & Asynchronous Transfers
- **`non_blocking=True`**: Data is moved to the GPU asynchronously. To maximize this, `BATCH_SIZE` is dynamically auto-scaled based on the GPU's Streaming Multiprocessor (SM) count to ensure the PCIe bus and GPU compute units overlap perfectly. **Do not hardcode BATCH_SIZE if `auto-adjust` logic is present.**

---

## 4. Expected AI Assistant Behavior
When instructed to modify this codebase:
1. **Always check the `config` object first**: Before hardcoding a value, check if it should be a configuration parameter.
2. **Respect Physical Units**: Wavelengths are in meters (e.g., `532e-9`), pixel sizes in meters (`8e-6`). Do not mix them with pixel counts.
3. **Preserve DDP Compatibility**: If modifying loss or metrics, ensure you use `dist.all_reduce` if `is_ddp` is true.
4. **Use the RAM cache**: If touching dataset code, ensure `_SHARED_CACHE` remains intact for batch processing efficiency.
