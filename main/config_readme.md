# ONN Configuration Guide (`config.json`)

This document explains all available configuration parameters used in `train.py`. You can copy `config.example.json` to `config.json` or `batch_config/your_config.json` and modify it.

## Basic Training
- **`exp_name`**: Name of the experiment. Used to name the output folder in `results_dir`.
- **`batch_train`**: (boolean) If `true`, the `train.py` script will redirect execution to `batch_train.py` to run all configs in `batch_config/`.
- **`dataset_path`**: Path to the dataset (can be relative to `train.py` or absolute).
- **`results_dir`**: Directory where training logs and models will be saved.
- **`epochs`**: Total number of training epochs.
- **`batch_size`**: Training batch size. (Note: The script may auto-reduce this if VRAM is insufficient).
- **`learning_rate`**: Initial learning rate for the Adam optimizer.
- **`train_num_workers`**: Number of CPU cores to use for data loading in the training set.
- **`val_num_workers`**: Number of CPU cores to use for data loading in the validation set.
- **`prefetch_factor`**: Number of batches loaded in advance by each worker.
- **`in_memory_dataset`**: (boolean) If `true`, the entire dataset will be pre-loaded into RAM as raw bytes to completely bypass disk I/O bottlenecks. Automatically falls back to disk if insufficient RAM is detected.

## Scheduler
- **`scheduler_metric`**: Metric to monitor for learning rate reduction. Recommended explicit values are:
  - `"val_acc"`: Reduce LR based on pure validation accuracy.
  - `"val_loss"`: Reduce LR based on validation loss.
  - `"composite_score"`: Reduce LR based on the composite score used by model selection.
  Legacy values `"acc"`, `"loss"`, and `"score"` are still accepted and mapped to the explicit names above.
- **`scheduler_patience`**: Number of epochs with no improvement after which learning rate will be reduced.

## Optical Model Settings
- **`num_layers`**: Number of diffractive layers (phase masks).
- **`distance_between_layers`**: Distance between consecutive phase masks (in meters).
- **`distance_to_detectors`**: Distance from the last phase mask to the detector plane (in meters).
- **`wavelength`**: Illumination wavelength (e.g., `5.32e-07` for 532nm).
- **`pixel_size`**: Physical size of a single pixel (e.g., `8e-06` for 8um).
- **`img_size`**: Resolution of the input image `[H, W]`.
- **`phase_mask_size`**: Resolution of the optical system `[H, W]`. Input images are padded to this size.

## Detector Settings
- **`num_classes`**: Number of physical detectors.
- **`label_num`**: Number of distinct input classes in the dataset.
- **`detector_shape`**: Shape of the detector (`"circle"` or `"square"`).
- **`detector_size`**: Width/Diameter of the detector in pixels.
- **`train_detector_pos`**: (boolean) Whether to optimize detector positions during training.
- **`train_detector_scale`**: (boolean) Whether to optimize the intensity scaling factor of each detector.
- **`train_detector_bias`**: (boolean) Whether to optimize the intensity bias of each detector.
- **`detector_pos`**: Initial `[x, y]` coordinates for the centers of the detectors.

## Classification Objective
- **`classification_loss_type`**: Selects the primary classification objective.
  Recommended values are:
  - `"competition"`: Use the old detector-competition margin as the main classification loss.
  - `"mse"`: Use the detector-normalized MSE loss as the main classification loss.
  When set to `"competition"`, the primary classification signal is directly driven by the old competition-style objective.

## Intensity-Related Losses & Composite Score
There are now two independent intensity-related terms:

- **Detector Competition Loss**: the old helper term. It increases the target detector's normalized share among all detectors and implicitly suppresses non-target detector responses.
- **Global Energy Concentration Loss**: the physical efficiency term. It increases the absolute energy concentration of the target detector relative to the entire output plane.

### Detector Competition Loss
- **`detector_competition_loss_weight`**: (float) Weight of the old helper loss.
- **`detector_competition_target_ratio`**: (float) Target ratio for the detector competition helper loss.

### Global Energy Concentration Loss
- **`global_energy_concentration_loss_weight`**: (float) Weight of the global concentration loss.
- **`global_energy_concentration_target_ratio`**: (float) Target ratio for the global concentration loss.  
  This is the desired *Average Intensity Ratio* (Average Intensity inside Detector / Average Intensity of entire plane). E.g., `50.0` means the light inside the target detector should be on average 50 times brighter than the background.
- **`global_concentration_start_acc`**: (float) Validation-accuracy threshold that must be reached before the global concentration loss is enabled. This is intended to keep early training classification-first.

### Backward Compatibility
- **`spatial_mask_loss_weight`** and **`auto_spatial_mask_target_ratio`** are still accepted as legacy fields.
- **`use_global_energy_ratio`** is still accepted to map the legacy single intensity loss onto one of the two new terms.

### Shared Optimization / Score Parameters
- **`aggressive_intensity_optimization`**: (boolean) If true, dynamically amplifies the penalty weight for energy loss once the classification accuracy reaches a specific threshold.
- **`aggressive_acc_threshold`**: (float) The validation accuracy threshold (e.g., `0.99`) required to trigger aggressive intensity optimization.
- **`aggressive_weight_multiplier`**: (float) The multiplier applied to both intensity-related loss weights when aggressive optimization is triggered.
- **`aggressive_leaky_slope`**: (float) The negative slope for the LeakyReLU applied to intensity-related losses in aggressive mode.
- **`default_leaky_slope`**: (float) The negative slope used before aggressive mode. Set to `0.0` to keep pure ReLU behavior before the accuracy threshold is met.
- **`best_model_metric`**: (string) Metric used to decide which checkpoint is also written to `best_model.pth`.
  Recommended values are:
  - `"val_acc"`: `best_model.pth` follows the best validation-accuracy checkpoint.
  - `"val_loss"`: `best_model.pth` follows the lowest validation-loss checkpoint.
  - `"composite_score"`: `best_model.pth` follows the composite score.
  Regardless of this setting, the trainer also saves `best_acc_model.pth`, `best_loss_model.pth`, and `best_score_model.pth`.
- **`best_model_acc_weight`**: (float) **Accuracy Importance in Model Selection.** 
  The weight given to Validation Accuracy when calculating the `Composite Score` to determine the best model. Default is `1.0`.
- **`best_model_intensity_weight`**: (float) **Intensity Importance in Model Selection.** 
  The weight given to the selected intensity ratio when calculating the `Composite Score` to determine the best model. Default is `0.5`.
- **`score_intensity_source`**: (string) Which ratio to use in score/model selection. Options: `"detector_competition"` or `"global_concentration"`.
- **`score_intensity_cap`**: (float) Caps the intensity contribution to `Composite Score` before aggressive optimization is triggered.

## Misalignment Simulation (Physical Error Modeling)
To simulate the physical manufacturing and assembly tolerances, you can inject random sub-pixel misalignment and tilt errors between layers.
- **`simulate_misalignment`**: (boolean) Master switch. If `true`, applies random affine transformations and phase gradients before each layer (except the first layer).
- **`misalignment_translation_max_pixels`**: (float) Maximum random in-plane translation (shift in X and Y) in pixels.
- **`misalignment_rotation_max_degrees`**: (float) Maximum random in-plane rotation around the optical axis (Z-axis) in degrees.
- **`misalignment_tilt_max_degrees`**: (float) Maximum random out-of-plane tilt around the X and Y axes in degrees. This simulates phase gradient ramps across the optical field.

## Distributed Data Parallel (DDP) Settings (Used in `batch_config/overall_config.json`)
When using `batch_train.py`, you can enable multi-GPU training by adding these keys to your `overall_config.json`:
- **`use_ddp`**: (boolean) If `true`, `batch_train.py` will launch training scripts using `torchrun` instead of standard python.
- **`nproc_per_node`**: (string or int) Number of GPUs to use per node. Usually set to `"gpu"` to use all available GPUs, or an integer like `2`.
- **`master_port`**: (int) The base port for DDP communication. `batch_train.py` will automatically increment this port for each parallel task to avoid collisions. Default is `29500`.

## Miscellaneous
- **`transform_intensity`**: Scales the magnitude of data augmentations (rotation, jitter, etc.). Set to `0` to disable all augmentations except padding/resizing.
- **`inherit_best_model`**: If true, automatically loads the weights from the latest `best_model.pth` found in `results_dir` before starting.
- **`inherit_model_path`**: (Optional) If `inherit_best_model` is true, this string determines which experiment's model to inherit. It matches experiment folders containing this string (ignoring dates). If left empty (`""`), it defaults to matching the current `exp_name`. If no matching model is found, it falls back to inheriting the absolute newest model across all experiments (and prints a warning).
- **`save_csv_logs`**: If true, saves the Phase Mask and Detector Positions as CSV files.
- **`save_csv_logs_every_epoch`**: If true, saves CSVs every epoch (WARNING: Very slow). If false, only saves at the last epoch.
