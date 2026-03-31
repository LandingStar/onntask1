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
- **`num_workers`**: Number of CPU cores to use for data loading. (Overrides auto-detection if set).
- **`prefetch_factor`**: Number of batches loaded in advance by each worker.

## Scheduler
- **`scheduler_metric`**: Metric to monitor for learning rate reduction (`"acc"` or `"loss"`).
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

## Spatial Mask & Dynamic Focusing (New Features)
These parameters balance Classification Accuracy and Energy Concentration (Diffraction Efficiency).

- **`target_encircled_energy`**: (float, 0.0 ~ 1.0) **Defines the Ideal Target Shape.** 
  Specifies the fraction of energy that *should* fall within the detector boundaries in an ideal scenario. For example, `0.8` means the target is a Gaussian spot where 80% of its energy is inside the detector. This generates the `Target Mask` for the loss function.
- **`spatial_mask_loss_weight`**: (float) **Initial Pulling Force.** 
  The starting weight applied to the spatial distribution loss (Pixel-wise MSE against the Target Mask). Default is `0.05`.

**Dynamic Adjustment:**
- **`auto_spatial_mask_weight`**: (boolean) Enables dynamic adjustment of `spatial_mask_loss_weight`.
- **`auto_spatial_mask_target_ratio`**: (float) **The Engineering Goal.** 
  The desired *Average Intensity Ratio* (Average Intensity inside Detector / Average Intensity of entire plane). E.g., `50.0` means the light inside the detector should be on average 50 times brighter than the background.
- **`auto_spatial_mask_acc_threshold`**: (float) **The Safety Threshold.** 
  The dynamic weight will *only* increase if the Validation Accuracy is above this threshold (e.g., `0.90`). If accuracy drops below `threshold - 0.1` (e.g., 0.80), the weight will automatically decrease to let the model recover.
- **`auto_spatial_mask_weight_step`**: (float) The multiplier used to increase the weight (e.g., `1.1` increases weight by 10% per epoch when conditions are met).

## Miscellaneous
- **`transform_intensity`**: Scales the magnitude of data augmentations (rotation, jitter, etc.). Set to `0` to disable all augmentations except padding/resizing.
- **`inherit_best_model`**: If true, automatically loads the weights from the latest `best_model.pth` found in `results_dir` before starting.
- **`save_csv_logs`**: If true, saves the Phase Mask and Detector Positions as CSV files.
- **`save_csv_logs_every_epoch`**: If true, saves CSVs every epoch (WARNING: Very slow). If false, only saves at the last epoch.