# ONN Configuration Guide (`config.json`)

## Reproducibility Notes
- The `config.json` copied into each `results/<run>/` directory is the source of truth for what actually ran. Do not assume a file under `test/` or `batch_config/` still matches the launched run if it was edited later.
- If `inherit_best_model = true`, the trainer first searches `results_dir` for `best_model.pth` matching `inherit_model_path` (or `exp_name` if the path is empty). If nothing matches, it falls back to the absolute newest `best_model.pth` in `results_dir`.
- `best_model_metric`, `scheduler_metric`, and `score_intensity_source` together decide which checkpoint is saved to `best_model.pth` and which signal drives LR scheduling. Always verify these three fields in the saved run config before interpreting results.
- In the compact epoch log line, `Int.Ratio` shows the ratio selected by `score_intensity_source`. For detailed interpretation of physical metrics, prefer `metrics.csv` columns such as `avg_global_concentration_ratio`, `avg_outside_average_ratio`, and `avg_physical_focus_ratio`.
- For newer physical-objective experiments, verify the actual saved `metrics.csv` header in the run directory before assuming a metric is available. A metric may exist in the current code but still be absent from an older or mismatched run artifact.
- If saved run artifacts contradict the current on-disk `train.py` behavior, assume the run was launched from a different code snapshot until that mismatch is explained. Treat the run directory as the source of truth and re-check the exact launcher / file provenance before drawing conclusions from code.
- `train.py` now supports a code fingerprint guard via `train_code_fingerprint`. If the config field is non-empty, both `train.py` and `batch_train.py` validate it against the current `train.py` file content and skip the run with a warning when it does not match.
- Current `train.py` fingerprint: `3c7af0d6c717`.
- Workflow rule:
  - every time `train.py` is modified, recompute the fingerprint
  - update `train_code_fingerprint` in every config you still plan to run
  - if you intentionally want a config to ignore fingerprint matching, leave `train_code_fingerprint` as an empty string

This document explains all available configuration parameters used in `train.py`. You can copy `config.example.json` to `config.json` or `batch_config/your_config.json` and modify it.

## Basic Training
- **`exp_name`**: Name of the experiment. Used to name the output folder in `results_dir`.
- **`batch_train`**: (boolean) If `true`, the `train.py` script will redirect execution to `batch_train.py` to run all configs in `batch_config/`.
- **`train_code_fingerprint`**: (string) Optional code fingerprint guard. If non-empty, the launcher verifies that the current `train.py` fingerprint matches this value before running. If it does not match, the run is skipped with a warning.
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
- **`detector_competition_loss_mode`**: (string) Controls the exact detector-helper semantics.
  Recommended values are:
  - `"raw_margin"`: Uses the signed margin `target_ratio - current_ratio`. Once the target is exceeded, this term becomes negative and keeps rewarding further detector-ratio growth.
  - `"legacy_relu_helper"`: Restores the historical helper behavior. Before aggressive mode it uses ReLU and stops contributing once the target ratio is met; in aggressive mode it switches to LeakyReLU to keep pushing concentration.
  Default is `"raw_margin"` to preserve the current experiment behavior.

### Global Energy Concentration Loss
- **`global_energy_concentration_loss_weight`**: (float) Weight of the global concentration loss.
- **`global_energy_concentration_target_ratio`**: (float) Target ratio for the global concentration loss.  
  This is the desired *Average Intensity Ratio* (Average Intensity inside Detector / Average Intensity of entire plane). E.g., `50.0` means the light inside the target detector should be on average 50 times brighter than the background.
- **`global_physical_objective_mode`**: (string) Controls the exact physical objective used by the global term.
  Recommended values are:
  - `"global_ratio"`: The existing objective. Pushes the target detector's average intensity relative to the entire plane.
  - `"target_inside_outside"`: A two-term physical objective. It rewards target-detector average intensity and penalizes the average intensity outside the target detector.
  - `"target_attract_ring"`: A spatial objective for phase-2 tuning. It uses a continuous attraction kernel centered on the target detector and optionally penalizes the detector's outer ring region to suppress bright halos around detector edges.
  - `"target_inside_core_edge"`: Rewards detector-inside and inner-core energy while explicitly penalizing bright edge bands just inside and just outside the detector boundary.
- **`global_inside_reward_weight`**: (float) Relative weight of the target-inside reward when `global_physical_objective_mode = "target_inside_outside"`.
- **`global_outside_penalty_weight`**: (float) Relative weight of the outside-energy penalty when `global_physical_objective_mode = "target_inside_outside"`.
- **`global_attract_reward_weight`**: (float) Relative weight of the continuous attraction term when `global_physical_objective_mode = "target_attract_ring"`.
- **`global_outer_ring_penalty_weight`**: (float) Relative weight of the detector-outer-ring penalty when `global_physical_objective_mode = "target_attract_ring"`.
- **`global_attract_sigma_ratio`**: (float) Attraction-kernel width relative to `detector_size`. Larger values create a wider spatial attraction basin.
- **`global_attract_extent_ratio`**: (float) Extra support radius of the local attraction crop relative to `detector_size`.
- **`global_outer_ring_width_ratio`**: (float) Width of the penalized outer ring relative to `detector_size`.
- **`global_inner_edge_penalty_weight`**: (float) Only used by `target_inside_core_edge`. Penalizes a narrow bright band just inside the detector boundary.
- **`global_outer_edge_penalty_weight`**: (float) Only used by `target_inside_core_edge`. Penalizes a narrow bright band just outside the detector boundary.
- **`global_inner_edge_width_ratio`**: (float) Width of the inner-edge penalty band relative to `detector_size`.
- **`global_outer_edge_width_ratio`**: (float) Width of the outer-edge penalty band relative to `detector_size`.
- **`global_concentration_start_acc`**: (float) Validation-accuracy threshold that must be reached before the global concentration loss is enabled. This is intended to keep early training classification-first.
- **`global_concentration_hold_epochs`**: (int) Optional sticky window for the global loss. Once the validation accuracy reaches `global_concentration_start_acc`, the global loss stays enabled for this many subsequent epochs even if accuracy dips slightly below the threshold. Default is `0` (no sticky window).

### Stage-Transition Controls
- **`detector_competition_decay_after_global`**: (boolean) If true, once validation accuracy first reaches `global_concentration_start_acc`, the detector competition helper weight starts decaying across later epochs.
- **`detector_competition_decay_epochs`**: (int) Number of epochs used for the detector-helper decay schedule after global activation starts.
- **`detector_competition_decay_floor_ratio`**: (float) Final ratio of the original detector helper weight after decay completes. For example, `0.25` means the helper decays down to 25% of its original weight and then stays there.

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
- **`score_intensity_source`**: (string) Which ratio to use in score/model selection. Options include `"detector_competition"`, `"global_concentration"`, `"physical_focus"`, `"spatial_focus"`, `"soft_inside_core_focus"`, `"inside_core_edge_focus"`, and `"inside_core_outside_focus"`.
- **`score_intensity_cap`**: (float) Caps the intensity contribution to `Composite Score` before aggressive optimization is triggered when using the legacy hard-cap score path.
- **`score_use_soft_intensity`**: (boolean) If `true`, replaces the hard cap with a soft `tanh` normalization before intensity enters `Composite Score`. Defaults to `true` for `score_intensity_source = "spatial_focus"`.
- **`score_intensity_reference`**: (float) Reference scale for the soft-normalized score path. Larger values make the intensity contribution grow more slowly.
- **`score_acc_gate`**: (float) Accuracy threshold for unlocking the intensity contribution in the soft-normalized score path.
- **`score_acc_gate_width`**: (float) Transition width for the accuracy gate. With `score_acc_gate = 0.995` and `score_acc_gate_width = 0.005`, the intensity term ramps from suppressed to fully active between `0.995` and `1.000` validation accuracy.
- **`score_acc_cap`**: (float) Accuracy contribution cap used by the composite score. If positive, accuracy above this value stops increasing the score directly.
- **`score_acc_floor`**: (float) Optional hard floor for aggressive physical optimization. Below this accuracy, the score falls back to accuracy-first behavior; at or above it, the score switches to a physical-selection regime.
- **`score_physical_transition_width`**: (float) Width of the transition from the accuracy floor into the physics-dominant score regime. For example, with `score_acc_floor = 0.80` and `score_physical_transition_width = 0.10`, the physical term ramps in between `0.80` and `0.90` validation accuracy.
- **`score_min_acc_for_selection`**: (float) Minimum validation accuracy required for a checkpoint to participate in composite-score model selection. Below this threshold the score is forced to a very low fallback value, so collapsed checkpoints do not overwrite `best_score_model.pth`.
- **`score_tiered_save_min_accs`**: (list[float] or comma-separated string) Optional extra accuracy thresholds for tiered checkpointing. For each threshold, the trainer also writes `best_score_acc_ge_<threshold>_model.pth`, saving the highest-score checkpoint whose validation accuracy stays at or above that threshold.
- **`score_outside_penalty_weight`**: (float) Only used when `score_intensity_source = "physical_focus"`. It sets how strongly outside average intensity reduces the selection score.
- **`score_ring_penalty_weight`**: (float) Only used when `score_intensity_source = "spatial_focus"`. It sets how strongly target outer-ring intensity reduces the selection score.
- **`score_inner_edge_penalty_weight`**: (float) Only used when `score_intensity_source = "inside_core_edge_focus"`. It sets how strongly bright intensity just inside the detector boundary reduces the selection score.
- **`score_outer_edge_penalty_weight`**: (float) Only used when `score_intensity_source = "inside_core_edge_focus"`. It sets how strongly bright intensity just outside the detector boundary reduces the selection score.
- Recommended note for `spatial_focus`:
  - do not blindly reuse the old `global_ratio` cap logic
  - prefer `score_use_soft_intensity = true`
  - give it its own `score_intensity_reference`
- Recommended note for aggressive `inside_*` experiments:
  - use `score_acc_floor` to define the minimum acceptable accuracy (for example `0.80`)
  - use `score_min_acc_for_selection` when you want best-model selection to hard-filter checkpoints below that accuracy
  - use `score_tiered_save_min_accs` when you want side-by-side physical-best checkpoints for multiple acceptable accuracy bands such as `0.99` and `0.985`
  - set `score_acc_gate` near the same floor so the physical term starts contributing much earlier
  - prefer larger `score_intensity_reference` values than the old gentle settings to avoid immediate `tanh` saturation

### Classification-Loss Relaxation
- **`classification_loss_relax_enabled`**: (boolean) If `true`, scales down the primary classification loss once batch accuracy is high enough.
- **`classification_loss_relax_acc_threshold`**: (float) Accuracy threshold where classification-loss relaxation begins.
- **`classification_loss_relax_transition_width`**: (float) Width of the relaxation transition. For example, with threshold `0.80` and width `0.10`, the classification loss ramps down between `0.80` and `0.90` batch accuracy.
- **`classification_loss_relax_floor_ratio`**: (float) Minimum fraction of the original classification loss kept after relaxation fully activates. Smaller values are more aggressive and allow the physical objective to dominate.
- **`classification_loss_relax_hold_epochs`**: (int) Optional validation-accuracy hold window. If the previous epoch reaches `classification_loss_relax_acc_threshold`, the trainer keeps the classification loss pinned at the relaxed floor for this many subsequent epochs, even if batch accuracy later dips.
- **`detector_competition_relax_acc_threshold`**: (float) Validation-accuracy threshold that activates a temporary detector-helper retreat window. Defaults to `classification_loss_relax_acc_threshold`.
- **`detector_competition_relax_hold_epochs`**: (int) Number of epochs to keep the detector competition helper in the relaxed state after the validation threshold is reached.
- **`detector_competition_relax_floor_ratio`**: (float) Ratio of the original detector competition weight kept during the temporary retreat window. Set to `0.0` to fully mute the helper while the hold is active.

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
- **`transform_intensity`**: Legacy combined augmentation strength. If `physical_transform_intensity` and `visual_transform_intensity` are not explicitly set, both groups inherit this value for backward compatibility.
- **`physical_transform_intensity`**: Strength of physically motivated input perturbations. This controls small input-plane rotation, translation, and the additive input noise used to mimic propagation/coupling fluctuations.
- **`physical_input_noise_std`**: Base standard deviation of additive Gaussian noise for the input field. The effective noise is `physical_input_noise_std * physical_transform_intensity`.
- **`visual_transform_intensity`**: Strength of generic image-style augmentations. This controls input scaling, brightness/contrast jitter, perspective warping, and sharpness perturbation.
- **`inherit_best_model`**: If true, automatically loads the weights from the latest `best_model.pth` found in `results_dir` before starting.
- **`inherit_model_path`**: (Optional) If `inherit_best_model` is true, this string determines which experiment's model to inherit. It matches experiment folders containing this string (ignoring dates). If left empty (`""`), it defaults to matching the current `exp_name`. If no matching model is found, it falls back to inheriting the absolute newest model across all experiments (and prints a warning).
- **`save_csv_logs`**: If true, saves the Phase Mask and Detector Positions as CSV files.
- **`save_csv_logs_every_epoch`**: If true, saves CSVs every epoch (WARNING: Very slow). If false, only saves at the last epoch.
