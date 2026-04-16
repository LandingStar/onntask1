# Current Status Review

## 1. Source Of Truth

- Actual run behavior must be judged from `results/<run>/config.json`, `training_log.txt`, and `metrics.csv`.
- Files under `test/` or `batch_config/` are only templates. If they are edited later, they may no longer match the run that already happened.
- Recent experiments showed this clearly: a config intended as a short phase-2 run was actually saved with `epochs = 256` and `learning_rate = 1e-3` in the run directory. Always trust the saved run config.

## 2. Current Effective Training Chain

### 2.1 Input Perturbation

- Input augmentation is now split into two groups:
  - `physical_transform_intensity`
  - `visual_transform_intensity`
- Physical group currently covers:
  - small input rotation
  - small input translation
  - additive Gaussian input noise (`physical_input_noise_std`)
  - model-side misalignment simulation:
    - `misalignment_translation_max_pixels`
    - `misalignment_rotation_max_degrees`
    - `misalignment_tilt_max_degrees`
- Visual group currently covers:
  - input scaling
  - brightness / contrast jitter
  - perspective warp
  - sharpness perturbation
- Validation data does not use the training-time random augmentation chain.

### 2.2 Forward / Detector Metrics

- `raw_detectors` gives per-detector energy before the final normalized label vector.
- `output_vec` / `out_label` is the detector-space classification vector used for class prediction.
- The current code tracks three baseline physically relevant validation metrics:
  - `avg_detector_competition_ratio`
  - `avg_global_concentration_ratio`
  - `avg_outside_average_ratio`
- A derived engineering metric is also tracked:
  - `avg_physical_focus_ratio = global_concentration_ratio - score_outside_penalty_weight * outside_average_ratio`
- For the newer spatial objective path, the code also computes:
  - `avg_kernel_attract_ratio`
  - `avg_target_ring_average_ratio`
  - `avg_spatial_focus_ratio = kernel_attract_ratio - score_ring_penalty_weight * target_ring_average_ratio`

### 2.3 Loss Structure

- `classification_loss_type = "mse"`
  - primary classification term is detector-vector MSE (`loss_vec`)
- `classification_loss_type = "competition"`
  - primary classification term becomes the detector competition share
  - detector helper weight is zeroed in that branch
- Detector helper currently supports:
  - `detector_competition_loss_mode = "legacy_relu_helper"`
    - historically effective path
  - `detector_competition_loss_mode = "raw_margin"`
    - keeps rewarding ratio growth after target is exceeded
- Global physical objective currently supports:
  - `global_physical_objective_mode = "global_ratio"`
    - thresholded ratio-style objective
  - `global_physical_objective_mode = "target_inside_outside"`
    - reward target-inside average intensity
    - penalize outside average intensity
  - `global_physical_objective_mode = "target_attract_ring"`
    - reward a continuous attraction kernel centered on the target detector
    - optionally penalize the detector outer-ring region to suppress bright halos

### 2.4 Stage Control

- Global term starts only after `global_concentration_start_acc` is met, unless the threshold is non-positive.
- `global_concentration_hold_epochs` can keep the global term active for a sticky window.
- `detector_competition_decay_after_global` can gradually reduce detector-helper weight after global activation.
- `aggressive_intensity_optimization` can multiply active intensity-related weights after a second threshold is met.

### 2.5 Scheduling And Checkpoint Selection

- `scheduler_metric` controls LR scheduling:
  - `val_acc`
  - `val_loss`
  - `composite_score`
- `best_model_metric` controls which checkpoint is copied to `best_model.pth`:
  - `val_acc`
  - `val_loss`
  - `composite_score`
- `score_intensity_source` controls which intensity metric participates in `composite_score`:
  - `detector_competition`
  - `global_concentration`
  - `physical_focus`
  - `spatial_focus`
- Current implementation detail:
  - `composite_score = val_acc * best_model_acc_weight + min(intensity, score_intensity_cap) * best_model_intensity_weight`
  - for `spatial_focus`, this is risky because the metric scale is not guaranteed to match the older `global_ratio` scale
  - a hard cap can therefore flatten score dynamics almost immediately after accuracy saturates

### 2.6 Inheritance

- If `inherit_best_model = true`, training tries to initialize from an existing `best_model.pth`.
- If `inherit_model_path` is set, matching is done using that string.
- If it is empty, the code falls back to matching `exp_name`, and if still unmatched it may fall back to the newest available `best_model.pth`.
- Because of this, inheritance must always be verified in the saved run config and training log.

## 3. What Has Been Verified So Far

### 3.1 Strongly Supported Findings

- The historical high-accuracy path is real:
  - `classification_loss_type = "mse"`
  - `detector_competition_loss_mode = "legacy_relu_helper"`
- This path restored the one-layer model to the `99%+` accuracy regime.

- `raw_margin` is not equivalent to the historical helper path.
- It keeps rewarding detector-ratio growth even after the target is exceeded.
- In practice it tended to drag `val_loss` negative and stall accuracy around the low-90% range.

### 3.2 Global-Ratio Schedule Tuning

- Repeated schedule experiments were run:
  - later start
  - earlier start
  - hold windows
  - helper decay
  - stronger global weights
  - phase-2 fine-tuning
- The consistent outcome was:
  - accuracy can often be preserved
  - `avg_global_concentration_ratio` only moves slightly
  - changing schedule alone does not produce a meaningful physical gain

### 3.3 Phase-2 With `target_inside_outside`

- The aggressive version proved the new objective has very strong drive.
- With large weights, it can quickly destroy classification while pushing the new physical objective.
- The short mild / medium / strong sweeps then showed the opposite side:
  - stable training
  - only very small gains
- Current interpretation:
  - the direction is not obviously nonsense
  - but in its current form it does not look like a high-leverage improvement path

### 3.4 Train / Val Gap

- The persistent `val_acc > train_acc` pattern is likely explained by training-only perturbations:
  - data augmentation
  - model-side misalignment simulation
- This does not automatically mean harmful overfitting.
- It does mean the train distribution is harder than the validation distribution.

### 3.5 Phase-2 A/B With `target_attract_ring`

- Two control-variable phase-2 runs were completed:
  - `floating_5det_one_phase2_jitter005_attract_only_20260416_0003`
  - `floating_5det_one_phase2_jitter005_attract_ring_20260416_0154`
- Shared settings included:
  - `classification_loss_type = "mse"`
  - `detector_competition_loss_mode = "legacy_relu_helper"`
  - `scheduler_metric = "composite_score"`
  - `best_model_metric = "composite_score"`
  - `score_intensity_source = "spatial_focus"`
  - `inherit_best_model = true`
  - `inherit_model_path = "floating_5det_one_compare_helper_legacy_relu_mse_50ep"`
  - `position_jitter = 0.05`
- The only intended difference was ring suppression:
  - `attract_only`: `global_outer_ring_penalty_weight = 0.0`, `score_ring_penalty_weight = 0.0`
  - `attract_ring`: `global_outer_ring_penalty_weight = 0.2`, `score_ring_penalty_weight = 0.2`

- The outcome was weak:
  - both runs reached `val_acc = 1.0`
  - `evaluation_samples.png` looked visually very similar
  - no clear halo suppression benefit was visible in the ring-penalty run
- On the saved run metrics, `attract_ring` did not beat `attract_only`:
  - best `val_loss` was slightly worse
  - `avg_global_concentration_ratio` was slightly lower
  - `avg_outside_average_ratio` was not clearly improved
- Current interpretation:
  - the continuous attraction idea may still be valid
  - this mild ring penalty setting does not currently show meaningful added value
  - `target_attract_ring` should not yet replace the current high-accuracy baseline

### 3.6 Evidence Gap In The Saved Metrics

- The current `train.py` writes CSV columns for:
  - `avg_kernel_attract_ratio`
  - `avg_target_ring_average_ratio`
  - `avg_spatial_focus_ratio`
- However, the two completed `phase2_jitter005` run directories did not actually save those columns in their `metrics.csv`.
- Because of that, the most important A/B question cannot yet be answered from saved artifacts alone:
  - did the continuous attraction kernel improve the intended target metric
  - and did ring penalty reduce the outer-ring intensity in a measurable way
- Practical consequence:
  - for this family of experiments, always verify the actual saved CSV header in the run directory before assuming the new metric chain is available.

### 3.7 Three Follow-Up Runs On The `target_attract_ring` Family

- Three additional phase-2 follow-up runs were completed:
  - `floating_5det_one_phase2_jitter005_attract_only_score_resolved_20260416_0934`
  - `floating_5det_one_phase2_jitter005_attract_only_wider_basin_20260416_1104`
  - `floating_5det_one_phase2_jitter005_attract_ring_stronger_20260416_1234`
- Intended purposes:
  - `score_resolved`: increase `composite_score` sensitivity
  - `wider_basin`: test a broader continuous attraction basin
  - `ring_stronger`: test clearly stronger outer-ring suppression

- Outcome summary:
  - all three runs reached `val_acc = 1.0`
  - best `val_loss` improved monotonically:
    - `8.738984`
    - `8.693155`
    - `8.647181`
  - but the saved physical proxy metrics did not improve
  - `avg_global_concentration_ratio` slightly drifted downward across the three runs
  - `avg_outside_average_ratio` slightly worsened
  - `evaluation_samples.png` remained visually very similar across runs
- Current interpretation:
  - these changes appear to improve numerical optimization more than physical concentration behavior
  - there is still no clear evidence that the detector halo problem was meaningfully reduced

### 3.8 Likely Reason The Spatial CSV Columns Were Still Missing

- The saved `metrics.csv` header from the new runs was:
  - detailed enough to include loss breakdowns and `avg_physical_focus_ratio`
  - but still missing `avg_kernel_attract_ratio`, `avg_target_ring_average_ratio`, and `avg_spatial_focus_ratio`
- This saved header matches neither:
  - the current on-disk `train.py`, which writes the three spatial columns
  - nor `train-old.py`, which writes only a minimal six-column CSV
- No launcher under `task1/main` was found that explicitly redirects normal runs to `train-old.py`.
- Strongest current explanation:
  - the completed runs were executed from a stale or alternate code snapshot of `train.py`
  - for example, an unsaved editor buffer or another local copy that was not the same as the current on-disk file
- Practical consequence:
  - when saved artifacts disagree with the current source code, trust the artifacts first and treat code provenance as unresolved until the mismatch is explained

## 4. Current Best Understanding

- The classification-success chain is now understood reasonably well.
- The bottleneck is no longer "how to recover accuracy".
- The bottleneck is:
  - how to improve physically meaningful concentration / coupling-related behavior
  - without damaging the already recovered classification path

- Evidence so far suggests that:
  - schedule changes alone are weak
  - weight scaling alone is weak or unstable
  - the exact physical objective is probably the main unresolved issue
- The latest A/B also suggests:
  - adding a mild outer-ring penalty on top of the attraction kernel does not automatically create a visible improvement
  - the present bottleneck is partly objective design, and partly measurement / selection resolution
- The three later follow-up runs strengthen that conclusion:
  - the current `target_attract_ring` family can reduce `val_loss`
  - but it has not yet demonstrated a corresponding physical improvement on the saved proxies

## 5. Practical Rules For Next Experiments

- Always check `results/<run>/config.json` first before interpreting any result.
- Also check the actual saved `metrics.csv` header before relying on a metric that exists in the current code.
- When testing a new physical objective, avoid changing both:
  - the training loss shape
  - and the checkpoint selection rule
  in the same first-pass experiment unless that is explicitly the goal.
- For short phase-2 sweeps, prefer:
  - low LR
  - explicit inheritance
  - a small grid of 2-3 configs
- If a compact epoch log looks confusing, read `metrics.csv` rather than relying only on the single-line `Int.Ratio` summary.
- If `score_intensity_cap` is small and `val_acc` is already saturated, `composite_score` can flatten very early and stop distinguishing later physical improvements.
- If a run artifact disagrees with the current `train.py` behavior, assume the run used a different code snapshot until proven otherwise.
- For `spatial_focus`, do not reuse the old `global_ratio` cap / target scale blindly. Its metric is area-normalized and can live on a materially different numeric range.

## 6. Recommended Focus Right Now

- Do not spend many more runs only tuning thresholds, hold windows, or decay lengths around the same objective.
- Treat the current accuracy-recovery path as a stable baseline.
- Evaluate future changes against that baseline using:
  - best validation accuracy
  - average global concentration ratio
  - average outside average ratio
  - physical focus ratio
- The next high-value direction should still be a better physical target definition, but it must be paired with a verified metric chain that actually lands in saved run artifacts.
- Before launching more `target_attract_ring` sweeps, first resolve:
  - why saved runs are still missing the new spatial CSV columns
  - and whether the launched script is guaranteed to match the current on-disk `train.py`
- Do not spend many more runs tuning the current mild `ring penalty = 0.2` setting unless:
  - the saved metrics for `kernel_attract_ratio`, `target_ring_average_ratio`, and `spatial_focus_ratio` are confirmed
  - or a stronger, more clearly separated ring-suppression setting is being tested on purpose
- A better next scoring design would be:
  - use an accuracy gate first, e.g. only let intensity dominate once `val_acc` exceeds a threshold such as `0.995`
  - use a soft normalization for intensity, such as `tanh` or `log1p`, instead of a hard cap
  - give `spatial_focus` its own reference scale rather than reusing `global_energy_concentration_target_ratio = 0.15`
  - keep scheduler and best-model selection on the same post-gate signal during phase-2

## 7. Next-Chat Handoff

- Current effective baseline:
  - `classification_loss_type = "mse"`
  - `detector_competition_loss_mode = "legacy_relu_helper"`
- This path is the currently validated way to restore the one-layer model back into the `99%+` accuracy regime.

- What has already been learned:
  - `raw_margin` is not equivalent to the historical helper path and usually hurts recovery.
  - Re-tuning the old `global_ratio` schedule mostly gives small physical gains.
  - `target_inside_outside` has real drive, but the aggressive version breaks accuracy and the mild version is too weak.

- Current main question:
  - not how to recover accuracy
  - but how to move energy into the detector more naturally without damaging the recovered classification path
- Working hypothesis:
  - the hard detector-style objective lacks a continuous attraction gradient outside the detector support
  - this likely explains the observed bright halo around detector boundaries

- Latest A/B result:
  - `config_phase2_jitter005_attract_only.json` and `config_phase2_jitter005_attract_ring.json` were run
  - both reached `val_acc = 1.0`
  - visual difference in `evaluation_samples.png` was very small
  - the mild `ring penalty = 0.2` setup did not show clear benefit over `attract_only`
- Important caveat:
  - the saved run `metrics.csv` files did not include `avg_kernel_attract_ratio`, `avg_target_ring_average_ratio`, or `avg_spatial_focus_ratio`
  - so the new objective family is still not fully measurable from saved artifacts alone

- Recommended next step:
  - first ensure the saved run artifacts really contain the new spatial metrics
  - then decide whether to keep exploring the attraction-kernel direction or redesign the physical target again
