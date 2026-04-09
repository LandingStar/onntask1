# Task1 Training Objective And Model Selection Design Notes

## 1. Purpose

This document records the current understanding of Task1's training objective design, with a focus on:

- classification loss
- detector competition helper loss
- global energy concentration loss
- best model selection logic
- scheduler trigger logic
- known failure modes and future refactoring direction

The goal is to preserve the reasoning behind recent debugging work so future changes do not repeat the same confusion.

## 2. High-Level Conclusion

The model capacity is not the root problem.

This was verified by the historical experiment in `task1/main/results/floating_5det_one_20260405_0338`, where the model reached about `99%+` validation accuracy before the recent loss refactor.

Therefore, the main source of later accuracy stagnation is not:

- insufficient ONN capacity
- single-layer physical limit
- dataset intrinsically capped at 90%

The main source is the design and coupling of:

- auxiliary intensity-related losses
- best model score
- learning rate scheduler metric

## 3. Historical Baseline

### 3.1 Old Successful Baseline

The historical run stored in:

- `task1/main/results/floating_5det_one_20260405_0338`

shows the following:

- validation accuracy can rise to about `0.99+`
- intensity ratio can also rise continuously
- the old training setup did not inherently prevent high accuracy

### 3.2 Important Interpretation

The old intensity-related helper term was not merely a physical efficiency term.

It effectively worked in detector space:

- increasing the target detector's normalized share
- decreasing non-target detector responses

This made it much closer to a classification-aligned auxiliary loss than to a pure global optical efficiency objective.

## 4. Current Loss Architecture

Current training logic in `task1/main/train.py` contains three conceptually distinct parts:

### 4.1 Classification Loss

The project now supports two explicit modes for the primary classification objective:

- `competition`
- `mse`

Current intended direction:

- use `competition` as the primary classification objective
- keep `mse` mainly as a reference / comparison path

`competition` means the main classification signal is directly driven by the old detector-space competition term.

`mse` means the main classification signal is driven by detector-normalized MSE.

### 4.2 Detector Competition Loss

This is the old helper term, now explicitly separated.

Definition:

- target detector share among all detectors
- implicitly suppresses non-target detectors

Interpretation:

- classification-aligned helper
- detector-space margin shaping
- useful for making detector outputs more one-hot

Important current design choice:

- this term should remain a plain linear margin
- it should not be wrapped in ReLU / LeakyReLU

Reason:

- historically this helper likely worked because it continued to shape detector competition even after crossing a threshold
- clipping it destroys part of its original behavior

### 4.3 Global Energy Concentration Loss

This is the new physical efficiency term.

Definition:

- target detector energy relative to the entire output plane energy

Interpretation:

- global optical efficiency objective
- encourages background cleanup
- encourages more absolute energy to enter the target detector

Important difference from detector competition:

- detector competition asks: "is target detector brighter than other detectors?"
- global concentration asks: "is target detector receiving enough of the whole plane's energy?"

These are related, but not equivalent.

## 5. Why The New Behavior Caused Confusion

### 5.1 Old Helper And Classification Were Largely Aligned

Both of these used detector-normalized outputs:

- classification MSE
- old detector competition helper

So in practice, the helper often improved classification margin rather than competing with it.

### 5.2 New Global Concentration Is A Different Objective

Global concentration can be beneficial physically, but it is not guaranteed to improve detector classification.

It may instead push the model toward:

- stronger optical concentration
- lower background
- more centered spots

without necessarily improving detector ranking margin.

### 5.3 Small-Weight A/B Experiments Were Misleading

The A/B experiments comparing:

- only detector competition
- only global concentration
- hybrid

gave very similar results in a low-weight regime.

This does **not** prove the two terms are equivalent.

It only shows that under those specific conditions:

- both terms were weak
- both were correlated with the same main classification objective
- the regime was too mild to strongly separate their effects

### 5.4 Why Competition Can Beat One-Hot MSE In A Single-Layer ONN

This is now considered a core design insight for this project.

For this task, the relevant issue is not merely "optimization cost" but also "representation compatibility".

Key reasoning:

- the single-layer ONN does not have the kind of intermediate nonlinear activations used in ordinary deep neural networks
- the optical propagation + phase modulation + detector readout pipeline behaves as a continuous mapping from the sample manifold to detector outputs
- if samples inside a class family carry intrinsic continuity, their detector-space image is also expected to preserve continuous structure

Therefore:

- forcing detector outputs to match a strict one-hot template asks the model to collapse a continuous output structure onto isolated simplex vertices
- for a single-layer continuous optical system, this is not a natural target and may be structurally incompatible with the model's inductive bias
- even if partial approximation is possible, it spends model freedom on removing class-internal output structure rather than improving actual detector-level discrimination

In contrast, detector competition only requires:

- the correct detector should dominate
- the target detector share should increase
- non-target detectors do not need to follow a rigid zero-template shape

This makes detector competition more compatible with:

- continuous sample variation
- single-layer optical constraints
- the actual `compute_acc()` criterion, which mainly cares about detector ranking rather than perfect one-hot geometry

Practical interpretation:

- `loss_vec` is template-fitting supervision
- `competition_loss` is detector-ranking supervision

For this project, the latter is now believed to be a better primary classification objective.

### 5.5 Discovery History

This debugging chain is important and should be preserved because it explains why the design direction changed.

Observed sequence:

1. the old concentrating-loss logic was replaced with a newer global concentration objective
2. after the replacement, training behavior degraded and model-selection behavior became suspicious
3. score / best-model / scheduler interactions were inspected and partially corrected
4. old-style detector competition behavior was reintroduced and tested
5. low-weight A/B tests showed the two intensity terms could look deceptively similar
6. deeper comparison with historical results showed the old detector competition path was likely helping classification itself
7. switching the primary classification objective back toward detector competition immediately produced a much faster accuracy rise, including very high accuracy in the first epoch of the new test
8. a later bug was found in the LR scheduler migration: `step()` used the new explicit metric semantics, but `ReduceLROnPlateau(mode=...)` was still initialized using the old `"acc"` naming assumption, so `scheduler_metric="val_acc"` incorrectly behaved like a `min` metric and the LR never decayed as intended

Current interpretation:

- the main issue was not model capacity
- the main issue was that one-hot-style detector MSE overconstrained the detector output shape for a single-layer ONN
- the older detector competition objective matched the optical model and the accuracy criterion better

## 6. Best Model And Scheduler Problem

This was one of the most important findings.

### 6.1 Current Best Model Logic

Current best model saving uses a composite score:

- validation accuracy
- plus an intensity term

This is not the same as pure validation accuracy.

### 6.2 Observed Failure Mode

In experiments such as:

- `task1/main/results/ab_hybrid_two_term_50ep_20260409_0119`

`Best Model Saved` appeared only in very early epochs, even though later epochs had much better validation accuracy.

Why:

- the chosen score still preferred early high intensity
- later checkpoints improved accuracy but did not beat the early score

### 6.3 Scheduler Naming Problem

Historically, the config name `scheduler_metric = "acc"` was semantically misleading.

It could still route into composite-score behavior rather than pure `val_acc`.

This caused two problems:

- users thought scheduler followed accuracy only
- actual LR decay behavior could still be influenced by intensity

Current direction:

- use explicit metric names
- prefer `val_acc`, `val_loss`, and `composite_score`
- keep legacy aliases only for backward compatibility

## 7. Current Design Principles

The following principles should guide future work.

### 7.1 Separate Training Objective From Model Selection

These must be independent:

- backprop objective
- best model save criterion
- scheduler criterion

They should never be implicitly coupled through reused variables with different meanings.

### 7.2 Keep Detector Competition As Classification Helper

Detector competition should be treated as:

- a detector-space helper
- an accuracy-supporting auxiliary term

It is not primarily a physical global-efficiency term.

### 7.3 Treat Global Concentration As Later-Stage Objective

Global concentration should usually be:

- off in early classification-first training
- introduced only after validation accuracy reaches a threshold
- or weighted very carefully

Current intended direction:

- gate it by `global_concentration_start_acc`

Otherwise it can distort the optimization target.

### 7.4 Never Let Early Intensity Dominate Best Model Selection

If classification is the main target, early large intensity values must not dominate:

- checkpoint saving
- scheduler decisions

## 8. Recommended Refactor Plan

### Phase 1: Clarify Metric Semantics

Introduce explicit names:

- `best_model_metric`
- `scheduler_metric`

Allowed values should be explicit and honest:

- `val_acc`
- `val_loss`
- `composite_score`

Do not overload `"acc"` to mean a score that already contains intensity.

### Phase 2: Save Multiple Best Checkpoints

Recommended outputs:

- `best_acc_model.pth`
- `best_score_model.pth`

Optional:

- `best_global_concentration_model.pth`

This prevents early intensity-heavy checkpoints from overwriting classification-best checkpoints.

### Phase 3: Make Stage-Gated Training Explicit

Recommended logic:

- early stage: `loss_vec + detector_competition_loss`
- later stage after accuracy threshold: add `global_concentration_loss`

This can be done either by:

- explicit threshold gating
- or phased training

### Phase 4: Improve Logging

The following should be written explicitly every epoch:

- `loss_vec`
- `detector_competition_loss`
- `global_concentration_loss`
- `target_penalty`
- `avg_detector_competition_ratio`
- `avg_global_concentration_ratio`
- `current_score`
- actual metric used for scheduler
- actual metric used for best-model comparison

Without this, future debugging will keep repeating.

### Phase 5: Reduce Legacy Config Ambiguity

Legacy fields still accepted now:

- `spatial_mask_loss_weight`
- `auto_spatial_mask_target_ratio`
- `use_global_energy_ratio`

These should eventually be deprecated in favor of explicit fields:

- `detector_competition_loss_weight`
- `detector_competition_target_ratio`
- `global_energy_concentration_loss_weight`
- `global_energy_concentration_target_ratio`

## 9. Practical Working Assumptions Going Forward

Until the refactor is completed, future work should assume:

- detector competition is the more likely source of old high-accuracy behavior
- global concentration is useful, but should not be assumed classification-friendly
- composite score should not be trusted as a proxy for pure classification quality
- any experiment comparing old/new intensity terms must explicitly log both ratios
- the "single-layer continuous mapping vs one-hot vertex fitting" explanation is strong enough to mention in task handoff / progress reports, but should still be described as a design conclusion rather than a formal proof

## 10. Current Status Snapshot

As of this document:

- detector competition and global concentration have been separated in code
- detector competition has been restored as a linear margin term without ReLU/LeakyReLU
- `classification_loss_type` now supports making detector competition the primary classification objective
- the old successful training behavior strongly suggests detector competition was helping classification
- `scheduler_metric` is being migrated toward explicit semantics: `val_acc`, `val_loss`, `composite_score`
- best-checkpoint saving is being split into dedicated files such as `best_acc_model.pth`, `best_loss_model.pth`, and `best_score_model.pth`
- `best_model.pth` should represent the checkpoint chosen by explicit `best_model_metric`
- per-term train/validation loss components and both intensity ratios are now being logged for easier debugging
- `global_concentration_loss` is being moved toward explicit stage-gated activation by validation accuracy

## 11. Next Recommended Code Changes

The next safe, high-value changes should be:

1. validate the new `best_model_metric` and explicit `scheduler_metric` behavior on real runs
2. confirm `best_model.pth` now matches user intent under `val_acc`, `val_loss`, and `composite_score`
3. verify the new logging fields are sufficient to explain future training plateaus
4. validate whether `classification_loss_type="competition"` restores the historical high-accuracy behavior

These changes should happen before any more complicated loss redesign.
