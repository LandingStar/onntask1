# Configuration Contract

## Purpose

This document records the intended semantics of config fields used by `task1`.

It should explain what a field means, not only where it is read.

## Naming Principles

- One field should have one meaning.
- Metric names should match actual runtime behavior.
- Legacy compatibility fields should be explicitly marked.

## Training Objective Fields

- `detector_competition_loss_weight`
  - Intended meaning:
    - To be filled.
- `detector_competition_target_ratio`
  - Intended meaning:
    - To be filled.
- `global_energy_concentration_loss_weight`
  - Intended meaning:
    - To be filled.
- `global_energy_concentration_target_ratio`
  - Intended meaning:
    - To be filled.

## Model Selection Fields

- `best_model_metric`
  - Status:
    - Planned / to be added.
- `best_model_acc_weight`
  - Intended meaning:
    - To be filled.
- `best_model_intensity_weight`
  - Intended meaning:
    - To be filled.
- `score_intensity_source`
  - Intended meaning:
    - To be filled.
- `score_intensity_cap`
  - Intended meaning:
    - To be filled.

## Scheduler Fields

- `scheduler_metric`
  - Current ambiguity:
    - To be filled.
- `scheduler_patience`
  - Intended meaning:
    - To be filled.

## Legacy Compatibility Fields

- `spatial_mask_loss_weight`
- `auto_spatial_mask_target_ratio`
- `use_global_energy_ratio`

Notes:

- These are still accepted for backward compatibility.
- Their future deprecation path should be documented here.
