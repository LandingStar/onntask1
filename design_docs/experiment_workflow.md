# Experiment Workflow

## Purpose

This document defines how experiments should be designed, compared, and interpreted.

## Standard Experiment Checklist

- State the hypothesis.
- State what config fields are changed.
- State what fields are frozen.
- Define primary metric.
- Define secondary metrics.
- Define what result would falsify the hypothesis.

## Recommended Comparison Types

### A/B Loss Comparison

- What changed:
  - To be filled.
- What must remain fixed:
  - To be filled.
- What to compare:
  - `val_acc`
  - intensity metrics
  - best-model selection behavior

### Scheduler / Model Selection Comparison

- What changed:
  - To be filled.
- What to compare:
  - epoch of LR decay
  - epoch of best-model save
  - final vs saved checkpoint quality

## Required Outputs To Preserve

- `config.json`
- `training_log.txt`
- `metrics.csv`
- `loss_acc.png`

## Interpretation Rules

- Do not infer causality from one run.
- Do not compare experiments if multiple unrelated fields changed.
- If two metrics disagree, identify which metric actually drove checkpoint saving and scheduler behavior.
