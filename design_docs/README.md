# Task1 Design Docs

## Purpose

This folder stores project-level design notes for `task1`.

It is intended to capture:

- stable architecture decisions
- training objective design
- configuration semantics
- experiment methodology
- unresolved technical questions

The goal is to avoid repeatedly reconstructing context from chats, logs, and code.

## Suggested Reading Order

1. `project_overview.md`
2. `training_objective_and_model_selection.md`
3. `configuration_contract.md`
4. `experiment_workflow.md`
5. `open_questions.md`

## Document Map

### Core

- [project_overview.md](file:///e:/workspace/onn%20training/task1/design_docs/project_overview.md)
  - Project purpose, code boundaries, key modules, and terminology.

- [training_objective_and_model_selection.md](file:///e:/workspace/onn%20training/task1/design_docs/training_objective_and_model_selection.md)
  - Current understanding of classification loss, detector competition loss, global concentration loss, best-model selection, and scheduler issues.

- [configuration_contract.md](file:///e:/workspace/onn%20training/task1/design_docs/configuration_contract.md)
  - Naming rules and intended semantics of major config fields.

- [experiment_workflow.md](file:///e:/workspace/onn%20training/task1/design_docs/experiment_workflow.md)
  - How to design, compare, and interpret experiments.

- [open_questions.md](file:///e:/workspace/onn%20training/task1/design_docs/open_questions.md)
  - Active uncertainties and next questions to validate.

## Maintenance Rules

- Prefer adding short, stable conclusions over chat-style notes.
- When a design decision changes, update the corresponding document rather than creating scattered one-off notes.
- If a finding is experiment-specific and not yet stable, record it under `open_questions.md` first.
- If a config field changes meaning, update both this folder and `main/config_readme.md`.
