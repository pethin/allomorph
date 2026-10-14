# Neural Trainer Subpackage — Agent Routing Guide

Covers Neural Amp Modeler (NAM) Architecture 2 (A2) training engines (`native` in-VRAM zero-copy PyTorch engine and `stock` PyTorch Lightning engine), loss formulations, and dual-tier slimmable `.nam` export.

## Key Modules
- [`constants.py`](constants.py) & [`core.py`](core.py): Canonical hyperparameter defaults (`NATIVE_DEFAULT_*`, `STOCK_DEFAULT_*`), hardware auto-tuning, and `.nam` container validation.
- [`native/`](native/): High-throughput in-VRAM 2D index-gather training loop, unified 11-channel block-masked WaveNet A2 model (`channels_3` + `channels_8`), Blackman FIR highpass regularizer, and Anti-Drift Pareto early stopping.
- [`stock/`](stock/): Upstream PyTorch Lightning / NAM 200-epoch baseline wrapper.

## Required Reading (Single Source of Truth)
Before modifying neural architectures, loss functions, learning rate schedules, or data loading in this directory, read:
1. [`docs/trainer/README.md`](../../../docs/trainer/README.md) — Master NAM pairing theory, dual-engine overview, hyperparameter standards, and CLI workflows.
2. [`docs/trainer/native/engine_architecture.md`](../../../docs/trainer/native/engine_architecture.md) & [`docs/trainer/native/spectral_loss_and_dynamics.md`](../../../docs/trainer/native/spectral_loss_and_dynamics.md) — Native A2 architecture and loss specifications.
3. [`docs/trainer/native/what_did_not_work_and_regression_prevention.md`](../../../docs/trainer/native/what_did_not_work_and_regression_prevention.md) — Forensic analysis of anti-patterns and regression prevention rules.

## Verification
```bash
uv run pytest tests/pipeline/ tests/integration/
```
