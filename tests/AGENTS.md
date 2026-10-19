# Automated Test Suite — Agent Routing Guide

Covers unit, property-based (Hypothesis), integration, and architectural invariant tests across the three isolated package suites (`allomorph`, `reamper`, and `denoiser`).

## Key Suites
- [`allomorph/`](allomorph/): Digital Twin simulation, DSP, MNA circuits, visualizer, and Tone3000 packaging tests:
  - [`test_guardrails.py`](allomorph/test_guardrails.py): Programmatic verification of cross-cutting mathematical, physical, naming, and documentation invariants.
  - [`circuit/`](allomorph/circuit/), [`config/`](allomorph/config/), [`dsp/`](allomorph/dsp/), [`physics/`](allomorph/physics/), [`pipeline/`](allomorph/pipeline/), [`tone3000/`](allomorph/tone3000/), [`visualizer/`](allomorph/visualizer/), [`integration/`](allomorph/integration/): Domain-specific test suites.
  - [`strategies.py`](allomorph/strategies.py): Shared Hypothesis property-based testing strategies.
- [`reamper/`](reamper/): Hardware USB re-amping, closed-loop gain calibration, latency alignment, and re-amped rig NAM A2 trainer tests.
- [`denoiser/`](denoiser/): Universal Bass Denoiser (50/60 Hz hum + SMPS noise synthesis, $C^\infty$ envelope loss, and native A2 trainer/CLI) tests.

## Required Reading (Single Source of Truth)
Before adding or modifying tests in this directory, read:
1. [`CONTRIBUTING.md`](../CONTRIBUTING.md) (Section 4) — Test coverage targets, fast execution budget, bounded synthetic buffer rules (`tmp_path`), TDD workflow, and strict `pyrefly` typing requirements for test files.
2. [`docs/architectural_guardrails.md`](../docs/architectural_guardrails.md) — Primacy of first-principles physical equations over legacy hardcoded test thresholds and the Master Invariant Registry.

## Verification
```bash
uv run ruff check tests/
uv run pyrefly check
uv run pytest
```
