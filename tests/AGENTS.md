# Automated Test Suite — Agent Routing Guide

Covers unit, property-based (Hypothesis), integration, and architectural invariant tests across all Allomorph subpackages.

## Key Suites
- [`test_guardrails.py`](test_guardrails.py): Programmatic verification of cross-cutting mathematical, physical, naming, and documentation invariants.
- [`circuit/`](circuit/), [`config/`](config/), [`dsp/`](dsp/), [`physics/`](physics/), [`pipeline/`](pipeline/), [`tone3000/`](tone3000/), [`visualizer/`](visualizer/), [`integration/`](integration/): Domain-specific test suites.
- [`strategies.py`](strategies.py): Shared Hypothesis property-based testing strategies.

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
