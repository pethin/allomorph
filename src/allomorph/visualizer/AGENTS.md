# Visualizer Subpackage — Agent Routing Guide

Covers Polars dataframe generation, Altair (Vega-Lite) frequency response chart compilation, and the self-contained interactive HTML documentation page (`docs/frequency_responses.html`).

## Key Modules
- [`dataframe.py`](dataframe.py): Vectorized continuum frequency response evaluation, global 1D forward curve caching (`_VOICING_CURVE_CACHE`), and voicing comparison data assembly (`build_voicings_comparison_data`).
- [`charts.py`](charts.py): Declarative Altair Vega-Lite chart builders and self-contained interactive HTML Voicing Comparator (`generate_voicings_page`).
- [`schema.py`](schema.py): Declarative `config/visualizer.toml` schema (`VisualizerConfig`, `VisualizerChipConfig`).

## Required Reading (Single Source of Truth)
Before modifying dataframe generation, downsampling steps, or chart builders in this directory, read:
1. [`docs/architectural_guardrails.md`](../../../docs/architectural_guardrails.md) — Visualizer comparator contracts, global target caching mandate, payload bounding, and latency invariants.

## Verification
```bash
uv run pytest tests/visualizer/ tests/test_guardrails.py
```
