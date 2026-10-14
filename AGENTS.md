# Allomorph — Agent Development & Discovery Guide

**Allomorph** is a forward electroacoustic Digital Twin simulation and Neural Amp Modeler (NAM) A2 training pipeline. It simulates standalone **Forward Digital Twins** of source and target bass instruments from a shared string excitation (`audio/input.wav`), pairing `Source Digital Twin → Target Digital Twin` so a trained NAM model in **Darkglass Anagram Block 1** transforms the physical source bass into the target instrument (`Physical Twin A ≈ Digital Twin A → NAM → Digital Twin B ≈ Physical Twin B`).

## 1. Global Tech Stack & Tooling Rules

- **Environment:** Always use **`uv`** (`uv run`, `uv add`).
- **Core Stack:** Strictly use **`numpy`**, **`polars`**, **`altair`**, **`pedalboard`**, **`pydantic`**, and **`numba`**. Never add `scipy`, `pandas`, `matplotlib`, or `soundfile`.
- **Lint & Format:** Pass `uv run ruff check` and `uv run ruff format --check` targeting Python 3.14 (`py314`).
- **Strict Typing:** Pass `uv run pyrefly check` (`preset = "strict"` in [`pyproject.toml`](pyproject.toml)). Never disable rules, downscale presets, or use `# type: ignore`.
- **Testing:** Run `uv run pytest`. All changes must pass [`test_guardrails.py`](tests/test_guardrails.py).
- **Git & CLA:** Sign off commits via `git commit -s` per [`CLA.md`](CLA.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Search Hygiene:** Always exclude `*.nam` and `*.wav` files when searching (`rg`, `git grep`, `grep`).

## 2. Core Cross-Cutting Principles

1. **Fail-Fast Integrity:** Never silently substitute default scales, pickups, voices, or circuits on missing/invalid configs; raise explicit `ValueError` or `KeyError` exceptions.
2. **$C^\infty$ Smoothness:** Never use piecewise `np.where` kinks on physical or DSP curves; use smooth analytic transitions per [`architectural_guardrails.md`](docs/architectural_guardrails.md).
3. **Catalog Clamp Validation:** Empirically validate all clamps across the catalog matrix and register them in [`clamp_and_limit_registry.md`](docs/clamp_and_limit_registry.md).
4. **Physical Primacy:** First-principles equations are ground truth; update legacy test expectations when refining physical formulas rather than warping equations.

## 3. Progressive Disclosure & Codebase Discovery Map

Consult the directory-scoped `AGENTS.md` and Single-Source-of-Truth (SSOT) docs before modifying a subsystem:

| Subsystem / Domain | Scoped `AGENTS.md` | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| **Invariants & $C^\infty$ Math** | [`AGENTS.md`](AGENTS.md) | [`architectural_guardrails.md`](docs/architectural_guardrails.md), [`clamp_and_limit_registry.md`](docs/clamp_and_limit_registry.md) |
| **Spatial & Scale Physics** | [`physics/AGENTS.md`](src/allomorph/physics/AGENTS.md) | [`aperture_math.md`](docs/aperture_math.md) |
| **MNA Circuits & Saturation** | [`circuit/AGENTS.md`](src/allomorph/circuit/AGENTS.md) | [`circuit_theory.md`](docs/circuit_theory.md), [`dsp_simulation_engine.md`](docs/dsp_simulation_engine.md) |
| **Excitation, FIR, LUFS & I/O** | [`dsp/AGENTS.md`](src/allomorph/dsp/AGENTS.md) | [`dsp_synthesis_engine.md`](docs/dsp_synthesis_engine.md) |
| **NAM A2 Neural Trainer** | [`trainer/AGENTS.md`](src/allomorph/trainer/AGENTS.md) | [`trainer/README.md`](docs/trainer/README.md) |
| **Polars + Altair Visualizer** | [`visualizer/AGENTS.md`](src/allomorph/visualizer/AGENTS.md) | [`architectural_guardrails.md`](docs/architectural_guardrails.md) |
| **Packaging, Naming & CLI** | [`pipeline/AGENTS.md`](src/allomorph/pipeline/AGENTS.md) | [`naming.md`](docs/tone_packs/naming.md), [`ranking.md`](docs/tone_packs/ranking.md), [`catalog.md`](docs/tone_packs/catalog.md) |
| **TOML Configs & Voices** | [`config/AGENTS.md`](config/AGENTS.md) | [`configuration_reference.md`](docs/configuration_reference.md), [`voice_catalog.md`](docs/voice_catalog.md), [`anagram_workflow.md`](docs/anagram_workflow.md) |
| **Test Suite & Guardrails** | [`tests/AGENTS.md`](tests/AGENTS.md) | [`CONTRIBUTING.md`](CONTRIBUTING.md), [`test_guardrails.py`](tests/test_guardrails.py) |

## 4. Maintaining `AGENTS.md` & Documentation

- **Zero Duplication:** Keep `AGENTS.md` files strictly as routing maps (`< 5,000` B root, `< 2,500` B scoped). Never copy formulas, numeric constants, metallurgy tables, or vocabulary lists into `AGENTS.md`.
- **Single Source of Truth:** Document every equation, clamp (`CL-XX`), or schema change in its canonical `docs/` file per [`CONTRIBUTING.md` §5](CONTRIBUTING.md#5-maintaining-agentsmd--documentation-architecture) and link to it.
