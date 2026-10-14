# Configuration Directory — Agent Routing Guide

Covers declarative TOML configurations for physical source instruments, target voicings, Tone3000 packs, active preamps, scale lengths, and string mechanics presets, loaded via [`src/allomorph/config/`](../src/allomorph/config/).

## Key Files & Directories
- [`instruments/`](instruments/): Source bass definitions with embedded `[pickups.<id>]`, `[harnesses.<id>]`, and `[voicings.<id>]`.
- [`packs/`](packs/): Declarative Tone3000 storefront pack definitions and bundle mappings.
- [`preamps.toml`](preamps.toml), [`scales.toml`](scales.toml), [`strings.toml`](strings.toml): Reusable active preamp EQ topologies, scale wave speeds, and string core/wrap presets.

## Required Reading (Single Source of Truth)
Before adding or modifying TOML configurations or Pydantic config loaders in [`src/allomorph/config/`](../src/allomorph/config/), read:
1. [`docs/configuration_reference.md`](../docs/configuration_reference.md) — Complete TOML schema specification, required fields, physical units, and hardware constraints.
2. [`docs/voice_catalog.md`](../docs/voice_catalog.md) — Master catalog of target pickup voicings, RLC parameters, and harnesses.
3. [`docs/tone_packs/naming.md`](../docs/tone_packs/naming.md) — Normative tone naming rules and taxonomy vocabulary for `tone_name` fields.
4. [`docs/architectural_guardrails.md`](../docs/architectural_guardrails.md) — Transducer `sensor_type` taxonomy and fail-fast declarative validation rules.

## Verification
```bash
uv run pytest tests/config/ tests/tone3000/ tests/test_guardrails.py
```
