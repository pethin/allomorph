# Pipeline & Tone Pack Orchestration Subpackage — Agent Routing Guide

Covers Tone3000 upload bundle partitioning, storefront tone pack assembly, and cryptographic recipe-token manifest provenance.

## Key Modules
- [`pack.py`](pack.py) & [`schema.py`](schema.py): Tone3000 storefront pack generation, description formatting, and Pydantic schema validation.
- [`bundle.py`](bundle.py): Pickup position affinity partitioning (`bundles/<pickup>/`) and dry/wet stem staging.
- [`manifest.py`](manifest.py): Cryptographic token provenance and audio QA telemetry verification.
- [`../naming.py`](../naming.py) & [`../version.py`](../version.py): Tone naming grammar, character budget enforcement, and tri-part semantic versioning (`v[dsp].[inst].[voice]`).

## Required Reading (Single Source of Truth)
Before modifying tone naming, bundle partitioning, storefront copy, or manifest schemas, read:
1. [`docs/tone_packs/naming.md`](../../../docs/tone_packs/naming.md) — Normative tone naming grammar, taxonomy dictionary, and pedalboard/Tone3000 character budgets.
2. [`docs/tone_packs/ranking.md`](../../../docs/tone_packs/ranking.md) & [`docs/tone_packs/catalog.md`](../../../docs/tone_packs/catalog.md) — Storefront voicing ranking architecture and active pack catalog.
3. [`docs/trainer/README.md`](../../../docs/trainer/README.md) — Training audio pairings and `allomorph audit` QA telemetry standards.

## Verification
```bash
uv run pytest tests/pipeline/ tests/tone3000/ tests/test_guardrails.py
```
