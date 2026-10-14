# Physics Subpackage — Agent Routing Guide

Covers spatial pickup aperture integration, multi-scale string continuum mechanics, inharmonicity dispersion, and causal minimum-phase prefilter synthesis.

## Key Modules
- [`aperture.py`](aperture.py): 2D cylindrical rod vs 1D blade sensing apertures, standing-wave comb envelopes, close-bridge saddle stiffness, bridge displacement excursion scaling, and luthier pickup isolation leveling.
- [`strings.py`](strings.py): Log-spaced wave-speed continuum routing, Gaussian RBF stiffness inharmonicity solver, and viscoelastic string wrap damping.
- [`schema.py`](schema.py): Pydantic schemas and type definitions for spatial pickup geometry and string continuum parameters.

## Required Reading (Single Source of Truth)
Before modifying equations, spatial coordinates, or continuum routing in this directory, read:
1. [`docs/aperture_math.md`](../../../docs/aperture_math.md) — Canonical derivations for aperture kernels, wave-speed continuum, multi-pickup delays, saddle stiffness, and upright bridge force sensors.
2. [`docs/architectural_guardrails.md`](../../../docs/architectural_guardrails.md) — Cross-cutting $C^\infty$ smoothness primitives (`cinf_smoothstep`, `alg4`) and Guardrail 5.1 contracts.
3. [`docs/clamp_and_limit_registry.md`](../../../docs/clamp_and_limit_registry.md) — Registered bounds (`CL-04`, `CL-19`, `CL-20`, `CL-26`).

## Verification
```bash
uv run pytest tests/physics/ tests/test_guardrails.py
```
