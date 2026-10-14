# Circuit & Virtual Analog Simulation Subpackage — Agent Routing Guide

Covers declarative Modified Nodal Analysis (MNA) circuit solving, complex impedance dispersion, oversampled state-space magnetic/op-amp non-linear saturation, and Forward Digital Twin waveform synthesis.

## Key Modules
- [`solver.py`](solver.py) & [`mna/`](mna/): Vectorized AC nodal transfer solver, active preamp EQ, Foster core eddy ladder, Jordan relaxation, Alnico skin dispersion, Cole-Davidson dielectric relaxation, and coupled $2\times 2$ mutual coil matrices.
- [`saturation.py`](saturation.py): Numba-accelerated (`@njit`) ODE state-space magnetic saturation, Lenz velocity drag, Dahl hysteresis, 2-block active op-amp headroom, and order-8 `fsqrt` rail limiting.
- [`forward.py`](forward.py) & [`sweeps.py`](sweeps.py): Branch-wise pre-summing Forward Digital Twin audio pipeline and parametric circuit sweeps.
- [`parser.py`](parser.py) & [`schema.py`](schema.py): SPICE unit parsing, potentiometer tapers, and `MAGNET_PROPERTIES` metallurgy registry.

## Required Reading (Single Source of Truth)
Before modifying circuit equations, metallurgy parameters, or saturation kernels in this directory, read:
1. [`docs/circuit_theory.md`](../../../docs/circuit_theory.md) — Canonical MNA circuit equations, active preamp topologies, impedance dispersion, and 15-parameter non-linear magnetic physics.
2. [`docs/dsp_simulation_engine.md`](../../../docs/dsp_simulation_engine.md) — SIMD execution pipeline, frequency-domain stage fusion, 2x/4x passband-constrained oversampling, and `allomorph-sim` CLI reference.
3. [`docs/architectural_guardrails.md`](../../../docs/architectural_guardrails.md) & [`docs/clamp_and_limit_registry.md`](../../../docs/clamp_and_limit_registry.md) — Cross-cutting $C^\infty$ primitives and active clamp registry (`CL-10`, `CL-13`, `CL-14`, `CL-21`, `CL-22`, `CL-29`).

## Verification
```bash
uv run pytest tests/circuit/ tests/test_guardrails.py
```
