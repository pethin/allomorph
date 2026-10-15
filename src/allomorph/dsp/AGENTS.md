# DSP Synthesis & Audio I/O Subpackage — Agent Routing Guide

Covers master dry string excitation synthesis (`audio/input.wav`), biaxial modal string plucks, homomorphic real-cepstrum minimum-phase FIR generation, ITU-R BS.1770-4 gated loudness calibration, and 24-bit PCM WAV I/O.

## Key Modules
- [`dry.py`](dry.py), [`plucks.py`](plucks.py), [`sweeps.py`](sweeps.py): Master excitation assembler, 2D modal plucks with pitch sag, and physical string-tilted logarithmic chirps.
- [`fir.py`](fir.py) & [`conditioning.py`](conditioning.py): Pure real-FFT homomorphic cepstrum minimum-phase FIR synthesis with `cinf_smoothstep` tail windowing (`fir.py`), and 8 Hz DC blocking, RLC-colored Johnson-Nyquist thermal dither (`-108 dBFS`), and calibrated LUFS/peak normalization (`conditioning.py`).
- [`loudness.py`](loudness.py): ITU-R BS.1770-4 K-weighted gated LUFS measurement, 4x oversampled FFT sinc true-peak detection, and linear peak-based safety scaling.
- [`convolution.py`](convolution.py) & [`io.py`](io.py): Overlap-save SIMD FFT convolution, C-view 24-bit little-endian byte packing, and `pedalboard` WAV I/O.

## Required Reading (Single Source of Truth)
Before modifying excitation signals, FIR synthesis, loudness normalization, or PCM packing in this directory, read:
1. [`docs/dsp_synthesis_engine.md`](../../../docs/dsp_synthesis_engine.md) — Canonical specification for string tilt laws, modal plucks, BS.1770-4 LUFS leveling, true-peak safety, and real-cepstrum FIR synthesis.
2. [`docs/architectural_guardrails.md`](../../../docs/architectural_guardrails.md) & [`docs/clamp_and_limit_registry.md`](../../../docs/clamp_and_limit_registry.md) — $C^\infty$ mollifier (`cinf_smoothstep`) and registered limits (`CL-16`, `CL-17`, `CL-23`, `CL-27`, `CL-28`, `CL-33`, `CL-34`).

## Verification
```bash
uv run pytest tests/dsp/ tests/test_guardrails.py
```
