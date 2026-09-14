"""
Tests for minimum-phase FIR synthesis, prefilter pipeline, and multi-pickup audio export.
"""

import math

from allomorph.dsp import NUM_TAPS
from allomorph.physics import compute_voice_prefilter_firs


def test_compute_voice_prefilter_fir_30in():
    voice_id = "precision_active"
    firs = compute_voice_prefilter_firs(
        voice_id, src_scale="30in", num_taps=NUM_TAPS, normalize=True
    )
    assert len(firs) == 1
    fir = firs[0]

    assert len(fir) == NUM_TAPS
    max_peak = max(abs(x) for x in fir)
    assert math.isclose(max_peak, 0.99, rel_tol=1e-3)

    # Causal minimum phase: early energy should dominate late energy
    early_energy = sum(x**2 for x in fir[:256])
    late_energy = sum(x**2 for x in fir[1024:])
    assert early_energy > late_energy * 5

    # Tail should taper to near zero
    assert abs(fir[-1]) < 0.01


def test_compute_voice_prefilter_fir_32in():
    voice_id = "stingray_parallel"
    firs = compute_voice_prefilter_firs(
        voice_id, src_scale="32in", num_taps=NUM_TAPS, normalize=True
    )
    assert len(firs) == 1
    fir = firs[0]

    assert len(fir) == NUM_TAPS
    max_peak = max(abs(x) for x in fir)
    assert math.isclose(max_peak, 0.99, rel_tol=1e-3)

    early_energy = sum(x**2 for x in fir[:256])
    late_energy = sum(x**2 for x in fir[1024:])
    assert early_energy > late_energy * 5


def test_compute_voice_prefilter_multiscale():
    voice_id = "dingwall_bridge"
    firs = compute_voice_prefilter_firs(
        voice_id, src_scale="30in", num_taps=NUM_TAPS, normalize=True
    )
    assert len(firs) == 1
    fir = firs[0]

    assert len(fir) == NUM_TAPS
    max_peak = max(abs(x) for x in fir)
    assert math.isclose(max_peak, 0.99, rel_tol=1e-3)

    early_energy = sum(x**2 for x in fir[:256])
    late_energy = sum(x**2 for x in fir[1024:])
    assert early_energy > late_energy * 5


def test_compute_voice_prefilter_firs_single_and_multi():
    # Single-pickup voice -> exactly 1 channel
    firs_single = compute_voice_prefilter_firs("precision_active", src_scale="30in", num_taps=512)
    assert len(firs_single) == 1
    assert len(firs_single[0]) == 512
    max_peak_single = max(abs(x) for x in firs_single[0])
    assert math.isclose(max_peak_single, 0.99, rel_tol=1e-3)

    # Multi-pickup voice (Jazz pair) -> exactly 2 channels (Neck and Bridge)
    firs_multi = compute_voice_prefilter_firs("jazz_pair_open", src_scale="30in", num_taps=512)
    assert len(firs_multi) == 2
    assert len(firs_multi[0]) == 512
    assert len(firs_multi[1]) == 512
    global_max = max(max(abs(x) for x in f) for f in firs_multi)
    assert math.isclose(global_max, 0.99, rel_tol=1e-3)

    # Multi-pickup voice (P/MM series) -> exactly 2 channels
    firs_pmm = compute_voice_prefilter_firs("p_mm_series", src_scale="32in", num_taps=512)
    assert len(firs_pmm) == 2
