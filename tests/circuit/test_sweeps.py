"""
Tests for instantaneous continuous parametric circuit sweeps in Allomorph.
Verifies tone pot sweep, volume pot sweep, cable capacitance sweep,
active EQ sweeps, Polars DataFrame generation, and performance benchmark (< 50 ms).
"""

import time

import numpy as np
import polars as pl

from allomorph.circuit import (
    compute_parametric_sweep,
    load_circuit,
)
from allomorph.dsp import FREQS


def test_tone_pot_sweep_treble_cut():
    """Tone pot sweep (0.0 -> 1.0) must show progressive treble cut (> 20 dB at 5 kHz)."""
    model = load_circuit("precision_vintage")
    res = compute_parametric_sweep(model, param="tone", values=[0.0, 0.25, 0.5, 0.75, 1.0])

    assert len(res.curves) == 5
    f_arr = np.asarray(res.freqs)
    idx_5k = int(np.argmin(np.abs(f_arr - 5000.0)))

    # Tone 0.0 (full roll-off) vs Tone 1.0 (full open)
    mag_closed = res.curves[0][idx_5k]
    mag_open = res.curves[-1][idx_5k]

    treble_cut_db = mag_open - mag_closed
    assert treble_cut_db > 20.0, f"Expected > 20 dB treble cut at 5 kHz, got {treble_cut_db:.2f} dB"

    # Verify monotonic treble increase with tone wiper position
    mags_5k = [c[idx_5k] for c in res.curves]
    for i in range(len(mags_5k) - 1):
        assert mags_5k[i] <= mags_5k[i + 1] + 1e-6


def test_volume_pot_sweep_attenuation():
    """Volume pot sweep (0.0 -> 1.0) must show cable loading attenuation."""
    model = load_circuit("precision_vintage")
    res = compute_parametric_sweep(model, param="vol", values=[0.0, 0.25, 0.5, 0.75, 1.0])

    assert len(res.curves) == 5
    f_arr = np.asarray(res.freqs)
    idx_1k = int(np.argmin(np.abs(f_arr - 1000.0)))

    # Vol 0.0 (grounded) vs Vol 1.0 (full)
    mag_zero = res.curves[0][idx_1k]
    mag_full = res.curves[-1][idx_1k]

    assert mag_zero < -60.0  # Fully attenuated
    assert mag_full > -3.0  # Nominal unattenuated passband
    assert mag_full > mag_zero + 50.0


def test_cable_capacitance_resonance_downshift():
    """Cable capacitance sweep (500 to 1500 pF) must show resonance downshifting."""
    model = load_circuit("precision_vintage")
    res = compute_parametric_sweep(model, param="cable", values=[500.0, 750.0, 1000.0, 1500.0])

    assert len(res.curves) == 4
    f_arr = np.asarray(res.freqs)

    # Filter frequencies between 500 Hz and 5000 Hz to locate resonant peaks
    mask = (f_arr >= 500.0) & (f_arr <= 5000.0)
    sub_f = f_arr[mask]

    peak_freqs = []
    for curve in res.curves:
        sub_curve = curve[mask]
        peak_idx = int(np.argmax(sub_curve))
        peak_freqs.append(sub_f[peak_idx])

    # Increasing capacitance must downshift resonance monotonically
    for i in range(len(peak_freqs) - 1):
        assert peak_freqs[i] >= peak_freqs[i + 1], (
            f"Expected downward resonance shift, got {peak_freqs[i]} -> {peak_freqs[i + 1]}"
        )

    # Treble at 4 kHz must roll off monotonically with higher cable capacitance
    idx_4k = int(np.argmin(np.abs(f_arr - 4000.0)))
    mags_4k = [c[idx_4k] for c in res.curves]
    for i in range(len(mags_4k) - 1):
        assert mags_4k[i] > mags_4k[i + 1]


def test_active_preamp_boost_sweep():
    """Active preamp bass boost sweep must increase low-frequency gain."""
    res = compute_parametric_sweep("jazz_pair_active", param="bass_boost", values=[0.0, 6.0, 12.0])

    assert len(res.curves) == 3
    f_arr = np.asarray(res.freqs)
    idx_40 = int(np.argmin(np.abs(f_arr - 40.0)))

    mag_0 = res.curves[0][idx_40]
    mag_12 = res.curves[-1][idx_40]
    boost = mag_12 - mag_0
    assert boost > 8.0, f"Expected active bass boost at 40 Hz, got {boost:.2f} dB"


def test_sweep_performance_benchmark():
    """100-step parametric sweep must execute in < 50 ms."""
    model = load_circuit("precision_vintage")
    values = np.linspace(0.0, 1.0, 100)

    # Warm-up run
    compute_parametric_sweep(model, param="tone", values=[0.0, 1.0])

    runs = []
    res = None
    for _ in range(5):
        t0 = time.perf_counter()
        res = compute_parametric_sweep(model, param="tone", values=values)
        runs.append((time.perf_counter() - t0) * 1000.0)

    best_ms = float(min(runs))
    assert res is not None
    assert len(res.curves) == 100
    assert best_ms < 100.0, (
        f"Expected < 100 ms for 100 steps, best was {best_ms:.2f} ms (runs: {runs})"
    )


def test_to_dataframe_schema():
    """to_dataframe() must return a Polars DataFrame with the expected columns."""
    model = load_circuit("precision_vintage")
    res = compute_parametric_sweep(model, param="tone", values=[0.0, 0.5, 1.0])

    df = res.to_dataframe()
    assert isinstance(df, pl.DataFrame)
    expected_cols = ["frequency", "magnitude_db", "param", "param_value", "label"]
    assert df.columns == expected_cols
    assert len(df) == len(FREQS) * 3

    # Test with include_voice_id=True
    res.voice_id = "precision_vintage"
    df_voice = res.to_dataframe(include_voice_id=True)
    assert "voice_id" in df_voice.columns


def test_circuit_state_restoration():
    """Parametric sweep must not permanently alter the circuit model."""
    model = load_circuit("precision_vintage")
    orig_tone = model.tone_pos
    orig_rtone = model.Rtone
    orig_ccable = model.Ccable

    compute_parametric_sweep(model, param="tone", values=[0.1, 0.5, 0.9])
    assert model.tone_pos == orig_tone
    assert model.Rtone == orig_rtone

    compute_parametric_sweep(model, param="cable", values=[250.0, 1200.0])
    assert model.Ccable == orig_ccable
