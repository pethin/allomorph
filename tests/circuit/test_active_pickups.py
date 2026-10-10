"""
Tests for first-principles active pickup modeling:
- Resonant frequency peak accuracy of EMG X-Series (MMTWX, PX, PCSX) and Classic Non-X series
- Dynamic leveling compression and saturation thresholds
- Small-signal linearity bypass (peak <= 0.10)
- Low-impedance line driver output stages (2k vs 10k)
"""

import math

import numpy as np

from allomorph.circuit import (
    apply_active_pickup_dynamics,
    solve_mna_harness,
)
from allomorph.config import INSTRUMENTS


def test_active_pickup_resonant_frequency_accuracy():
    """Verify that loaded active pickup circuits match their declared resonant peaks within +-50 Hz."""
    freqs = np.linspace(20.0, 20000.0, 8000)

    test_cases = [
        ("30in_emg_mmtw", "dual_mode", "mmtwx", 2500.0),
        ("30in_emg_mmtw", "single_mode", "mmtwx", 3500.0),
        ("32in_custom_pmm", "px_solo", "px", 3200.0),
        ("32in_custom_pmm", "mm_single", "mmtwx", 3500.0),
        ("32in_custom_pmm", "mm_dual", "mmtwx", 2500.0),
        ("32in_fretless_pmm", "pcsx_solo", "pcsx", 2610.0),
        ("34in_active_emg", "neck_solo", "neck", 4150.0),
        ("34in_active_emg", "bridge_solo", "bridge", 4150.0),
    ]

    for inst_id, v_id, pickup_key, expected_fr in test_cases:
        inst = INSTRUMENTS[inst_id]
        v = inst.voicings[v_id]
        h = inst.harnesses[v.harness]
        curves = solve_mna_harness(inst, h, v, freqs=freqs)
        assert pickup_key in curves, f"Pickup {pickup_key} not found in curves for {inst_id}:{v_id}"
        curve = curves[pickup_key]
        peak_idx = int(np.argmax(curve))
        peak_f = float(freqs[peak_idx])

        error_hz = abs(peak_f - expected_fr)
        assert error_hz <= 50.0, (
            f"{inst_id}:{v_id}:{pickup_key} resonant peak {peak_f:.1f} Hz deviates by {error_hz:.1f} Hz "
            f"from target {expected_fr:.1f} Hz (tolerance +-50 Hz)"
        )


def test_x_series_linear_passband_dynamics():
    """Verify EMG X-Series pickup dynamics (k_level=0.0, vsat=2.40V) provide completely uncompressed linearity up to 2.4V headroom."""
    sr = 48000
    t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
    # Sine wave with peak amplitude 0.80 (well above 0.10 small-signal threshold, below 2.40V rail)
    audio = 0.80 * np.sin(2.0 * math.pi * 100.0 * t).astype(np.float32)

    processed = apply_active_pickup_dynamics(
        audio,
        vsat=2.40,
        k_level=0.0,
        tau_att=0.003,
        tau_rel=0.045,
        sr=sr,
    )

    # In X-series with k_level=0.0 and amplitude below vsat, dynamics should be virtually linear (identical to input)
    max_diff = float(np.max(np.abs(processed - audio)))
    assert max_diff < 1e-4, f"X-Series with k_level=0.0 should not compress: max diff {max_diff}"


def test_classic_emg_dynamic_leveling_compression():
    """Verify Classic Non-X EMG dynamics (k_level=0.25, vsat=0.50V) introduce smooth leveling compression and soft clipping."""
    sr = 48000
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False)
    # Strong burst: peak amplitude 1.20 V
    audio = 1.20 * np.sin(2.0 * math.pi * 100.0 * t).astype(np.float32)

    processed = apply_active_pickup_dynamics(
        audio,
        vsat=0.50,
        k_level=0.25,
        tau_att=0.003,
        tau_rel=0.045,
        sr=sr,
    )

    # Classic EMG output must be constrained by vsat=0.50V soft knee
    max_peak = float(np.max(np.abs(processed)))
    assert max_peak <= 0.50, f"Classic EMG exceeded vsat ceiling: peak {max_peak} > 0.50"

    # Sustained envelope must be leveled down compared to raw input
    rms_in = float(np.sqrt(np.mean(audio**2)))
    rms_out = float(np.sqrt(np.mean(processed**2)))
    assert rms_out < rms_in * 0.5, (
        f"Classic EMG did not compress strong signal: rms_in={rms_in:.3f}, rms_out={rms_out:.3f}"
    )


def test_small_signal_linearity_bypass():
    """Verify that test sweeps and small calibration impulses (peak <= 0.10) bypass non-linear processing identically."""
    sr = 48000
    # Small signal with peak 0.08
    small_signal = np.array([0.0, 0.05, -0.08, 0.02, -0.01], dtype=np.float32)

    # Both X-Series and Classic variants should bypass identically
    out_x = apply_active_pickup_dynamics(
        small_signal,
        vsat=2.40,
        k_level=0.0,
        sr=sr,
    )
    out_classic = apply_active_pickup_dynamics(
        small_signal,
        vsat=0.50,
        k_level=0.25,
        sr=sr,
    )

    np.testing.assert_array_equal(out_x, small_signal)
    np.testing.assert_array_equal(out_classic, small_signal)


def test_active_pickup_output_impedances():
    """Verify that X-Series declares 2k line driver impedance and Classic declares 10k output impedance."""
    mmtwx_p = INSTRUMENTS["30in_emg_mmtw"].pickups["mmtwx"]
    assert mmtwx_p.has_internal_buffer
    assert mmtwx_p.buffer_output_impedance == 2000.0

    classic_p = INSTRUMENTS["34in_active_emg"].pickups["neck"]
    assert classic_p.has_internal_buffer
    assert classic_p.buffer_output_impedance == 10000.0
