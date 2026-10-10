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


def test_active_pickup_mna_resonant_frequency_generic():
    """Verify that MNA harness solver calculates the resonant peak of an active pickup circuit
    matching the theoretical RLC resonance fr = 1 / (2*pi*sqrt(L*C)) within +-50 Hz using generic data.
    """
    from allomorph.config.schema import (
        CoilConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        VoicingConfig,
    )

    L = 2.0
    C = 1.0e-9
    f_expected = 1.0 / (2.0 * math.pi * math.sqrt(L * C))

    inst = InstrumentConfig(
        id="generic_active",
        name="Generic Active Bass",
        scale_length_in=34.0,
        electronics="active",
        pickups={
            "act": PickupConfig(
                name="Active Pickup",
                position_from_bridge_m=0.10,
                has_internal_buffer=True,
                buffer_output_impedance=2000.0,
                coils=[
                    CoilConfig(
                        id="act_c",
                        position_from_bridge_m=0.10,
                        L=L,
                        Rdc=1000.0,
                        Ccoil=C,
                        Reddy=100000.0,
                    )
                ],
            )
        },
        harnesses={
            "direct": HarnessConfig(
                name="Direct",
                type="passive",
                wiring=[
                    ["pickups.act.hot", "out"],
                    ["pickups.act.cold", "GND"],
                ],
            )
        },
        voicings={"v_act": VoicingConfig(id="v_act", name="Active Voicing", harness="direct")},
    )

    freqs = np.linspace(20.0, 10000.0, 4000)
    curves = solve_mna_harness(inst, inst.harnesses["direct"], inst.voicings["v_act"], freqs=freqs)
    mag = next(iter(curves.values()))
    peak_idx = int(np.argmax(mag))
    peak_f = float(freqs[peak_idx])

    error_hz = abs(peak_f - f_expected)
    assert error_hz <= 50.0, (
        f"Generic active pickup resonant peak {peak_f:.1f} Hz deviates by {error_hz:.1f} Hz "
        f"from theoretical {f_expected:.1f} Hz (tolerance +-50 Hz)"
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
    """Verify PickupConfig modeling of internal buffer line driver output impedances using generic data."""
    from allomorph.config.schema import PickupConfig

    p_x = PickupConfig(name="x_series", has_internal_buffer=True, buffer_output_impedance=2000.0)
    assert p_x.has_internal_buffer
    assert p_x.buffer_output_impedance == 2000.0

    p_classic = PickupConfig(
        name="classic", has_internal_buffer=True, buffer_output_impedance=10000.0
    )
    assert p_classic.has_internal_buffer
    assert p_classic.buffer_output_impedance == 10000.0
