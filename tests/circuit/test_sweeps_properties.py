"""
Property-based tests for parametric sweeps and metrics extraction in allomorph.circuit.sweeps.
Verifies mathematical and electroacoustic invariants:
1. Volume monotonicity: higher volume wiper position yields equal or higher passband gain.
2. Cable capacitance shift: higher cable capacitance monotonically shifts resonant peak lower.
3. Metrics record invariants: computed metrics are finite, with consistent peak boost and loss relations.
4. Parameter override immutability: applying parameter overrides never mutates input harness or voicing.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.circuit.sweeps import (
    apply_sweep_parameter_override,
    compute_parametric_sweep,
    compute_sweep_curve_metrics,
    extract_channel_response,
)
from tests.conftest import make_generic_instrument_config
from tests.strategies import st_pot_wipers


@given(
    v1=st_pot_wipers(),
    v2=st_pot_wipers(),
)
def test_sweep_volume_attenuation_monotonicity(v1: float, v2: float) -> None:
    """Volume pot attenuation is monotonic: v1 <= v2 implies gain(v1) <= gain(v2) + eps."""
    low_v = min(v1, v2)
    high_v = max(v1, v2)

    inst = make_generic_instrument_config()
    res = compute_parametric_sweep(inst, param="vol", values=[low_v, high_v])

    f_arr = np.asarray(res.freqs)
    idx_1k = int(np.argmin(np.abs(f_arr - 1000.0)))

    mag_low = res.curves[0][idx_1k]
    mag_high = res.curves[1][idx_1k]

    assert mag_low <= mag_high + 1e-4, (
        f"Volume monotonicity violated at 1 kHz: vol={low_v} ({mag_low:.2f} dB) > vol={high_v} ({mag_high:.2f} dB)"
    )


@given(
    c1=st.floats(min_value=500.0, max_value=900.0, allow_nan=False),
    c2=st.floats(min_value=1000.0, max_value=2000.0, allow_nan=False),
)
def test_sweep_cable_resonance_downshift(c1: float, c2: float) -> None:
    """Increasing cable capacitance (c1 < c2) downshifts resonant peak and attenuates treble."""
    inst = make_generic_instrument_config()
    res = compute_parametric_sweep(inst, param="cable", values=[c1, c2])

    f_arr = np.asarray(res.freqs)
    mask = (f_arr >= 500.0) & (f_arr <= 5000.0)
    sub_f = f_arr[mask]

    peak_idx1 = int(np.argmax(res.curves[0][mask]))
    peak_idx2 = int(np.argmax(res.curves[1][mask]))

    f_res1 = sub_f[peak_idx1]
    f_res2 = sub_f[peak_idx2]

    # c2 > c1 implies f_res2 <= f_res1
    assert f_res2 <= f_res1 + 10.0, (
        f"Cable capacitance downshift violated: c1={c1:.0f} pF -> f_res1={f_res1} Hz, "
        f"c2={c2:.0f} pF -> f_res2={f_res2} Hz"
    )

    # Treble at 5 kHz must roll off monotonically with higher cable capacitance
    idx_5k = int(np.argmin(np.abs(f_arr - 5000.0)))
    assert res.curves[0][idx_5k] > res.curves[1][idx_5k]


@given(
    peak_gain=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False),
    loss_db=st.floats(min_value=-30.0, max_value=0.0, allow_nan=False),
)
def test_compute_sweep_curve_metrics_invariants(peak_gain: float, loss_db: float) -> None:
    """Computed circuit metrics are finite and satisfy peak_boost_db == peak_db - insertion_loss_db."""
    f = np.linspace(20.0, 20000.0, 512)
    # Synthetic curve with low-frequency baseline and resonant bump
    f_res = 2500.0
    curve = loss_db + peak_gain * np.exp(-0.5 * ((f - f_res) / 400.0) ** 2)

    rec = compute_sweep_curve_metrics(
        freqs=f,
        curve=curve,
        param="tone",
        param_value=0.5,
        label="Test Curve",
    )

    assert math.isfinite(rec.insertion_loss_db)
    assert math.isfinite(rec.peak_db)
    assert math.isfinite(rec.peak_boost_db)
    assert math.isclose(rec.peak_boost_db, rec.peak_db - rec.insertion_loss_db, abs_tol=0.02)
    assert math.isfinite(rec.hf_slope_db_oct)


def test_apply_sweep_parameter_override_immutability() -> None:
    """apply_sweep_parameter_override creates modified copies without mutating originals."""
    inst = make_generic_instrument_config()
    voicing = inst.voicings["generic_voice"]
    harness = inst.harnesses[voicing.harness]

    orig_vol = float(voicing.controls["vol"])
    orig_taper = harness.controls["vol"].taper

    h_iter, v_iter = apply_sweep_parameter_override(
        harness=harness,
        voicing=voicing,
        param="vol",
        value=0.25,
        actual_taper="linear",
    )

    # Modified copies have updated values
    assert v_iter.controls["vol"] == 0.25
    assert h_iter.controls["vol"].taper == "linear"

    # Original configurations remain completely unchanged
    assert voicing.controls["vol"] == orig_vol
    assert harness.controls["vol"].taper == orig_taper


def test_extract_channel_response_invariants() -> None:
    """extract_channel_response extracts designated channel or sums across all channels."""
    c1 = np.ones(64, dtype=np.complex128) * 1.0
    c2 = np.ones(64, dtype=np.complex128) * 2.0
    curves_dict = {"neck": c1, "bridge": c2}

    # Sum evaluation
    total = extract_channel_response(curves_dict, pickup_channel="sum")
    assert np.allclose(total, 3.0)

    # Designated index evaluation
    ch0 = extract_channel_response(curves_dict, pickup_channel=0)
    assert np.allclose(ch0, 1.0)
    ch1 = extract_channel_response(curves_dict, pickup_channel=1)
    assert np.allclose(ch1, 2.0)
