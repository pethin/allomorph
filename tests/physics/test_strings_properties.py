"""
Property-based tests for physical string dispersion, inharmonicity Gaussian RBF,
and wrap damping acoustics.
Verifies mathematical invariants: DC transmission identity, inharmonicity positivity,
wave-speed continuum monotonicity, and high-frequency damping boundedness.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.config.schema import StringPresetConfig
from allomorph.physics.strings import (
    compute_dispersive_wave_speed,
    compute_forward_string_transfer,
    generate_wave_speed_continuum,
    get_inharmonicity_for_f0,
)
from tests.strategies import st_frequency_arrays


@given(
    st_frequency_arrays(min_len=5, max_len=30, min_f=0.0, max_f=20000.0),
    st.floats(min_value=2000.0, max_value=8000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_forward_string_transfer_invariants(
    freqs: np.ndarray, damping_cutoff_hz: float, damping_order: float
) -> None:
    """Forward wrap damping must have H(0) == 1.0, stay in (0, 1], and attenuate monotonically."""
    string_cfg = StringPresetConfig(
        name="Test Preset",
        type="roundwound",
        wrap="stainless_steel",
        core="hex_steel",
        tension_lbs=42.0,
        damping_cutoff_hz=damping_cutoff_hz,
        damping_order=damping_order,
    )

    h_wrap = compute_forward_string_transfer(freqs, string_cfg, sensor_type="magnetic")
    assert np.all(np.isfinite(h_wrap))
    assert np.all(h_wrap > 0.0)
    assert np.all(h_wrap <= 1.000001)

    if freqs[0] == 0.0:
        assert math.isclose(float(h_wrap[0]), 1.0, abs_tol=1e-6)

    # Monotonic attenuation with frequency
    diffs = np.diff(h_wrap)
    assert np.all(diffs <= 1e-8)

    # Sensor type 'direct' must be exact 1.0 identity
    h_direct = compute_forward_string_transfer(freqs, string_cfg, sensor_type="direct")
    assert np.all(h_direct == 1.0)


@given(
    st.floats(min_value=15.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_inharmonicity_rbf_positivity(f0: float) -> None:
    """Inharmonicity constant B_s(f0) must be strictly positive and within physical boundaries."""
    b_s = float(get_inharmonicity_for_f0(f0))
    assert math.isfinite(b_s)
    # Bass string flexural stiffness B_s is strictly positive and bounded by physical range
    assert 0.0 < b_s < 0.01


@given(
    st.floats(min_value=0.70, max_value=0.95, allow_nan=False, allow_infinity=False),
    st.sampled_from([12, 24, 36]),
)
def test_generate_wave_speed_continuum_invariants(scale_m: float, num_points: int) -> None:
    """Wave-speed continuum must produce sorted f0 values and weights that partition unity."""
    continuum = generate_wave_speed_continuum(scale_m, num_points=num_points)
    assert len(continuum) == num_points

    total_weight = sum(pt.weight for pt in continuum)
    assert math.isclose(total_weight, 1.0, abs_tol=1e-6)

    # Strictly increasing fundamental frequencies from ~30.87 Hz to 100.0 Hz
    f0_vals = [pt.f0 for pt in continuum]
    for i in range(1, len(f0_vals)):
        assert f0_vals[i] > f0_vals[i - 1]

    # Wave speeds v0 = 2 * L * f0
    for pt in continuum:
        expected_v = 2.0 * pt.scale_m * pt.f0
        assert math.isclose(pt.v0, expected_v, rel_tol=1e-4)


@given(
    st_frequency_arrays(min_len=5, max_len=30, min_f=20.0, max_f=20000.0),
    st.floats(min_value=50.0, max_value=180.0, allow_nan=False, allow_infinity=False),
)
def test_compute_dispersive_wave_speed_stiffness(freqs: np.ndarray, v0: float) -> None:
    """Dispersive wave speed v(f) >= v0 and increases monotonically with frequency."""
    v_disp = compute_dispersive_wave_speed(freqs, v0=v0, scale_length_m=0.8636)
    assert np.all(np.isfinite(v_disp))
    assert np.all(v_disp >= v0 - 1e-6)

    diffs = np.diff(v_disp)
    assert np.all(diffs >= -1e-8)
