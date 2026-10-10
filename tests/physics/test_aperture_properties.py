"""
Property-based tests for acoustic sensing aperture, multi-coil spatial response,
and bridge displacement low-shelf filtering.
Verifies mathematical invariants: DC conservation, monotonic spatial lowpass attenuation,
partition of unity, and asymmetric algebraic limiter headroom.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.config.schema import CoilConfig
from allomorph.physics.aperture import (
    build_wave_speed_continuum_partitions,
    combine_coherent_incoherent_aperture,
    compute_coil_aperture_sinc,
    compute_displacement_proximity_shelf,
    compute_pickup_isolation_leveling,
    compute_string_dispersion_tensor,
    soft_clamp_displacement_ratio,
)
from tests.strategies import st_frequency_arrays


@given(
    st.floats(min_value=0.70, max_value=1.05, allow_nan=False, allow_infinity=False),
)
def test_wave_speed_continuum_partitions_partition_of_unity(scale_length_m: float) -> None:
    """Wave-speed continuum partitions must partition points with sum(weights) == 1.0."""
    partitions = build_wave_speed_continuum_partitions(scale_length_m=scale_length_m)
    assert len(partitions) > 0
    total_weight = sum(pt.weight for pts in partitions.values() for pt in pts)
    assert math.isclose(total_weight, 1.0, abs_tol=1e-6)

    for reg, pts in partitions.items():
        assert reg in ("lower", "upper", "all")
        assert len(pts) > 0
        for pt in pts:
            assert pt.f0 > 0.0
            assert pt.v0 > 0.0
            assert pt.scale_m > 0.0
            assert pt.weight > 0.0


@given(
    st_frequency_arrays(min_len=5, max_len=30, min_f=20.0, max_f=20000.0),
    st.floats(min_value=30.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_string_dispersion_tensor_stiffness_invariants(
    freqs: np.ndarray, f0: float, v0: float
) -> None:
    """String dispersion tensor v_disp(f) >= v0, v_disp(0) == v0, and increases monotonically with f."""
    f0_arr = np.array([f0], dtype=np.float64)
    v0_arr = np.array([v0], dtype=np.float64)

    v_disp = compute_string_dispersion_tensor(freqs, f0_arr, v0_arr)
    assert v_disp.shape == (1, len(freqs))

    # Phase velocity increases with flexural stiffness: v(f) >= v0
    assert np.all(v_disp >= v0 - 1e-6)

    # Monotonicity with frequency: higher harmonics travel faster due to bending stiffness
    diffs = np.diff(v_disp[0])
    assert np.all(diffs >= -1e-8)


@given(
    st_frequency_arrays(min_len=5, max_len=30, min_f=0.0, max_f=20000.0),
    st.floats(min_value=0.005, max_value=0.05, allow_nan=False, allow_infinity=False),
    st.sampled_from(["rod", "blade"]),
)
def test_coil_aperture_sinc_lowpass_invariants(
    freqs: np.ndarray, w_m: float, pole_type: str
) -> None:
    """Coil spatial aperture must evaluate to 1.0 at DC, stay in (0, 1], and attenuate monotonically."""
    v_disp = np.full((1, len(freqs)), 100.0, dtype=np.float64)
    ap = compute_coil_aperture_sinc(freqs, v_disp, w_m, pole_type=pole_type)[0]

    assert np.all(np.isfinite(ap))
    assert np.all(ap > 0.0)
    assert np.all(ap <= 1.000001)

    # DC identity check if 0 Hz is present
    if freqs[0] == 0.0:
        assert math.isclose(float(ap[0]), 1.0, abs_tol=1e-6)

    # Monotonic attenuation
    diffs = np.diff(ap)
    assert np.all(diffs <= 1e-8)


@given(
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_soft_clamp_displacement_ratio_algebraic_limits(delta_g: float) -> None:
    """Asymmetric alg4 limiter bounds positive excursion <= +12 dB and negative excursion >= -16 dB."""
    clamped = float(soft_clamp_displacement_ratio(delta_g))
    assert clamped <= 12.0001
    assert clamped >= -16.0001
    assert math.isfinite(clamped)

    # Near 0 dB, behaves linearly with no slope kink
    if abs(delta_g) <= 1.0:
        assert math.isclose(clamped, delta_g, rel_tol=0.05, abs_tol=0.01)


@given(
    st_frequency_arrays(min_len=5, max_len=30, min_f=20.0, max_f=20000.0),
    st.floats(min_value=0.03, max_value=0.30, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.70, max_value=0.95, allow_nan=False, allow_infinity=False),
)
def test_displacement_proximity_shelf_high_frequency_transparency(
    freqs: np.ndarray, pos_m: float, scale_m: float
) -> None:
    """Low-shelf displacement filter H_pos(f) must approach 1.0 (0.00 dB) at high frequencies."""
    h_pos = compute_displacement_proximity_shelf(freqs, pos_m, scale_m)
    assert np.all(np.isfinite(h_pos))
    assert np.all(h_pos > 0.0)

    # High-frequency asymptote transparency: near 20 kHz, ratio to 1.0 is negligible
    hf_val = float(h_pos[-1])
    assert abs(hf_val - 1.0) < 0.05


@given(
    st.floats(min_value=0.04, max_value=0.25, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.04, max_value=0.25, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.76, max_value=0.94, allow_nan=False, allow_infinity=False),
)
def test_pickup_isolation_leveling_bounds_and_identity(
    pos_m: float, ref_pos_m: float, scale_m: float
) -> None:
    """Pickup isolation leveling factor is 1.0 when pos_m >= ref_pos_m, and bounded in [1.0, 2.0]."""
    k_iso = compute_pickup_isolation_leveling(pos_m, scale_m, ref_pos_m=ref_pos_m)
    assert math.isfinite(k_iso)
    assert k_iso >= 1.0 - 1e-6
    assert k_iso <= 10.0 ** (6.0 / 20.0) + 1e-4

    if pos_m >= ref_pos_m:
        assert math.isclose(k_iso, 1.0, abs_tol=1e-5)


@given(
    st_frequency_arrays(min_len=5, max_len=30, min_f=20.0, max_f=20000.0),
    st.floats(min_value=0.04, max_value=0.20, allow_nan=False, allow_infinity=False),
)
def test_combine_coherent_incoherent_aperture_properties(freqs: np.ndarray, pos_m: float) -> None:
    """Coherent + incoherent aperture combination stays non-negative and finite."""
    coil = CoilConfig(
        id="c1",
        position_from_bridge_m=pos_m,
        aperture_width_in=0.75,
        L=4.0,
        Rdc=6000.0,
        Reddy=200000.0,
        Ccoil=5e-11,
        weight=1.0,
        polarity=1.0,
        strings=[1, 2, 3, 4],
    )
    coil_sum = np.ones((1, len(freqs)), dtype=np.complex128)
    p_incoh = np.ones((1, len(freqs)), dtype=np.float64)
    v0_arr = np.array([80.0], dtype=np.float64)

    m_blend = combine_coherent_incoherent_aperture(
        coil_sum=coil_sum,
        p_incoh=p_incoh,
        active_coils=[coil],
        v0_arr=v0_arr,
        freqs=freqs,
        total_w=1.0,
    )
    assert np.all(np.isfinite(m_blend))
    assert np.all(m_blend >= 0.0)
