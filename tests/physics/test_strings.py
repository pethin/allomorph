"""
Tests for physical string modeling, core/wrap presets, dispersion,
alternate tunings, inharmonicity, and multi-scale wave speed continuums.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from allomorph.config import (
    SCALES,
    STRINGS,
    VOICES,
    get_instrument_string,
    get_voice_string,
    load_instrument,
)
from allomorph.config.schema import CoilConfig
from allomorph.dsp import FREQS
from allomorph.physics.aperture import numpy_pickup_acoustic_response
from allomorph.physics.strings import (
    INHARMONICITY_ANCHORS_BS,
    INHARMONICITY_ANCHORS_F0,
    MEAN_BASS_F0,
    NOTE_NAMES,
    STRING_FUNDAMENTALS,
    compute_dispersive_wave_speed,
    compute_forward_string_transfer,
    generate_wave_speed_continuum,
    get_inharmonicity_for_f0,
    infer_string_names,
    pitch_to_note_name,
    resolve_scale_length,
)


def test_strings_catalog_loading():
    """Verify that all core physical string presets exist and have valid physical bounds."""
    expected_presets = [
        "roundwound_nickel_standard",
        "roundwound_nickel_6string",
        "roundwound_stainless_clank",
        "flatwound_low_tension",
        "flatwound_vintage_heavy",
        "double_bass_spirocore",
    ]
    for p in expected_presets:
        assert p in STRINGS, f"Preset '{p}' missing from STRINGS catalog"
        s = STRINGS[p]
        assert s.tension_lbs > 100.0
        assert s.damping_cutoff_hz >= 1500.0
        assert s.damping_order >= 1.0

    # 6-string nickel roundwound tension and material parity
    s6 = STRINGS["roundwound_nickel_6string"]
    assert math.isclose(s6.tension_lbs, 230.0, abs_tol=1e-3)
    assert s6.wrap == "nickel"
    assert s6.core == "hex"
    assert math.isclose(s6.damping_cutoff_hz, 8500.0, abs_tol=1e-3)


def test_instrument_string_resolution():
    """Verify source instrument string resolution and default fallback."""
    # 32in fretless explicitly declares La Bella Low Tension Flats
    inst_fretless = load_instrument("32in_fretless_pmm")
    str_fretless = get_instrument_string(inst_fretless)
    assert str_fretless.preset == "flatwound_low_tension"
    assert str_fretless.brand == "La Bella"
    assert str_fretless.model == "LTF-4A"
    assert math.isclose(str_fretless.tension_lbs, 132.0, abs_tol=1e-3)
    assert math.isclose(str_fretless.damping_cutoff_hz, 2800.0, abs_tol=1e-3)

    # 30in and 34in default to roundwound_nickel_standard
    inst_30 = load_instrument("30in_emg_mmtw")
    str_30 = get_instrument_string(inst_30)
    assert str_30.type == "roundwound"
    assert math.isclose(str_30.damping_cutoff_hz, 8500.0, abs_tol=1e-3)

    inst_34 = load_instrument("34in_standard_p")
    str_34 = get_instrument_string(inst_34)
    assert str_34.type == "roundwound"


def test_target_voice_strings():
    """Verify target voice goal string mappings."""
    v14 = VOICES["upright_piezo"]
    str_v14 = get_voice_string(v14)
    assert str_v14.type == "double_bass"
    assert str_v14.tension_lbs == 265.0

    v13 = VOICES["dingwall_bridge"]
    str_v13 = get_voice_string(v13)
    assert str_v13.type == "roundwound"
    assert str_v13.wrap == "stainless"

    v05c = VOICES["precision_warm"]
    str_v05c = get_voice_string(v05c)
    assert str_v05c.type == "flatwound"

    # Standard voices default to roundwound_nickel_standard
    v04 = VOICES["precision_active"]
    str_v04 = get_voice_string(v04)
    assert str_v04.type == "roundwound"


def test_forward_string_wrap_damping():
    """Verify forward viscoelastic wrap damping: flatwounds roll off early (3.2 kHz),

    nickel roundwounds at 8.5 kHz, and stainless steel extends to 12.0 kHz.
    """
    freqs = np.asarray(FREQS, dtype=np.float64)
    s_flats = STRINGS["flatwound_vintage_heavy"]
    s_nickel = STRINGS["roundwound_nickel_standard"]
    s_stainless = STRINGS["roundwound_stainless_clank"]

    h_wrap_flats = compute_forward_string_transfer(freqs, s_flats)
    h_wrap_nickel = compute_forward_string_transfer(freqs, s_nickel)
    h_wrap_stainless = compute_forward_string_transfer(freqs, s_stainless)

    # All wrap damping curves must be finite, real, and <= 1.0
    for h_w in (h_wrap_flats, h_wrap_nickel, h_wrap_stainless):
        assert len(h_w) == len(freqs)
        assert np.all(np.isfinite(h_w))
        assert np.all(h_w <= 1.0)
        assert math.isclose(h_w[0], 1.0, abs_tol=1e-5)

    # At 10 kHz, transmission order must strictly reflect wrap metallurgy:
    # stainless > nickel > flatwounds
    idx_10k = np.argmin(np.abs(freqs - 10000.0))
    assert h_wrap_stainless[idx_10k] > h_wrap_nickel[idx_10k] > h_wrap_flats[idx_10k]

    # At its cutoff frequency f_damp, wrap damping must be at -3 dB (1 / sqrt(2) ≈ 0.707)
    f_damp_flats = float(s_flats.damping_cutoff_hz)
    idx_damp = np.argmin(np.abs(freqs - f_damp_flats))
    assert math.isclose(h_wrap_flats[idx_damp], 1.0 / math.sqrt(2.0), abs_tol=0.08)


def test_forward_sensor_compliance_bypass():
    """Verify that direct DI bypasses viscoelastic string wrap damping."""
    freqs = np.asarray(FREQS, dtype=np.float64)
    s_clank = STRINGS["roundwound_stainless_clank"]

    # Direct DI must evaluate bit-exact 1.0000 across all frequencies
    h_direct = compute_forward_string_transfer(
        freqs, s_clank, scale_length_inches=34.0, sensor_type="direct"
    )
    assert np.allclose(h_direct, 1.0, atol=1e-12)

    # Magnetic sensors apply wrap damping (passband <= 1.0, rolling off at high frequencies)
    h_mag = compute_forward_string_transfer(
        freqs, s_clank, scale_length_inches=34.0, sensor_type="magnetic"
    )
    assert np.all(h_mag <= 1.0)
    assert h_mag[-1] < 0.90


def test_per_string_acoustic_dispersion():
    """
    Verify per-string acoustic inharmonicity dispersion.
    Wave speed v(f) increases smoothly with frequency due to flexural stiffness,
    thick low strings disperse more than thin high strings, and response is bounded.
    """
    freqs = np.asarray(FREQS, dtype=np.float64)

    # 1. Low-E dispersion
    v0_e = 71.16
    v_disp_e = compute_dispersive_wave_speed(freqs, v0_e, "E")
    # At DC, v(0) == v0
    assert math.isclose(v_disp_e[0], v0_e, rel_tol=1e-5)
    # Strictly monotonically increasing
    assert np.all(np.diff(v_disp_e) >= -1e-6)
    # Bounded physical dispersion (< 12% rise at 20 kHz)
    assert v_disp_e[-1] <= 1.12 * v0_e

    # 2. High-G dispersion
    v0_g = 169.27
    v_disp_g = compute_dispersive_wave_speed(freqs, v0_g, "G")
    assert math.isclose(v_disp_g[0], v0_g, rel_tol=1e-5)
    assert np.all(np.diff(v_disp_g) >= -1e-6)
    assert v_disp_g[-1] <= 1.05 * v0_g

    # 3. Low strings experience greater percentage wave speed dispersion than high strings
    disp_ratio_e = v_disp_e[-1] / v0_e
    disp_ratio_g = v_disp_g[-1] / v0_g
    assert disp_ratio_e > disp_ratio_g


def test_infer_string_names_and_split_coil_high_c():
    """
    Verify string name inference across 4, 5, and 6 string basses,
    and that split-coil pickups bind High-C to the treble half and Low-B to the bass half.
    """
    # 4-string standard
    assert infer_string_names([71.16, 95.0, 126.81, 169.27]) == ["E", "A", "D", "G"]

    # 5-string Low-B (standard 34" and 37" multiscale)
    assert infer_string_names([53.28, 71.16, 95.0, 126.81, 169.27]) == ["B", "E", "A", "D", "G"]
    assert infer_string_names([58.02, 75.88, 99.19, 129.60, 169.27]) == ["B", "E", "A", "D", "G"]

    # 5-string High-C (E-A-D-G-C on standard 34" and 30" short scale)
    assert infer_string_names([71.16, 95.0, 126.81, 169.27, 225.69]) == ["E", "A", "D", "G", "C"]
    assert infer_string_names([62.79, 83.82, 111.89, 149.35, 199.36]) == ["E", "A", "D", "G", "C"]

    # 6-string
    assert infer_string_names([53.28, 71.16, 95.0, 126.81, 169.27, 225.69]) == [
        "B",
        "E",
        "A",
        "D",
        "G",
        "C",
    ]

    # Split-coil P-Bass response with 5-string High-C:
    # Forward coil: E/A; Rearward coil: D/G. High-C should bind with D/G.
    freqs = np.asarray(FREQS, dtype=np.float64)
    split_p_coils = [
        CoilConfig(
            position_from_bridge_m=0.138,
            aperture_width_in=1.0,
            weight=1.0,
            polarity=1.0,
            strings=[3, 4],
        ),
        CoilConfig(
            position_from_bridge_m=0.112,
            aperture_width_in=1.0,
            weight=1.0,
            polarity=1.0,
            strings=[1, 2],
        ),
    ]
    # High-C 5-string speeds
    high_c_speeds = [71.16, 95.0, 126.81, 169.27, 225.69]
    resp_high_c = numpy_pickup_acoustic_response(freqs, split_p_coils, high_c_speeds)
    assert len(resp_high_c) == len(freqs)
    assert np.all(np.isfinite(resp_high_c))
    assert np.all(resp_high_c > 0.0)


def test_alternate_tunings_dispersion_and_split_coil():
    """
    Verify support for alternate and dropped tunings:
    - Drop D (D-A-D-G), Drop C (C-G-C-F), C Standard (C-F-A#-D#), D Standard (D-G-C-F), Drop A (A-E-A-D-G)
    - Continuous inharmonicity interpolation B_s(f0)
    - Split-coil (P-Bass) register routing ensuring dropped strings map to the bass coil half
    """
    # 1. Tuning string name inference
    drop_d_speeds = [63.42, 95.00, 126.81, 169.27]
    assert infer_string_names(drop_d_speeds) == ["D", "A", "D", "G"]

    drop_c_speeds = [56.49, 84.63, 112.98, 150.82]
    assert infer_string_names(drop_c_speeds) == ["C", "G", "C", "F"]

    c_std_speeds = [56.49, 75.40, 100.65, 134.35]
    assert infer_string_names(c_std_speeds) == ["C", "F", "A#", "D#"]

    d_std_speeds = [63.42, 84.63, 112.98, 150.82]
    assert infer_string_names(d_std_speeds) == ["D", "G", "C", "F"]

    drop_a_5_speeds = [47.50, 71.16, 95.00, 126.81, 169.27]
    assert infer_string_names(drop_a_5_speeds) == ["A", "E", "A", "D", "G"]

    # 2. Continuous inharmonicity interpolation B_s(f0)
    bs_a0 = get_inharmonicity_for_f0(27.50)
    bs_c1 = get_inharmonicity_for_f0(32.70)
    bs_d1 = get_inharmonicity_for_f0(36.71)
    bs_e1 = get_inharmonicity_for_f0(41.20)
    bs_a1 = get_inharmonicity_for_f0(55.00)
    bs_d2 = get_inharmonicity_for_f0(73.42)
    bs_g2 = get_inharmonicity_for_f0(98.00)
    bs_c3 = get_inharmonicity_for_f0(130.81)

    # Strictly monotonic decrease with frequency / pitch
    assert bs_a0 > bs_c1 > bs_d1 > bs_e1 > bs_a1 > bs_d2 > bs_g2 > bs_c3
    # Bounded physical stiffness
    assert 1e-6 < bs_c1 < 5e-5
    assert 1e-6 < bs_d1 < 5e-5

    # 3. Wave speed dispersion for Drop D and Drop C
    freqs = np.asarray(FREQS, dtype=np.float64)
    v_disp_drop_d = compute_dispersive_wave_speed(freqs, 63.42)
    assert math.isclose(v_disp_drop_d[0], 63.42, rel_tol=1e-5)
    assert np.all(np.diff(v_disp_drop_d) >= -1e-6)
    assert v_disp_drop_d[-1] <= 1.15 * 63.42

    v_disp_drop_c = compute_dispersive_wave_speed(freqs, 56.49)
    assert math.isclose(v_disp_drop_c[0], 56.49, rel_tol=1e-5)
    assert np.all(np.diff(v_disp_drop_c) >= -1e-6)
    assert v_disp_drop_c[-1] <= 1.15 * 56.49

    # 4. Split-coil P-Bass response under Drop D and Drop C
    # Forward coil: 139mm (E/A strings); Rearward coil: 111mm (D/G strings)
    split_p_coils = [
        CoilConfig(
            position_from_bridge_m=0.1390,
            aperture_width_in=1.0,
            weight=1.0,
            polarity=1.0,
            strings=[3, 4],
        ),
        CoilConfig(
            position_from_bridge_m=0.1110,
            aperture_width_in=1.0,
            weight=1.0,
            polarity=1.0,
            strings=[1, 2],
        ),
    ]

    resp_drop_d = numpy_pickup_acoustic_response(freqs, split_p_coils, drop_d_speeds)
    assert len(resp_drop_d) == len(freqs)
    assert np.all(np.isfinite(resp_drop_d))
    assert np.all(resp_drop_d > 0.0)
    assert math.isclose(resp_drop_d[0], 1.0, rel_tol=1e-3)

    resp_drop_c = numpy_pickup_acoustic_response(freqs, split_p_coils, drop_c_speeds)
    assert len(resp_drop_c) == len(freqs)
    assert np.all(np.isfinite(resp_drop_c))
    assert np.all(resp_drop_c > 0.0)
    assert math.isclose(resp_drop_c[0], 1.0, rel_tol=1e-3)

    # Verify that string 0 in Drop D (named "D") binds to the forward coil (0.139m), not rearward coil (0.111m)
    single_string_0_d = numpy_pickup_acoustic_response(
        freqs, split_p_coils, [63.42], string_names=["D"]
    )
    fwd_coil_resp = numpy_pickup_acoustic_response(freqs, [split_p_coils[0]], [63.42])
    assert np.allclose(single_string_0_d, fwd_coil_resp, rtol=1e-4)


@given(
    st.floats(min_value=25.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False),
)
def test_inharmonicity_continuous_monotonicity_property(f0: float, df: float) -> None:
    """Property test verifying inharmonicity RBF solver is strictly decreasing across bass fundamentals."""
    b1 = float(get_inharmonicity_for_f0(f0))
    b2 = float(get_inharmonicity_for_f0(f0 + df))
    assert b1 > b2
    assert 1e-7 < b1 < 1e-3


@given(
    st.floats(min_value=30.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_dispersive_wave_speed_property(c0: float) -> None:
    """Property test verifying wave speed dispersion is monotonically non-decreasing with frequency."""
    freqs = np.array([20.0, 100.0, 500.0, 2000.0, 10000.0], dtype=np.float64)
    v_disp = compute_dispersive_wave_speed(freqs, c0)
    assert math.isclose(v_disp[0], c0, rel_tol=1e-3)
    assert np.all(np.diff(v_disp) >= -1e-6)


def test_multiscale_wave_speed_continuum_endpoints():
    """Verify that 34"-37" multi-scale Dingwall wave speeds continuously interpolate
    from 37" scale at Low B (30.87 Hz) to 34" scale at High G (100.0 Hz)."""
    inst = load_instrument("37in_multiscale_dingwall")
    continuum = generate_wave_speed_continuum(inst, num_points=24)

    # First point: f0 = 30.87 Hz (Low B), scale = 37.0" (0.9398 m)
    pt_low = continuum[0]
    assert math.isclose(pt_low.f0, 30.87, abs_tol=0.01)
    assert math.isclose(pt_low.scale_m, 37.0 * 0.0254, abs_tol=1e-4)
    expected_v_low = 2.0 * (37.0 * 0.0254) * pt_low.f0
    assert math.isclose(pt_low.v0, expected_v_low, abs_tol=0.01)

    # Last point: f0 = 100.0 Hz (High G), scale = 34.0" (0.8636 m)
    pt_high = continuum[-1]
    assert math.isclose(pt_high.f0, 100.00, abs_tol=0.01)
    assert math.isclose(pt_high.scale_m, 34.0 * 0.0254, abs_tol=1e-4)
    expected_v_high = 2.0 * (34.0 * 0.0254) * pt_high.f0
    assert math.isclose(pt_high.v0, expected_v_high, abs_tol=0.01)

    # All intermediate scale lengths must monotonically decrease from 37" to 34"
    scales = [pt.scale_m for pt in continuum]
    assert all(scales[i] >= scales[i + 1] for i in range(len(scales) - 1))


def test_multiscale_sp1_wave_speed_continuum_endpoints():
    """Verify that 32"-35" multi-scale Dingwall SP1 wave speeds continuously interpolate
    from 35" scale at Low B (30.87 Hz) to 32" scale at High G (100.0 Hz)."""
    inst = load_instrument("34in_dingwall_sp1")
    continuum = generate_wave_speed_continuum(inst, num_points=24)

    # First point: f0 = 30.87 Hz, scale = 35.0" (0.8890 m)
    pt_low = continuum[0]
    assert math.isclose(pt_low.f0, 30.87, abs_tol=0.01)
    assert math.isclose(pt_low.scale_m, 35.0 * 0.0254, abs_tol=1e-4)
    expected_v_low = 2.0 * (35.0 * 0.0254) * pt_low.f0
    assert math.isclose(pt_low.v0, expected_v_low, abs_tol=0.01)

    # Last point: f0 = 100.0 Hz, scale = 32.0" (0.8128 m)
    pt_high = continuum[-1]
    assert math.isclose(pt_high.f0, 100.00, abs_tol=0.01)
    assert math.isclose(pt_high.scale_m, 32.0 * 0.0254, abs_tol=1e-4)
    expected_v_high = 2.0 * (32.0 * 0.0254) * pt_high.f0
    assert math.isclose(pt_high.v0, expected_v_high, abs_tol=0.01)

    # All intermediate scale lengths must monotonically decrease from 35" to 32"
    scales = [pt.scale_m for pt in continuum]
    assert all(scales[i] >= scales[i + 1] for i in range(len(scales) - 1))


def test_inharmonicity_rbf_exact_reproduction():
    """Verify that the C^inf Gaussian RBF reproduces empirical inharmonicity anchors exactly."""
    for f0_expected, bs_expected in zip(INHARMONICITY_ANCHORS_F0, INHARMONICITY_ANCHORS_BS):
        bs_interp = get_inharmonicity_for_f0(float(f0_expected))
        assert math.isclose(bs_interp, float(bs_expected), rel_tol=1e-6)

    # Test vectorized interpolation
    arr_interp = get_inharmonicity_for_f0(INHARMONICITY_ANCHORS_F0)
    assert np.allclose(arr_interp, INHARMONICITY_ANCHORS_BS, rtol=1e-6)


def test_constants_and_pitch_conversion():
    """Verify module-level string constants and pitch to note name conversion."""
    assert STRING_FUNDAMENTALS == [41.20, 55.00, 73.42, 98.00]
    assert NOTE_NAMES == ["E", "A", "D", "G"]
    assert math.isclose(MEAN_BASS_F0, 66.9045, abs_tol=1e-3)

    assert pitch_to_note_name(41.2) == "E"
    assert pitch_to_note_name(55.0) == "A"
    assert pitch_to_note_name(73.4) == "D"
    assert pitch_to_note_name(98.0) == "G"
    assert pitch_to_note_name(30.87) == "B"
    assert pitch_to_note_name(130.81) == "C"


def test_resolve_scale_length_fallbacks():
    """Verify resolve_scale_length edge cases and scale identification."""
    # Custom speeds with explicit length
    assert resolve_scale_length([70.0, 90.0], scale_length_m=0.8128) == pytest.approx(0.8128)
    assert resolve_scale_length([70.0, 90.0], scale_length_m=(0.8128, 0.8636)) == pytest.approx(
        0.8382
    )

    # 4-string matching 34in
    assert resolve_scale_length([71.16, 95.0, 126.81, 169.27]) == pytest.approx(0.8636)
    # Default fallback
    assert resolve_scale_length([10.0, 20.0]) == pytest.approx(0.8636)

    # 5-string matching 30in or 32in
    sc_30 = SCALES["30in"]
    assert resolve_scale_length(list(sc_30.speeds) + [225.0]) == pytest.approx(0.762)
    sc_32 = SCALES["32in"]
    assert resolve_scale_length(list(sc_32.speeds) + [225.0]) == pytest.approx(0.8128)


def test_compute_dispersive_wave_speed_inferred_f0():
    """Verify compute_dispersive_wave_speed infers f0 properly when f0 is None or non-positive."""
    freqs = np.asarray(FREQS, dtype=np.float64)
    v_inferred = compute_dispersive_wave_speed(freqs, 71.16, f0=None, scale_length_m=0.8636)
    assert len(v_inferred) == len(freqs)
    assert math.isclose(v_inferred[0], 71.16, rel_tol=1e-5)

    v_default_scale = compute_dispersive_wave_speed(freqs, 71.16, f0=0.0, scale_length_m=None)
    assert len(v_default_scale) == len(freqs)
    assert math.isclose(v_default_scale[0], 71.16, rel_tol=1e-5)


def test_generate_wave_speed_continuum_types():
    """Verify generate_wave_speed_continuum handles various input types."""
    # From scale name string
    cont_str = generate_wave_speed_continuum("34in")
    assert len(cont_str) == 24
    assert cont_str[0].scale_m == pytest.approx(0.8636)

    # From ScaleConfig
    cont_cfg = generate_wave_speed_continuum(SCALES["30in"])
    assert len(cont_cfg) == 24
    assert cont_cfg[0].scale_m == pytest.approx(0.762)

    # From single float
    cont_float = generate_wave_speed_continuum(0.8128, num_points=12)
    assert len(cont_float) == 12
    assert cont_float[0].scale_m == pytest.approx(0.8128)
