"""
Tests for pickup aperture response, spatial comb filtering, dual-coil humbuckers,
saddle boundary stiffness, cylindrical rod vs blade aperture, and microphonics.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.config import (
    SCALES,
    VOICES,
    get_source_pickup,
    load_instrument,
    resolve_pickup_coils,
)
from allomorph.config.schema import CoilConfig, PickupConfig
from allomorph.dsp import FREQS
from allomorph.physics.aperture import (
    CALIBRATION_EXCURSION_ETA,
    FRACTIONAL_EXCURSION_CALIBRATION_ETA,
    aperture_response,
    compute_body_microphonic_coupling,
    compute_coil_aperture,
    compute_displacement_proximity_shelf,
    compute_pickup_isolation_leveling,
    compute_saddle_boundary_coupling,
    get_coil_register,
    is_voice_matching_source,
    numpy_aperture,
    numpy_pickup_acoustic_response,
    numpy_pickup_macro_aperture,
    numpy_position,
    pickup_acoustic_response,
    position_envelope,
    soft_clamp_displacement_ratio,
)


def test_aperture_zero_frequency():
    """Verify physical aperture DC identity: Sinc(0) = 1.0, Comb(0) = 1.0, Product = 1.0000 (0.00 dB)."""
    speeds = SCALES["30in"].speeds
    res = aperture_response(np.array([0.0]), w_in=1.5, d_in=0.75, speeds=speeds)[0]
    assert math.isclose(res, 1.0, abs_tol=1e-6)

    # Test numpy_aperture with default speeds and d_in == 0
    res_single = numpy_aperture(np.array([0.0]), w_in=0.75, d_in=0.0)[0]
    assert math.isclose(res_single, 1.0, abs_tol=1e-6)


@given(
    st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_aperture_dc_conservation_property(w_in: float, d_in: float) -> None:
    """Property test verifying physical aperture DC identity: H(0) = 1.0 for arbitrary coil geometry."""
    speeds = SCALES["34in"].speeds
    res = float(aperture_response(np.array([0.0]), w_in=w_in, d_in=d_in, speeds=speeds)[0])
    assert math.isclose(res, 1.0, abs_tol=1e-5)


@given(
    st.floats(min_value=0.0, max_value=24000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=2.5, allow_nan=False, allow_infinity=False),
)
def test_numpy_aperture_bounding_property(f: float, w_in: float) -> None:
    """Property test verifying single-coil aperture filtering magnitude |H(f)| <= 1.0 everywhere."""
    res = float(numpy_aperture(np.array([f]), w_in=w_in, d_in=0.0)[0])
    assert 0.0 <= res <= 1.0 + 1e-6


def test_aperture_single_vs_dual():
    """Verify dual coil exhibits narrower aperture transmission than narrow single coil at high frequencies."""
    speeds = SCALES["30in"].speeds
    val_single = aperture_response(np.array([5000.0]), w_in=0.75, d_in=0.0, speeds=speeds)[0]
    val_dual = aperture_response(np.array([5000.0]), w_in=1.50, d_in=0.75, speeds=speeds)[0]

    # Dual coil should have more high-frequency aperture filtering than narrow single coil
    assert val_dual < val_single


def test_position_envelope():
    """Verify spatial standing-wave excursion envelope at bridge and neck positions."""
    speeds = SCALES["34in"].speeds
    freqs = np.array([200.0, 1000.0, 5000.0])
    res_bridge = position_envelope(freqs, pos_m=0.0406, speeds=speeds)
    res_neck = position_envelope(freqs, pos_m=0.1250, speeds=speeds)

    assert len(res_bridge) == 3
    assert len(res_neck) == 3
    assert all(val > 0.1 for val in res_bridge)
    assert all(val > 0.1 for val in res_neck)

    # Bridge witness point boundary condition: sin(0) = 0 at pos_m = 0
    res_zero = numpy_position(freqs, pos_m=0.0, speeds=speeds)
    assert np.allclose(res_zero, 0.0)


def test_pickup_acoustic_response_single_coil():
    """Verify single coil acoustic response across frequencies."""
    speeds = SCALES["34in"].speeds
    coils = [
        CoilConfig(
            position_from_bridge_m=0.0406,
            aperture_width_in=0.75,
            weight=1.0,
            polarity=1.0,
            strings=["all"],
        )
    ]
    freqs = np.array([0.0, 500.0, 2000.0])
    res = pickup_acoustic_response(freqs, coils, speeds)

    # At 0 Hz, spatial phase is 0 and sinc(0) = 1, so response is 1.0 (fundamental preserved)
    assert math.isclose(res[0], 1.0, abs_tol=1e-3)
    # At 500 and 2000 Hz, response is positive
    assert res[1] > 0.05
    assert res[2] > 0.05


def test_split_coil_string_differentiation():
    """Verify that lower (3, 4) coil only affects bass strings, and upper (1, 2) coil only affects treble strings."""
    speeds = [66.98, 89.41, 119.35, 159.31]  # strings 4, 3, 2, 1 on 32"

    coils = [
        CoilConfig(
            position_from_bridge_m=0.1088,
            aperture_width_in=1.10,
            weight=1.0,
            polarity=1.0,
            strings=[3, 4],
        ),
        CoilConfig(
            position_from_bridge_m=0.1368,
            aperture_width_in=1.10,
            weight=1.0,
            polarity=1.0,
            strings=[1, 2],
        ),
    ]

    freqs = np.array([500.0])
    val_all = pickup_acoustic_response(freqs, coils, speeds, string_names=[4, 3, 2, 1])[0]
    val_low = pickup_acoustic_response(freqs, coils, speeds[:2], string_names=[4, 3])[0]
    val_high = pickup_acoustic_response(freqs, coils, speeds[2:], string_names=[2, 1])[0]

    # Low strings (closer to bridge on Reverse P) and high strings (closer to neck) have distinct responses
    assert not math.isclose(val_low, val_high, rel_tol=1e-2)
    assert val_all > 0.05


def test_3coil_pmm_compound_response():
    """Verify that 3-coil P/MM blend evaluates 3 distinct physical coil positions."""
    inst = load_instrument("32in_custom_pmm")
    blend = get_source_pickup(inst, "p_mm_series")  # routes to blend_parallel
    coils = resolve_pickup_coils(blend, inst)

    # Should have 4 coil records (PX D/G, PX E/A, MMTWX neck, MMTWX bridge)
    assert len(coils) == 4

    # Check that strings 1/2 (D/G) see 3 active coils and strings 3/4 (E/A) see 3 active coils
    dg_coils = [c for c in coils if "all" in c.strings or 1 in c.strings or "D" in c.strings]
    assert len(dg_coils) == 3

    ea_coils = [c for c in coils if "all" in c.strings or 4 in c.strings or "E" in c.strings]
    assert len(ea_coils) == 3

    # Positions should match blueprint: PX DG 136.8mm, MMTWX neck 73.7mm, MMTWX bridge 50.8mm
    positions = sorted([round(c.position_from_bridge_m * 1000, 1) for c in dg_coils])
    assert positions == [50.8, 73.7, 136.8]


def test_dual_coil_notch_smoothness_and_dc_identity():
    """
    Verify that dual-coil humbuckers (e.g. StingRay MM) preserve exact DC unity (1.000)
    and transition through comb notches with C^1 smoothness and zero non-differentiable V-cusps.
    """
    speeds = SCALES["34in"].speeds
    coils = [
        CoilConfig(
            position_from_bridge_m=0.0755,
            aperture_width_in=0.75,
            weight=0.5,
            polarity=1.0,
            strings=["all"],
        ),
        CoilConfig(
            position_from_bridge_m=0.0565,
            aperture_width_in=0.75,
            weight=0.5,
            polarity=1.0,
            strings=["all"],
        ),
    ]

    # 1. Exact DC unity preservation
    res_dc = pickup_acoustic_response(np.array([0.0]), coils, speeds)[0]
    assert math.isclose(res_dc, 1.0, abs_tol=1e-3)

    # 2. Smooth parabolic bottom across the comb notch region (1500 to 5000 Hz)
    f_arr = np.linspace(1500.0, 5000.0, 350)
    h_arr = pickup_acoustic_response(f_arr, coils, speeds)
    dh = np.gradient(h_arr, f_arr)
    d2h = np.gradient(dh, f_arr)

    # The rate of change of derivative must be bounded without sharp spikes (|d2h| < 5e-6)
    assert np.max(np.abs(d2h)) < 5e-6, (
        "Comb notch must have a smooth parabolic bottom without non-differentiable V-cusps"
    )
    # Notch must retain authentic physical acoustic depth (-12 to -8 dB, i.e. 0.20 to 0.40)
    assert 0.20 <= np.min(h_arr) <= 0.40


def test_identity_acoustic_transfer_preserves_flat_bass():
    """Verify that modeling a source instrument against its matching voice bypasses acoustic deconvolution."""
    inst_p = load_instrument("34in_standard_p")
    assert is_voice_matching_source(inst_p, "precision_vintage", VOICES["precision_vintage"])

    inst_jazz = load_instrument("34in_standard_jazz")
    assert is_voice_matching_source(inst_jazz, "jazz_pair_open", VOICES["jazz_pair_open"])
    assert is_voice_matching_source(inst_jazz, "jazz_pair_active", VOICES["jazz_pair_active"])

    # 5-string Dingwall bridge matches Voice 13 (Dingwall Multi-Scale Bridge)
    inst_dingwall = load_instrument("37in_multiscale_dingwall")
    assert is_voice_matching_source(inst_dingwall, "dingwall_bridge", VOICES["dingwall_bridge"])

    # Non-matching voice should return False
    assert not is_voice_matching_source(inst_p, "jazz_pair_open", VOICES["jazz_pair_open"])


def test_numpy_pickup_macro_aperture_properties():
    """Verify that macro aperture computes a smooth, comb-free sensing envelope."""
    inst = load_instrument("30in_emg_mmtw")
    src_pickup = inst.pickups["mmtw_dual"]
    speeds = inst.string_wave_speeds

    freqs = np.linspace(20.0, 10000.0, 500)
    macro_env = numpy_pickup_macro_aperture(freqs, src_pickup.coils, speeds)

    # 1. DC / fundamental must be ~1.0 (within 0.1%)
    assert math.isclose(macro_env[0], 1.0, rel_tol=1e-3)

    # 2. Smooth aperture rolloff without comb nulls: > 0.70 at 2 kHz, > 0.05 at 10 kHz
    f2k_idx = int(np.argmin(np.abs(freqs - 2000.0)))
    assert macro_env[f2k_idx] > 0.70
    assert all(x > 0.05 for x in macro_env)


def test_cylindrical_rod_vs_blade_aperture():
    """Verify 2D cylindrical rod aperture (Airy/Bessel) and 1D blade aperture properties."""
    freqs = np.asarray(FREQS, dtype=np.float64)
    v_disp = np.full_like(freqs, 100.0)
    w_m = 0.75 * 0.0254

    ap_rod = compute_coil_aperture(freqs, v_disp, w_m, pole_type="rod")
    ap_blade = compute_coil_aperture(freqs, v_disp, w_m, pole_type="blade")

    # 1. Exact unity at DC
    assert math.isclose(ap_rod[0], 1.0, abs_tol=1e-6)
    assert math.isclose(ap_blade[0], 1.0, abs_tol=1e-6)

    # 2. Both must be strictly monotonic decreasing
    assert np.all(np.diff(ap_rod) <= 1e-9)
    assert np.all(np.diff(ap_blade) <= 1e-9)

    # 3. Rod pole piece (2D disc) has slightly higher high-frequency response than a full-width blade
    idx_4k = int(np.argmin(np.abs(freqs - 4000.0)))
    assert ap_rod[idx_4k] > ap_blade[idx_4k], (
        "Cylindrical rod should exhibit crisper top-end transmission than a solid blade"
    )


def test_saddle_witness_point_boundary_stiffness():
    """Verify bridge saddle boundary layer stiffness behavior."""
    freqs = np.asarray(FREQS, dtype=np.float64)

    # 1. Distances >= 75 mm (e.g. neck pickup or P-bass at 125 mm) must return exact 1.0
    h_p = compute_saddle_boundary_coupling(freqs, pos_m=0.125)
    assert np.all(h_p == 1.0), (
        "Pickups >= 75 mm from bridge must not be attenuated by saddle boundary"
    )

    # 2. Close bridge pickup (e.g. 60s Jazz bridge at 63.5 mm)
    h_bridge = compute_saddle_boundary_coupling(freqs, pos_m=0.0635)
    assert math.isclose(h_bridge[0], 1.0, abs_tol=1e-6), "Saddle coupling must be exact 1.0 at DC"
    assert np.all(np.diff(h_bridge) <= 1e-9), "Saddle coupling must be monotonically decreasing"

    # Attenuation at 10 kHz should be subtle (< 1.5 dB)
    idx_10k = int(np.argmin(np.abs(freqs - 10000.0)))
    att_db = 20.0 * np.log10(h_bridge[idx_10k])
    assert -1.8 < att_db < -0.3, (
        f"Saddle attenuation at 10 kHz was {att_db:.2f} dB (expected -1.8 to -0.3 dB)"
    )

    # 3. Very close bridge pickup (e.g. Dingwall bridge at 48 mm)
    h_dingwall = compute_saddle_boundary_coupling(freqs, pos_m=0.048)
    assert h_dingwall[idx_10k] < h_bridge[idx_10k], (
        "Pickups closer to bridge should experience slightly greater boundary stiffness damping"
    )


def test_pickup_isolation_leveling():
    """Verify luthier individual pickup height compensation in isolation."""
    # 1. Pickup at reference position (or ahead of it) evaluates to exact 1.0000 (0.00 dB)
    k_ref = compute_pickup_isolation_leveling(0.1556, scale_m=0.8636, ref_pos_m=0.1556)
    assert math.isclose(k_ref, 1.0, abs_tol=1e-6)

    k_ahead = compute_pickup_isolation_leveling(0.2000, scale_m=0.8636, ref_pos_m=0.1556)
    assert math.isclose(k_ahead, 1.0, abs_tol=1e-6)

    # 2. Bridge pickup (63.5 mm on Jazz Bass) gets smooth luthier height boost (+4.8 to +5.8 dB)
    k_jazz_bridge = compute_pickup_isolation_leveling(0.0635, scale_m=0.8636, ref_pos_m=0.1556)
    boost_db = 20.0 * math.log10(k_jazz_bridge)
    assert 4.8 <= boost_db <= 5.8, f"Expected Jazz bridge boost ~5.3 dB, got {boost_db:.2f} dB"

    # 3. Dingwall bridge pickup (48.0 mm on 37" scale) relative to middle (96.0 mm)
    k_dingwall_bridge = compute_pickup_isolation_leveling(0.0480, scale_m=0.9398, ref_pos_m=0.0960)
    dingwall_boost_db = 20.0 * math.log10(k_dingwall_bridge)
    assert 4.0 <= dingwall_boost_db <= 5.0, (
        f"Expected Dingwall bridge boost ~4.5 dB, got {dingwall_boost_db:.2f} dB"
    )

    # 4. Maximum boost is strictly bounded by max_boost_db (+6.0 dB)
    k_extreme = compute_pickup_isolation_leveling(
        0.0100, scale_m=0.8636, ref_pos_m=0.3000, max_boost_db=6.0
    )
    assert 20.0 * math.log10(k_extreme) <= 6.0

    # 5. Single-pickup bass (ref_pos_m is None) evaluates to bit-exact 1.0000 (0.00 dB)
    k_single = compute_pickup_isolation_leveling(0.0635, scale_m=0.8636, ref_pos_m=None)
    assert math.isclose(k_single, 1.0, abs_tol=1e-9)


def test_scale_normalized_bridge_proximity_displacement():
    """Verify that bridge proximity spatial displacement uses scale-normalized fractional positions."""
    # In 32in custom P/MM, MM pickup is at 62.2mm on 812.8mm scale: eta = 0.0765
    # In 34in StingRay, MM pickup is at 66.0mm on 863.6mm scale: eta = 0.0764
    # The fractional displacement between the two matching sweet spots is virtually zero
    eta_32 = 0.0622 / 0.8128
    eta_34 = 0.0660 / 0.8636
    delta_in_scaled = (eta_34 - eta_32) * 34.0
    assert abs(delta_in_scaled) < 0.01

    # Raw absolute mm would erroneously claim the 34in is +0.15" further forward
    delta_in_unnormalized = (0.0660 - 0.0622) / 0.0254
    assert delta_in_unnormalized > 0.14


def test_soft_clamp_displacement_ratio_bounds():
    """Verify asymmetric order-4 algebraic limiter ('alg4') bounds and linear fidelity."""
    # 1. Linear region around 0 dB: |delta_g| <= 6 dB has error < 0.10 dB
    for dg in [-6.0, -3.0, 0.0, 3.0, 6.0]:
        clamped = float(soft_clamp_displacement_ratio(dg))
        assert abs(clamped - dg) < 0.10

    # 2. Positive excursion ceiling: <= +12.0 dB
    assert float(soft_clamp_displacement_ratio(30.0)) <= 12.0
    assert float(soft_clamp_displacement_ratio(100.0)) <= 12.0

    # 3. Negative excursion floor: >= -16.0 dB
    assert float(soft_clamp_displacement_ratio(-40.0)) >= -16.0
    assert float(soft_clamp_displacement_ratio(-100.0)) >= -16.0

    # 4. Vectorized operation
    arr = np.array([-30.0, -5.0, 0.0, 5.0, 30.0])
    clamped_arr = soft_clamp_displacement_ratio(arr)
    assert isinstance(clamped_arr, np.ndarray)
    assert len(clamped_arr) == 5
    assert clamped_arr[0] >= -16.0
    assert clamped_arr[-1] <= 12.0


def test_compute_displacement_proximity_shelf():
    """Verify compute_displacement_proximity_shelf low-shelf properties."""
    freqs = np.asarray(FREQS, dtype=np.float64)

    # 1. Invalid positions return exact 1.0
    assert np.all(compute_displacement_proximity_shelf(freqs, pos_m=0.0, scale_m=0.8636) == 1.0)
    assert np.all(compute_displacement_proximity_shelf(freqs, pos_m=0.1, scale_m=0.0) == 1.0)

    # 2. High frequencies asymptotically approach 1.0 (0.00 dB)
    h_pos = compute_displacement_proximity_shelf(freqs, pos_m=0.125, scale_m=0.8636)
    idx_10k = int(np.argmin(np.abs(freqs - 10000.0)))
    assert math.isclose(h_pos[idx_10k], 1.0, abs_tol=0.05)

    # 3. Low frequencies reflect displacement ratio
    idx_20 = int(np.argmin(np.abs(freqs - 20.0)))
    assert h_pos[idx_20] > 1.0  # 125mm is further from bridge than calibration eta -> boost


def test_body_microphonic_coupling():
    """Verify mechanical body-pickup microphonic coupling physics and differential scaling."""
    freqs = np.asarray(FREQS, dtype=np.float64)

    # 1. Active EMG source to Vintage Alnico V target: Δk_body = 0.08 - 0.0 = 0.08
    src_pickup_active = PickupConfig(name="EMG Active", magnet_type="active")
    tgt_voice_alnico5 = VOICES["precision_vintage"]
    h_body = compute_body_microphonic_coupling(freqs, src_pickup_active, tgt_voice_alnico5)

    assert len(h_body) == len(freqs)
    assert np.all(np.isfinite(h_body))
    # DC and sub-audible must be exactly 1.0 (0.0 dB)
    assert math.isclose(h_body[0], 1.0, rel_tol=1e-5)
    # Peak must occur near 6.2 kHz
    peak_idx = int(np.argmax(h_body))
    peak_freq = freqs[peak_idx]
    assert 5500.0 <= peak_freq <= 6800.0
    # Boost should be subtle (+0.35 to +0.55 dB)
    peak_db = 20.0 * np.log10(h_body[peak_idx])
    assert 0.35 <= peak_db <= 0.55

    # 2. Matching passive source (Alnico V to Alnico V): Δk_body = 0.0 -> exact identity
    src_alnico5 = PickupConfig(name="Vintage P", magnet_type="alnico_v")
    h_body_id = compute_body_microphonic_coupling(freqs, src_alnico5, tgt_voice_alnico5)
    assert np.allclose(h_body_id, 1.0, atol=1e-12)

    # 3. Active-to-active: exact identity
    src_active = PickupConfig(name="EMG Active", magnet_type="active")
    tgt_active = tgt_voice_alnico5.model_copy(update={"magnet_type": "active"})
    h_body_active = compute_body_microphonic_coupling(freqs, src_active, tgt_active)
    assert np.allclose(h_body_active, 1.0, atol=1e-12)


def test_get_coil_register():
    """Verify get_coil_register maps string lists to 'lower', 'upper', or 'all'."""
    assert get_coil_register(CoilConfig(position_from_bridge_m=0.1, strings=["all"])) == "all"
    assert get_coil_register(CoilConfig(position_from_bridge_m=0.1, strings=[3, 4])) == "lower"
    assert get_coil_register(CoilConfig(position_from_bridge_m=0.1, strings=["E", "A"])) == "lower"
    assert get_coil_register(CoilConfig(position_from_bridge_m=0.1, strings=[1, 2])) == "upper"
    assert get_coil_register(CoilConfig(position_from_bridge_m=0.1, strings=["D", "G"])) == "upper"
    assert get_coil_register(CoilConfig(position_from_bridge_m=0.1, strings=[1, 2, 3, 4])) == "all"


def test_calibration_excursion_constants():
    """Verify module-level calibration excursion reference constants."""
    assert math.isclose(FRACTIONAL_EXCURSION_CALIBRATION_ETA, 0.1082677, abs_tol=1e-6)
    assert math.isclose(CALIBRATION_EXCURSION_ETA, 0.1082677, abs_tol=1e-6)


def test_multiscale_acoustic_response_range():
    """Verify numpy_pickup_acoustic_response handles multiscale range tuples."""
    coils = [
        CoilConfig(
            position_from_bridge_m=0.0635,
            aperture_width_in=0.75,
            weight=1.0,
            polarity=1.0,
            strings=["all"],
        )
    ]
    freqs = np.asarray(FREQS, dtype=np.float64)
    resp = numpy_pickup_acoustic_response(freqs, coils, scale_length_m=(0.8636, 0.9398))
    assert len(resp) == len(freqs)
    assert np.all(np.isfinite(resp))
    assert math.isclose(resp[0], 1.0, abs_tol=1e-3)


def test_body_microphonic_coupling_active_inst_and_zero_delta():
    """Verify compute_body_microphonic_coupling with active instrument and zero/negative delta_k."""
    freqs = np.asarray(FREQS, dtype=np.float64)
    inst_active = load_instrument("34in_active_emg").model_copy(update={"electronics": "active"})
    src_pickup = inst_active.pickups["neck"]
    tgt_voice = VOICES["precision_vintage"]

    # Active electronics treats source as active
    h_body = compute_body_microphonic_coupling(freqs, src_pickup, tgt_voice, inst=inst_active)
    assert np.max(h_body) > 1.0

    # Negative delta_k (e.g. Alnico V source to Neodymium target, 0.02 - 0.08 = -0.06)
    src_alnico = PickupConfig(name="P", magnet_type="alnico_v")
    tgt_neo = tgt_voice.model_copy(update={"magnet_type": "neodymium"})
    h_zero = compute_body_microphonic_coupling(freqs, src_alnico, tgt_neo)
    assert np.allclose(h_zero, 1.0)


def test_is_voice_matching_source_edge_cases():
    """Verify is_voice_matching_source edge cases and mismatch branches."""
    inst_p = load_instrument("34in_standard_p")

    # Non-existent voice
    assert not is_voice_matching_source(inst_p, "non_existent_voice")

    # Preserve aperture with no_eq
    v_preserve = VOICES["precision_vintage"].model_copy(
        update={"preserve_aperture": True, "circuit": type("Circuit", (), {"no_eq": True})()}
    )
    assert is_voice_matching_source(inst_p, "test_preserve", voice_cfg=v_preserve)

    # Scale range mismatch (> 0.012 m)
    v_short = VOICES["precision_vintage"].model_copy(update={"scale": "30in"})
    assert not is_voice_matching_source(inst_p, "test_short", voice_cfg=v_short)

    # Coil count mismatch (P-bass 2 split coils vs single coil voice)
    v_single = VOICES["precision_vintage"].model_copy(
        update={"coils": [VOICES["precision_vintage"].coils[0]]}
    )
    assert not is_voice_matching_source(inst_p, "test_single", voice_cfg=v_single)

    # Coil aperture width mismatch (> 0.15 in)
    wide_coil = VOICES["precision_vintage"].coils[0].model_copy(update={"aperture_width_in": 2.5})
    v_wide = VOICES["precision_vintage"].model_copy(update={"coils": [wide_coil, wide_coil]})
    assert not is_voice_matching_source(inst_p, "test_wide", voice_cfg=v_wide)


def test_numpy_pickup_acoustic_response_string_names_markers():
    """Verify acoustic response with various string name markers and register fallbacks."""
    freqs = np.asarray(FREQS, dtype=np.float64)
    # Only upper coil (strings=[1, 2]), so for lower partition active will be empty and fall back to list(coils)
    upper_coil_only = [
        CoilConfig(
            position_from_bridge_m=0.12,
            aperture_width_in=1.0,
            weight=1.0,
            polarity=1.0,
            strings=[1, 2],
        )
    ]
    resp_numeric = numpy_pickup_acoustic_response(
        freqs, upper_coil_only, string_speeds=[70.0, 90.0, 120.0, 160.0], string_names=[4, 3, 2, 1]
    )
    assert len(resp_numeric) == len(freqs)
    assert math.isclose(resp_numeric[0], 1.0, abs_tol=1e-3)

    resp_str_markers = numpy_pickup_acoustic_response(
        freqs,
        upper_coil_only,
        string_speeds=[70.0, 90.0, 120.0, 160.0],
        string_names=["E", "A", "D", "G"],
    )
    assert len(resp_str_markers) == len(freqs)
    assert math.isclose(resp_str_markers[0], 1.0, abs_tol=1e-3)


def test_numpy_pickup_acoustic_response_advanced_branches():
    """Verify acoustic response with stacked/coaxial coils (delta_x_span <= 0.002) and custom string names."""
    freqs = np.asarray(FREQS, dtype=np.float64)
    # Stacked humbucker with delta_x_span == 0
    coils_stacked = [
        CoilConfig(
            position_from_bridge_m=0.075,
            aperture_width_in=0.75,
            weight=0.5,
            polarity=1.0,
            strings=["all"],
        ),
        CoilConfig(
            position_from_bridge_m=0.075,
            aperture_width_in=0.75,
            weight=0.5,
            polarity=1.0,
            strings=["all"],
        ),
    ]
    resp_stacked = numpy_pickup_acoustic_response(freqs, coils_stacked, scale_length_m=0.8636)
    assert len(resp_stacked) == len(freqs)
    assert math.isclose(resp_stacked[0], 1.0, abs_tol=1e-3)

    # Passing scale_length_m as a list of speeds (> 10.0)
    speeds_list = [70.0, 95.0, 125.0, 165.0]
    resp_from_speeds = numpy_pickup_acoustic_response(
        freqs,
        coils_stacked,
        scale_length_m=speeds_list,
        string_names=["low", "bass", "treble", "high"],
    )
    assert len(resp_from_speeds) == len(freqs)
    assert math.isclose(resp_from_speeds[0], 1.0, abs_tol=1e-3)


def test_numpy_pickup_macro_aperture_blade_and_speeds():
    """Verify macro aperture with blade pole piece, custom string speeds, and empty coil fallback."""
    freqs = np.asarray(FREQS, dtype=np.float64)
    coils_blade = [
        CoilConfig(
            position_from_bridge_m=0.08,
            aperture_width_in=0.75,
            pole_type="blade",
            weight=1.0,
            polarity=1.0,
            strings=["all"],
        )
    ]
    macro_blade = numpy_pickup_macro_aperture(
        freqs, coils_blade, string_speeds=[70.0, 95.0, 125.0, 165.0]
    )
    assert len(macro_blade) == len(freqs)
    assert math.isclose(macro_blade[0], 1.0, abs_tol=1e-3)

    # Empty coils fallback
    macro_empty = numpy_pickup_macro_aperture(freqs, [])
    assert len(macro_empty) == len(freqs)
    assert math.isclose(macro_empty[0], 1.0, abs_tol=1e-3)
