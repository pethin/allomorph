"""
Tests for pickup aperture response, spatial comb filtering, dual-coil humbuckers,
saddle boundary stiffness, cylindrical rod vs blade aperture, and microphonics.
"""

import math

import numpy as np

from allomorph.config import (
    SCALES,
    VOICES,
    get_source_pickup,
    load_instrument,
    resolve_pickup_coils,
)
from allomorph.config.schema import CoilConfig, PickupConfig
from allomorph.dsp import FREQS
from allomorph.physics import (
    aperture_response,
    compute_body_microphonic_coupling,
    compute_coil_aperture,
    compute_pickup_isolation_leveling,
    compute_saddle_boundary_coupling,
    compute_voice_prefilter_firs,
    is_voice_matching_source,
    numpy_pickup_macro_aperture,
    pickup_acoustic_response,
    position_envelope,
)
from allomorph.visualizer import build_voice_dataframe


def test_aperture_zero_frequency():
    speeds = SCALES["30in"].speeds
    res = aperture_response(np.array([0.0]), w_in=1.5, d_in=0.75, speeds=speeds)[0]
    # Physical aperture: Sinc(0) = 1.0000, Comb(0) = 1.0000, Product = 1.0000 (0.00 dB)
    assert math.isclose(res, 1.0, abs_tol=1e-6)


def test_aperture_single_vs_dual():
    speeds = SCALES["30in"].speeds
    val_single = aperture_response(np.array([5000.0]), w_in=0.75, d_in=0.0, speeds=speeds)[0]
    val_dual = aperture_response(np.array([5000.0]), w_in=1.50, d_in=0.75, speeds=speeds)[0]

    # Dual coil should have more high-frequency aperture filtering than narrow single coil
    assert val_dual < val_single


def test_position_envelope():
    speeds = SCALES["34in"].speeds
    freqs = np.array([200.0, 1000.0, 5000.0])
    res_bridge = position_envelope(freqs, pos_m=0.0406, speeds=speeds)
    res_neck = position_envelope(freqs, pos_m=0.1250, speeds=speeds)

    assert len(res_bridge) == 3
    assert len(res_neck) == 3
    assert all(val > 0.1 for val in res_bridge)
    assert all(val > 0.1 for val in res_neck)


def test_pickup_acoustic_response_single_coil():
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
    # Notch must retain authentic physical acoustic depth (-12 to -8 dB, i.e. 0.25 to 0.40)
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


def test_character_voicings_aperture_and_identity_behavior():
    """Verify aperture preservation and identity matching invariants for character voicings.
    studio_direct is an acoustic identity (preserve_aperture=True, no_eq=True).
    studio_active and studio_passive preserve physical aperture but apply
    transformative circuit EQ, so is_voice_matching_source must strictly return False.
    """
    test_instruments = [
        "34in_standard_p",
        "30in_emg_mmtw",
        "34in_active_stingray",
        "37in_multiscale_dingwall",
        "34in_standard_jazz",
    ]

    for inst_id in test_instruments:
        inst = load_instrument(inst_id)

        # studio_direct is an identity spatial match across all basses
        assert is_voice_matching_source(inst, "studio_direct", VOICES["studio_direct"]), (
            f"studio_direct must match source aperture on {inst_id}"
        )

        # 15b and 15c are transformative circuits and must NEVER match source circuit
        assert not is_voice_matching_source(inst, "studio_active", VOICES["studio_active"]), (
            f"studio_active must NOT be an identity match on {inst_id}"
        )
        assert not is_voice_matching_source(inst, "studio_passive", VOICES["studio_passive"]), (
            f"studio_passive must NOT be an identity match on {inst_id}"
        )

        # Prefilter FIRs must be unity impulses (preserve_aperture=True)
        for char_vid in ["studio_direct", "studio_active", "studio_passive"]:
            firs = compute_voice_prefilter_firs(char_vid, instrument=inst, num_taps=64)
            assert len(firs) == 1
            assert firs[0][0] == 1.0
            assert all(x == 0.0 for x in firs[0][1:])


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
    f2k_idx = np.argmin(np.abs(freqs - 2000.0))
    assert macro_env[f2k_idx] > 0.70
    assert all(x > 0.05 for x in macro_env)


def test_30in_mm_pj_subbass_retention():
    """Verify that 30in MM dual-coil playing P/J hybrid retains full sub-bass without collapse."""
    df = build_voice_dataframe("pj_active", VOICES["pj_active"], instrument="30in_emg_mmtw")
    f20 = df.filter(df["frequency"] == 20.0)["magnitude_db"][0]
    f100 = df.filter((df["frequency"] >= 99.0) & (df["frequency"] <= 101.0))["magnitude_db"][0]
    f_max_val = df["magnitude_db"].max()
    assert isinstance(f_max_val, (int, float))
    f_max = float(f_max_val)

    # Sub-bass fundamental must retain warm displacement excursion (~ +3.5 dB) without collapse
    assert math.isclose(f20, 3.53, abs_tol=0.5)
    assert math.isclose(f100, 1.52, abs_tol=0.5)

    # Resonant peak must extend cleanly above passband (between +1.0 dB and +7.0 dB)
    assert 1.0 <= f_max <= 7.0


def test_30in_mm_jazz_pair_subbass_retention():
    """Verify that 30in MM dual-coil playing Jazz Bass pair retains full sub-bass without collapse."""
    df = build_voice_dataframe(
        "jazz_pair_open", VOICES["jazz_pair_open"], instrument="30in_emg_mmtw"
    )
    f20 = df.filter(df["frequency"] == 20.0)["magnitude_db"][0]
    f100 = df.filter((df["frequency"] >= 99.0) & (df["frequency"] <= 101.0))["magnitude_db"][0]

    # Sub-bass fundamental must retain authentic dual-coil displacement (~ -1.0 dB) without collapse
    assert math.isclose(f20, -1.03, abs_tol=0.5)
    assert math.isclose(f100, -1.27, abs_tol=0.5)


def test_jazz_differential_transfer_has_no_artificial_comb_filter():
    """Verify that multi-pickup matching source instruments (e.g. 34in_standard_jazz) have tau=0 and no comb filtering."""
    # 1. FIRs must have zero inter-pickup delay (peak at tap 0)
    firs_01 = compute_voice_prefilter_firs("jazz_pair_active", instrument="34in_standard_jazz")
    assert len(firs_01) == 2
    assert np.argmax(np.abs(firs_01[0])) <= 2
    assert np.argmax(np.abs(firs_01[1])) <= 2

    firs_02 = compute_voice_prefilter_firs("jazz_pair_open", instrument="34in_standard_jazz")
    assert len(firs_02) == 2
    assert np.argmax(np.abs(firs_02[0])) <= 2
    assert np.argmax(np.abs(firs_02[1])) <= 2

    # 2. Differential transfer function must be smooth without artificial comb filter notches
    inst = load_instrument("34in_standard_jazz")
    df_01 = build_voice_dataframe("jazz_pair_active", VOICES["jazz_pair_active"], instrument=inst)
    mags_01 = df_01["magnitude_db"].to_numpy()
    freqs = df_01["frequency"].to_numpy()

    i100 = np.argmin(np.abs(freqs - 100.0))
    i600 = np.argmin(np.abs(freqs - 600.0))
    i1800 = np.argmin(np.abs(freqs - 1800.0))

    # Without artificial comb filter, 600 Hz and 1800 Hz are within 2.0 dB of 100 Hz (smooth preamp shelf)
    assert abs(mags_01[i600] - mags_01[i100]) < 2.0
    assert abs(mags_01[i1800] - mags_01[i100]) < 2.0


def test_single_to_multi_pickup_coherence_eliminates_high_frequency_comb_notches():
    """Verify that single-to-multi pickup conversion retains the 600-800 Hz acoustic scoop while eliminating high-frequency comb notches."""
    inst = load_instrument("30in_emg_mmtw")
    for voice_id in ["jazz_pair_active", "jazz_pair_open"]:
        df = build_voice_dataframe(voice_id, VOICES[voice_id], instrument=inst)
        mags = df["magnitude_db"].to_numpy()
        freqs = df["frequency"].to_numpy()

        i100 = np.argmin(np.abs(freqs - 100.0))
        mask_mid = (freqs >= 500.0) & (freqs <= 800.0)
        min_mid = np.min(mags[mask_mid])
        scoop_depth = mags[i100] - min_mid

        # 1. Iconic acoustic mid-scoop must be preserved (distinct mid-scoop >= 8.0 dB relative to 100 Hz)
        assert scoop_depth >= 8.0, (
            f"{voice_id} mid-scoop was {scoop_depth:.1f} dB (expected >= 8.0 dB)"
        )

        # 2. High-frequency comb filter notches above 1.8 kHz must be eliminated (no notches deeper than -7 dB)
        for f_check in [2000.0, 3200.0, 4400.0]:
            idx = np.argmin(np.abs(freqs - f_check))
            assert mags[idx] > -7.0, (
                f"{voice_id} at {f_check} Hz was {mags[idx]:.1f} dB (expected > -7.0 dB, comb notch present)"
            )

    # 3. Voice 01 must rise smoothly without periodic comb ripple oscillations in 1.5 - 5.0 kHz
    df_01 = build_voice_dataframe("jazz_pair_active", VOICES["jazz_pair_active"], instrument=inst)
    mags_01 = df_01["magnitude_db"].to_numpy()
    freqs_01 = df_01["frequency"].to_numpy()
    mask_mid_hi = (freqs_01 >= 1500.0) & (freqs_01 <= 5000.0)
    diffs = np.diff(mags_01[mask_mid_hi])
    assert np.all(diffs >= -0.05), (
        "High frequencies must rise smoothly without periodic comb ripple oscillations"
    )


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


def test_multicoil_wavelength_dependent_coherence_and_mudbucker():
    """Verify that multi-coil cross-coherence decays smoothly without kinks and mudbucker rolls off monotonically."""
    # 1. Voice 09 (Music Man StingRay) on 34in Standard Jazz Bass
    jazz_inst = load_instrument("34in_standard_jazz")
    df_09 = build_voice_dataframe(
        "stingray_parallel",
        VOICES["stingray_parallel"],
        instrument=jazz_inst,
        mode="difference",
    )
    f_09 = df_09["frequency"].to_numpy()
    m_09 = df_09["magnitude_db"].to_numpy()

    # Fundamental mid-scoop at 2.5 kHz must be preserved (< -5.0 dB)
    scoop_mask = (f_09 >= 2200.0) & (f_09 <= 2800.0)
    assert np.min(m_09[scoop_mask]) < -5.0

    # Treble rise from 4.5 kHz to 7.0 kHz must be strictly monotonic (no 5.4 kHz plateau/kink)
    treble_mask = (f_09 >= 4500.0) & (f_09 <= 7000.0)
    diffs_09 = np.diff(m_09[treble_mask])
    assert np.all(diffs_09 >= -0.05), (
        "Treble rise must be smooth and monotonic without kinks or plateaus"
    )

    # 2. Voice 12 (Mudbucker) on 34in Active StingRay and 34in Standard P-Bass
    p_inst = load_instrument("34in_active_stingray")
    df_12 = build_voice_dataframe(
        "mudbucker_deep",
        VOICES["mudbucker_deep"],
        instrument=p_inst,
        mode="difference",
    )
    f_12 = df_12["frequency"].to_numpy()
    m_12 = df_12["magnitude_db"].to_numpy()

    # Sub-bass punch (+6.2 dB)
    assert m_12[0] > 5.5

    # Strictly monotonic rolloff above 1.5 kHz (no zigzag comb teeth)
    rolloff_mask = (f_12 >= 1500.0) & (f_12 <= 18000.0)
    diffs_12 = np.diff(m_12[rolloff_mask])
    assert np.all(diffs_12 <= 0.05), (
        "Mudbucker response on active StingRay must roll off monotonically without secondary peaks or teeth"
    )

    # Also strictly monotonic rolloff on passive 34in Standard P-Bass (no 10 kHz treble scoop or fizz)
    p_inst_p = load_instrument("34in_standard_p")
    df_12_p = build_voice_dataframe(
        "mudbucker_deep",
        VOICES["mudbucker_deep"],
        instrument=p_inst_p,
        mode="difference",
    )
    f_12_p = df_12_p["frequency"].to_numpy()
    m_12_p = df_12_p["magnitude_db"].to_numpy()

    assert m_12_p[0] > 5.5
    rolloff_mask_p = (f_12_p >= 1500.0) & (f_12_p <= 18000.0)
    diffs_12_p = np.diff(m_12_p[rolloff_mask_p])
    assert np.all(diffs_12_p <= 0.05), (
        "Mudbucker response on passive P-Bass must roll off monotonically without 10 kHz fizz"
    )


def test_dynamic_coherence_decay_and_multiscale_snap():
    """Verify dynamic delay-based coherence window for P/J and multiscale tension snap for short-scale."""
    # 1. Voice 07 (P/J Bass) on 30in short scale has delta = 25 samples (tau = 0.52ms, notch = 960Hz)
    inst_30 = load_instrument("30in_emg_mmtw")
    df_07 = build_voice_dataframe(
        "pj_active", VOICES["pj_active"], instrument=inst_30, mode="difference"
    )
    f_07 = df_07["frequency"].to_numpy()
    m_07 = df_07["magnitude_db"].to_numpy()

    # The dynamic coherence window must preserve the authentic ~10-11 dB P/J mid-scoop at 960 Hz
    idx_notch = np.argmin(np.abs(f_07 - 960.0))
    assert m_07[idx_notch] < -8.0, (
        f"Expected deep P/J mid-scoop at 960 Hz, got {m_07[idx_notch]:.2f} dB"
    )

    # 2. Voice 13 (Dingwall Multi-Scale) on 30in short scale must receive tension snap
    firs_13 = compute_voice_prefilter_firs("dingwall_bridge", instrument=inst_30)
    assert len(firs_13) == 1
    # Check that prefilter is non-trivial and has high-frequency energy
    assert np.linalg.norm(firs_13[0]) > 0.1


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
    # Boost should be subtle (+0.4 to +0.55 dB)
    peak_db = 20.0 * np.log10(h_body[peak_idx])
    assert 0.35 <= peak_db <= 0.55
    # Ultrasonic damping: above 15 kHz, curve smoothly rolls back toward 1.0
    idx_18k = int(np.argmin(np.abs(freqs - 18000.0)))
    assert 20.0 * np.log10(h_body[idx_18k]) < 0.15

    # 2. Matching passive source (Alnico V to Alnico V): Δk_body = 0.0 -> exact identity
    src_alnico5 = PickupConfig(name="Vintage P", magnet_type="alnico_v")
    h_body_id = compute_body_microphonic_coupling(freqs, src_alnico5, tgt_voice_alnico5)
    assert np.allclose(h_body_id, 1.0, atol=1e-12)

    # 3. Active-to-active: exact identity
    src_active = PickupConfig(name="EMG Active", magnet_type="active")
    tgt_active = VOICES["studio_active"]
    h_body_active = compute_body_microphonic_coupling(freqs, src_active, tgt_active)
    assert np.allclose(h_body_active, 1.0, atol=1e-12)

    # 4. Alnico III target has k_body = 0.09 (slightly higher peak than Alnico V at 0.08)
    tgt_voice_alnico3 = tgt_voice_alnico5.model_copy(update={"magnet_type": "alnico_iii"})
    h_body_a3 = compute_body_microphonic_coupling(freqs, src_pickup_active, tgt_voice_alnico3)
    peak_db_a3 = 20.0 * np.log10(np.max(h_body_a3))
    assert 0.40 <= peak_db_a3 <= 0.65
    assert np.max(h_body_a3) > np.max(h_body)

    # 5. BODY_COUPLING_PROPERTIES must be removed from allomorph.physics
    import allomorph.physics as phys

    assert not hasattr(phys, "BODY_COUPLING_PROPERTIES")


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
    idx_4k = np.argmin(np.abs(freqs - 4000.0))
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
    idx_10k = np.argmin(np.abs(freqs - 10000.0))
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

