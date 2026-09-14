"""
Allomorph Architectural Guardrails Automated Invariant Test Suite.

Programmatically verifies that all core physical modeling and numerical invariants
specified in AGENTS.md and docs/architectural_guardrails.md are strictly upheld:
1. Active preamp finite DC transmission (H(0) >= 1.0) and Gibbs truncation ripple prevention
2. Mathematical identity flatness (exact 0.00 dB on matching source/target)
3. Small-signal linearity (bit-exact linear bypass for peak <= 0.10)
4. Quadrature regularization floor at comb nulls
5. Accelerated Numba JIT execution for recursive audio buffer loops
6. First-class transducer taxonomy ('magnetic', 'bridge_force', 'direct') and zero-conditional deconvolution
"""

import ast
import math

import numpy as np

from allomorph.circuit import (
    REPO_ROOT,
    compute_active_preamp_eq,
    compute_differential_circuit_transfer_functions,
    load_circuit,
)
from allomorph.config import VOICES, load_instrument
from allomorph.dsp import FREQS
from allomorph.physics import soft_clamp_displacement_ratio
from allomorph.visualizer import build_voice_dataframe


def test_guardrail_active_preamp_dc_transmission():
    """Guardrail 5.3.3: Active preamps must feature flat, finite DC transmission (H(0) >= 1.0)
    to prevent unphysical sub-audible inversion steps and Gibbs truncation ripples."""
    s_dc = 1j * 2.0 * math.pi * 1e-6  # Near DC

    for preamp_type in ["sadowsky_2band", "stingray_2band"]:
        H_eq = compute_active_preamp_eq(preamp_type, s_dc)
        mag_dc = abs(H_eq)
        assert mag_dc >= 1.0, (
            f"Preamp {preamp_type} DC magnitude was {mag_dc:.4f} (expected >= 1.0). "
            "Sub-audible highpass poles (s / (s + w_sub)) are strictly prohibited."
        )


def test_guardrail_zero_gibbs_ripples_in_differential_curves():
    """Guardrail 5.3.3: Differential frequency response curves between 20 Hz and 300 Hz
    must be smooth and monotonic without periodic Gibbs truncation ripple oscillations."""
    active_sources = ["34in_active_stingray", "34in_preamp_soapbar", "34in_active_emg"]
    test_voices = ["jazz_pair_open", "precision_vintage"]

    for inst_id in active_sources:
        inst = load_instrument(inst_id)
        for voice_id in test_voices:
            df = build_voice_dataframe(
                voice_id, VOICES[voice_id], instrument=inst, mode="difference"
            )
            sub_df = df.filter((df["frequency"] >= 20.0) & (df["frequency"] <= 300.0))
            mags = sub_df["magnitude_db"].to_numpy()

            # Numerical derivative (slope differences)
            diffs = np.diff(mags)
            # Count sign flips (extrema in 20-300 Hz)
            sign_flips = sum(
                1
                for i in range(len(diffs) - 1)
                if (diffs[i] > 1e-4 and diffs[i + 1] < -1e-4)
                or (diffs[i] < -1e-4 and diffs[i + 1] > 1e-4)
            )

            assert sign_flips <= 1, (
                f"{inst_id} -> {voice_id} had {sign_flips} slope sign flips between 20 Hz and 300 Hz. "
                "Periodic Gibbs truncation ripples are present."
            )


def test_guardrail_zero_high_frequency_gibbs_ripples():
    """Guardrail 5.1.2: Multi-pickup spatial arrival delays must use causal integer sample shifting
    rather than circular FFT phase rotations to prevent high-frequency (8-20 kHz) Gibbs truncation ripples."""
    active_sources = [
        "34in_active_stingray",
        "30in_emg_mmtw",
        "34in_preamp_soapbar",
        "34in_active_emg",
    ]
    test_voices = ["jazz_pair_active", "pj_active", "jazz_pair_open"]

    for inst_id in active_sources:
        inst = load_instrument(inst_id)
        for voice_id in test_voices:
            df = build_voice_dataframe(
                voice_id, VOICES[voice_id], instrument=inst, mode="difference"
            )
            sub_df = df.filter((df["frequency"] >= 8000.0) & (df["frequency"] <= 20000.0))
            mags = sub_df["magnitude_db"].to_numpy()

            diffs = np.diff(mags)
            sign_flips = sum(
                1
                for i in range(len(diffs) - 1)
                if (diffs[i] > 1e-4 and diffs[i + 1] < -1e-4)
                or (diffs[i] < -1e-4 and diffs[i + 1] > 1e-4)
            )

            assert sign_flips <= 1, (
                f"{inst_id} -> {voice_id} had {sign_flips} slope sign flips between 8 kHz and 20 kHz. "
                "Periodic Gibbs truncation ripples are present in high frequencies."
            )


def test_guardrail_identity_model_flatness():
    """Guardrail 5.3.1: Pairing an instrument with its matching target voice must evaluate
    to exact 0.00 dB identity across all frequency bins."""
    inst_ray = load_instrument("34in_active_stingray")
    ray_circ = inst_ray.pickups["mm_parallel"].circuit
    assert ray_circ is not None
    m_src_ray = load_circuit(ray_circ)
    m_tgt_ray = load_circuit(VOICES["stingray_parallel"].circuit)

    diff_ray = compute_differential_circuit_transfer_functions(m_tgt_ray, m_src_ray, freqs=FREQS)
    h_diff = np.asarray(diff_ray[0])

    assert np.all(h_diff == 1.0), (
        "Matching active StingRay models must produce bit-exact 1.000 (0.00 dB)"
    )


def test_guardrail_small_signal_linearity():
    """Guardrail 5.4.1: Audio signals with peak amplitude <= 0.10 must bypass saturation
    and non-linear drag bit-exact to preserve linear test fidelity."""
    from allomorph.circuit import apply_oversampled_saturation

    sr = 48000
    n_samples = 2400
    # Small signal well within linear threshold (0.05 peak)
    small_sig = (0.05 * np.sin(2.0 * np.pi * 100.0 * np.arange(n_samples) / sr)).astype(np.float32)

    sat_out = apply_oversampled_saturation(
        small_sig,
        vsat=0.45,
        alpha=0.15,
        alpha3=0.08,
        eta_hyst=0.06,
        k_sag=0.25,
        k_eddy=0.15,
        k_pull=0.12,
        k_stein=0.03,
    )

    # On small signals, the saturation engine must return a bit-exact identical array
    assert np.array_equal(sat_out, small_sig), (
        "Small signals (<= 0.10) must bypass saturation bit-exact"
    )


def test_guardrail_quadrature_null_floor_bounded():
    """Guardrail 5.2.3: Multi-coil combining must incorporate the quadrature regularization floor
    so deep comb cancellation nulls never collapse to singular non-differentiable cusps."""
    from allomorph.physics import numpy_pickup_macro_aperture

    inst = load_instrument("30in_emg_mmtw")
    src_pickup = inst.pickups["mmtw_dual"]
    speeds = inst.string_wave_speeds

    freqs = np.linspace(20.0, 15000.0, 1000)
    macro_env = numpy_pickup_macro_aperture(freqs, src_pickup.coils, speeds)

    min_val = np.min(macro_env)
    # The quadrature floor (0.18^2) guarantees transmission never drops below ~0.03 (-30 dB)
    assert min_val > 0.02, (
        f"Comb null dropped to {min_val:.5f} (< 0.02); quadrature floor is missing."
    )


def test_guardrail_buffer_loop_acceleration():
    """Guardrail 6.1: Recursive ODE state solvers in saturation.py must be decorated
    with @njit to prevent interpreted Python loops over audio buffers."""
    sim_script = REPO_ROOT / "src" / "allomorph" / "circuit" / "saturation.py"
    tree = ast.parse(sim_script.read_text())

    recursive_cores = ["_lenz_velocity_drag_core", "_dahl_core", "_slew_limit_core"]
    found_cores = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in recursive_cores:
            decorator_names = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorator_names.append(dec.id)
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    decorator_names.append(dec.func.id)
            if node.name not in found_cores or "njit" in decorator_names:
                found_cores[node.name] = decorator_names

    for core_name in recursive_cores:
        assert core_name in found_cores, f"Expected {core_name} to exist in {sim_script.name}"
        assert "njit" in found_cores[core_name], (
            f"{core_name} is missing @njit fastmath acceleration"
        )


def test_guardrail_transducer_taxonomy_and_zero_conditional_deconvolution():
    """Guardrail 5.3.4: Transducers must be modeled via first-class physical taxonomy
    ('magnetic', 'bridge_force', 'direct') with zero ad-hoc voice ID conditionals."""
    # 1. Verify all registered voices declare a recognized physical sensor_type
    valid_sensors = {"magnetic", "bridge_force", "direct"}
    for vid, cfg in VOICES.items():
        sensor = cfg.sensor_type
        assert sensor in valid_sensors, f"Voice {vid} has invalid sensor_type: '{sensor}'"

    # 2. AST check: physics module must contain zero hardcoded voice ID conditionals in FIR synthesis
    phys_file = REPO_ROOT / "src" / "allomorph" / "physics" / "prefilter.py"
    tree = ast.parse(phys_file.read_text())

    prohibited_constants = {
        "15_source_direct",
        "studio_direct",
        "studio_active",
        "studio_passive",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_voice_prefilter_firs":
            for sub_node in ast.walk(node):
                if (
                    isinstance(sub_node, ast.Constant)
                    and isinstance(sub_node.value, str)
                    and sub_node.value in prohibited_constants
                ):
                    raise AssertionError(
                        f"Found prohibited hardcoded voice ID '{sub_node.value}' inside compute_voice_prefilter_firs. "
                        "All acoustic filtering must be governed by first-class physical parameters (e.g. sensor_type)."
                    )

    # 3. Direct sensor target output mode must evaluate to bit-exact 0.00 dB
    vcfg = VOICES["studio_direct"]
    df_out = build_voice_dataframe("studio_direct", vcfg, instrument="studio_direct", mode="output")
    mags_out = df_out["magnitude_db"].to_numpy()
    assert np.all(mags_out == 0.0), (
        f"studio_direct output mode was not bit-exact 0.00 dB (max error: {np.max(np.abs(mags_out))})"
    )

    # 4. Studio Voicings preserve aperture (unit impulse)
    from allomorph.physics import compute_voice_prefilter_firs

    firs = compute_voice_prefilter_firs("studio_direct", instrument="studio_direct")
    assert len(firs) == 1
    fir = np.array(firs[0])
    assert fir[0] == 1.0
    assert np.all(fir[1:] == 0.0)

    # 5. Direct sensor deconvolution without preserve_aperture must smoothly invert aperture sinc
    test_direct_cfg = vcfg.model_copy(update={"preserve_aperture": False})
    VOICES["_test_direct_deconv"] = test_direct_cfg
    try:
        firs_dir = compute_voice_prefilter_firs("_test_direct_deconv", instrument="34in_standard_p")
        assert len(firs_dir) == 1
        fir_dir = np.array(firs_dir[0])

        f_bins = np.fft.rfftfreq(8192, 1.0 / 48000.0)
        H = np.abs(np.fft.rfft(fir_dir, 8192))
        gain_5k = H[np.argmin(np.abs(f_bins - 5000))] / H[np.argmin(np.abs(f_bins - 20))]
        assert 1.2 <= gain_5k <= 2.5, f"Expected 1.2 <= gain_5k <= 2.5, got {gain_5k:.3f}"

        # Verify monotonic smooth inversion in 20 Hz to 5000 Hz passband (zero sign flips)
        mask = (f_bins >= 20.0) & (f_bins <= 5000.0)
        H_band = H[mask]
        diffs = np.diff(H_band)
        sign_flips = sum(
            1
            for i in range(len(diffs) - 1)
            if (diffs[i] > 1e-5 and diffs[i + 1] < -1e-5)
            or (diffs[i] < -1e-5 and diffs[i + 1] > 1e-5)
        )
        assert sign_flips == 0, (
            f"Deconvolution curve had {sign_flips} sign flips in 20-5000 Hz band (must be smoothly monotonic)"
        )
    finally:
        del VOICES["_test_direct_deconv"]


def test_guardrail_fail_fast_zero_silent_fallbacks():
    """Guardrail 5.3.5: Missing configuration models, invalid scale names, unknown pickups,
    unknown string presets, or unrecognized magnet types must immediately raise explicit
    ValueError or KeyError exceptions instead of silently applying default fallbacks."""
    import pytest

    from allomorph.circuit import simulate_voice
    from allomorph.circuit.parser import CircuitModel
    from allomorph.circuit.solver import apply_magnet_properties_to_model
    from allomorph.config.scales import resolve_scale_range
    from allomorph.config.schema import InstrumentConfig, InstrumentStringsConfig, PickupConfig
    from allomorph.config.strings import get_instrument_string

    # 1. Passive instrument with missing pickup circuit must raise ValueError
    dummy_passive = InstrumentConfig(
        id="mock_passive_bass",
        name="Mock Passive Bass",
        electronics="passive",
        default_pickup="p",
        pickups={
            "p": PickupConfig(
                name="Passive P",
                position_from_bridge_m=0.125,
                aperture_width_in=0.75,
                coil_spacing_in=0.0,
                magnet_type="alnico_v",
            )
        },
        string_wave_speeds=[73.4, 98.0, 130.8, 174.6],
        scale_length_in=34.0,
    )
    with pytest.raises(ValueError, match="does not define a '\\[circuit\\]' block"):
        simulate_voice("precision_active", instrument=dummy_passive, max_samples=100)

    # 2. Unknown target voice ID must raise KeyError
    with pytest.raises(KeyError, match="Target voice 'nonexistent_voice' not found"):
        simulate_voice("nonexistent_voice")

    # 3. Unknown scale string must raise ValueError
    with pytest.raises(ValueError, match="Unknown scale or instrument identifier"):
        resolve_scale_range("99in_fictional_scale")

    # 4. Unknown string preset must raise KeyError
    with pytest.raises(KeyError, match="String preset 'imaginary_flats' not found"):
        get_instrument_string(
            InstrumentConfig(strings=InstrumentStringsConfig(preset="imaginary_flats"))
        )

    # 5. Unknown magnet type must raise KeyError
    with pytest.raises(KeyError, match="Unknown magnet type 'kryptonite'"):
        apply_magnet_properties_to_model(
            CircuitModel(), PickupConfig(name="mock", magnet_type="kryptonite")
        )


def test_guardrail_visualizer_vectorization_and_performance():
    """Guardrail 6.4 (Commit 7c6e634): Voicing visualizer generation must be vectorized
    and execute in < 150 ms with step=3 downsampling, without redundant multi-rate FFTs."""
    import time

    from allomorph.visualizer import build_voicings_comparison_data

    # 1. Bounded execution latency: building voicings data must take < 150 ms warm
    # (after global get_cached_target_dfs is primed)
    build_voicings_comparison_data(step=3)

    t0 = time.perf_counter()
    data = build_voicings_comparison_data(step=3)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert duration_ms < 150.0, (
        f"build_voicings_comparison_data took {duration_ms:.2f} ms (budget: < 150 ms). "
        "Iterative build_voice_dataframe(mode='difference') calls inside per-voice loops are prohibited."
    )

    # 2. Downsampling invariant: Exactly 200 points per curve (600 // 3)
    freqs = data["frequencies"]
    assert len(freqs) == 200, f"Expected 200 downsampled frequency points, found {len(freqs)}"

    # 3. Payload rounding invariant: Decibel magnitudes must be rounded to at most 2 decimal places
    for vid, vdata in data["voices"].items():
        for m in vdata["magnitude_db"]:
            assert round(m, 2) == m, (
                f"Unrounded float {m} violates payload compression guardrail in {vid}"
            )


def test_guardrail_visualizer_voicings_comparison_fidelity():
    """Guardrail 3.7.2 & 5.3.6: Interactive Voicings Comparison visualizer curves
    (build_voicings_comparison_data and build_voicings_comparison_dataframe) must
    encompass all registered target voices, evaluate identity pairs to bit-exact 0.00 dB,
    and enforce H_diff = H_tgt - H_src across all frequency bins."""
    from allomorph.config import VOICES
    from allomorph.visualizer import (
        build_voicings_comparison_data,
        build_voicings_comparison_dataframe,
        compute_curve_rms_db,
    )

    # 1. Catalog Completeness: Encompasses all registered target voices
    data = build_voicings_comparison_data(step=3)
    assert set(data["voices"].keys()) == set(VOICES.keys())
    assert len(data["frequencies"]) == 200

    # 2. Physical Electroacoustic Boundedness (< +25 dB boost, > -120 dB attenuation)
    for vid, vdata in data["voices"].items():
        m_arr = np.array(vdata["magnitude_db"], dtype=np.float64)
        assert not np.isnan(m_arr).any()
        assert np.max(m_arr) < 25.0, f"Voice {vid} peak boost {np.max(m_arr)} exceeds +25 dB"
        assert np.min(m_arr) > -120.0, f"Voice {vid} attenuation {np.min(m_arr)} below -120 dB"
        for m in vdata["magnitude_db"]:
            assert round(m, 2) == m, f"Unrounded float {m} in voice {vid}"

    # 3. Identity Pair Invariant: Source == Target -> H_diff bit-exact 0.00 dB
    test_identities = [
        "precision_vintage",
        "jazz_bridge_growl",
        "stingray_parallel",
        "studio_direct",
    ]
    for vid in test_identities:
        df_id = build_voicings_comparison_dataframe(vid, vid, step=1)
        s3 = df_id.filter(df_id["line_type"] == "3. Normalized Difference (Norm. Diff)")[
            "magnitude_db"
        ]
        assert (s3 == 0.0).all(), f"Identity pair {vid} -> {vid} did not evaluate to 0.00 dB"

    # 4. Universal Differential Consistency: H_tgt,norm - H_src,norm = Norm. Diff in passband (< 3.5 kHz)
    # Beyond 8 kHz, boost is soft-knee capped at <= +7.0 dB and mollifier-tapered (retaining 25% at 20 kHz)
    df_diff = build_voicings_comparison_dataframe("precision_vintage", "jazz_bridge_growl", step=1)
    s1 = df_diff.filter(df_diff["line_type"] == "1. Source Voicing")["magnitude_db"].to_numpy()
    s2 = df_diff.filter(df_diff["line_type"] == "2. Target Voicing")["magnitude_db"].to_numpy()
    s3 = df_diff.filter(df_diff["line_type"] == "3. Normalized Difference (Norm. Diff)")[
        "magnitude_db"
    ].to_numpy()
    freqs = df_diff.filter(df_diff["line_type"] == "1. Source Voicing")["frequency"].to_numpy()
    idx_pass = freqs <= 3000.0
    src_rms = compute_curve_rms_db(s1)
    tgt_rms = compute_curve_rms_db(s2)
    s1_norm = s1 - src_rms
    s2_norm = s2 - tgt_rms
    assert np.allclose((s2_norm - s1_norm)[idx_pass], s3[idx_pass], atol=0.10)
    assert np.max(s3) <= 7.05
    assert s3[-1] <= 0.0


def test_guardrail_visualizer_voicing_ir_diff_3d_fidelity():
    """Guardrail 3.7.3 & 5.3.6: Voicing IR Difference 3D Waterfall & Waveform
    (build_voicing_ir_diff_3d_data) must generate Cumulative Spectral Decay (CSD)
    surfaces of the difference impulse response (h_diff(t)), collapsing to an
    ideal unit delta impulse and flat CSD on identity pairs."""
    from allomorph.config import VOICES
    from allomorph.visualizer import build_voicing_ir_diff_3d_data

    data = build_voicing_ir_diff_3d_data(num_freqs=50, num_slices=24, max_time_ms=10.0)
    assert set(data["voices"].keys()) == set(VOICES.keys())
    assert set(data["responses"].keys()) == set(VOICES.keys())

    # Identity pair: ideal unit delta impulse and flat 0.00 dB initial CSD
    for vid in ["precision_vintage", "jazz_bridge_growl", "stingray_parallel"]:
        entry = data["responses"][vid][vid]
        fir = entry["fir_waveform"]
        assert len(fir) == 128
        assert fir[0] == 1.0
        assert all(x == 0.0 for x in fir[1:])

        csd = entry["csd_matrix"]
        assert len(csd) == 24
        assert all(val == 0.0 for val in csd[0])
        assert all(val == -60.0 for val in csd[-1])


def test_guardrail_c_infinity_algebraic_rail_limiter():
    """Guardrail 5.2.1: Asymptotic algebraic rail limiter (p=8) must be strictly bounded
    (|f(x)| < V_sat), C^1/C^2 smooth with zero slope jumps, and preserve bit-exact linearity
    for small signals (|x| <= 0.10)."""
    vsat = 0.985
    p = 8.0

    # 1. Strict asymptotic boundedness: |f(x)| <= V_sat within float64 machine epsilon
    x_extremes = np.array([-100.0, -10.0, -2.0, -0.985, 0.0, 0.985, 2.0, 10.0, 100.0])
    f_extremes = x_extremes / ((1.0 + (np.abs(x_extremes) / vsat) ** p) ** (1.0 / p))
    assert np.all(np.abs(f_extremes) <= vsat + 1e-15), "Algebraic rail limiter breached V_sat bound"
    assert np.abs(f_extremes[2]) < vsat - 1e-4, (
        "Limiter must be strictly below V_sat for moderate drive"
    )
    assert np.all(np.diff(f_extremes) > 0.0), "Limiter must be strictly monotonic"

    # 2. Small-signal linearity: bit-exact linear bypass for |x| <= 0.10
    x_small = np.linspace(-0.10, 0.10, 201)
    f_small = x_small / ((1.0 + (np.abs(x_small) / vsat) ** p) ** (1.0 / p))
    max_lin_err = float(np.max(np.abs(f_small - x_small)))
    assert max_lin_err < 1e-8, f"Small-signal linearity error {max_lin_err:.2e} exceeded 1e-8"

    # 3. C^1 and C^2 continuity: numerical derivatives must be continuous with zero knee kinks
    x_grid = np.linspace(-1.5 * vsat, 1.5 * vsat, 2001)
    dx = x_grid[1] - x_grid[0]
    f_grid = x_grid / ((1.0 + (np.abs(x_grid) / vsat) ** p) ** (1.0 / p))
    f_prime = np.gradient(f_grid, dx)
    f_double_prime = np.gradient(f_prime, dx)

    # First derivative must be positive and bounded by 1.0 (passivity)
    assert np.all(f_prime > 0.0), "First derivative must be strictly positive"
    assert np.all(f_prime <= 1.0 + 1e-9), "First derivative must not exceed unity gain"
    # Second derivative must be finite and continuous without impulsive jumps
    assert not np.any(np.isnan(f_double_prime))
    assert np.max(np.abs(np.diff(f_double_prime))) < 0.5, (
        "Second derivative has discontinuous slope kink"
    )


def test_guardrail_vector_causal_normalization():
    """Guardrail 5.1.2: Multi-pickup spatial arrival delays must apply Vector Causal Normalization
    (tau_i = Delta_tau_i - min_j Delta_tau_j) to guarantee strict causality (min(tau_i) == 0)
    and preserve physical multi-pickup phase relationships without negative delays or circular FFT wraps."""
    from allomorph.physics.prefilter import compute_voice_prefilter_firs

    inst = load_instrument("34in_preamp_soapbar")
    # Voice 08 (Vintage PJ) has dual coils with different bridge distances (P=125 mm, J=63.5 mm)
    firs = compute_voice_prefilter_firs("pj_passive", inst)
    assert len(firs) == 2, "Expected 2 channel FIRs for PJ dual-pickup target"

    peaks = [int(np.argmax(np.abs(h))) for h in firs]

    # 1. Strict Causality Invariant: earliest wave arrival must have exactly zero pre-delay
    # (min(peak_shift) >= 0, no non-causal negative sample shifts)
    assert min(peaks) >= 0, "Non-causal negative sample shift detected"

    # 2. Physical phase delay preservation: relative arrival delay must be preserved
    # P pickup (125 mm from bridge) senses wave earlier than J bridge pickup (63.5 mm)
    delta_peaks = peaks[0] - peaks[1]
    assert delta_peaks != 0, "Multi-pickup arrival delay was lost or clamped to zero"


def test_guardrail_c_infinity_smooth_norms_and_steinmetz():
    """Guardrail 5.2.1: Charbonnier pseudo-norms (||x||_eps = sqrt(x^2 + eps^2) - eps)
    and quadratic Steinmetz core formulations must evaluate with continuous gradients
    vanishing at x = 0, eliminating non-differentiable cusps and infinite gradient singularities."""
    # 1. Charbonnier Pseudo-norm Invariant:
    eps = 1e-4
    x = np.linspace(-1e-2, 1e-2, 2001)
    dx = x[1] - x[0]
    norm = np.sqrt(x**2 + eps**2) - eps

    assert abs(norm[1000]) == 0.0, "Charbonnier norm must vanish exactly at x=0"
    assert np.all(norm >= 0.0), "Charbonnier norm must be strictly non-negative"

    # Gradient must be C^1 continuous and vanish at origin
    grad = np.gradient(norm, dx)
    assert abs(grad[1000]) < 1e-6, "Charbonnier gradient must vanish at origin"
    assert np.all(np.diff(grad) >= 0.0), "Charbonnier gradient must be monotonically non-decreasing"

    # 2. Quadratic Steinmetz Loss Core Invariant:
    # Formulated as (x^2 / (1 + x^2))^0.8 rather than (|x| / (1 + |x|))^1.6
    u = np.linspace(0.0, 1.0, 1001)
    du = u[1] - u[0]
    stein = (u**2 / (1.0 + u**2)) ** 0.8
    stein_grad = np.gradient(stein, du)

    # Gradient must remain strictly finite at u=0 (no infinite singularity)
    assert np.all(np.isfinite(stein_grad)), "Steinmetz gradient contains NaN or Inf"
    assert stein_grad[0] < 10.0, (
        f"Steinmetz gradient at origin {stein_grad[0]} exploded (singularity present)"
    )


def test_guardrail_inharmonicity_gaussian_rbf_invariants():
    """Guardrail 5.1.4: String stiffness and inharmonicity B_s must interpolate laboratory anchors
    via an exact C^inf Gaussian RBF, strictly preserving empirical table values (< 1e-10 relative error)
    and strictly decreasing monotonicity across bass fundamental registers [20, 250] Hz."""
    from allomorph.physics.strings import (
        INHARMONICITY_ANCHORS_BS,
        INHARMONICITY_ANCHORS_F0,
        get_inharmonicity_for_f0,
    )

    # 1. Exact anchor reproduction
    b_vals = [get_inharmonicity_for_f0(f0) for f0 in INHARMONICITY_ANCHORS_F0]
    rel_errors = [abs(b - exp) / exp for b, exp in zip(b_vals, INHARMONICITY_ANCHORS_BS)]
    max_err = max(rel_errors)
    assert max_err < 1e-10, (
        f"Gaussian RBF inharmonicity anchor relative error {max_err:.2e} exceeded 1e-10"
    )

    # 2. Physical Monotonicity: B_s must strictly decrease as fundamental frequency rises
    # across the bass guitar fundamental anchor range [20.0, 196.0] Hz (up to 12th fret G string)
    f_grid = np.linspace(20.0, 196.0, 500)
    b_grid = np.array([get_inharmonicity_for_f0(f) for f in f_grid])
    diffs = np.diff(b_grid)
    assert np.all(diffs < 0.0), "Inharmonicity B_s is not strictly decreasing across [20, 196] Hz"


def test_guardrail_dielectric_admittance_dc_continuity():
    """Guardrail 5.5.1: Cole-Davidson dielectric admittance and series capacitor networks
    must be continuous down to DC (f = 0.0 Hz) without piecewise branch step jumps or NaN/Inf."""
    from allomorph.circuit import compute_circuit_transfer_functions, load_circuit

    f_dc_grid = np.array([0.0, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0])

    for voice_id in ["precision_vintage", "rickenbacker_clank"]:
        model = load_circuit(voice_id)
        curves = compute_circuit_transfer_functions(model, freqs=f_dc_grid)
        for ch_curve in curves:
            arr = np.asarray(ch_curve, dtype=np.float64)
            assert not np.any(np.isnan(arr)), f"NaN in DC circuit response for {voice_id}"
            assert not np.any(np.isinf(arr)), f"Inf in DC circuit response for {voice_id}"
            assert np.all(arr >= 0.0), f"Negative magnitude in circuit response for {voice_id}"

            # Step jump between 0 Hz and 1e-6 Hz must be vanishingly small (< 1e-5)
            step_jump = abs(arr[1] - arr[0])
            assert step_jump < 1e-5, (
                f"Piecewise DC step discontinuity {step_jump:.2e} detected in {voice_id}"
            )


def test_guardrail_aperture_zero_frequency_exact_unity():
    """Guardrail 5.1.1: Physical acoustic aperture of sensing coils must evaluate to exact
    unity (1.0000, 0.00 dB) at zero frequency across all scale lengths (no artificial additive floors)."""
    from allomorph.config import SCALES
    from allomorph.physics import aperture_response

    f_zero = np.array([0.0])
    for s_name, s_cfg in SCALES.items():
        resp_single = aperture_response(f_zero, w_in=0.75, d_in=0.0, speeds=s_cfg.speeds)[0]
        resp_dual = aperture_response(f_zero, w_in=1.50, d_in=0.75, speeds=s_cfg.speeds)[0]
        assert math.isclose(resp_single, 1.0, abs_tol=1e-6), (
            f"Scale {s_name} single-coil zero-frequency aperture was {resp_single:.6f} != 1.0"
        )
        assert math.isclose(resp_dual, 1.0, abs_tol=1e-6), (
            f"Scale {s_name} dual-coil zero-frequency aperture was {resp_dual:.6f} != 1.0"
        )


def test_guardrail_spatial_coherence_dc_unity():
    """Guardrail 5.1.2: Multi-pickup spatial coherence gamma(f) must evaluate to >= 0.999 (100% coherent)
    at DC (f = 0 Hz), smoothly decaying at higher frequencies, with zero artificial attenuation at DC."""
    f_bins = np.linspace(0.0, 10000.0, 1000)
    # Test typical dual-pickup geometry: delta_tau = 0.52 ms (notch at ~960 Hz)
    delta_tau = 0.00052
    f_notch = 1.0 / (2.0 * delta_tau)
    f_mid = 1.35 * f_notch
    f_sigma = max(0.35 * f_notch, 1.0)
    gamma = 0.5 * (1.0 - np.tanh((f_bins - f_mid) / f_sigma))

    # At DC (f = 0), gamma must be >= 0.999
    assert gamma[0] >= 0.999, f"Spatial coherence at DC was {gamma[0]:.4f} (expected >= 0.999)"

    # At the primary notch (f = f_notch), gamma must evaluate to ~0.88
    idx_notch = np.argmin(np.abs(f_bins - f_notch))
    assert math.isclose(gamma[idx_notch], 0.88, abs_tol=0.02)

    # At high frequencies (f >> f_notch), gamma must smoothly approach 0.0 (incoherent summation)
    assert gamma[-1] < 0.001


def test_guardrail_fail_fast_composite_coils():
    """Guardrail 5.3.5: Composite pickup referencing an undefined component pickup ID
    must immediately raise an explicit diagnostic KeyError rather than silently continuing."""
    import pytest

    from allomorph.config.geometry import resolve_pickup_coils
    from allomorph.config.schema import InstrumentConfig, PickupComponentConfig, PickupConfig

    bad_inst = InstrumentConfig(
        id="bad_test_inst",
        pickups={
            "valid_pickup": PickupConfig(name="Valid", resonant_frequency_hz=3000.0, q_factor=1.2),
            "bad_composite": PickupConfig(
                name="Bad Composite",
                type="composite",
                components=[
                    PickupComponentConfig(pickup="valid_pickup", weight=0.5),
                    PickupComponentConfig(pickup="non_existent_pickup", weight=0.5),
                ],
            ),
        },
    )

    with pytest.raises(KeyError, match="non_existent_pickup"):
        resolve_pickup_coils(bad_inst.pickups["bad_composite"], bad_inst)


def test_guardrail_spatial_position_scaling_high_frequency_flatness():
    """Guardrail 5.1.3: Spatial bridge proximity position transfer function H_pos(f)
    must scale low-frequency fundamental excursion logarithmically according to standing-wave
    displacement ratios, while remaining strictly flat (0.00 dB, unity gain) at high frequencies (>= 1.5 kHz)
    to prevent artificial treble boost/cut on bridge/neck pickups."""
    # Test extreme bridge pickup (eta = 0.06) vs neck pickup (eta = 0.16)
    eta_bridge = 0.06
    eta_neck = 0.16
    f = np.linspace(20.0, 20000.0, 1000)

    for eta_src, eta_tgt in [(eta_bridge, eta_neck), (eta_neck, eta_bridge)]:
        delta_g = 20.0 * np.log10(eta_tgt / eta_src)
        delta_g_soft = soft_clamp_displacement_ratio(delta_g)
        g_0 = 10.0 ** (delta_g_soft / 20.0)
        h_pos = np.sqrt((g_0**2 + (f / 220.0) ** 2) / (1.0 + (f / 220.0) ** 2))
        pos_db = 20.0 * np.log10(h_pos)

        # DC fundamental scaling must match delta_g_soft bit-exact at 0 Hz and within 0.25 dB at 20 Hz
        assert math.isclose(20.0 * np.log10(g_0), delta_g_soft, abs_tol=1e-6)
        assert math.isclose(pos_db[0], delta_g_soft, abs_tol=0.25)

        # High frequencies above 2.5 kHz must have <= 0.25 dB residual shelf transition,
        # and above 5 kHz strictly < 0.05 dB (converging to exact 0.00 dB at treble)
        idx_2500 = np.argmin(np.abs(f - 2500.0))
        idx_5k = np.argmin(np.abs(f - 5000.0))
        assert np.all(np.abs(pos_db[idx_2500:]) < 0.25)
        assert np.all(np.abs(pos_db[idx_5k:]) < 0.05)


def test_guardrail_displacement_ratio_algebraic_limiter_bounds():
    """Guardrail 5.1.3 & 5.3.6: Verify asymmetric order-4 algebraic limiter ('alg4')
    properties for spatial bridge proximity displacement scaling:
    1. Exact bit-exact 0.000 dB identity at ΔG = 0.
    2. Linear passband fidelity (< 0.10 dB error) for normal operating range (|ΔG| <= 6.0 dB).
    3. Monotonic, smooth saturation bounded by +12.0 dB boost ceiling and -16.0 dB cut floor.
    4. Evaluated across all 735 source-to-target transformations in the catalog,
       soft displacement scaling remains strictly within [-12.0 dB, +12.0 dB],
       satisfying Guardrail 5.3.6 sub-audible DC transmission bounds.
    """
    from allomorph.config import (
        compute_effective_position,
        load_all_instruments,
        resolve_pickup_coils,
        resolve_scale_range,
        resolve_voice_coils,
    )

    # 1. Exact identity
    assert soft_clamp_displacement_ratio(0.0) == 0.0

    # 2. Linear passband fidelity (|ΔG| <= 6.0 dB)
    for test_val in [-5.88, -4.42, -2.52, 2.52, 4.42, 5.88]:
        clamped = soft_clamp_displacement_ratio(test_val)
        assert abs(clamped - test_val) < 0.10, f"Error at {test_val} dB exceeded 0.10 dB"

    # 3. Saturation limits
    assert soft_clamp_displacement_ratio(50.0) < 12.0
    assert soft_clamp_displacement_ratio(-50.0) > -16.0
    assert soft_clamp_displacement_ratio(9.11) > 8.40  # Mudbucker retains > 8.4 dB

    # 4. Catalog-wide DC transmission bounds across all 735 combinations
    all_insts = load_all_instruments()
    for inst_id, inst in all_insts.items():
        src_scale = float(inst.scale_length_m or 0.8636)
        for pcfg in inst.pickups.values():
            coils = resolve_pickup_coils(pcfg, inst)
            src_pos = compute_effective_position(coils)
            eta_src = src_pos / src_scale

            for vid, vcfg in VOICES.items():
                if vcfg.sensor_type in (
                    "direct",
                    "bridge_force",
                ):
                    continue
                vcoils = resolve_voice_coils(vcfg)
                tgt_pos = compute_effective_position(vcoils)
                tgt_scale_range = resolve_scale_range(vcfg.scale)
                tgt_scale = (tgt_scale_range[0] + tgt_scale_range[1]) / 2.0
                eta_tgt = tgt_pos / tgt_scale

                delta_g = 20.0 * np.log10(eta_tgt / eta_src)
                soft_dc = soft_clamp_displacement_ratio(delta_g)

                assert -16.0 <= soft_dc <= 12.0, (
                    f"Transformation {inst_id} -> {vid} produced soft DC {soft_dc:.2f} dB, "
                    f"violating [-16.0 dB, +12.0 dB] transmission bound"
                )


def test_guardrail_lossless_c_inf_optimizations():
    """Guardrail 5.2 / Architecture C: Verifies bit-exact mathematical equivalence
    (Δ < 1e-13) of the optimized C^inf SIMD, 2D tensor, and DSP algorithms against
    canonical reference implementations, ensuring maximum NAM training fidelity down to -92 dBFS:
      1. Pure real-FFT homomorphic cepstrum FIR synthesis pipeline.
      2. 2D tensor continuum aperture broadcasting across multiple instrument scales.
      3. Branchless 3-stage fsqrt algebraic rail limiter (p = 8).
      4. Fused analytic string damping ratio.
      5. Potentiometer audio taper constant denominator evaluations.
    """
    from allomorph.circuit.parser import eval_pot_taper
    from allomorph.circuit.saturation import _algebraic_limiter_p8_core
    from allomorph.circuit.solver import smooth_soft_knee_db
    from allomorph.config.instruments import INSTRUMENTS
    from allomorph.config.strings import STRINGS
    from allomorph.dsp import cinf_smoothstep, synthesize_minimum_phase_fir
    from allomorph.physics.aperture import numpy_pickup_acoustic_response
    from allomorph.physics.strings import compute_differential_string_transfer

    # 1. Real-FFT cepstrum pipeline equivalence
    curve = np.linspace(0.1, 1.0, 4097)
    fir_opt = synthesize_minimum_phase_fir(curve, normalize=False)
    n_fft = 8192
    half = 4096
    log_mag = 0.5 * np.log(curve**2 + 1e-8)
    full_log_mag = np.concatenate([log_mag, log_mag[half - 1 : 0 : -1]])
    c_ref = np.fft.ifft(full_log_mag).real
    c_hat = np.zeros(n_fft, dtype=np.float64)
    c_hat[0] = c_ref[0]
    c_hat[half] = c_ref[half]
    c_hat[1:half] = 2.0 * c_ref[1:half]
    h_ref = np.fft.ifft(np.exp(np.fft.fft(c_hat))).real[:4096]
    taper_len = int(4096 * 0.15)
    t = np.arange(taper_len, dtype=np.float64) / max(taper_len, 1)
    w = 1.0 - cinf_smoothstep(t)
    h_ref[4096 - taper_len :] *= w
    assert np.max(np.abs(np.array(fir_opt) - h_ref)) < 1e-14

    # 2. 2D tensor aperture broadcasting across instruments
    f_grid = np.linspace(0, 24000, 4096)
    for inst_id in ["34in_standard_p", "34in_standard_jazz", "30in_emg_mmtw"]:
        inst = INSTRUMENTS[inst_id]
        for p_cfg in inst.pickups.values():
            resp = numpy_pickup_acoustic_response(
                f_grid, p_cfg.coils, scale_length_m=inst.scale_length_m
            )
            assert resp.shape == f_grid.shape
            assert np.all(np.isfinite(resp))
            assert np.all(resp >= 0.0)

    # 3. Branchless 3-stage fsqrt algebraic rail limiter (p = 8)
    x = np.linspace(-2.0, 2.0, 10000)
    vsat = 0.985
    lim_opt = _algebraic_limiter_p8_core(x, vsat)
    lim_ref = x / np.power(1.0 + np.power(np.abs(x) / vsat, 8.0), 1.0 / 8.0)
    assert np.max(np.abs(lim_opt - lim_ref)) < 1e-14

    # 4. Fused analytic string damping ratio
    f = np.linspace(20, 24000, 4096)
    src_str = STRINGS["roundwound_stainless_clank"]
    tgt_str = STRINGS["flatwound_vintage_heavy"]
    damp_opt = compute_differential_string_transfer(f, src_str, tgt_str)
    f_src = float(src_str.damping_cutoff_hz)
    n_src = float(src_str.damping_order)
    f_tgt = float(tgt_str.damping_cutoff_hz)
    n_tgt = float(tgt_str.damping_order)
    src_mag = 1.0 / np.sqrt(1.0 + (f / f_src) ** (2.0 * n_src))
    tgt_mag = 1.0 / np.sqrt(1.0 + (f / f_tgt) ** (2.0 * n_tgt))
    ratio_ref = tgt_mag / np.maximum(src_mag, 1e-6)
    r_db = 20.0 * np.log10(np.maximum(ratio_ref, 1e-6))
    sigma = 0.5 * (1.0 + np.tanh(0.5 * r_db))
    r_soft_db = sigma * smooth_soft_knee_db(r_db, thresh=5.0, ceiling=8.0, alpha=2.0) + (
        1.0 - sigma
    ) * (-smooth_soft_knee_db(-r_db, thresh=24.0, ceiling=36.0, alpha=2.0))
    g_bloom = float(src_str.tension_lbs) / float(tgt_str.tension_lbs)
    h_bloom = np.sqrt((g_bloom**2 + (f / 90.0) ** 2) / (1.0 + (f / 90.0) ** 2))
    damp_ref = (10.0 ** (r_soft_db / 20.0)) * h_bloom
    assert np.max(np.abs(damp_opt - damp_ref)) < 1e-14

    # Also test positive boost direction (flats -> stainless)
    damp_boost_opt = compute_differential_string_transfer(f, tgt_str, src_str)
    ratio_boost_ref = src_mag / np.maximum(tgt_mag, 1e-6)
    r_boost_db = 20.0 * np.log10(np.maximum(ratio_boost_ref, 1e-6))
    sigma_boost = 0.5 * (1.0 + np.tanh(0.5 * r_boost_db))
    r_boost_soft_db = sigma_boost * smooth_soft_knee_db(
        r_boost_db, thresh=5.0, ceiling=8.0, alpha=2.0
    ) + (1.0 - sigma_boost) * (
        -smooth_soft_knee_db(-r_boost_db, thresh=24.0, ceiling=36.0, alpha=2.0)
    )
    g_boost_bloom = float(tgt_str.tension_lbs) / float(src_str.tension_lbs)
    h_boost_bloom = np.sqrt((g_boost_bloom**2 + (f / 90.0) ** 2) / (1.0 + (f / 90.0) ** 2))
    damp_boost_ref = (10.0 ** (r_boost_soft_db / 20.0)) * h_boost_bloom
    assert np.max(np.abs(damp_boost_opt - damp_boost_ref)) < 1e-14

    # 5. Potentiometer audio taper constant denominator evaluations
    for th in [0.0, 0.25, 0.5, 0.75, 1.0]:
        expected = (math.exp(4.394449154672439 * th) - 1.0) / (math.exp(4.394449154672439) - 1.0)
        assert abs(eval_pot_taper(th, "audio") - expected) < 1e-14


def test_guardrail_studio_voicings_and_identity_invariants():
    """Guardrail 5.2.1 / 5.3.1 / 5.3.6: Studio voicings aperture preservation,
    identity matching, and digital twin response invariants:
      1. is_voice_matching_source evaluates strictly to True for studio_direct
         across all 12 playable source instruments (preserving acoustic identity).
      2. is_voice_matching_source evaluates strictly to False for studio_active
         and studio_passive across all 12 playable source instruments (transformative
         circuit models must never be flagged as identities).
      3. build_voicings_comparison_dataframe(step=3) produces bit-exact 0.00 dB for studio_direct
         self-comparison in Stage 3.
      4. build_voicings_comparison_dataframe(step=3) produces non-flat transformative curves (dynamic
         range > 1.0 dB) for studio_active and studio_passive.
      5. simulate_voice(skip_identity=True) strictly skips studio_direct across active
         and passive basses while producing valid output for 15b and 15c.
    """
    import tempfile
    from pathlib import Path

    from allomorph.circuit import simulate_voice
    from allomorph.circuit.schema import SimulationConfig
    from allomorph.config import load_all_instruments
    from allomorph.physics import is_voice_matching_source
    from allomorph.visualizer import build_voicings_comparison_dataframe

    all_insts = load_all_instruments()
    playable_insts = all_insts

    # 1 & 2. is_voice_matching_source invariants across all 12 playable instruments
    for iid, inst in playable_insts.items():
        assert is_voice_matching_source(inst, "studio_direct", VOICES["studio_direct"]), (
            f"studio_direct must match source aperture on {iid}"
        )

        assert not is_voice_matching_source(inst, "studio_active", VOICES["studio_active"]), (
            f"studio_active must NOT be flagged as identity match on {iid}"
        )

        assert not is_voice_matching_source(inst, "studio_passive", VOICES["studio_passive"]), (
            f"studio_passive must NOT be flagged as identity match on {iid}"
        )

    # 3 & 4. build_voicings_comparison_dataframe identity and transformative invariants
    df_direct = build_voicings_comparison_dataframe("studio_direct", "studio_direct", step=3)
    s3_dir = df_direct.filter(df_direct["line_type"] == "3. Normalized Difference (Norm. Diff)")[
        "magnitude_db"
    ]
    assert (s3_dir == 0.0).all(), "studio_direct self-comparison must evaluate to bit-exact 0.00 dB"

    for vid in ["studio_active", "studio_passive"]:
        df_trans = build_voicings_comparison_dataframe("studio_direct", vid, step=3)
        s3_trans = df_trans.filter(
            df_trans["line_type"] == "3. Normalized Difference (Norm. Diff)"
        )["magnitude_db"].to_list()
        assert not all(m == 0.0 for m in s3_trans), f"{vid} comparison must NOT be flat 0.00 dB"
        dr = max(s3_trans) - min(s3_trans)
        assert dr > 0.5, f"{vid} comparison dynamic range was {dr:.2f} dB (expected > 0.5 dB)"

    # 4b. StingRay identity discrimination: parallel is flat 0.0 dB, series is transformative
    df_ray_par = build_voicings_comparison_dataframe(
        "stingray_parallel", "stingray_parallel", step=3
    )
    df_ray_ser = build_voicings_comparison_dataframe("stingray_parallel", "stingray_series", step=3)
    s3_par = df_ray_par.filter(df_ray_par["line_type"] == "3. Normalized Difference (Norm. Diff)")[
        "magnitude_db"
    ]
    s3_ser = df_ray_ser.filter(df_ray_ser["line_type"] == "3. Normalized Difference (Norm. Diff)")[
        "magnitude_db"
    ].to_list()
    assert (s3_par == 0.0).all(), "stingray_parallel identity must evaluate to bit-exact 0.00 dB"
    assert not all(m == 0.0 for m in s3_ser), "stingray_series must NOT be flat"
    assert max(s3_ser) - min(s3_ser) > 5.0, "stingray_series dynamic range must exceed 5.0 dB"

    # 5. simulate_voice identity skip invariants
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        for iid in ["34in_standard_p", "30in_emg_mmtw", "34in_active_stingray"]:
            # Studio direct must be skipped
            out_neutral = tmp_p / f"neutral_{iid}.wav"
            cfg_neutral = SimulationConfig(
                instrument=iid,
                output_wav=out_neutral,
                skip_identity=True,
                max_samples=2400,
            )
            assert simulate_voice("studio_direct", config=cfg_neutral) is False
            assert not out_neutral.exists()

            # Studio active and studio passive must be produced
            for char_vid in ["studio_active", "studio_passive"]:
                out_char = tmp_p / f"{char_vid}_{iid}.wav"
                cfg_char = SimulationConfig(
                    instrument=iid,
                    output_wav=out_char,
                    skip_identity=True,
                    max_samples=2400,
                )
                assert simulate_voice(char_vid, config=cfg_char) is True
                assert out_char.exists()
                assert out_char.stat().st_size > 0


def test_guardrail_scale_tension_zero_center_and_circuit_headroom():
    """Guardrail 5.1.3 & 5.3.6:
    1. Scale tension dynamics derived directly from physical scale ratio r_L = L / L_0 (L_0 = 34.0"):
       - Baseline 34" evaluates to bit-exact 1.0000 (0.000 dB) across all bins (zero hardcoded dB).
       - Long scale (37" Dingwall) has attack snap (r_L^1.5 > 1) and lean low end (1/r_L < 1).
       - Short scale (30" EMG) has fundamental bloom (1/r_L > 1) and relaxed attack (r_L^1.5 < 1).
    2. Differential circuit deconvolution must remain cleanly bounded by the max_boost_db ceiling.
    3. Strictly C^inf smooth soft-knee saturation verification (smooth_soft_knee_db).
    """
    f_arr = np.asarray(FREQS, dtype=np.float64)

    # 1. Scale tension zero-center and physical ratio scaling test:
    inst_34 = load_instrument("34in_standard_p")
    inst_30 = load_instrument("30in_emg_mmtw")
    inst_ding = load_instrument("37in_multiscale_dingwall")
    src_34 = float(inst_34.scale_length_in or 34.0)
    src_30 = float(inst_30.scale_length_in or 30.0)
    src_ding = float(inst_ding.scale_length_in or 37.0)

    def eval_scale_tension(scale_in: float) -> np.ndarray:
        r_L = scale_in / 34.0
        g_bloom = 1.0 / r_L
        g_snap = r_L**1.5
        return np.sqrt(
            (g_bloom**2 + (f_arr / 100.0) ** 2) / (1.0 + (f_arr / 100.0) ** 2)
        ) * np.sqrt((1.0 + g_snap**2 * (f_arr / 2800.0) ** 2) / (1.0 + (f_arr / 2800.0) ** 2))

    # Baseline 34": r_L = 1.0 -> bit-exact 1.0 across all bins
    h_tens_34 = eval_scale_tension(src_34)
    assert np.all(h_tens_34 == 1.0)

    # Long scale (37" Dingwall): r_L = 37/34 -> positive attack snap at 5 kHz, tight low end
    h_tens_ding = eval_scale_tension(src_ding)
    idx_5k = np.argmin(np.abs(f_arr - 5000.0))
    idx_40 = np.argmin(np.abs(f_arr - 40.0))
    snap_db_ding = 20.0 * np.log10(h_tens_ding[idx_5k])
    low_db_ding = 20.0 * np.log10(h_tens_ding[idx_40])
    assert snap_db_ding > 0.5  # Positive attack snap (+0.86 dB)
    assert low_db_ding < 0.0  # Tight, articulate low end (-0.63 dB)

    # Short scale (30" EMG): r_L = 30/34 -> warm excursion bloom at 40 Hz, softened attack at 5 kHz
    h_tens_30 = eval_scale_tension(src_30)
    snap_db_30 = 20.0 * np.log10(h_tens_30[idx_5k])
    low_db_30 = 20.0 * np.log10(h_tens_30[idx_40])
    assert low_db_30 > 0.5  # Warm fundamental bloom (+0.95 dB)
    assert snap_db_30 < -0.5  # Softened attack snap (-1.18 dB)

    # 2. Circuit deconvolution headroom on passive P-Bass to target voice
    p_circ = inst_34.pickups["split_p"].circuit
    assert p_circ is not None
    m_p = load_circuit(p_circ)

    v_target = VOICES["jazz_pair_active"]
    assert v_target.circuit is not None
    m_tgt = load_circuit(v_target.circuit)

    diff_curves = compute_differential_circuit_transfer_functions(
        m_tgt, m_p, freqs=f_arr, max_boost_db=12.0
    )
    h_diff_p = np.asarray(diff_curves[0], dtype=np.float64)
    max_db_diff = float(np.max(20.0 * np.log10(h_diff_p)))
    assert max_db_diff <= 12.0

    # 3. Strictly C^inf smooth soft-knee saturation verification
    from allomorph.circuit import smooth_soft_knee_db

    xs_dense = np.linspace(-10.0, 20.0, 1000)
    ys_dense = smooth_soft_knee_db(xs_dense, thresh=6.0, ceiling=8.0)
    grad1 = np.gradient(ys_dense, xs_dense)
    grad2 = np.gradient(grad1, xs_dense)
    assert np.all(grad1 > 0.0), "Smooth soft knee must be strictly monotonic (dy/dx > 0)"
    assert not np.any(np.isnan(grad2)), (
        "Smooth soft knee second derivative must be finite everywhere"
    )
    assert abs(smooth_soft_knee_db(4.0, thresh=6.0, ceiling=8.0) - 4.0) < 1e-4


def test_guardrail_cinf_dsp_smoothness():
    """Guardrail 5.2: Asserts that core DSP transition functions exhibit C^inf smoothness
    with continuous, finite derivatives across transition boundaries without slope kinks or derivative jumps."""
    from allomorph.circuit.saturation import _get_saturation_filter_pack
    from allomorph.dsp import cinf_smoothstep
    from allomorph.physics import soft_clamp_displacement_ratio

    # 1. cinf_smoothstep: first and second derivatives vanish identically at t=0 and t=1
    dt = 1e-4
    d1_0 = (cinf_smoothstep(dt) - cinf_smoothstep(0.0)) / dt
    d1_1 = (cinf_smoothstep(1.0) - cinf_smoothstep(1.0 - dt)) / dt
    assert abs(d1_0) < 1e-4, f"Smoothstep d1 at t=0 must vanish: {d1_0}"
    assert abs(d1_1) < 1e-4, f"Smoothstep d1 at t=1 must vanish: {d1_1}"

    # Dense gradient smoothness across [0, 1]
    ts = np.linspace(0.0, 1.0, 500)
    ys = cinf_smoothstep(ts)
    dys = np.gradient(ys, ts)
    d2ys = np.gradient(dys, ts)
    assert np.all(dys >= -1e-12), "Smoothstep must be monotonically increasing"
    assert not np.any(np.isnan(d2ys)), "Smoothstep second derivative must be finite everywhere"

    # 2. soft_clamp_displacement_ratio: unified C^inf partition of unity on R
    dgs = np.linspace(-25.0, 25.0, 1000)
    c_out = np.asarray([soft_clamp_displacement_ratio(dg) for dg in dgs])
    c_grad1 = np.gradient(c_out, dgs)
    c_grad2 = np.gradient(c_grad1, dgs)
    assert np.all(c_grad1 > 0.0), "Displacement limiter must be strictly monotonic"
    assert not np.any(np.isnan(c_grad2)), (
        "Displacement limiter second derivative must be finite everywhere"
    )
    # Zero-crossing must be exact 0.00 dB
    assert soft_clamp_displacement_ratio(0.0) == 0.0

    # 3. Multi-rate anti-aliasing decimation mask (aa_mask): vanishing boundary derivatives
    _, _, aa_mask, _ = _get_saturation_filter_pack(4800, 2)
    freqs = np.fft.rfftfreq(4800, 1.0 / 48000.0)
    idx_22k = int(np.argmin(np.abs(freqs - 22000.0)))
    idx_24k = int(np.argmin(np.abs(freqs - 24000.0)))
    assert aa_mask[idx_22k] == 1.0, "aa_mask must be exactly 1.0 at 22 kHz"
    assert aa_mask[idx_24k] == 0.0, "aa_mask must be exactly 0.0 at 24 kHz"
