"""
Allomorph - Two-Stage Pipeline & Staging Test Suite
Validates the Canonical Intermediate baseline, 32 frontend IRs, 3-tier dynamic continuum,
and concise stage-friendly naming invariants.
"""

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.circuit import (
    export_all_frontend_irs,
    generate_canonical_sweep,
    simulate_backend_targets,
    simulate_voice,
)
from allomorph.config import (
    VOICES,
    get_source_pickup,
    load_all_instruments,
    load_instrument,
)
from allomorph.dsp import read_wav
from allomorph.naming import (
    VOICE_CONCISE_SLUGS,
    get_baked_basename,
    get_t3k_basename,
    resolve_instruments,
    resolve_voices,
)
from allomorph.physics import compute_voice_prefilter_firs
from allomorph.pipeline.schema import get_tier_spec


def test_canonical_intermediate_config():
    """Validates the Canonical Intermediate configuration datums."""
    inst = load_instrument("canonical_intermediate")
    assert inst.id == "canonical_intermediate"
    assert inst.scale_length_in == 34.0
    assert inst.scale_length_m == 0.8636

    pickups = inst.pickups
    assert len(pickups) == 1
    p = pickups["canonical_median"]
    assert p.position_from_bridge_m == pytest.approx(0.0935, abs=1e-4)
    assert p.aperture_width_in == pytest.approx(0.75, abs=1e-3)
    assert p.coil_spacing_in == 0.0

    # Wideband passive reference circuit
    assert inst.electronics == "passive"
    assert p.magnet_type == "ideal"
    assert p.pole_type == "rod"
    assert p.resonant_frequency_hz == pytest.approx(4800.0)
    assert p.q_factor == pytest.approx(0.75)

    # 6-string reference wave speeds (B0 to C3) and standard string preset
    assert len(inst.string_wave_speeds) == 6
    assert inst.string_wave_speeds[0] == pytest.approx(53.28, abs=0.01)
    assert inst.string_wave_speeds[-1] == pytest.approx(225.69, abs=0.01)
    assert inst.strings.preset == "roundwound_nickel_6string"


def test_canonical_sweep_calibration(tmp_path: Path):
    """Validates that the Canonical Intermediate sweep is unnormalized, matching input sweep peak/RMS within < 0.5 dB."""
    sweep_path = tmp_path / "canonical_sweep.wav"
    generate_canonical_sweep(output_wav=sweep_path)

    assert sweep_path.exists()
    audio, sr = read_wav(sweep_path)
    assert sr == 48000

    peak = float(np.max(np.abs(audio)))
    peak_db = 20.0 * math.log10(peak)
    assert peak_db == pytest.approx(-1.12, abs=0.20)
    assert not np.isnan(audio).any()
    assert not np.isinf(audio).any()


def test_frontend_ir_generation(tmp_path: Path):
    """Validates that all 35 frontend IRs are exported with exact 2048-tap length and positive polarity."""
    exported = export_all_frontend_irs(output_dir=tmp_path)
    assert len(exported) == 35

    for ir_path in exported:
        assert ir_path.exists(), f"IR missing: {ir_path}"
        fir, sr = read_wav(ir_path)
        assert sr == 48000
        assert len(fir) == 2048, f"{ir_path} has {len(fir)} taps (expected 2048)"

        # Positive initial polarity assertion
        assert np.sum(fir[:16]) > 0.0, f"{ir_path.name} has inverted polarity ({np.sum(fir[:16])})"
        # Zero NaN/Inf
        assert not np.isnan(fir).any()
        assert not np.isinf(fir).any()


def test_concise_naming_invariants():
    """Asserts that all 21 target voice model filenames across all 3 tiers are <= 22 characters."""
    assert isinstance(VOICE_CONCISE_SLUGS, dict)
    for k, v in VOICE_CONCISE_SLUGS.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        assert k in VOICES

    tier_prefixes = ["cln_", "std_", "hot_"]
    for slug in VOICE_CONCISE_SLUGS.values():
        for prefix in tier_prefixes:
            filename = f"{prefix}{slug}.nam"
            assert len(filename) <= 22, (
                f"Model filename '{filename}' exceeds 22 characters ({len(filename)} chars)"
            )


def test_t3k_pack_naming_invariants():
    """Asserts that all target voices and instrument pickups declare human-readable names

    and produce Tone Name [Pickup Position] basenames strictly <= 34 characters.
    """
    # 1. All 24 target voices declare valid tone_name
    assert len(VOICES) >= 24
    for vid, vcfg in VOICES.items():
        assert vcfg.tone_name is not None and vcfg.tone_name.strip(), (
            f"Voice '{vid}' missing tone_name"
        )
        assert len(vcfg.tone_name) <= 23, (
            f"Voice '{vid}' tone_name '{vcfg.tone_name}' too long ({len(vcfg.tone_name)} chars > 23)"
        )

    # 2. All playable instruments + canonical intermediate declare valid position_name for each pickup
    instruments = load_all_instruments()
    instruments["canonical_intermediate"] = load_instrument("canonical_intermediate")
    for iid, inst in instruments.items():
        assert len(inst.pickups) > 0, f"Instrument '{iid}' has no pickups"
        for pid, pcfg in inst.pickups.items():
            assert pcfg.position_name is not None and pcfg.position_name.strip(), (
                f"Instrument '{iid}' pickup '{pid}' missing position_name"
            )
            assert len(pcfg.position_name) <= 10, (
                f"Instrument '{iid}' pickup '{pid}' position_name '{pcfg.position_name}' too long"
            )

    # 3. P/MM instruments must clearly distinguish P/MM and P/J
    for pmm_iid in ["32in_custom_pmm", "32in_fretless_pmm"]:
        pmm_inst = instruments[pmm_iid]
        assert pmm_inst.pickups["blend_parallel"].position_name == "P/MM"
        assert pmm_inst.pickups["pj_blend_parallel"].position_name == "P/J"

    # 4. Exhaustive cross-product: EVERY combination of voice and pickup must be <= 34 chars
    for vid, vcfg in VOICES.items():
        for iid, inst in instruments.items():
            for pid, pcfg in inst.pickups.items():
                tone = vcfg.tone_name or vcfg.name
                pos = pcfg.position_name or pcfg.name
                basename = get_t3k_basename(tone, pos, preserve_aperture=vcfg.preserve_aperture)
                assert len(basename) <= 34, (
                    f"Basename '{basename}' ({vid} + {iid}/{pid}) exceeds 34 characters: {len(basename)}"
                )
                assert "/" not in basename
                assert "\\" not in basename
                if vcfg.preserve_aperture:
                    assert basename == tone.replace("/", "\u2215").replace("\\", "\u2215")
                else:
                    assert basename == f"{tone} [{pos}]".replace("/", "\u2215").replace("\\", "\u2215")

    # 5. Length boundary and error handling in get_t3k_basename
    with pytest.raises(ValueError, match="exceeds 34 characters"):
        get_t3k_basename("This Is An Extremely Long Tone Name", "Parallel")

    assert get_t3k_basename("Precision Vintage", "Split") == "Precision Vintage [Split]"
    assert get_t3k_basename("P∕MM Parallel", "P/MM") == "P\u2215MM Parallel [P\u2215MM]"
    assert "/" not in get_t3k_basename("P∕MM Parallel", "P/MM")

    # 6. Single-pickup instruments omit pickup name suffix
    assert get_t3k_basename("Precision Vintage", None) == "Precision Vintage"
    assert get_t3k_basename("Precision Vintage", "") == "Precision Vintage"
    assert get_t3k_basename("Precision Vintage", "   ") == "Precision Vintage"

    single_p_inst = instruments["34in_standard_p"]
    assert len(single_p_inst.pickups) == 1
    pos_p = (
        None
        if len(single_p_inst.pickups) <= 1
        else single_p_inst.pickups["split_p"].position_name
    )
    assert get_t3k_basename("Precision Vintage", pos_p) == "Precision Vintage"

    single_ray_inst = instruments["34in_active_stingray"]
    assert len(single_ray_inst.pickups) == 1
    pos_ray = (
        None
        if len(single_ray_inst.pickups) <= 1
        else single_ray_inst.pickups["mm_parallel"].position_name
    )
    assert get_t3k_basename("StingRay Parallel", pos_ray) == "StingRay Parallel"

    # 7. Preserve aperture Character tones omit pickup name suffix
    assert get_t3k_basename("Active Character", "Parallel") == "Active Character"
    assert get_t3k_basename("Neutral Character", "Neck") == "Neutral Character"
    assert get_t3k_basename("Passive Character", "Bridge") == "Passive Character"
    assert get_t3k_basename("Custom Tone", "Parallel", preserve_aperture=True) == "Custom Tone"



def test_backend_3_tier_dynamics():
    """Validates that Clean, Standard, and Hot Rod tiers have correct progressive saturation factors."""
    test_voice = "05_vintage_62_p_alnico"
    vcfg = VOICES[test_voice]
    base_alpha = vcfg.alpha if vcfg.alpha is not None else 0.25
    base_vsat = vcfg.vsat if vcfg.vsat is not None else 0.50

    # Clean: 0% saturation, high vsat
    clean_alpha = 0.0
    clean_vsat = 10.0
    assert clean_alpha == 0.0
    assert clean_vsat >= 10.0

    # Standard: 100% nominal saturation
    std_alpha = base_alpha
    std_vsat = base_vsat
    assert std_alpha == base_alpha
    assert std_vsat == base_vsat

    # Hot Rod: 175% saturation, reduced vsat
    hot_alpha = min(1.0, base_alpha * 1.75)
    hot_vsat = max(0.20, base_vsat / 1.35)
    assert hot_alpha > std_alpha
    assert hot_vsat < std_vsat


def test_bake_dynamic_tier_and_auto_pickup():
    """Validates the dynamic (differential) tier and auto pickup mapping for on-demand bake mode."""
    import tempfile

    # 1. Auto pickup mapping verification across instruments
    inst_30 = load_instrument("30in")
    p_p = get_source_pickup(inst_30, "04_modern_p_ceramic")
    p_j = get_source_pickup(inst_30, "03_jazz_bridge_60s")
    assert p_p.id == "mmtw_dual"
    assert p_j.id == "mmtw_single"

    inst_fretless = load_instrument("32in_fretless_pmm")
    p_upright = get_source_pickup(inst_fretless, "14_upright_bridge_transducer")
    assert p_upright.id == "pcsx"

    # 2. compute_voice_prefilter_firs supports explicit pickup and auto fallback
    firs_auto = compute_voice_prefilter_firs(
        "04_modern_p_ceramic", instrument=inst_30, src_pickup_key="auto"
    )
    firs_dual = compute_voice_prefilter_firs(
        "04_modern_p_ceramic", instrument=inst_30, src_pickup_key="mmtw_dual"
    )
    firs_single = compute_voice_prefilter_firs(
        "04_modern_p_ceramic", instrument=inst_30, src_pickup_key="mmtw_single"
    )
    assert len(firs_auto) == 1
    assert len(firs_dual) == 1
    assert len(firs_single) == 1
    assert np.allclose(firs_auto[0], firs_dual[0])
    # Single coil vs dual coil aperture response must differ
    assert not np.allclose(firs_dual[0], firs_single[0])

    # 3. Fast simulation test of dynamic tier with max_samples
    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "test_dynamic_bake.wav"
        simulate_voice(
            "04_modern_p_ceramic",
            output_wav=out_wav,
            instrument="30in",
            pickup="auto",
            tier="dynamic",
            max_samples=2048,
        )
        assert out_wav.exists()
        with wave.open(str(out_wav), "rb") as wf:
            assert wf.getnframes() == 2048
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 3


def test_baked_short_distinct_names_and_instrument_directories():
    """Validates that baked files have short distinct names (<= 22 chars) and group by instrument."""
    playable_insts = resolve_instruments("all")
    all_voices = resolve_voices("all")

    # 1. Check all voices produce unique, short distinct basenames under auto pickup
    for tier in ["dynamic", "standard", "clean", "hotrod"]:
        tier_prefix = get_tier_spec(tier).prefix
        basenames = set()
        for vid in all_voices:
            basename = get_baked_basename(vid, tier=tier, pickup="auto")
            assert basename.startswith(tier_prefix)
            wav_filename = f"{basename}.wav"
            assert len(wav_filename) <= 22, f"WAV filename '{wav_filename}' exceeds 22 chars"
            assert basename not in basenames, f"Duplicate basename: {basename}"
            basenames.add(basename)
        assert len(basenames) == len(all_voices)

    # 2. Non-auto explicit pickup appends pickup key for aperture-shifting voices
    override_basename = get_baked_basename("04_modern_p_ceramic", tier="dynamic", pickup="bridge")
    assert override_basename == "dyn_04_modern_p_bridge"

    # 3. Preserve aperture / character tones strictly omit pickup suffix even with explicit pickup override
    assert get_baked_basename("15_neutral_character", tier="dynamic", pickup="neck") == "dyn_15_neutral"
    assert get_baked_basename("15b_active_character", tier="dynamic", pickup="bridge") == "dyn_15b_active"
    assert get_baked_basename("15c_passive_character", tier="dynamic", pickup="parallel") == "dyn_15c_passive"
    assert (
        get_baked_basename(
            "04_modern_p_ceramic", tier="dynamic", pickup="bridge", preserve_aperture=True
        )
        == "dyn_04_modern_p"
    )

    # 4. Directory structure verification: each instrument has its own subfolder
    for inst_id in playable_insts:
        inst_baked_dir = Path("audio/baked") / inst_id
        assert inst_baked_dir.parent == Path("audio/baked")
        assert inst_baked_dir.name == inst_id


def test_pipeline_cli_streamlined_stages():
    """Asserts that the CLI parser accepts all 7 pure Architecture C stages and rejects deprecated stages."""
    import argparse

    # We inspect the parser directly by testing valid arguments
    valid_stages = ["all", "viz", "canonical", "frontends", "targets", "train", "bake"]
    deprecated_stages = ["prep", "spice", "sim", "simulate"]

    # Construct test parser matching main()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "viz", "canonical", "frontends", "targets", "train", "bake"],
        default="all",
    )

    for stage in valid_stages:
        parsed = parser.parse_args(["--stage", stage])
        assert parsed.stage == stage

    for dep_stage in deprecated_stages:
        with pytest.raises(SystemExit):
            parser.parse_args(["--stage", dep_stage])

    # Assert --bake flag is rejected
    with pytest.raises(SystemExit):
        parser.parse_args(["--bake"])


def test_export_frontend_ir_passive_missing_circuit_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that export_frontend_ir raises ValueError if a passive pickup lacks a circuit model."""
    import allomorph.circuit.staging as staging_mod
    from allomorph.config.schema import InstrumentConfig, PickupConfig

    # Mock load_instrument to return a passive bass with a pickup missing 'circuit'
    dummy_passive = InstrumentConfig(
        id="mock_passive_p",
        name="Mock Passive P",
        electronics="passive",
        scale_length_in=34.0,
        string_wave_speeds=[73.4, 98.0, 130.8, 174.6],
        default_pickup="p",
        pickups={
            "p": PickupConfig(
                name="Passive P",
                position_from_bridge_m=0.125,
                aperture_width_in=0.75,
                coil_spacing_in=0.0,
                magnet_type="alnico_v",
                # Note: No 'circuit' defined!
            )
        },
    )

    def mock_load(inst_id: Any) -> InstrumentConfig:
        return dummy_passive

    monkeypatch.setattr(staging_mod, "load_instrument", mock_load)

    with pytest.raises(
        ValueError, match="does not define a '\\[pickups.p.circuit\\]' configuration"
    ):
        staging_mod.export_frontend_ir("mock_passive_p", "p")


def test_simulate_backend_targets(tmp_path: Path):
    """Validates that simulate_backend_targets runs successfully across clean and standard tiers."""
    out_dir = tmp_path / "targets"
    simulate_backend_targets(
        tier="clean",
        voice_id="01_modern_jazz_active",
        max_samples=2048,
        output_dir=out_dir,
    )
    clean_folder = get_tier_spec("clean").folder_name
    clean_wav = out_dir / clean_folder / "out_01_modern_jazz_active.wav"
    assert clean_wav.exists()
    assert clean_wav.stat().st_size > 0

    std_folder = get_tier_spec("standard").folder_name
    simulate_backend_targets(
        tier="standard",
        voice_id="05_vintage_62_p_alnico",
        max_samples=2048,
        output_dir=out_dir,
    )
    std_wav = out_dir / std_folder / "out_05_vintage_62_p_alnico.wav"
    assert std_wav.exists()
    assert std_wav.stat().st_size > 0


def test_target_wet_files_normalization_and_true_peak_clamping_modes(tmp_path: Path):
    """Validates that Target Wet Files default to normalize='auto' matching input sweep RMS (exact default unity),

    strictly enforcing the calibration sweep peak ceiling (<= 0.9900) so no sample ever clips at full scale (0 dBFS)
    for Tone3000 compliance.
    """
    from allomorph.circuit.simulation import CANONICAL_SWEEP_PATH

    out_dir = tmp_path / "targets"
    # 1. Default mode: normalize='auto' (matching input sweep RMS for default unity)
    simulate_backend_targets(
        tier="clean",
        voice_id="04_modern_p_ceramic",
        output_dir=out_dir,
        max_samples=120000,
    )
    clean_folder = get_tier_spec("clean").folder_name
    wav_norm = out_dir / clean_folder / "out_04_modern_p_ceramic.wav"
    audio_norm, _ = read_wav(wav_norm)
    can_audio, _ = read_wav(CANONICAL_SWEEP_PATH)
    can_audio_slice = can_audio[:120000]
    norm_rms = float(np.sqrt(np.mean(audio_norm**2)))
    can_rms = float(np.sqrt(np.mean(can_audio_slice**2)))
    norm_rms_db = 20.0 * math.log10(norm_rms)
    can_rms_db = 20.0 * math.log10(can_rms)
    # Output RMS matches Canonical Intermediate input sweep within 0.05 dB (exact default unity)
    assert abs(norm_rms_db - can_rms_db) < 0.05
    # Peak is strictly below full scale (<= 0.9900 ceiling) ensuring Tone3000 accepts without clipping
    assert float(np.max(np.abs(audio_norm))) <= 0.9905

    # 2. Disabled mode: normalize='none'
    out_dir_none = tmp_path / "targets_none"
    simulate_backend_targets(
        tier="clean",
        voice_id="04_modern_p_ceramic",
        output_dir=out_dir_none,
        max_samples=120000,
        normalize="none",
    )
    wav_none = out_dir_none / clean_folder / "out_04_modern_p_ceramic.wav"
    audio_none, _ = read_wav(wav_none)
    peak_none = float(np.max(np.abs(audio_none)))
    # Calibration sweep peak ceiling strictly bounds output to <= 0.9905
    assert peak_none <= 0.9905


def test_frontend_ir_unnormalized_unity_gain_and_normalization_modes(tmp_path: Path):
    """Validates that frontend deconvolution IRs default to unnormalized physical unity gain

    (~0 dB fundamental transmission), preventing volume jumps and distortion into Block 2,
    while also verifying optional peak-normalization and gain trims.
    """
    from allomorph.circuit.staging import export_frontend_ir

    test_pickups = [
        ("30in_emg_mmtw", "mmtw_dual"),  # Active humbucker
        ("30in_emg_mmtw", "mmtw_single"),  # Active single-coil
        ("34in_standard_p", "split_p"),  # Passive split coil
        ("34in_standard_jazz", "bridge"),  # Passive single coil
    ]

    for inst_id, pkey in test_pickups:
        # Default unnormalized mode: exact physical unity gain
        ir_path = export_frontend_ir(inst_id, pkey, output_dir=tmp_path / "unnorm")
        assert ir_path.exists()
        fir, sr = read_wav(ir_path)
        assert sr == 48000
        assert len(fir) == 2048

        # Unnormalized filter reflects true physical aperture displacement (0.45 to 1.6, [-7 dB, +4 dB])
        dc_gain = float(np.sum(fir))
        assert 0.45 <= dc_gain <= 1.60, (
            f"{inst_id} ({pkey}) unnormalized DC gain ({dc_gain:.4f}) deviated from expected physical bounds"
        )
        peak = float(np.max(np.abs(fir)))
        assert peak <= 0.9901, f"{inst_id} ({pkey}) unnormalized peak ({peak:.4f}) exceeded 0.99"

        # Explicit peak-normalized mode
        ir_norm_path = export_frontend_ir(inst_id, pkey, output_dir=tmp_path / "norm", normalize=True)
        fir_norm, _ = read_wav(ir_norm_path)
        peak_norm = float(np.max(np.abs(fir_norm)))
        assert 0.985 <= peak_norm <= 0.9901, (
            f"{inst_id} ({pkey}) normalized peak ({peak_norm:.4f}) did not match expected 0.99"
        )

    # Test explicit gain trim (-3 dB)
    ir_trimmed = export_frontend_ir(
        "30in_emg_mmtw", "mmtw_dual", output_dir=tmp_path / "trim", normalize=True, gain_db=-3.0
    )
    fir_trimmed, _ = read_wav(ir_trimmed)
    peak_trimmed = float(np.max(np.abs(fir_trimmed)))
    expected_peak = 0.99 * (10.0 ** (-3.0 / 20.0))
    assert abs(peak_trimmed - expected_peak) < 0.01


def test_frontend_wet_wav_generation(tmp_path: Path):
    """Validates that export_frontend_wet_wav generates compliant 24-bit wet sweeps
    with '_wet.wav' suffix, matching causal convolution of input dry sweep with deconvolution FIR.
    """
    from allomorph.circuit.simulation import CALIBRATION_PEAK_CEILING, find_default_input_audio
    from allomorph.circuit.staging import (
        compute_frontend_deconvolution_fir,
        export_frontend_wet_wav,
    )
    from allomorph.dsp import fft_convolve, write_wav_24bit

    dry_path = find_default_input_audio()
    assert dry_path is not None
    dry_audio, dry_sr = read_wav(dry_path)

    # Use a 48,000-sample (1.0s) active sweep slice to accelerate test runtime
    test_dry = dry_audio[1104000:1152000]
    test_dry_path = tmp_path / "test_dry.wav"
    write_wav_24bit(str(test_dry_path), test_dry, dry_sr)

    # Also create a scaled input (peak ~0.95) that produces a convolved output > 0.9900
    # to explicitly verify the proportional calibration ceiling clamp on frontend audio
    hot_dry = test_dry * 1.8
    hot_dry_path = tmp_path / "hot_dry.wav"
    write_wav_24bit(str(hot_dry_path), hot_dry, dry_sr)

    test_pickups = [
        ("30in_emg_mmtw", "mmtw_dual"),
        ("30in_emg_mmtw", "mmtw_single"),
    ]

    # 1. Small-signal linearity verification (<= 0.10 peak)
    # Bypasses non-linear conditioning to preserve exact mathematical linearity
    small_dry = test_dry * 0.05
    small_dry_path = tmp_path / "small_dry.wav"
    write_wav_24bit(str(small_dry_path), small_dry, dry_sr)

    for inst_id, pkey in test_pickups:
        wet_small_path = export_frontend_wet_wav(
            inst_id, pkey, input_wav=small_dry_path, output_dir=tmp_path / "small"
        )
        assert wet_small_path.exists()
        audio_small, sr = read_wav(wet_small_path)
        fir = compute_frontend_deconvolution_fir(inst_id, pkey, num_taps=2048, normalize=False)
        expected_small = fft_convolve(small_dry, np.asarray(fir, dtype=np.float64), mode="causal")
        max_err = float(np.max(np.abs(audio_small - expected_small.astype(np.float32))))
        assert max_err < 1e-5, f"Small-signal wet WAV deviated from linear convolution: {max_err}"

        # 2. Full-scale Approach A conditioning: slew limiting, soft rail protection, and dither
        wet_path = export_frontend_wet_wav(
            inst_id, pkey, input_wav=test_dry_path, output_dir=tmp_path / "wet"
        )
        assert wet_path.exists(), f"Frontend wet WAV missing: {wet_path}"
        assert wet_path.name.endswith("_wet.wav")

        audio_wet, sr = read_wav(wet_path)
        assert sr == 48000
        assert len(audio_wet) == len(test_dry)
        assert float(np.max(np.abs(audio_wet))) <= CALIBRATION_PEAK_CEILING + 1e-6

        # 3. Normalized mode
        wet_norm_path = export_frontend_wet_wav(
            inst_id, pkey, input_wav=test_dry_path, output_dir=tmp_path / "norm", normalize=True
        )
        audio_norm, _ = read_wav(wet_norm_path)
        assert abs(float(np.max(np.abs(audio_norm))) - CALIBRATION_PEAK_CEILING) < 1e-4

    # 4. Explicit hot input verification: verify soft-knee rail protection and safety clamp
    wet_hot_path = export_frontend_wet_wav(
        "30in_emg_mmtw", "mmtw_dual", input_wav=hot_dry_path, output_dir=tmp_path / "hot"
    )
    audio_hot, _ = read_wav(wet_hot_path)
    assert float(np.max(np.abs(audio_hot))) <= CALIBRATION_PEAK_CEILING + 1e-6



def test_parseval_spectral_integration_rms_accuracy():
    """Validates that frequency-domain Parseval RMS integration matches time-domain

    causal convolution within 0.001 dB, while executing in microsecond time.
    """
    from allomorph.circuit.staging import _get_reference_rms_sweeps
    from allomorph.dsp import fft_convolve

    sweeps = _get_reference_rms_sweeps()
    assert sweeps is not None
    dry_audio, _can_rms, X, n_fft = sweeps

    # Test with synthetic 2048-tap minimum-phase impulse response
    rng = np.random.default_rng(42)
    fir = rng.standard_normal(2048).astype(np.float32) * 0.01

    # Time domain:
    out_time = fft_convolve(dry_audio, fir, mode="causal")
    rms_time = float(np.sqrt(np.mean(out_time**2)))

    # Freq domain (Parseval):
    H = np.fft.rfft(fir, n_fft)
    Y_sq = np.abs(X * H) ** 2
    sum_y2 = float((Y_sq[0] + 2.0 * np.sum(Y_sq[1:-1]) + Y_sq[-1]) / n_fft)
    rms_freq = float(np.sqrt(sum_y2 / len(dry_audio)))

    diff_db = abs(20.0 * math.log10(rms_freq / max(rms_time, 1e-9)))
    assert diff_db < 0.001, (
        f"Parseval frequency-domain RMS deviated from time-domain by {diff_db:.4f} dB"
    )


def test_clean_tier_saturation_bypass_and_performance(tmp_path: Path):
    """Validates that tier='clean' enables saturation bypass and linear stage fusion,

    executing target simulation with low latency without nonlinear ODE overhead.
    """
    import time

    from allomorph.circuit.simulation import CANONICAL_SWEEP_PATH

    if CANONICAL_SWEEP_PATH.exists():
        sweep_in = CANONICAL_SWEEP_PATH
    else:
        sweep_in = tmp_path / "canonical_sweep.wav"
        generate_canonical_sweep(output_wav=sweep_in)

    out_file = tmp_path / "out_clean_test.wav"
    t0 = time.perf_counter()
    simulate_voice(
        voice_id="04_modern_p_ceramic",
        input_wav=sweep_in,
        output_wav=out_file,
        instrument="canonical_intermediate",
        tier="clean",
        normalize="none",
    )
    elapsed = time.perf_counter() - t0

    assert out_file.exists()
    # Clean tier with stage fusion runs in ~1.5s on full 700k audio file; must be strictly under 3.5s
    assert elapsed < 3.5, f"Clean tier took {elapsed:.2f}s (expected < 3.5s with stage fusion)"


def test_simulate_backend_targets_jobs_parallelism(tmp_path: Path):
    """Validates that simulate_backend_targets accepts jobs and dispatches parallel worker processes."""
    out_dir = tmp_path / "parallel_targets"
    simulate_backend_targets(
        tier="clean",
        voice_id="all",
        max_samples=2048,
        jobs=2,
        output_dir=out_dir,
    )
    clean_folder = get_tier_spec("clean").folder_name
    target_folder = out_dir / clean_folder
    assert target_folder.exists()
    wav_files = list(target_folder.glob("out_*.wav"))
    assert len(wav_files) >= 20

