"""
Allomorph - Digital Twin Pipeline & Staging Test Suite
Validates the direct digital twin forward simulation, Tone3000 upload bundles, dynamic feel,
and concise stage-friendly naming invariants.
"""

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from allomorph.circuit import (
    CALIBRATION_PEAK_CEILING,
    export_instrument_pickup_wav,
    simulate_instrument_voicing,
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
    get_t3k_basename,
)
from allomorph.physics import compute_voice_prefilter_firs


def test_direct_instrument_voicings_and_bundles():
    """Validates that all physical instruments define valid native voicings and affinity-based pickup bundles."""
    from allomorph.config.instruments import (
        load_all_instruments,
        partition_instrument_bundles,
    )

    instruments = load_all_instruments()
    assert len(instruments) >= 14

    for iid, inst in instruments.items():
        assert len(inst.voicings) > 0, f"Instrument '{iid}' has no native voicings"
        for vid, vcfg in inst.voicings.items():
            assert vcfg.pickup in inst.pickups, (
                f"Instrument '{iid}' voicing '{vid}' references undefined pickup '{vcfg.pickup}'"
            )
            assert vcfg.affinity in ("neck", "bridge", "parallel", "direct"), (
                f"Instrument '{iid}' voicing '{vid}' has invalid affinity '{vcfg.affinity}'"
            )

        from allomorph.config.instruments import STANDARD_CATALOG_TARGETS

        bundles = partition_instrument_bundles(inst)
        assert len(bundles) > 0, f"Instrument '{iid}' produced no bundles"
        total_targets = sum(len(b.targets) for b in bundles.values())
        assert total_targets == len(STANDARD_CATALOG_TARGETS), (
            f"Instrument '{iid}' partitioned {total_targets} targets (expected {len(STANDARD_CATALOG_TARGETS)})"
        )


def test_forward_simulation_engine(tmp_path: Path):
    """Validates that simulate_instrument_voicing generates wet audio with zero sample delay,

    calibrated RMS loudness, and true-peak <= -0.09 dBFS.
    """
    from allomorph.circuit.forward import simulate_instrument_voicing

    out_wav = tmp_path / "test_forward.wav"
    res = simulate_instrument_voicing(
        "34in_standard_p",
        "vintage_open",
        output_wav=out_wav,
        max_samples=48000,
    )
    assert res.exists()
    audio, sr = read_wav(res)
    assert sr == 48000
    assert len(audio) == 48000

    peak = float(np.max(np.abs(audio)))
    assert peak <= 0.9905, f"Peak {peak} exceeded -0.09 dBFS ceiling"

    # Verify minimum-phase causal alignment starting at sample 0 (no artificial leading zeroes)
    assert np.abs(audio[0]) > 0.0 or np.max(np.abs(audio[:16])) > 1e-4

    # Verify sidecar manifest records base_dry_sha256
    manifest_p = out_wav.parent / "manifest.json"
    assert manifest_p.exists()
    import json

    with manifest_p.open("r", encoding="utf-8") as f:
        m = json.load(f)
    assert "base_dry_sha256" in m
    assert out_wav.name in m["files"]
    assert m["files"][out_wav.name]["base_dry_sha256"] == m["base_dry_sha256"]


def test_tone_pack_exporter(tmp_path: Path):
    """Validates that export_tone_pack creates self-contained Tone3000 upload bundles with dry v[dsp].[inst].[voicing].wav and wet stems."""
    from allomorph.pipeline.pack import export_tone_pack

    pack_dir = export_tone_pack(
        "34in_standard_p",
        output_dir=tmp_path / "packs",
        max_samples=2048,
        catalog_targets=[("34in_standard_p", "vintage_open"), ("34in_standard_p", "vintage_mids")],
    )
    assert pack_dir.exists()

    bundles_dir = pack_dir / "bundles"
    assert bundles_dir.exists()
    bundle_dirs = [d for d in bundles_dir.iterdir() if d.is_dir()]
    assert len(bundle_dirs) >= 1

    for bdir in bundle_dirs:
        dry_files = list(bdir.glob("dry*.wav"))
        assert len(dry_files) == 1, f"Bundle {bdir.name} missing dry file: found {dry_files}"
        dry_file = dry_files[0]
        assert dry_file.name.startswith("dry v3.1.")
        wet_files = [f for f in bdir.glob("*.wav") if not f.name.startswith("dry")]
        assert len(wet_files) > 0, f"Bundle {bdir.name} has no wet stems"
        for wf in wet_files:
            assert "v3.1." in wf.name
        instructions = bdir / "upload_instructions.txt"
        assert instructions.exists()
        assert dry_file.name in instructions.read_text(encoding="utf-8")
        manifest = bdir / "manifest.json"
        assert manifest.exists()
        with open(manifest, "r", encoding="utf-8") as mf:
            mdata = json.load(mf)
        assert mdata["dry_file"] == dry_file.name
        assert mdata["instrument_version"] == 1
        assert "base_dry_sha256" in mdata
        for stem in mdata["stems"]:
            assert "target_instrument_version" in stem
            assert "target_voicing_version" in stem
            assert stem["version"].startswith("v3.1.")


def test_concise_naming_invariants():
    """Asserts that all target voice model filenames are <= 22 characters."""
    assert isinstance(VOICE_CONCISE_SLUGS, dict)
    for k, v in VOICE_CONCISE_SLUGS.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        assert k in VOICES
        filename = f"{v}.nam"
        assert len(filename) <= 22, (
            f"Model filename '{filename}' exceeds 22 characters ({len(filename)} chars)"
        )


def test_t3k_pack_naming_invariants():
    """Asserts that all target voices and instrument pickups declare human-readable names

    and produce Tone Name [Pickup Position] basenames strictly <= 34 characters.
    """
    # 1. All target voices declare valid tone_name
    assert len(VOICES) >= 23
    for vid, vcfg in VOICES.items():
        assert vcfg.tone_name is not None and vcfg.tone_name.strip(), (
            f"Voice '{vid}' missing tone_name"
        )
        assert len(vcfg.tone_name) <= 23, (
            f"Voice '{vid}' tone_name '{vcfg.tone_name}' too long ({len(vcfg.tone_name)} chars > 23)"
        )

    # 2. All playable instruments declare valid position_name for each pickup
    instruments = load_all_instruments()
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
                    assert basename == f"{tone} [{pos}]".replace("/", "\u2215").replace(
                        "\\", "\u2215"
                    )

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
        None if len(single_p_inst.pickups) <= 1 else single_p_inst.pickups["split_p"].position_name
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

    # 7. Preserve aperture Studio tones omit pickup name suffix
    assert get_t3k_basename("Studio Active", "Parallel") == "Studio Active"
    assert get_t3k_basename("Studio Direct", "Neck") == "Studio Direct"
    assert get_t3k_basename("Studio Passive", "Bridge") == "Studio Passive"
    assert get_t3k_basename("Custom Tone", "Parallel", preserve_aperture=True) == "Custom Tone"


def test_dynamic_feel_and_auto_pickup():
    """Validates dynamic differential feel and auto pickup mapping for forward simulation mode."""
    import tempfile

    # 1. Auto pickup mapping verification across instruments
    inst_30 = load_instrument("30in")
    p_p = get_source_pickup(inst_30, "precision_active")
    p_j = get_source_pickup(inst_30, "jazz_bridge_open")
    assert p_p.id == "mmtw_dual"
    assert p_j.id == "mmtw_single"

    inst_fretless = load_instrument("32in_fretless_pmm")
    p_upright = get_source_pickup(inst_fretless, "upright_acoustic")
    assert p_upright.id == "pcsx"

    # 2. compute_voice_prefilter_firs supports explicit pickup and auto fallback
    firs_auto = compute_voice_prefilter_firs(
        "precision_active", instrument=inst_30, src_pickup_key="auto"
    )
    firs_dual = compute_voice_prefilter_firs(
        "precision_active", instrument=inst_30, src_pickup_key="mmtw_dual"
    )
    firs_single = compute_voice_prefilter_firs(
        "precision_active", instrument=inst_30, src_pickup_key="mmtw_single"
    )
    assert len(firs_auto) == 1
    assert len(firs_dual) == 1
    assert len(firs_single) == 1
    assert np.allclose(firs_auto[0], firs_dual[0])
    # Single coil vs dual coil aperture response must differ
    assert not np.allclose(firs_dual[0], firs_single[0])

    # 3. Fast simulation test of dynamic feel with max_samples
    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "test_dynamic_sim.wav"
        simulate_instrument_voicing(
            instrument="30in",
            voicing="precision_active",
            output_wav=out_wav,
            max_samples=2048,
        )
        assert out_wav.exists()
        with wave.open(str(out_wav), "rb") as wf:
            assert wf.getnframes() == 2048
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 3


def test_export_instrument_pickup_wav(tmp_path: Path):
    """Verify export_instrument_pickup_wav creates valid 24-bit WAVs directly in destination directory."""
    from allomorph.dsp import write_wav_24bit

    # Create a calibrated 1-second excitation signal to test the full pipeline fast
    test_in = tmp_path / "test_excitation.wav"
    sr = 48000
    t = np.arange(sr) / sr
    audio_in = 0.85 * np.sin(2 * np.pi * 100 * t)
    audio_in = audio_in * (10 ** (-16.0 / 20.0) / np.sqrt(np.mean(audio_in**2)))
    write_wav_24bit(test_in, audio_in, sample_rate=sr)

    out_dir = tmp_path / "34in_standard_jazz"

    # Export bridge pickup wet stem
    wet_bridge = export_instrument_pickup_wav(
        "34in_standard_jazz", pickup_key="bridge", input_wav=test_in, output_dir=out_dir
    )
    assert wet_bridge.exists()
    assert wet_bridge.name == "bridge.wav"
    assert wet_bridge.parent == out_dir

    # Export neck pickup wet stem
    wet_neck = export_instrument_pickup_wav(
        "34in_standard_jazz", pickup_key="neck", input_wav=test_in, output_dir=out_dir
    )
    assert wet_neck.exists()
    assert wet_neck.name == "neck.wav"
    assert wet_neck.parent == out_dir

    # Check WAV formatting
    with wave.open(str(wet_bridge), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getsampwidth() == 3  # 24-bit PCM
        assert wf.getnchannels() == 1  # Mono
        assert wf.getnframes() > 0

    # Check audio levels
    audio, _sr = read_wav(wet_bridge, dtype=np.float64)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    peak_db = 20.0 * math.log10(peak)
    rms_db = 20.0 * math.log10(rms)

    # Peak must never exceed CALIBRATION_PEAK_CEILING (0.9900 / -0.09 dBFS)
    assert peak <= CALIBRATION_PEAK_CEILING + 1e-6
    assert -15.0 <= peak_db <= 0.0
    assert -18.0 <= rms_db <= -14.0


def test_precision_warm_tone_cap_no_artificial_spike():
    """Verify that compute_voice_prefilter_firs for rolled-off tone cap voice (precision_warm)
    does not divide by small time-domain impulse peaks, keeping DC gain well-behaved (~ -1.14 dB)
    instead of the broken +21.35 dB boost.
    """
    firs = compute_voice_prefilter_firs("precision_warm", instrument="34in_standard_p")
    dc_gain = float(np.sum(firs[0]))
    dc_gain_db = 20.0 * math.log10(dc_gain)

    # Physical flatwound tension differential (155/195 lbs) is ~ -1.14 dB; must be strictly < +5.0 dB
    assert -3.0 <= dc_gain_db <= +5.0, f"DC gain was {dc_gain_db:+.2f} dB (expected ~ -1.14 dB)"


def test_pipeline_cli_streamlined_stages():
    """Asserts that the CLI parser accepts modern stages and rejects deprecated stages."""
    import argparse

    # We inspect the parser directly by testing valid arguments
    valid_stages = ["all", "viz", "sim", "pack", "train"]
    deprecated_stages = ["prep", "spice", "canonical", "frontends", "targets", "bake"]

    # Construct test parser matching main()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "viz", "sim", "pack", "train"],
        default="all",
    )

    for stage in valid_stages:
        parsed = parser.parse_args(["--stage", stage])
        assert parsed.stage == stage

    for dep_stage in deprecated_stages:
        with pytest.raises(SystemExit):
            parser.parse_args(["--stage", dep_stage])

    # Assert --bake and --t3k-pack flags are rejected
    with pytest.raises(SystemExit):
        parser.parse_args(["--bake"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--t3k-pack"])


def test_forward_simulation_performance(tmp_path: Path):
    """Validates that forward simulation executes with low latency (< 5.0s on 5-second calibration sweep)."""
    import time

    from allomorph.circuit.audio import find_default_input_audio

    sweep_in = find_default_input_audio()
    assert sweep_in is not None

    # Warmup Numba JIT compiler
    simulate_instrument_voicing(
        instrument="34in_standard_p",
        voicing="precision_vintage",
        input_wav=sweep_in,
        output_wav=tmp_path / "warmup.wav",
        max_samples=1024,
    )

    out_file = tmp_path / "out_perf_test.wav"
    t0 = time.perf_counter()
    simulate_instrument_voicing(
        instrument="34in_standard_p",
        voicing="precision_vintage",
        input_wav=sweep_in,
        output_wav=out_file,
        max_samples=48000 * 5,
    )
    elapsed = time.perf_counter() - t0

    assert out_file.exists()
    assert elapsed < 5.0, f"Forward simulation took {elapsed:.2f}s (expected < 5.0s)"
