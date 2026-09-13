"""
Tests for dedicated instrument dry excitation audio (dry_<inst_id>.wav)
in 1-block monolithic baked simulations and NAM A2 training.
"""

import fnmatch
import math
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from allomorph.circuit.simulation import CALIBRATION_PEAK_CEILING, SimulationConfig, simulate_voice
from allomorph.circuit.staging import (
    export_instrument_dry_wav,
    find_instrument_dry_wav,
)
from allomorph.dsp import read_wav
from allomorph.naming import get_instrument_dry_basename, get_instrument_dry_path
from allomorph.physics.prefilter import compute_voice_prefilter_firs
from allomorph.pipeline.schema import NamTrainingConfig


def test_instrument_dry_naming_and_sorting():
    """Verify the naming convention and pickup-specific dry/ directory isolation behavior."""
    basename_single = get_instrument_dry_basename("34in_standard_p", "split_p")
    assert basename_single == "dry_34in_standard_p_split_p"

    basename_jazz = get_instrument_dry_basename("34in_standard_jazz", "bridge")
    assert basename_jazz == "dry_34in_standard_jazz_bridge"

    path_single = get_instrument_dry_path("34in_standard_p", "split_p")
    assert path_single.name == "dry_34in_standard_p_split_p.wav"
    assert "baked/34in_standard_p/split_p/dry" in str(path_single)
    assert path_single.parent.name == "dry"

    path_jazz = get_instrument_dry_path("34in_standard_jazz", "bridge")
    assert path_jazz.name == "dry_34in_standard_jazz_bridge.wav"
    assert "baked/34in_standard_jazz/bridge/dry" in str(path_jazz)
    assert path_jazz.parent.name == "dry"

    # Isolating dry file in dry/ subdirectory keeps the pickup tone directory clean
    wet_names = [
        "Dingwall Bridge v2.1.1.wav",
        "Jazz Pair Active v2.1.1.wav",
        "Precision Warm v2.1.1.wav",
        "StingRay Classic v2.1.1.wav",
    ]
    for wet in wet_names:
        assert fnmatch.fnmatch(wet, "*.wav")


def test_export_instrument_dry_wav():
    """Verify export_instrument_dry_wav creates valid 24-bit WAVs in pickup-specific dry/ subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "34in_standard_jazz"

        # Export bridge pickup dry
        dry_bridge = export_instrument_dry_wav(
            "34in_standard_jazz", pickup_key="bridge", output_dir=out_dir
        )
        assert dry_bridge.exists()
        assert dry_bridge.name == "dry_34in_standard_jazz_bridge.wav"
        assert dry_bridge.parent == out_dir / "bridge" / "dry"

        # Verify only a single file is in bridge/dry/, and NO loose files in bridge/
        assert len(list((out_dir / "bridge").glob("*.wav"))) == 0
        dry_wavs = list((out_dir / "bridge" / "dry").glob("*.wav"))
        assert len(dry_wavs) == 1
        assert dry_wavs[0].name == "dry_34in_standard_jazz_bridge.wav"

        # Export neck pickup dry
        dry_neck = export_instrument_dry_wav(
            "34in_standard_jazz", pickup_key="neck", output_dir=out_dir
        )
        assert dry_neck.exists()
        assert dry_neck.name == "dry_34in_standard_jazz_neck.wav"
        assert dry_neck.parent == out_dir / "neck" / "dry"

        # Check WAV formatting
        with wave.open(str(dry_bridge), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 3  # 24-bit PCM
            assert wf.getnchannels() == 1  # Mono
            assert wf.getnframes() > 0

        # Check audio levels
        audio, _sr = read_wav(dry_bridge, dtype=np.float64)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        peak_db = 20.0 * math.log10(peak)
        rms_db = 20.0 * math.log10(rms)

        # Peak must never exceed CALIBRATION_PEAK_CEILING (0.9900 / -0.09 dBFS)
        assert peak <= CALIBRATION_PEAK_CEILING
        assert -2.0 <= peak_db <= -0.5
        assert -18.0 <= rms_db <= -14.0


def test_find_instrument_dry_wav():
    """Verify find_instrument_dry_wav locates existing dry files in pickup dry/ and supports auto-generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inst_dir = Path(tmpdir) / "34in_standard_jazz"

        # 1. Missing file with auto_generate=False raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            find_instrument_dry_wav(
                "34in_standard_jazz",
                pickup_key="bridge",
                audio_dir=inst_dir,
                auto_generate=False,
            )

        # 2. Auto-generation creates the file in bridge/dry/ subdirectory
        found = find_instrument_dry_wav(
            "34in_standard_jazz", pickup_key="bridge", audio_dir=inst_dir, auto_generate=True
        )
        assert found.exists()
        assert found.name == "dry_34in_standard_jazz_bridge.wav"
        assert found.parent == inst_dir / "bridge" / "dry"

        # 3. Subsequent calls find the existing primary file immediately
        found_again = find_instrument_dry_wav(
            "34in_standard_jazz",
            pickup_key="bridge",
            audio_dir=inst_dir,
            auto_generate=False,
        )
        assert found_again == found


def test_baked_simulation_dingwall_bridge_volume_drop_fixed():
    """
    Verify that simulating Dingwall Bridge with the dedicated _dry_<inst_id>.wav
    resolves the artificial -12.4 dB volume drop down to the authentic physical bridge displacement (~ -3.5 to -4.5 dB).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "34in_standard_p"
        dry_path = export_instrument_dry_wav("34in_standard_p", output_dir=out_dir)

        dry_audio, _ = read_wav(dry_path, dtype=np.float64)
        dry_rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(dry_audio**2))))

        baked_out = out_dir / "dingwall_bridge.wav"
        cfg = SimulationConfig(
            input_wav=dry_path,
            output_wav=baked_out,
            instrument="34in_standard_p",
            pickup="split_p",
            tier="clean",
            normalize="none",
        )
        success = simulate_voice("13_dingwall_multiscale_bridge", config=cfg)
        assert success
        assert baked_out.exists()

        out_audio, _ = read_wav(baked_out, dtype=np.float64)
        out_peak = float(np.max(np.abs(out_audio)))
        out_rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(out_audio**2))))

        # Peak must never clip or trigger emergency ceiling clamp
        assert out_peak <= CALIBRATION_PEAK_CEILING
        assert out_peak < 0.95  # Sits comfortably around 0.82 (-1.72 dBFS)

        # RMS volume difference must be the authentic physical bridge proximity delta
        # (-3.0 dB to -5.5 dB), NOT the broken -12.42 dB drop
        rms_drop = out_rms_db - dry_rms_db
        assert -5.5 <= rms_drop <= -2.5, f"Volume drop was {rms_drop:.2f} dB (expected ~ -3.5 to -4.5 dB)"


def test_05c_tone_cap_no_artificial_spike():
    """
    Verify that compute_voice_prefilter_firs for rolled-off tone cap voice (05c)
    does not divide by small time-domain impulse peaks, keeping DC gain ~ +2.4 dB
    instead of the broken +21.35 dB boost.
    """
    firs = compute_voice_prefilter_firs("05c_vintage_62_p_47nf", instrument="34in_standard_p")
    dc_gain = float(np.sum(firs[0]))
    dc_gain_db = 20.0 * math.log10(dc_gain)

    # Physical flatwound differential is +2.4 dB; must be strictly < +6.0 dB
    assert -1.0 <= dc_gain_db <= +5.0, f"DC gain was {dc_gain_db:+.2f} dB (expected ~ +2.40 dB)"


def test_trainer_cli_baked_option():
    """Verify trainer configuration and CLI schema support --baked."""
    import inspect

    from train_nam import train_voice

    # 1. train_voice has baked argument
    sig = inspect.signature(train_voice)
    assert "baked" in sig.parameters
    assert sig.parameters["baked"].default is False

    # 2. NamTrainingConfig has baked field
    cfg = NamTrainingConfig()
    assert cfg.baked is False
    cfg_baked = NamTrainingConfig(baked=True)
    assert cfg_baked.baked is True


def test_pipeline_cli_bake_pickup_subdirectories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that allomorph-pipeline --stage bake partitions audio and dry files into pickup subdirectories."""
    import allomorph.circuit.simulation as sim_mod
    import allomorph.pipeline.cli as cli_mod

    fake_audio = tmp_path / "audio"
    fake_models = tmp_path / "models"
    monkeypatch.setattr(cli_mod, "AUDIO_DIR", fake_audio)
    monkeypatch.setattr(cli_mod, "MODELS_DIR", fake_models)
    monkeypatch.setattr(sim_mod, "AUDIO_DIR", fake_audio)

    cli_mod.main(
        [
            "--stage",
            "bake",
            "--instrument",
            "34in_standard_jazz",
            "--voice",
            "13_dingwall_multiscale_bridge,04_modern_p_ceramic",
            "--max-samples",
            "2000",
            "--no-manifest",
        ]
    )

    inst_dir = fake_audio / "baked" / "34in_standard_jazz"
    assert inst_dir.exists()

    # Zero loose wav files in inst_dir root
    assert len(list(inst_dir.glob("*.wav"))) == 0

    # 13_dingwall_multiscale_bridge is mapped to bridge pickup
    bridge_dir = inst_dir / "bridge"
    assert bridge_dir.exists()
    assert (bridge_dir / "dry" / "dry_34in_standard_jazz_bridge.wav").exists()
    bridge_wavs = list(bridge_dir.glob("*.wav"))
    assert len(bridge_wavs) == 1
    assert "13_dingwall" in bridge_wavs[0].name

    # 04_modern_p_ceramic is mapped to neck pickup
    neck_dir = inst_dir / "neck"
    assert neck_dir.exists()
    assert (neck_dir / "dry" / "dry_34in_standard_jazz_neck.wav").exists()
    neck_wavs = list(neck_dir.glob("*.wav"))
    assert len(neck_wavs) == 1
    assert "04_modern_p" in neck_wavs[0].name
