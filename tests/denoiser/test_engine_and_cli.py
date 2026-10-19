"""Integration tests for `denoiser` naming, WaveNet `.nam` loading, matrix trainer, evaluator, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from allomorph.dsp.constants import FS
from allomorph.dsp.io import read_wav, write_wav_24bit
from allomorph.trainer.native.models import AllomorphWaveNetA2
from denoiser.cli import apply_cmd, evaluate_cmd, train_cmd
from denoiser.constants import (
    resolve_denoiser_display_name,
    resolve_denoiser_gear_model,
    resolve_denoiser_model_stem,
    resolve_mains_targets,
    resolve_profile_targets,
)


def test_denoiser_naming_and_target_resolvers() -> None:
    """Denoiser model filenames and display names must encode mains and profile without redundant prefixes."""
    assert resolve_mains_targets("both") == ("both",)
    assert resolve_mains_targets("60Hz") == ("60",)
    assert resolve_mains_targets("50") == ("50",)
    assert resolve_mains_targets("all") == ("both", "60", "50")

    assert resolve_profile_targets("balanced") == ("balanced",)
    assert resolve_profile_targets("all") == ("gentle", "balanced", "aggressive")

    assert resolve_denoiser_model_stem("both", "balanced") == "universal_balanced"
    assert resolve_denoiser_model_stem("60", "aggressive") == "60hz_aggressive"
    assert resolve_denoiser_model_stem("50", "gentle") == "50hz_gentle"

    assert resolve_denoiser_display_name("both", "balanced") == "Universal Balanced"
    assert resolve_denoiser_display_name("60", "aggressive") == "60Hz Aggressive"
    assert resolve_denoiser_display_name("50", "gentle") == "50Hz Gentle"
    assert (
        resolve_denoiser_gear_model("both", "balanced")
        == "Universal Bass Denoiser (Universal Balanced)"
    )

    with pytest.raises(ValueError, match="Invalid mains mode"):
        resolve_mains_targets("400")
    with pytest.raises(ValueError, match="Invalid denoiser profile"):
        resolve_profile_targets("extreme")


def test_wavenet_a2_from_nam_file_roundtrip(tmp_path: Path) -> None:
    """AllomorphWaveNetA2.from_nam_file must reconstruct exact weights for both Ch3 Lite and Ch8 Full submodels."""
    torch.manual_seed(123)
    model_orig = AllomorphWaveNetA2()
    model_orig.mask_weights()
    model_orig.eval()

    nam_file = tmp_path / "roundtrip.nam"
    model_orig.export_nam(nam_file, sample_rate=float(FS), metadata={"name": "Universal Balanced"})

    model_loaded, meta, sr_loaded = AllomorphWaveNetA2.from_nam_file(nam_file, device="cpu")
    assert meta["name"] == "Universal Balanced"
    assert sr_loaded == FS

    x = torch.randn(1, 1, 6346 + 512, dtype=torch.float32) * 0.2
    with torch.no_grad():
        y_orig = model_orig(x)
        y_loaded = model_loaded(x)

    assert torch.allclose(y_orig, y_loaded, atol=1e-6)


def test_denoiser_cli_train_default_profiles_evaluate_and_apply(tmp_path: Path) -> None:
    """`denoiser train` with default `--mains both --profile all` must export all 3 universal models, sidecars, and storefront_description.txt."""
    sr = FS
    n_samples = (
        48_000  # 1.0s raw input -> >100k samples after multi-voice + probes corpus expansion
    )
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    dry = (0.30 * np.sin(2.0 * np.pi * 82.41 * t)).astype(np.float32)

    in_wav = tmp_path / "input.wav"
    out_dir = tmp_path / "models" / "denoiser"
    write_wav_24bit(in_wav, dry, sample_rate=sr)

    train_cmd(
        mains="both",
        profile="all",
        input_file=in_wav,
        output_dir=out_dir,
        epochs=1,
        min_epochs=1,
        batch_size=2,
        device="cpu",
        fast_dev_run=True,
    )

    expected_models = [
        ("universal_gentle.nam", "Universal Gentle", -50.0),
        ("universal_balanced.nam", "Universal Balanced", -44.0),
        ("universal_aggressive.nam", "Universal Aggressive", -36.0),
    ]
    for filename, exp_name, exp_thresh in expected_models:
        nam_path = out_dir / filename
        sidecar_path = nam_path.with_suffix(".json")
        assert nam_path.is_file(), f"Missing {nam_path}"
        assert sidecar_path.is_file(), f"Missing {sidecar_path}"

        nam_data = json.loads(nam_path.read_text(encoding="utf-8"))
        assert nam_data["architecture"] == "SlimmableContainer"
        assert nam_data["metadata"]["name"] == exp_name
        assert nam_data["metadata"]["gear_type"] == "pedal"
        assert nam_data["metadata"]["tone_type"] == "clean"
        assert nam_data["metadata"]["threshold_dbfs"] == exp_thresh

        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar_data["nam_file"] == filename
        assert sidecar_data["name"] == exp_name

    storefront_txt = out_dir / "storefront_description.txt"
    assert storefront_txt.is_file()
    desc_content = storefront_txt.read_text(encoding="utf-8")
    assert 100 < len(desc_content) <= 10000
    assert "Universal Balanced" in desc_content

    # Test evaluate_cmd and apply_cmd on universal_balanced.nam
    balanced_nam = out_dir / "universal_balanced.nam"
    evaluate_cmd(
        model_path=balanced_nam,
        mains="both",
        duration=0.25,
        device="cpu",
    )

    noisy_in = tmp_path / "noisy_bass.wav"
    denoised_out = tmp_path / "denoised_bass.wav"
    write_wav_24bit(noisy_in, dry[:12000], sample_rate=sr)
    apply_cmd(
        input_wav=noisy_in,
        output_wav=denoised_out,
        model_path=balanced_nam,
        device="cpu",
    )
    assert denoised_out.is_file()
    out_audio, out_sr = read_wav(denoised_out)
    assert out_sr == sr
    assert len(out_audio) == 12000
