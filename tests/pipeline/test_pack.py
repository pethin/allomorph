"""
Tests for Tone3000 Tone Pack Pipeline in allomorph.pipeline.pack.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.config.instruments import load_instrument, partition_instrument_bundles
from allomorph.dsp import write_wav_24bit
from allomorph.pipeline.pack import (
    _sha256_file,
    export_tone_pack,
    generate_storefront_description,
)


@pytest.fixture
def mini_dry_audio(tmp_path: Path) -> Path:
    """Creates a short, valid 24-bit 48kHz dry audio file for testing pack export."""
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)  # 50 ms
    # Multi-frequency test excitation
    audio = 0.3 * np.sin(2.0 * np.pi * 100.0 * t) + 0.1 * np.sin(2.0 * np.pi * 1000.0 * t)
    dry_path = tmp_path / "test_input.wav"
    write_wav_24bit(dry_path, audio.astype(np.float32), sample_rate=sr)
    return dry_path


def test_sha256_file(tmp_path: Path):
    """Verify _sha256_file computes valid SHA256 hex digest."""
    p = tmp_path / "hello.txt"
    p.write_bytes(b"Allomorph Tone Pack SHA test")
    digest = _sha256_file(p)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_generate_storefront_description():
    """Verify storefront description generation contains all required sections."""
    inst = load_instrument("34in_standard_p")
    bundles = partition_instrument_bundles(inst)
    desc = generate_storefront_description(inst, bundles)

    assert "ALLOMORPH" in desc
    assert "RECOMMENDED SIGNAL CHAIN" in desc
    assert "QUICK INSTRUMENT SETUP" in desc
    assert "LICENSE & DISCLAIMER" in desc
    assert inst.name in desc


def test_export_tone_pack_minimal(tmp_path: Path, mini_dry_audio: Path):
    """Verify exporting a tone pack with bounded samples creates valid bundle layout."""
    inst = load_instrument("30in_emg_mmtw")
    out_dir = tmp_path / "test_pack"

    # Restrict to two targets for high-speed unit testing
    targets = [
        ("34in_standard_p", "vintage_open"),
        ("34in_standard_jazz", "bridge_growl"),
    ]

    pack_dir = export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
        catalog_targets=targets,
        overwrite=True,
    )

    assert pack_dir == out_dir
    assert (pack_dir / "manifest.json").exists()
    assert (pack_dir / "storefront_description.txt").exists()

    bundles_dir = pack_dir / "bundles"
    assert bundles_dir.exists()
    bundle_folders = list(bundles_dir.iterdir())
    assert len(bundle_folders) >= 1

    for b_dir in bundle_folders:
        if b_dir.is_dir():
            assert (b_dir / "manifest.json").exists()
            assert (b_dir / "upload_instructions.txt").exists()
            dry_wavs = list(b_dir.glob("dry *.wav"))
            assert len(dry_wavs) == 1
            wet_wavs = list(b_dir.glob("*.wav"))
            # Should have dry file + 2 stems
            assert len(wet_wavs) >= 2


def test_export_tone_pack_existing_skip_and_repack(tmp_path: Path, mini_dry_audio: Path):
    """Verify existing pack returns early when overwrite=False, and repacks when overwrite=True."""
    inst = load_instrument("30in_emg_mmtw")
    out_dir = tmp_path / "test_pack_repack"
    targets = [("34in_standard_p", "vintage_open")]

    # Initial export
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
        catalog_targets=targets,
        overwrite=False,
    )
    manifest_p = out_dir / "manifest.json"
    mtime1 = manifest_p.stat().st_mtime_ns

    # Call again with overwrite=False -> early return, mtime untouched
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
        catalog_targets=targets,
        overwrite=False,
    )
    mtime2 = manifest_p.stat().st_mtime_ns
    assert mtime1 == mtime2

    # Call with overwrite=True -> forces repack
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
        catalog_targets=targets,
        overwrite=True,
    )
    assert (out_dir / "manifest.json").exists()


def test_generate_storefront_description_passive_multi_pickup():
    """Verify storefront description for a passive multi-pickup bass formats Master Tone/Vol instructions."""
    inst = load_instrument("34in_standard_jazz")
    bundles = partition_instrument_bundles(inst)
    desc = generate_storefront_description(inst, bundles)

    assert "Master Tone Knob: 100%" in desc
    assert "Physical Volume Knob: 100%" in desc


def test_export_tone_pack_parallel_bundles_and_cached_stems(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify parallel bundle processing with jobs > 1 and cached stem compilation."""
    from allomorph.version import write_manifest

    wet_dir = tmp_path / "wet"
    monkeypatch.setattr("allomorph.pipeline.pack.WET_AUDIO_DIR", wet_dir)

    # Mock simulate_instrument_voicing to quickly write a valid WAV file and manifest
    def _mock_sim_voicing(
        *args: Any,
        voicing: Any = None,
        output_wav: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        if output_wav is not None:
            p = Path(output_wav)
            p.parent.mkdir(parents=True, exist_ok=True)
            # Source instrument dry stem gets 0.35, target wet stems get 0.85
            is_dry = (
                "dry" in p.name
                or "30in" in str(p)
                or str(voicing) in ("dual", "single", "mmtw_dual", "mmtw_single")
            )
            val = 0.35 if is_dry else 0.85
            samples = np.array([0.0, val, -val], dtype=np.float32)
            write_wav_24bit(p, samples, sample_rate=48000)
            write_manifest(
                p.parent,
                "sim",
                [p],
                version_tag="v5.1.1",
                base_dry_file=mini_dry_audio.name,
                base_dry_sha256=_sha256_file(mini_dry_audio),
            )

    monkeypatch.setattr("allomorph.pipeline.pack.simulate_instrument_voicing", _mock_sim_voicing)

    inst = load_instrument("30in_emg_mmtw")
    out_dir = tmp_path / "test_parallel_pack"
    targets = [("34in_standard_p", "vintage_open")]

    pack_dir = export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=None,
        jobs=2,
        catalog_targets=targets,
        overwrite=True,
    )

    assert (pack_dir / "manifest.json").exists()
    bundles_dir = pack_dir / "bundles"
    assert bundles_dir.exists()
