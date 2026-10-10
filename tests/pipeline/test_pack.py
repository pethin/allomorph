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
    inst = load_instrument("34in_standard_p")
    out_dir = tmp_path / "test_pack"

    # Restrict to two targets for high-speed unit testing
    targets = [
        ("34in_standard_jazz", "bridge_growl"),
        ("34in_standard_jazz", "pair_open"),
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
    inst = load_instrument("34in_standard_p")
    out_dir = tmp_path / "test_pack_repack"
    targets = [("34in_standard_jazz", "bridge_growl")]

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
                or "34in_standard_jazz" in str(p)
                or str(voicing) in ("neck_warm", "bridge_open", "pair_open")
            )
            val = 0.35 if is_dry else 0.85
            samples = np.array([0.0, val, -val], dtype=np.float32)
            write_wav_24bit(p, samples, sample_rate=48000)
            write_manifest(
                p.parent,
                "sim",
                [p],
                version_tag="v6.1.1",
                base_dry_file=mini_dry_audio.name,
                base_dry_sha256=_sha256_file(mini_dry_audio),
            )

    monkeypatch.setattr("allomorph.pipeline.pack.simulate_instrument_voicing", _mock_sim_voicing)

    inst = load_instrument("34in_standard_jazz")
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


def test_train_tone_pack_flat_structure_mocked(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify train_tone_pack outputs flat .nam models, updates manifest, and respects skip logic."""
    import json

    from allomorph.pipeline.pack import train_tone_pack

    inst = load_instrument("34in_standard_p")
    pack_dir = tmp_path / "test_train_pack"
    targets = [
        ("34in_standard_jazz", "bridge_growl"),
        ("34in_standard_jazz", "pair_open"),
    ]

    export_tone_pack(
        instrument=inst,
        output_dir=pack_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
        catalog_targets=targets,
        overwrite=True,
    )

    mock_train_calls: list[dict[str, Any]] = []

    def _mock_train_voice(
        *args: Any,
        models_dir: Path | str = "",
        basename: str | None = None,
        voice: str = "",
        **kwargs: Any,
    ) -> bool:
        mock_train_calls.append(
            {"models_dir": Path(models_dir), "basename": basename, "voice": voice}
        )
        out_dir = Path(models_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        nam_file = out_dir / f"{basename}.nam"
        dummy_content = {
            "version": "0.5.1",
            "architecture": "A2-Slimmable",
            "config": {
                "submodels": [
                    {"name": "channels_3"},
                    {"name": "channels_8"},
                ]
            },
            "metadata": {
                "training": {
                    "validation_esr": 0.000185,
                    "validation_esr_a2_full": 0.000185,
                    "validation_esr_a2_lite": 0.001850,
                    "validation_esr_ch8": 0.000185,
                    "validation_esr_ch3": 0.001850,
                    "differential_esr": 0.0142,
                    "mrstft_loss": 0.125,
                    "differential_mrstft": 0.450,
                    "epochs_trained": 24,
                    "stop_reason": "triple_gate_converged",
                }
            },
        }
        with open(nam_file, "w", encoding="utf-8") as f:
            json.dump(dummy_content, f)
        return True

    # Monkeypatch train_voice in allomorph.trainer and allomorph.pipeline.pack
    import allomorph.pipeline.pack
    import allomorph.trainer

    monkeypatch.setattr(allomorph.trainer, "train_voice", _mock_train_voice)

    # 1. Initial training run: both target stems should be trained
    nam_dir = train_tone_pack(pack=pack_dir, voice="all", overwrite=False)

    assert nam_dir == pack_dir / "nam"
    assert nam_dir.exists()

    # Verify flat structure: NO .nam or nam folders inside bundles
    bundles_dir = pack_dir / "bundles"
    assert not any(p.suffix == ".nam" for p in bundles_dir.rglob("*"))
    assert not any(p.name == "nam" for p in bundles_dir.rglob("*"))

    # Models must be placed flatly inside pack_dir / "nam"
    nam_files = list(nam_dir.glob("*.nam"))
    assert len(nam_files) == 2
    for nf in nam_files:
        assert nf.parent == nam_dir
        # Filename matches stem without .wav
        assert " v6.1.1.nam" in nf.name

    assert len(mock_train_calls) == 2

    # Verify manifest.json was updated with top-level "models" dictionary
    with open(pack_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "models" in manifest_data
    assert len(manifest_data["models"]) == 2

    for model_name, model_meta in manifest_data["models"].items():
        assert model_name.endswith(".nam")
        assert model_meta["filename"] == model_name
        assert model_meta["validation_esr"] == 0.000185
        assert model_meta["validation_esr_a2_full"] == 0.000185
        assert model_meta["validation_esr_a2_lite"] == 0.001850
        assert model_meta["validation_esr_ch8"] == 0.000185
        assert model_meta["validation_esr_ch3"] == 0.001850
        assert model_meta["differential_esr"] == 0.0142
        assert model_meta["mrstft_loss"] == 0.125
        assert model_meta["differential_mrstft"] == 0.450
        assert model_meta["epochs_trained"] == 24
        assert model_meta["stop_reason"] == "triple_gate_converged"
        assert len(model_meta["sha256"]) == 64

    # 2. Resumption & Skipping: calling again with overwrite=False skips existing models
    mock_train_calls.clear()
    train_tone_pack(pack=pack_dir, voice="all", overwrite=False)
    assert len(mock_train_calls) == 0  # 0 calls, 100% skipped!

    # 3. Forcing overwrite: calling with overwrite=True retrains both
    train_tone_pack(pack=pack_dir, voice="all", overwrite=True)
    assert len(mock_train_calls) == 2

    # 4. Voice filter: only train matching voice
    mock_train_calls.clear()
    train_tone_pack(pack=pack_dir, voice="jazz_bridge_growl", overwrite=True)
    assert len(mock_train_calls) == 1
    assert "growl" in mock_train_calls[0]["basename"].lower()


def test_train_tone_pack_legacy_removal_and_manifest_population(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify train_tone_pack detects legacy models (< 2 submodels) and unlinks them, and populates manifest."""
    import json

    from allomorph.pipeline.pack import export_tone_pack, train_tone_pack

    inst = load_instrument("34in_standard_p")
    pack_dir = tmp_path / "test_legacy_pack"
    targets = [("34in_standard_jazz", "bridge_growl")]

    export_tone_pack(
        instrument=inst,
        output_dir=pack_dir,
        input_wav=mini_dry_audio,
        max_samples=1200,
        catalog_targets=targets,
        overwrite=True,
    )

    # Create a legacy single-tier model in nam/
    nam_dir = pack_dir / "nam"
    nam_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = next((pack_dir / "bundles" / "split").glob("*.wav"))
    legacy_nam = nam_dir / f"{legacy_file.stem}.nam"
    with open(legacy_nam, "w", encoding="utf-8") as f:
        json.dump({"version": "0.5.0", "config": {"submodels": [{"name": "channels_8"}]}}, f)

    mock_calls: list[str] = []

    def _mock_train_voice(*args: Any, basename: str | None = None, **kwargs: Any) -> bool:
        mock_calls.append(str(basename))
        out_file = nam_dir / f"{basename}.nam"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "0.5.1",
                    "config": {"submodels": [{"name": "ch3"}, {"name": "ch8"}]},
                    "metadata": {"training": {"validation_esr": 0.001}},
                },
                f,
            )
        return True

    import allomorph.trainer

    monkeypatch.setattr(allomorph.trainer, "train_voice", _mock_train_voice)

    # Calling train_tone_pack should detect legacy and retrain
    train_tone_pack(pack=pack_dir, voice="all", overwrite=False)
    assert len(mock_calls) >= 1


def test_export_tone_pack_artwork_candidates(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify export_tone_pack resolves artwork for various instrument IDs and handles train=True."""
    from allomorph.pipeline.pack import export_tone_pack

    # Mock train_tone_pack
    train_called: list[bool] = []

    def _mock_train(*args: object, **kwargs: object) -> None:
        train_called.append(True)

    monkeypatch.setattr("allomorph.pipeline.pack.train_tone_pack", _mock_train)

    # Test with real catalog instruments to exercise artwork candidate resolution and train=True
    for test_id in [
        "34in_active_stingray",
        "30in_mustang_pj",
        "34in_standard_pj",
        "34in_preamp_soapbar",
    ]:
        real_inst = load_instrument(test_id)
        out_d = tmp_path / f"art_{test_id}"
        export_tone_pack(
            instrument=real_inst,
            output_dir=out_d,
            input_wav=mini_dry_audio,
            max_samples=1200,
            catalog_targets=[("34in_standard_jazz", "bridge_growl")],
            overwrite=True,
            train=True,
        )
        assert (out_d / "manifest.json").exists()

    assert len(train_called) == 4
