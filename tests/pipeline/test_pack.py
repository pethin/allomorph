"""
Tests for Tone3000 Tone Pack Pipeline in allomorph.pipeline.pack.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.config.instruments import (
    load_instrument,
    load_tone_pack,
)
from allomorph.dsp import read_wav, write_wav_24bit
from allomorph.pipeline.pack import (
    _sha256_file,
    check_pack_needs_repacking,
    export_tone_pack,
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


def test_storefront_description_sections():
    """Verify tone pack full storefront description contains all required sections."""
    pack_cfg = load_tone_pack("34in_standard_p")
    desc = pack_cfg.full_storefront_description

    assert "ALLOMORPH" in desc
    assert "RECOMMENDED SIGNAL CHAIN" in desc
    assert "QUICK INSTRUMENT SETUP" in desc
    assert "LICENSE & DISCLAIMER" in desc
    assert "Standard Precision Bass" in desc


def _restrict_pack_targets(
    monkeypatch: pytest.MonkeyPatch,
    inst_id: str,
    targets: list[tuple[str, str]],
) -> None:
    from allomorph.config.instruments import TONE_PACKS

    pack_cfg = TONE_PACKS[inst_id]
    new_bundles = [
        b_cfg.model_copy(update={"targets": [f"{i}:{v}" for i, v in targets]})
        for b_cfg in pack_cfg.bundles
    ]
    monkeypatch.setitem(TONE_PACKS, inst_id, pack_cfg.model_copy(update={"bundles": new_bundles}))


def test_export_tone_pack_minimal(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify exporting a tone pack with bounded samples creates valid bundle layout."""
    inst = load_instrument("34in_standard_p")
    out_dir = tmp_path / "test_pack"

    # Restrict to two targets for high-speed unit testing
    targets = [
        ("34in_standard_jazz", "bridge_growl"),
        ("34in_standard_jazz", "pair_open"),
    ]
    _restrict_pack_targets(monkeypatch, "34in_standard_p", targets)

    pack_dir = export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
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


def test_export_tone_pack_existing_skip_and_repack(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify existing pack returns early when fresh, and resyncs when a stem is missing."""
    inst = load_instrument("34in_standard_p")
    out_dir = tmp_path / "test_pack_repack"
    targets = [("34in_standard_jazz", "bridge_growl")]
    _restrict_pack_targets(monkeypatch, "34in_standard_p", targets)

    # Initial export
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
    )
    manifest_p = out_dir / "manifest.json"
    mtime1 = manifest_p.stat().st_mtime_ns

    # Call again when fresh -> early return, mtime untouched
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
    )
    mtime2 = manifest_p.stat().st_mtime_ns
    assert mtime1 == mtime2

    # Invalidate by unlinking a bundle stem -> resyncs automatically
    stem_wav = next((out_dir / "bundles" / "split").glob("*.wav"))
    stem_wav.unlink()
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
    )
    assert stem_wav.exists()


def test_storefront_description_passive_multi_pickup():
    """Verify storefront description for a passive multi-pickup bass formats Master Tone instructions."""
    pack_cfg = load_tone_pack("34in_standard_jazz")
    desc = pack_cfg.full_storefront_description

    assert "Master Tone Knob: 100%" in desc
    assert "Neck Volume Knob: 100%" in desc


def test_export_tone_pack_parallel_bundles_and_cached_stems(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify parallel bundle processing with jobs > 1 and cached stem compilation."""
    from allomorph.version import write_manifest

    stems_dir = tmp_path / "stems"
    monkeypatch.setattr("allomorph.pipeline.pack.STEMS_AUDIO_DIR", stems_dir)

    # Mock simulate_instrument_voicing to quickly write a valid WAV file and manifest
    def _mock_sim_voicing(
        *args: Any,
        instrument: Any = None,
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
                or getattr(voicing, "id", str(voicing)) in ("neck_warm", "bridge_open", "pair_open")
            )
            val = 0.35 if is_dry else 0.85
            samples = np.array([0.0, val, -val], dtype=np.float32)
            write_wav_24bit(p, samples, sample_rate=48000)
            from allomorph.version import compute_recipe_token, resolve_tri_part_version

            recipe_tok = compute_recipe_token(
                instrument=instrument,
                voicing=voicing,
                base_dry_path=mini_dry_audio,
            )
            write_manifest(
                p.parent,
                "sim",
                [p],
                version_tag=resolve_tri_part_version(),
                base_dry_file=mini_dry_audio.name,
                recipe_token=recipe_tok,
            )

    monkeypatch.setattr("allomorph.pipeline.pack.simulate_instrument_voicing", _mock_sim_voicing)

    inst = load_instrument("34in_standard_jazz")
    out_dir = tmp_path / "test_parallel_pack"
    targets = [("34in_standard_p", "vintage_open")]
    _restrict_pack_targets(monkeypatch, "34in_standard_jazz", targets)

    pack_dir = export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=None,
        jobs=2,
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
    _restrict_pack_targets(monkeypatch, "34in_standard_p", targets)

    export_tone_pack(
        instrument=inst,
        output_dir=pack_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
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
        empty_weights: list[float] = []
        dummy_content = {
            "version": "0.7.0",
            "architecture": "SlimmableContainer",
            "config": {
                "submodels": [
                    {"max_value": 0.5, "model": {"weights": [0.01] * 1871}},
                    {"max_value": 1, "model": {"weights": [0.01] * 12146}},
                ]
            },
            "weights": empty_weights,
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
        from allomorph.version import DSP_GENERATION

        assert f" v{DSP_GENERATION}.1.1.nam" in nf.name

    assert len(mock_train_calls) == 2

    # Verify manifest.json was updated with top-level "models" dictionary
    with open(pack_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "models" in manifest_data
    assert len(manifest_data["models"]) == 2

    for b_name, b_meta in manifest_data["bundles"].items():
        assert "instrument_id" not in b_meta
        assert "instrument_version" not in b_meta
        assert "stems" not in b_meta
        assert b_meta["stem_count"] == 2
        # Bundle manifest retains the authoritative list of stems
        b_manifest_path = pack_dir / "bundles" / b_name / "manifest.json"
        assert b_manifest_path.exists()
        with open(b_manifest_path, "r", encoding="utf-8") as bf:
            b_data = json.load(bf)
        for stem in b_data["stems"]:
            assert "target" in stem
            assert "target_instrument" not in stem
            assert "target_voicing" not in stem

    for model_name, model_meta in manifest_data["models"].items():
        assert model_name.endswith(".nam")
        assert model_meta["filename"] == model_name
        assert "target" in model_meta
        assert "target_instrument" not in model_meta
        assert "target_voicing" not in model_meta
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
        assert "dry_recipe_token" in model_meta
        assert model_meta["dry_recipe_token"].startswith("alo_")
        assert "wet_recipe_token" in model_meta
        assert model_meta["wet_recipe_token"].startswith("alo_rec_")
        assert "wet_digest_token" in model_meta
        assert model_meta["wet_digest_token"].startswith("alo_stem_")
        assert "dry_sha256" not in model_meta
        assert "wet_sha256" not in model_meta

    # 2. Resumption & Skipping: calling again with overwrite=False skips existing models
    mock_train_calls.clear()
    train_tone_pack(pack=pack_dir, voice="all", overwrite=False)
    assert len(mock_train_calls) == 0  # 0 calls, 100% skipped!

    # 3. Forcing overwrite: calling with overwrite=True retrains both
    train_tone_pack(pack=pack_dir, voice="all", overwrite=True)
    assert len(mock_train_calls) == 2

    # 4. Voice filter: only train matching voice
    mock_train_calls.clear()
    train_tone_pack(pack=pack_dir, voice="34in_standard_jazz:bridge_growl", overwrite=True)
    assert len(mock_train_calls) == 1
    assert "growl" in mock_train_calls[0]["basename"].lower()


def test_train_tone_pack_legacy_removal_and_manifest_population(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify train_tone_pack detects legacy/invalid models and unlinks them, and populates manifest."""
    import json

    from allomorph.pipeline.pack import export_tone_pack, train_tone_pack

    inst = load_instrument("34in_standard_p")
    pack_dir = tmp_path / "test_legacy_pack"
    targets = [("34in_standard_jazz", "bridge_growl")]
    _restrict_pack_targets(monkeypatch, "34in_standard_p", targets)

    export_tone_pack(
        instrument=inst,
        output_dir=pack_dir,
        input_wav=mini_dry_audio,
        max_samples=1200,
    )

    # Create a legacy single-tier or invalid model in nam/
    nam_dir = pack_dir / "nam"
    nam_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = next((pack_dir / "bundles" / "split").glob("*.wav"))
    legacy_nam = nam_dir / f"{legacy_file.stem}.nam"
    with open(legacy_nam, "w", encoding="utf-8") as f:
        json.dump({"version": "0.5.0", "architecture": "SlimmableContainer", "weights": []}, f)

    mock_calls: list[str] = []

    def _mock_train_voice(*args: Any, basename: str | None = None, **kwargs: Any) -> bool:
        mock_calls.append(str(basename))
        out_file = nam_dir / f"{basename}.nam"
        empty_weights: list[float] = []
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "0.7.0",
                    "architecture": "SlimmableContainer",
                    "config": {
                        "submodels": [
                            {"max_value": 0.5, "model": {"weights": [0.01] * 1871}},
                            {"max_value": 1, "model": {"weights": [0.01] * 12146}},
                        ]
                    },
                    "weights": empty_weights,
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
        _restrict_pack_targets(monkeypatch, test_id, [("34in_standard_jazz", "bridge_growl")])
        out_d = tmp_path / f"art_{test_id}"
        export_tone_pack(
            instrument=real_inst,
            output_dir=out_d,
            input_wav=mini_dry_audio,
            max_samples=1200,
            train=True,
        )
        assert (out_d / "manifest.json").exists()

    assert len(train_called) == 4


def test_export_tone_pack_prunes_stale_bundles(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify that export_tone_pack prunes stale/extraneous bundle directories."""
    inst = load_instrument("34in_standard_p")
    out_dir = tmp_path / "test_prune_pack"

    # Pre-populate bundles_dir with a stale legacy bundle directory
    stale_bundle_dir = out_dir / "bundles" / "split_p"
    stale_bundle_dir.mkdir(parents=True)
    (stale_bundle_dir / "stale.txt").write_text("obsolete", encoding="utf-8")

    targets = [("34in_standard_jazz", "bridge_growl")]
    _restrict_pack_targets(monkeypatch, "34in_standard_p", targets)
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=1200,
    )

    bundles_dir = out_dir / "bundles"
    assert (bundles_dir / "split").exists()
    assert not stale_bundle_dir.exists()


def test_export_tone_pack_unconfigured_no_dir_leak(tmp_path: Path, mini_dry_audio: Path):
    """Verify that an instrument without a tone pack raises FileNotFoundError without creating directories."""
    inst = load_instrument("30in_emg_mmtw")
    out_dir = tmp_path / "test_unconfigured_pack"

    with pytest.raises(FileNotFoundError, match="Tone pack configuration not found"):
        export_tone_pack(
            instrument=inst,
            output_dir=out_dir,
            input_wav=mini_dry_audio,
            max_samples=1200,
        )

    assert not out_dir.exists()


def test_pack_copies_newer_stems_by_digest(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify pack sync copies newer stems from audio/stems based strictly on cryptographic digests."""
    from allomorph.version import (
        DSP_GENERATION,
        compute_recipe_token,
        resolve_tri_part_version,
        write_manifest,
    )

    stems_dir = tmp_path / "stems"
    monkeypatch.setattr("allomorph.pipeline.pack.STEMS_AUDIO_DIR", stems_dir)

    def _mock_sim_voicing(
        *args: Any,
        instrument: Any = None,
        voicing: Any = None,
        output_wav: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        if output_wav is not None:
            p = Path(output_wav)
            p.parent.mkdir(parents=True, exist_ok=True)
            is_dry = (
                "dry" in p.name
                or "34in_standard_p" in str(p)
                or getattr(voicing, "id", str(voicing)) in ("split_vintage",)
            )
            val = 0.35 if is_dry else 0.85
            samples = np.array([0.0, val, -val], dtype=np.float32)
            write_wav_24bit(p, samples, sample_rate=48000)
            recipe_tok = compute_recipe_token(
                instrument=instrument,
                voicing=voicing,
                base_dry_path=mini_dry_audio,
            )
            v_tag = resolve_tri_part_version(DSP_GENERATION, instrument.version, voicing.version)
            write_manifest(
                p.parent,
                "sim",
                [p],
                version_tag=v_tag,
                base_dry_file=mini_dry_audio.name,
                recipe_token=recipe_tok,
            )

    monkeypatch.setattr("allomorph.pipeline.pack.simulate_instrument_voicing", _mock_sim_voicing)

    inst = load_instrument("34in_standard_p")
    out_dir = tmp_path / "test_digest_sync_pack"
    targets = [("34in_standard_jazz", "bridge_growl")]
    _restrict_pack_targets(monkeypatch, "34in_standard_p", targets)

    # 1. Initial export
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=None,
    )

    # Pack should be fresh
    assert not check_pack_needs_repacking(
        inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
    )

    # Find the copied wet stem in bundle
    bundle_stem = next(
        p for p in (out_dir / "bundles" / "split").glob("*.wav") if not p.name.startswith("dry")
    )
    initial_audio, _ = read_wav(bundle_stem)
    assert np.allclose(initial_audio, np.array([0.0, 0.85, -0.85], dtype=np.float32))

    # 2. Update master stem in stems_dir with new audio (different digest)
    master_stem = stems_dir / "34in_standard_jazz" / "bridge_growl.wav"
    assert master_stem.exists()
    updated_samples = np.array([0.0, 0.55, -0.55, 0.1], dtype=np.float32)
    write_wav_24bit(master_stem, updated_samples, sample_rate=48000)
    tgt_jazz = load_instrument("34in_standard_jazz")
    tgt_recipe = compute_recipe_token(
        instrument=tgt_jazz,
        voicing=tgt_jazz.voicings["bridge_growl"],
        base_dry_path=mini_dry_audio,
    )
    tgt_v_tag = resolve_tri_part_version(
        DSP_GENERATION, tgt_jazz.version, tgt_jazz.voicings["bridge_growl"].version
    )
    write_manifest(
        master_stem.parent,
        "sim",
        [master_stem],
        version_tag=tgt_v_tag,
        base_dry_file=mini_dry_audio.name,
        recipe_token=tgt_recipe,
    )

    # Pack should now detect digest mismatch and need repacking
    assert check_pack_needs_repacking(
        inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
    )

    # 3. Re-export without force
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=None,
    )

    # Bundle stem should now contain updated audio
    updated_audio, _ = read_wav(bundle_stem)
    assert np.allclose(updated_audio, updated_samples)
    assert not check_pack_needs_repacking(
        inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
    )

    # 4. Extraneous file pruning test
    extraneous_wav = out_dir / "bundles" / "split" / "extraneous_stale.wav"
    write_wav_24bit(extraneous_wav, updated_samples, sample_rate=48000)
    assert extraneous_wav.exists()

    # Extraneous wav must trigger repacking
    assert check_pack_needs_repacking(
        inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
    )

    # Re-export must prune extraneous wav
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=None,
    )
    assert not extraneous_wav.exists()
    assert not check_pack_needs_repacking(
        inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
    )


def test_pack_preserves_nam_tokens_on_digest_mismatch(
    tmp_path: Path, mini_dry_audio: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify export_tone_pack preserves recorded model tokens when stem digests mismatch."""
    import json

    inst = load_instrument("34in_standard_p")
    out_dir = tmp_path / "test_model_digest_pack"
    targets = [("34in_standard_jazz", "bridge_growl")]
    _restrict_pack_targets(monkeypatch, "34in_standard_p", targets)

    # 1. Initial export
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
    )

    # 2. Inject an existing trained model with recorded tokens
    manifest_p = out_dir / "manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    model_key = "test_model.nam"
    manifest_data["models"] = {
        model_key: {
            "filename": model_key,
            "bundle": "split",
            "target": "34in_standard_jazz:bridge_growl",
            "tone_name": "Jazz Bridge Growl",
            "dry_file": "dry v6.1.1.wav",
            "wet_file": "old_stem_name.wav",
            "dry_digest_token": "alo_stem_RECORDED_OLD_DRY",
            "wet_digest_token": "alo_stem_RECORDED_OLD_WET",
            "dry_recipe_token": "alo_OLD_DRY_RECIPE",
            "wet_recipe_token": "alo_rec_OLD_WET_RECIPE",
            "version": "v1.0.0",
        }
    }
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 3. Invalidate bundle stem to force repacking
    bundle_stem = next(
        p for p in (out_dir / "bundles" / "split").glob("*.wav") if not p.name.startswith("dry")
    )
    bundle_stem.unlink()

    # Re-export pack
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
    )

    # 4. Assert recorded tokens and filenames remain untouched because digests mismatched
    with open(manifest_p, "r", encoding="utf-8") as f:
        repacked_manifest = json.load(f)

    entry = repacked_manifest["models"][model_key]
    assert entry["dry_digest_token"] == "alo_stem_RECORDED_OLD_DRY"
    assert entry["wet_digest_token"] == "alo_stem_RECORDED_OLD_WET"
    assert entry["dry_recipe_token"] == "alo_OLD_DRY_RECIPE"
    assert entry["wet_recipe_token"] == "alo_rec_OLD_WET_RECIPE"
    assert entry["wet_file"] == "old_stem_name.wav"
    assert entry["version"] == "v1.0.0"

    # 5. Now update recorded tokens to match current bundle digests
    b_man_p = out_dir / "bundles" / "split" / "manifest.json"
    with open(b_man_p, "r", encoding="utf-8") as bf:
        b_man = json.load(bf)

    matching_stem = next(
        s for s in b_man["stems"] if s["target"] == "34in_standard_jazz:bridge_growl"
    )
    entry["dry_digest_token"] = b_man["dry_digest_token"]
    entry["wet_digest_token"] = matching_stem["digest_token"]
    repacked_manifest["models"][model_key] = entry
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(repacked_manifest, f, indent=2)

    # Invalidate again to trigger repack
    bundle_stem.unlink()
    export_tone_pack(
        instrument=inst,
        output_dir=out_dir,
        input_wav=mini_dry_audio,
        max_samples=2400,
    )

    # 6. Because digests matched, metadata and filenames should be migrated
    with open(manifest_p, "r", encoding="utf-8") as f:
        matched_manifest = json.load(f)

    updated_entry = matched_manifest["models"][model_key]
    assert updated_entry["wet_file"] == matching_stem["filename"]
    assert updated_entry["wet_recipe_token"] == matching_stem["recipe_token"]
    assert updated_entry["version"] == matching_stem["version"]
