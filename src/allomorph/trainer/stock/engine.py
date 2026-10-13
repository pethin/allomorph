"""
Allomorph Trainer - PyTorch / Lightning Architecture 2 Engine
High-fidelity studio reference neural network training for Neural Amp Modeler (NAM)
Architecture 2 (A2) using PyTorch Lightning and the upstream neural-amp-modeler trainer.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from allomorph.config import (
    VOICES,
    InstrumentConfig,
    compute_effective_position,
    load_instrument,
    resolve_voice_coils,
)
from allomorph.pipeline.schema import (
    NamExportMetadata,
    NamSourceInstrumentMeta,
    NamSourceVoicingMeta,
    NamTargetVoicingMeta,
    NamTrainingMetadata,
)
from allomorph.trainer.callbacks import (
    EsrProgressCallback,
)
from allomorph.trainer.constants import (
    STOCK_DEFAULT_BATCH_SIZE,
    STOCK_DEFAULT_MAX_EPOCHS,
    STOCK_DEFAULT_NUM_WORKERS,
    STOCK_DEFAULT_PRECISION,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    compute_recipe_token,
    get_git_commit,
    is_wet_stem_valid,
    resolve_tri_part_version,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIO_DIR = REPO_ROOT / "audio"
MODELS_DIR = REPO_ROOT / "models"


def configure_a2_architecture(
    nam_core: Any,
    precision: str = STOCK_DEFAULT_PRECISION,
    num_workers: int | str = STOCK_DEFAULT_NUM_WORKERS,
) -> None:
    """Configure NAM Architecture 2 packed model submodels, hardware acceleration, and logging for stock training."""
    from allomorph.trainer.core import resolve_hardware_num_workers, resolve_hardware_precision

    resolved_precision = resolve_hardware_precision(precision)
    resolved_workers = resolve_hardware_num_workers(num_workers)

    try:
        import torch

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = torch.cuda.is_available()
    except ImportError:
        pass

    orig_detect_input_version = getattr(
        nam_core, "_orig_detect_input_version", nam_core._detect_input_version
    )
    nam_core._orig_detect_input_version = orig_detect_input_version

    def patched_detect_input_version(path: Any) -> tuple[Any, bool]:
        err_types = (
            getattr(nam_core, "_InputValidationError", ValueError),
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        )
        try:
            res = orig_detect_input_version(path)
            nam_core._is_dry_wet = False
            return res
        except err_types:
            nam_core._is_dry_wet = True
            version_cls = getattr(nam_core, "_Version", None)
            if version_cls is None:
                from nam.train._version import Version as version_cls

            return version_cls(3, 0, 0), False

    nam_core._detect_input_version = patched_detect_input_version

    orig_check_data = getattr(nam_core, "_orig_check_data", nam_core._check_data)
    nam_core._orig_check_data = orig_check_data

    def patched_check_data(*args: Any, **kwargs: Any) -> Any:
        if getattr(nam_core, "_is_dry_wet", False):
            import nam.train.metadata as meta

            return meta.DataChecks(passed=True, version=3)
        return orig_check_data(*args, **kwargs)

    nam_core._check_data = patched_check_data

    orig_get_data_config = getattr(nam_core, "_orig_get_data_config", nam_core._get_data_config)
    nam_core._orig_get_data_config = orig_get_data_config

    def patched_get_data_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cfg: dict[str, Any] = orig_get_data_config(*args, **kwargs)
        if getattr(nam_core, "_is_dry_wet", False):
            if "train" in cfg:
                cfg["train"]["start_samples"] = 0
            if "validation" in cfg:
                cfg["validation"]["require_input_pre_silence"] = False
        return cfg

    nam_core._get_data_config = patched_get_data_config

    orig_get_packed_model_config = getattr(
        nam_core, "_orig_get_packed_model_config", nam_core._get_packed_model_config
    )
    nam_core._orig_get_packed_model_config = orig_get_packed_model_config

    def get_configured_packed_model_config() -> dict[str, Any]:
        cfg: dict[str, Any] = orig_get_packed_model_config()
        cfg["net"]["config"]["submodels"] = [
            s
            for s in cfg["net"]["config"]["submodels"]
            if s["name"] in ("channels_3", "channels_8")
        ]
        return cfg

    nam_core._get_packed_model_config = get_configured_packed_model_config

    orig_get_configs = getattr(
        nam_core, "_orig_get_configs", getattr(nam_core, "_get_configs", None)
    )
    if orig_get_configs is not None:
        nam_core._orig_get_configs = orig_get_configs

        def patched_get_configs(
            *args: Any, **kwargs: Any
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            data_config, model_config, learning_config = orig_get_configs(*args, **kwargs)
            if resolved_precision in ("bf16-mixed", "16-mixed", "32-true", "32"):
                learning_config["trainer"]["precision"] = resolved_precision
            learning_config["train_dataloader"]["num_workers"] = resolved_workers
            if resolved_workers > 0:
                learning_config["train_dataloader"]["persistent_workers"] = True
            return data_config, model_config, learning_config

        nam_core._get_configs = patched_get_configs

    orig_get_callbacks = getattr(nam_core, "_orig_get_callbacks", nam_core.get_callbacks)
    nam_core._orig_get_callbacks = orig_get_callbacks

    def get_callbacks_with_logging(
        threshold_esr: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        callbacks: list[Any] = orig_get_callbacks(threshold_esr, *args, **kwargs)
        progress_cb = EsrProgressCallback(min_epochs=0)
        callbacks.append(progress_cb)
        nam_core._last_esr_callback = progress_cb
        nam_core._last_stopping_callback = None
        return callbacks

    nam_core.get_callbacks = get_callbacks_with_logging


def train_voice_stock(
    instrument: str | InstrumentConfig = "30in",
    voice: str = "precision_active",
    source_voicing: str | None = None,
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    reference_wav: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
    epochs: int = STOCK_DEFAULT_MAX_EPOCHS,
    batch_size: int | str = STOCK_DEFAULT_BATCH_SIZE,
    precision: str = STOCK_DEFAULT_PRECISION,
    num_workers: int | str = STOCK_DEFAULT_NUM_WORKERS,
    silent: bool = True,
    save_plot: bool = False,
    fast_dev_run: bool = False,
    basename: str | None = None,
    version_tag: str | None = "auto",
    no_manifest: bool = False,
    include_identity: bool = False,
    seed: int | None = None,
) -> bool:
    """Trains a Neural Amp Modeler Architecture 2 local model container using PyTorch Lightning."""
    try:
        import nam.train.core as nam_core
        import nam.train.metadata as train_meta
        import torch
        from nam.models.metadata import UserMetadata

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        os.environ.setdefault("HIPBLASLT_ALLOW_TF32", "1")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch, "backends"):
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = torch.cuda.is_available()
    except ImportError as e:
        print(f"Error: Failed to import 'neural-amp-modeler' ({e}).")
        print("Please run `uv sync` to ensure project dependencies are installed:")
        print(f"  uv run python -m allomorph.trainer --voice {voice}")
        return False

    if voice not in VOICES:
        raise KeyError(f"Target voice '{voice}' not found in catalog.")
    vcfg = VOICES[voice]
    voice_name = vcfg.name

    try:
        inst_cfg = (
            instrument if isinstance(instrument, InstrumentConfig) else load_instrument(instrument)
        )
    except FileNotFoundError, KeyError, ValueError, OSError:
        inst_cfg = InstrumentConfig(
            id=str(instrument),
            name=str(instrument),
            scale_length_in=34.0,
            scale_length_m=0.8636,
            string_wave_speeds=[58.02, 75.88, 99.19, 129.6],
            pickups={},
        )

    inst_ver = getattr(inst_cfg, "version", 1)
    voice_ver = getattr(vcfg, "version", 1)
    tri_part = resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
    actual_version_tag = (
        tri_part
        if (version_tag == "auto" or version_tag is True)
        else (version_tag if version_tag not in (None, "none", False) else None)
    )

    model_basename = basename or (f"{voice}_{actual_version_tag}" if actual_version_tag else voice)
    inst_id = inst_cfg.id
    inst_name = inst_cfg.name
    scale_length_in = inst_cfg.scale_length_in or 34.0

    from allomorph.config.voices import voicing_to_voice_config

    if source_voicing is not None:
        if source_voicing not in inst_cfg.voicings:
            raise KeyError(
                f"Source voicing '{source_voicing}' not found on instrument '{inst_cfg.id}'. "
                f"Available voicings: {list(inst_cfg.voicings.keys())}"
            )
        src_vcfg = inst_cfg.voicings[source_voicing]
        src_voicing_id = source_voicing
    elif voice in inst_cfg.voicings:
        src_vcfg = inst_cfg.voicings[voice]
        src_voicing_id = voice
    elif inst_cfg.voicings:
        src_voicing_id = next(iter(inst_cfg.voicings.keys()))
        src_vcfg = inst_cfg.voicings[src_voicing_id]
    else:
        src_voicing_id = "default"
        src_vcfg = None

    if src_vcfg is not None:
        src_voice_cfg = voicing_to_voice_config(inst_cfg, src_vcfg)
        src_coils = resolve_voice_coils(src_voice_cfg)
        src_eff_pos_m = compute_effective_position(src_coils) if src_coils else 0.0
        src_eff_pos_mm = src_eff_pos_m * 1000.0
        src_voicing_name = src_vcfg.name
        src_tone_name = src_vcfg.tone_name
        src_harness = src_vcfg.harness
        src_controls = dict(src_vcfg.controls)
        src_switches = dict(src_vcfg.switches)
        src_aperture = src_coils[0].aperture_width_in if src_coils else 0.75
        src_coil_spacing = (
            abs(src_coils[1].position_from_bridge_m - src_coils[0].position_from_bridge_m) / 0.0254
            if len(src_coils) >= 2
            else 0.0
        )
        src_type = "dual_coil" if len(src_coils) >= 2 else "single_coil"
    else:
        src_eff_pos_m = 0.0
        src_eff_pos_mm = 0.0
        src_voicing_name = "Default Voicing"
        src_tone_name = None
        src_harness = ""
        src_controls: dict[str, float] = {}
        src_switches: dict[str, str] = {}
        src_aperture = 0.75
        src_coil_spacing = 0.0
        src_type = "single_coil"

    src_voicing_meta = NamSourceVoicingMeta(
        id=src_voicing_id,
        name=src_voicing_name,
        tone_name=src_tone_name,
        harness=src_harness,
        position_from_bridge_m=src_eff_pos_m,
        position_from_bridge_mm=src_eff_pos_mm,
        effective_position_m=src_eff_pos_m,
        effective_position_mm=src_eff_pos_mm,
        aperture_width_in=src_aperture,
        coil_spacing_in=src_coil_spacing,
        type=src_type,
        controls=src_controls,
        switches=src_switches,
    )

    models_path = Path(models_dir)
    if models_path.name == "nam" or models_path.name == inst_id:
        inst_models_dir = models_path
    else:
        inst_models_dir = models_path / inst_id
    inst_models_dir.mkdir(parents=True, exist_ok=True)
    target_nam = inst_models_dir / f"{model_basename}.nam"

    from allomorph.trainer.core import (
        find_sweep_input,
        get_hardware_device_name,
        resolve_hardware_batch_size,
        resolve_hardware_num_workers,
        resolve_hardware_precision,
    )

    input_path = find_sweep_input(input_wav)

    if not output_wav:
        from allomorph.config.instruments import resolve_target_voicing

        tgt_inst, tgt_v = resolve_target_voicing(voice, instrument=inst_cfg)
        target_slug = (
            tgt_v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            if tgt_v.tone_name
            else (tgt_v.id or voice)
        )
        target_wet = AUDIO_DIR / "wet" / tgt_inst.id / f"{target_slug}.wav"
        target_recipe_token = compute_recipe_token(
            instrument=tgt_inst,
            voicing=tgt_v,
            base_dry_path=input_path,
        )
        if not (
            target_wet.exists()
            and is_wet_stem_valid(
                target_wet,
                expected_recipe_token=target_recipe_token,
                base_dry_path=input_path,
            )
        ):
            try:
                from allomorph.circuit import simulate_instrument_voicing

                print(
                    f"[NAM Trainer] Target wet stem missing or stale, auto-simulating: {target_wet.name}"
                )
                output_path = simulate_instrument_voicing(
                    instrument=tgt_inst, voicing=tgt_v, output_wav=target_wet
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as e:
                print(f"[NAM Trainer] Warning: Failed to auto-simulate {target_wet.name}: {e}")
                output_path = target_wet
        else:
            output_path = target_wet
    else:
        output_path = Path(output_wav)

    if not input_path or not input_path.exists():
        print(f"Error: Could not find training input file '{input_path}'.")
        return False

    if not output_path.exists():
        print(f"Error: Target output audio '{output_path}' does not exist.")
        print("Please run the simulation stage first:")
        print(f"  uv run allomorph stems generate --targets {voice}")
        return False

    # Resolve reference stem
    if reference_wav:
        reference_path = Path(reference_wav)
    else:
        candidate_source = AUDIO_DIR / "wet" / inst_id / f"{src_voicing_id}.wav"
        src_v = inst_cfg.voicings.get(src_voicing_id)
        if src_v is None:
            src_v = next(iter(inst_cfg.voicings.values()))
        src_recipe_token = compute_recipe_token(
            instrument=inst_cfg,
            voicing=src_v,
            base_dry_path=input_path,
        )
        if candidate_source.exists() and is_wet_stem_valid(
            candidate_source,
            expected_recipe_token=src_recipe_token,
            base_dry_path=input_path,
        ):
            reference_path = candidate_source
        else:
            try:
                from allomorph.circuit.forward import simulate_instrument_voicing

                print(
                    f"[NAM Trainer] Dedicated source stem missing or stale, auto-generating: {candidate_source.name}"
                )
                src_v = inst_cfg.voicings.get(src_voicing_id)
                if src_v is None:
                    src_v = next(iter(inst_cfg.voicings.values()))
                reference_path = simulate_instrument_voicing(
                    instrument=inst_cfg,
                    voicing=src_v,
                    output_wav=candidate_source,
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as e:
                print(
                    f"[NAM Trainer] Warning: Failed to auto-generate source voicing stem: {e}. Falling back to input sweep."
                )
                reference_path = input_path

    if not include_identity and reference_path.resolve() == output_path.resolve():
        print(
            f"[NAM Trainer] Skipping identity pair for '{voice}' on {inst_id}: "
            f"source and target audio stems are identical ({reference_path.name}). "
            "Use --include-identity to force training."
        )
        return True

    resolved_batch_size = resolve_hardware_batch_size(batch_size)
    resolved_precision = resolve_hardware_precision(precision)
    resolved_num_workers = resolve_hardware_num_workers(num_workers)
    device_name = get_hardware_device_name(engine="stock")

    configure_a2_architecture(
        nam_core,
        precision=resolved_precision,
        num_workers=resolved_num_workers,
    )

    train_work_dir = inst_models_dir / f".train_{model_basename}"
    train_work_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("  ALLOMORPH NAM LOCAL A2 TRAINER (Stock Lightning)")
    print(f'  Source Bass: {inst_name} ({inst_id}, {scale_length_in}")')
    print(f"  Source Voicing: {src_voicing_name} (pos={src_eff_pos_mm:.1f}mm)")
    print(f"  Target Voice:{voice} ({voice_name})")
    arch_display = "Architecture 2 Slimmable (channels_3 + channels_8)"
    print(f"  Model Tier:  {arch_display}")
    print(f"  Hardware:    {device_name}")
    steps_per_epoch = 1350 // resolved_batch_size
    print(f"  Batch Size:  {resolved_batch_size} (auto, {steps_per_epoch} steps/epoch)")
    print(f"  Precision:   {resolved_precision}")
    print(f"  Workers:     {resolved_num_workers} (zero-copy in-memory)")
    print(f"  Input Audio: {input_path.name}")
    print(f"  Output Audio:{output_path.name}")
    print(f"  Epochs:      {epochs}")
    print(f"  Destination: {target_nam}")
    print("========================================\n")

    model_title = f"{voice_name} [{inst_name}]"
    user_metadata = UserMetadata(
        name=model_title,
        modeled_by="Allomorph (Peter Nguyen <peter@phn.dev>)",
        gear_make=inst_name,
        gear_model=f"{src_voicing_name} -> {voice_name}",
        gear_type="preamp",
        tone_type="clean",
    )

    print("Validating dataset and calibration markers...")
    try:
        train_output = nam_core.train(
            input_path=str(input_path),
            output_path=str(output_path),
            train_path=str(train_work_dir),
            epochs=epochs,
            batch_size=resolved_batch_size,
            modelname=model_basename,
            silent=silent,
            save_plot=save_plot,
            local=False,
            threshold_esr=None,
            user_metadata=user_metadata,
            fast_dev_run=fast_dev_run,
            latency=0,
            ignore_checks=True,
        )
    finally:
        with contextlib.suppress(ImportError, RuntimeError):
            import matplotlib.pyplot as plt

            plt.close("all")

    if train_output is None or train_output.model is None:
        print("Error: Training did not produce a model.")
        return False

    cb = getattr(nam_core, "_last_esr_callback", None)
    stopping_cb = getattr(nam_core, "_last_stopping_callback", None)
    best_mrstft = (
        cb.best_mrstft
        if (cb is not None and cb.best_mrstft is not None and cb.best_mrstft < float("inf"))
        else None
    )
    epochs_trained = cb.last_epoch + 1 if (cb is not None and cb.last_epoch >= 0) else None
    stop_reason = stopping_cb.stop_reason if stopping_cb is not None else None

    raw_meta = train_output.metadata.model_dump()
    for k in list(raw_meta.keys()):
        if "esr" in k.lower():
            raw_meta.pop(k, None)

    best_studio_spec = (
        cb.best_spectral_ch8 if (cb is not None and cb.best_spectral_ch8 < float("inf")) else None
    )
    best_lite_spec = (
        cb.best_spectral_ch3 if (cb is not None and cb.best_spectral_ch3 < float("inf")) else None
    )
    best_esr_pre = (
        cb.best_esr_pre
        if (cb is not None and cb.best_esr_pre is not None and cb.best_esr_pre < float("inf"))
        else None
    )

    if best_studio_spec is not None:
        raw_meta["validation_spectral_loss"] = best_studio_spec
        raw_meta["validation_spectral_loss_a2_full"] = best_studio_spec
    if best_lite_spec is not None:
        raw_meta["validation_spectral_loss_a2_lite"] = best_lite_spec
    if best_esr_pre is not None:
        raw_meta["esr_pre_loss"] = best_esr_pre
    if best_mrstft is not None:
        raw_meta["mrstft_loss"] = best_mrstft
    if cb is not None and cb.best_ch3_mrstft is not None and cb.best_ch3_mrstft < float("inf"):
        raw_meta["mrstft_loss_ch3"] = cb.best_ch3_mrstft
    if epochs_trained is not None:
        raw_meta["epochs_trained"] = epochs_trained
    if stop_reason is not None:
        raw_meta["stop_reason"] = stop_reason
    raw_meta["batch_size"] = resolved_batch_size
    raw_meta["precision"] = resolved_precision
    raw_meta["num_workers"] = resolved_num_workers
    raw_meta["device_name"] = device_name

    target_coils = resolve_voice_coils(vcfg)
    target_eff_pos_m = compute_effective_position(target_coils) if target_coils else 0.0
    target_eff_pos_mm = target_eff_pos_m * 1000.0

    print("\nExporting Architecture 2 (.nam) model container with full instrument metadata...")
    nam_meta = NamExportMetadata(
        training=NamTrainingMetadata.model_validate(raw_meta),
        license="PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0)",
        copyright="Copyright 2026 Peter Nguyen <peter@phn.dev>. All commercial rights reserved.",
        author="Peter Nguyen <peter@phn.dev>",
        version=tri_part,
        dsp_version=DSP_GENERATION,
        instrument_version=inst_ver,
        voice_version=voice_ver,
        allomorph_version=ALLOMORPH_VERSION,
        git_commit=get_git_commit(),
        generated_at=datetime.now(UTC).isoformat(),
        source_instrument=NamSourceInstrumentMeta(
            id=inst_id,
            name=inst_name,
            scale_length_in=scale_length_in,
            scale_length_m=inst_cfg.scale_length_m,
            string_wave_speeds=inst_cfg.string_wave_speeds,
            voicing=src_voicing_meta,
        ),
        target_voicing=NamTargetVoicingMeta(
            id=voice,
            name=voice_name,
            tone_name=vcfg.tone_name,
            sensor_type=vcfg.sensor_type,
            resonant_frequency_hz=float(vcfg.fr),
            q_factor=float(vcfg.Q),
            effective_position_m=target_eff_pos_m,
            effective_position_mm=target_eff_pos_mm,
            coils=target_coils,
        ),
    )
    meta_dump = nam_meta.model_dump()
    other_metadata = {
        train_meta.TRAINING_KEY: meta_dump["training"],
        "license": meta_dump["license"],
        "copyright": meta_dump["copyright"],
        "author": meta_dump["author"],
        "version": meta_dump["version"],
        "dsp_version": meta_dump["dsp_version"],
        "instrument_version": meta_dump["instrument_version"],
        "voice_version": meta_dump["voice_version"],
        "allomorph_version": meta_dump["allomorph_version"],
        "git_commit": meta_dump["git_commit"],
        "generated_at": meta_dump["generated_at"],
        "source_instrument": meta_dump["source_instrument"],
        "target_voicing": meta_dump["target_voicing"],
    }

    export_net: Any = train_output.model.net
    export_net.export(
        str(inst_models_dir),
        basename=model_basename,
        user_metadata=user_metadata,
        other_metadata=other_metadata,
    )

    # Clean up temporary lightning checkpoint folder
    if train_work_dir.exists():
        shutil.rmtree(train_work_dir, ignore_errors=True)

    if target_nam.exists():
        if not no_manifest:
            write_manifest(
                output_dir=inst_models_dir,
                stage="train",
                files=[target_nam],
                version_tag=tri_part,
            )
        size_kb = target_nam.stat().st_size / 1024
        print("\n[Success] Architecture 2 Model exported successfully!")
        print(f"  Model Path:    {target_nam} ({size_kb:.1f} KB)")
        print(f"  Model Title:   {model_title}")
        print(f"  Source Bass:   {inst_name}")
        print(f"  Source Voicing: {src_voicing_name} ({src_eff_pos_mm:.1f}mm)")
        if best_studio_spec is not None:
            lite_disp = (
                f" | {best_lite_spec:.5f} (A2 Lite Ch3)" if best_lite_spec is not None else ""
            )
            print(f"  Validation Spectral Loss: {best_studio_spec:.5f} (A2 Full Ch8){lite_disp}")
            if best_esr_pre is not None:
                print(f"  Validation ESR_pre:       {best_esr_pre:.5f}")
            if best_mrstft is not None:
                print(f"  Validation MRSTFT:        {best_mrstft:.6f}")
            if stop_reason:
                print(f"  Termination:              {stop_reason}")
        print("  Ready for Darkglass Anagram Block 1 (Preamp) loading.")
        return True
    else:
        print(f"Warning: Expected model file at {target_nam} not found.")
        return False


__all__ = [
    "AUDIO_DIR",
    "MODELS_DIR",
    "configure_a2_architecture",
    "train_voice_stock",
]
