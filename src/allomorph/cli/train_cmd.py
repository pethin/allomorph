"""
Allomorph CLI - Train Subcommand
Coordinates local NAM Architecture 2 model training with dynamic engine options,
strict parameter validation, target routing, and automatic stale model pruning.
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Group, Parameter

from allomorph.cli.console import (
    log_error,
    log_step,
    log_success,
    log_warning,
)
from allomorph.cli.targets import resolve_targets
from allomorph.config.instruments import (
    INSTRUMENTS,
    load_instrument,
    partition_instrument_bundles,
    resolve_target_voicing,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.config.voices import VOICES
from allomorph.pipeline.pack import PACKS_DIR, train_tone_pack
from allomorph.trainer import train_voice
from allomorph.trainer.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENGINE,
    DEFAULT_ETA_MIN,
    DEFAULT_LR_T_MAX,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PRECISION,
    DEFAULT_SEED,
    NATIVE_DEFAULT_FIR_HP_FC,
    NATIVE_DEFAULT_FIR_HP_WEIGHT,
    NATIVE_DEFAULT_LITE_WEIGHT,
    NATIVE_DEFAULT_LR,
    NATIVE_DEFAULT_LR_SCHEDULER,
    NATIVE_DEFAULT_MAX_EPOCHS,
    NATIVE_DEFAULT_MIN_DELTA,
    NATIVE_DEFAULT_MIN_EPOCHS,
    NATIVE_DEFAULT_MRSTFT_WEIGHT,
    NATIVE_DEFAULT_MSE_WEIGHT,
    NATIVE_DEFAULT_PATIENCE,
    NATIVE_DEFAULT_SINE_REGULARIZER,
    NATIVE_DEFAULT_SLICE_LEN,
    NATIVE_DEFAULT_VAL_INTERVAL,
    NATIVE_DEFAULT_WEIGHT_DECAY,
    STOCK_DEFAULT_MAX_EPOCHS,
    STOCK_DEFAULT_VAL_LOSS_METRIC,
)
from allomorph.trainer.core import resolve_trainer_engine

MODELS_DIR = REPO_ROOT / "models"

native_group = Group("Native Engine Options")
stock_group = Group("Stock Engine Options")

NATIVE_PARAM_FLAGS = {
    "--slice-len",
    "--min-epochs",
    "--patience",
    "--min-delta",
    "--lr",
    "--weight-decay",
    "--mse-weight",
    "--mrstft-weight",
    "--lite-weight",
    "--fir-hp-weight",
    "--fir-hp-fc",
    "--sine-regularizer",
    "--no-sine-regularizer",
    "--lr-scheduler",
    "--eta-min",
    "--lr-t-max",
    "--val-interval",
}

STOCK_PARAM_FLAGS = {
    "--val-loss-metric",
}


def prune_stale_pack_models(pack_id: str, dry_run: bool = False) -> list[str]:
    """Prunes stale or orphan .nam models from a tone pack's nam/ directory."""
    pack_dir = PACKS_DIR / pack_id
    nam_dir = pack_dir / "nam"
    manifest_path = pack_dir / "manifest.json"
    if not nam_dir.exists():
        return []

    inst = INSTRUMENTS.get(pack_id)
    if not inst:
        return []

    bundles = partition_instrument_bundles(inst)
    valid_model_basenames: set[str] = set()
    for b in bundles.values():
        for t_v in b.targets:
            v_cfg = t_v.voicing
            stem_slug = (
                v_cfg.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
                if v_cfg.tone_name
                else (v_cfg.id or "default")
            )
            valid_model_basenames.add(stem_slug)

    pruned: list[str] = []
    for nam_file in sorted(nam_dir.glob("*.nam")):
        base = nam_file.stem
        # Check if basename matches any valid bundle target stem
        is_valid = any(
            base.endswith(slug) or slug.endswith(base) or base == slug
            for slug in valid_model_basenames
        )
        if not is_valid:
            action = "[DRY-RUN] Would prune" if dry_run else "Pruned stale model"
            log_step("PRUNE", f"{action}: {nam_file.name}")
            pruned.append(nam_file.name)
            if not dry_run:
                nam_file.unlink(missing_ok=True)

    # Clean up manifest entries if models were deleted
    if pruned and not dry_run and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "models" in data and isinstance(data["models"], dict):
                for p in pruned:
                    data["models"].pop(p, None)
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except json.JSONDecodeError, OSError:
            pass

    return pruned


def prune_stale_instrument_models(
    inst_id: str, target_models_dir: Path, dry_run: bool = False
) -> list[str]:
    """Prunes orphan .nam models from an instrument models directory."""
    inst = INSTRUMENTS.get(inst_id)
    if not inst or not target_models_dir.exists():
        return []

    valid_slugs = {
        v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
        if v.tone_name
        else (v.id or "default")
        for v in inst.voicings.values()
    }

    pruned: list[str] = []
    for nam_file in sorted(target_models_dir.glob("*.nam")):
        base = nam_file.stem
        if not any(slug in base for slug in valid_slugs):
            action = "[DRY-RUN] Would prune" if dry_run else "Pruned orphan model"
            log_step("PRUNE", f"{action}: {nam_file.name}")
            pruned.append(nam_file.name)
            if not dry_run:
                nam_file.unlink(missing_ok=True)

    return pruned


def train(
    targets: tuple[str, ...] = ("packs",),
    *,
    engine: Literal["auto", "native", "stock"] = DEFAULT_ENGINE,
    epochs: int | None = None,
    batch_size: str = DEFAULT_BATCH_SIZE,
    precision: str = DEFAULT_PRECISION,
    num_workers: str = DEFAULT_NUM_WORKERS,
    force: bool = False,
    prune: bool = True,
    fast_dev_run: bool = False,
    seed: int | None = DEFAULT_SEED,
    models_dir: Path | None = None,
    # Native Engine Options
    slice_len: Annotated[int, Parameter(group=native_group)] = NATIVE_DEFAULT_SLICE_LEN,
    min_epochs: Annotated[int, Parameter(group=native_group)] = NATIVE_DEFAULT_MIN_EPOCHS,
    patience: Annotated[int, Parameter(group=native_group)] = NATIVE_DEFAULT_PATIENCE,
    min_delta: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_MIN_DELTA,
    lr: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_LR,
    weight_decay: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_WEIGHT_DECAY,
    mse_weight: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_MSE_WEIGHT,
    mrstft_weight: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_MRSTFT_WEIGHT,
    lite_weight: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_LITE_WEIGHT,
    fir_hp_weight: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_FIR_HP_WEIGHT,
    fir_hp_fc: Annotated[float, Parameter(group=native_group)] = NATIVE_DEFAULT_FIR_HP_FC,
    sine_regularizer: Annotated[
        bool, Parameter(group=native_group)
    ] = NATIVE_DEFAULT_SINE_REGULARIZER,
    lr_scheduler: Annotated[str, Parameter(group=native_group)] = NATIVE_DEFAULT_LR_SCHEDULER,
    eta_min: Annotated[float, Parameter(group=native_group)] = DEFAULT_ETA_MIN,
    lr_t_max: Annotated[int, Parameter(group=native_group)] = DEFAULT_LR_T_MAX,
    val_interval: Annotated[int, Parameter(group=native_group)] = NATIVE_DEFAULT_VAL_INTERVAL,
    # Stock Engine Options
    val_loss_metric: Annotated[str, Parameter(group=stock_group)] = STOCK_DEFAULT_VAL_LOSS_METRIC,
) -> None:
    """Trains studio-grade Neural Amp Modeler (.nam) models across packs or instruments."""
    # 1. Strict Engine Parameter Validation
    cli_tokens = set(sys.argv[1:])

    if engine == "stock":
        for flag in NATIVE_PARAM_FLAGS:
            if flag in cli_tokens or any(t.startswith(f"{flag}=") for t in cli_tokens):
                raise ValueError(
                    f"Option '{flag}' is specific to the native engine and cannot be used with '--engine stock'."
                )
        if slice_len != NATIVE_DEFAULT_SLICE_LEN:
            raise ValueError(
                "Option '--slice-len' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if min_epochs != NATIVE_DEFAULT_MIN_EPOCHS:
            raise ValueError(
                "Option '--min-epochs' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if patience != NATIVE_DEFAULT_PATIENCE:
            raise ValueError(
                "Option '--patience' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if min_delta != NATIVE_DEFAULT_MIN_DELTA:
            raise ValueError(
                "Option '--min-delta' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if lr != NATIVE_DEFAULT_LR:
            raise ValueError(
                "Option '--lr' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if weight_decay != NATIVE_DEFAULT_WEIGHT_DECAY:
            raise ValueError(
                "Option '--weight-decay' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if mse_weight != NATIVE_DEFAULT_MSE_WEIGHT:
            raise ValueError(
                "Option '--mse-weight' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if mrstft_weight != NATIVE_DEFAULT_MRSTFT_WEIGHT:
            raise ValueError(
                "Option '--mrstft-weight' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if lite_weight != NATIVE_DEFAULT_LITE_WEIGHT:
            raise ValueError(
                "Option '--lite-weight' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if fir_hp_weight != NATIVE_DEFAULT_FIR_HP_WEIGHT:
            raise ValueError(
                "Option '--fir-hp-weight' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if fir_hp_fc != NATIVE_DEFAULT_FIR_HP_FC:
            raise ValueError(
                "Option '--fir-hp-fc' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if sine_regularizer != NATIVE_DEFAULT_SINE_REGULARIZER:
            raise ValueError(
                "Option '--sine-regularizer' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if lr_scheduler != NATIVE_DEFAULT_LR_SCHEDULER:
            raise ValueError(
                "Option '--lr-scheduler' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if eta_min != DEFAULT_ETA_MIN:
            raise ValueError(
                "Option '--eta-min' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if lr_t_max != DEFAULT_LR_T_MAX:
            raise ValueError(
                "Option '--lr-t-max' is specific to the native engine and cannot be used with '--engine stock'."
            )
        if val_interval != NATIVE_DEFAULT_VAL_INTERVAL:
            raise ValueError(
                "Option '--val-interval' is specific to the native engine and cannot be used with '--engine stock'."
            )
    elif engine == "native":
        for flag in STOCK_PARAM_FLAGS:
            if flag in cli_tokens or any(t.startswith(f"{flag}=") for t in cli_tokens):
                raise ValueError(
                    f"Option '{flag}' is specific to the stock engine and cannot be used with '--engine native'."
                )
        if val_loss_metric != STOCK_DEFAULT_VAL_LOSS_METRIC:
            raise ValueError(
                "Option '--val-loss-metric' is specific to the stock engine and cannot be used with '--engine native'."
            )

    resolved_engine = resolve_trainer_engine(engine)
    resolved_epochs = (
        epochs
        if epochs is not None
        else (
            NATIVE_DEFAULT_MAX_EPOCHS if resolved_engine == "native" else STOCK_DEFAULT_MAX_EPOCHS
        )
    )

    # 2. Target Resolution
    resolved = resolve_targets(targets, default="packs", allow_shorthand=True)
    if not resolved:
        log_warning("No targets matched for training.")
        return

    log_step(
        "TRAIN",
        f"Initiating NAM Architecture 2 training for {len(resolved)} target(s) "
        f"[engine={resolved_engine}, epochs={resolved_epochs}, force={force}, prune={prune}]...",
    )

    total_trained = 0
    total_pruned = 0

    for tgt in resolved:
        if tgt.kind == "pack":
            pack_id = tgt.id
            if prune:
                pruned_models = prune_stale_pack_models(pack_id)
                total_pruned += len(pruned_models)

            log_step(
                "TRAIN", f"Training tone pack '{pack_id}' (tone: {tgt.sub_target or 'all'})..."
            )
            try:
                train_tone_pack(
                    pack=pack_id,
                    voice=tgt.sub_target or "all",
                    overwrite=force,
                    epochs=resolved_epochs,
                    min_epochs=min_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    mse_weight=mse_weight,
                    mrstft_weight=mrstft_weight,
                    batch_size=batch_size,
                    precision=precision,
                    num_workers=num_workers,
                    lr_scheduler=lr_scheduler,
                    eta_min=eta_min,
                    lr_t_max=lr_t_max,
                    fast_dev_run=fast_dev_run,
                    engine=resolved_engine,
                    seed=seed,
                )
                total_trained += 1
            except Exception as e:  # noqa: BLE001
                log_error(f"Failed to train pack '{pack_id}': {e}")
                sys.exit(1)

        elif tgt.kind == "instrument":
            inst_id = tgt.id
            out_models_dir = (
                Path(models_dir) / inst_id if models_dir is not None else MODELS_DIR / inst_id
            )
            out_models_dir.mkdir(parents=True, exist_ok=True)

            if prune:
                pruned_models = prune_stale_instrument_models(inst_id, out_models_dir)
                total_pruned += len(pruned_models)

            try:
                inst_cfg = load_instrument(inst_id)
            except Exception as e:  # noqa: BLE001
                log_error(f"Failed to load instrument '{inst_id}': {e}")
                sys.exit(1)

            voices_to_train: list[str] = []
            if tgt.sub_target is not None:
                _, v_cfg = resolve_target_voicing(tgt.sub_target, instrument=inst_cfg)
                slug = (
                    v_cfg.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
                    if v_cfg.tone_name
                    else (v_cfg.id or tgt.sub_target)
                )
                voices_to_train.append(slug if slug in VOICES else tgt.sub_target)
            else:
                for v_cfg in inst_cfg.voicings.values():
                    slug = (
                        v_cfg.tone_name.lower()
                        .replace(" ", "_")
                        .replace("∕", "_")
                        .replace("/", "_")
                        if v_cfg.tone_name
                        else (v_cfg.id or "")
                    )
                    voices_to_train.append(slug if slug in VOICES else (v_cfg.id or ""))

            for v in voices_to_train:
                log_step("TRAIN", f"Training instrument model {inst_id} -> {v}...")
                ok = train_voice(
                    instrument=inst_cfg,
                    voice=v,
                    models_dir=out_models_dir,
                    epochs=resolved_epochs,
                    min_epochs=min_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    mse_weight=mse_weight,
                    mrstft_weight=mrstft_weight,
                    batch_size=batch_size,
                    precision=precision,
                    num_workers=num_workers,
                    fast_dev_run=fast_dev_run,
                    engine=resolved_engine,
                    seed=seed,
                    lr=lr,
                    weight_decay=weight_decay,
                    slice_len=slice_len,
                    lite_weight=lite_weight,
                    val_interval=val_interval,
                    sine_regularizer=sine_regularizer,
                    val_loss_metric=val_loss_metric,
                )
                if ok:
                    total_trained += 1
                else:
                    log_error(f"Training failed for {inst_id} -> {v}")
                    sys.exit(1)

    prune_str = f", pruned {total_pruned} stale model(s)" if total_pruned > 0 else ""
    log_success(f"Training complete: {total_trained} target task(s) processed{prune_str}.")
