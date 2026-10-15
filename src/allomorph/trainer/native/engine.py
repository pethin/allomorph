"""Native PyTorch training engine for Allomorph Architecture 2 (A2) NAM models.

Zero-external-framework training loop featuring:
- In-VRAM zero-copy tensor slicing (loading stem audio once directly on GPU/accelerator).
- Self-Normalizing Multi-Band High-Pass ESR Loss with waveform MSE anchoring.
- Anti-Drift Pareto early stopping with in-memory checkpoint restoration.
- Native export to compliant Darkglass Anagram Block 1 slimmable .nam JSON.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss
from pedalboard.io import AudioFile

from allomorph.config import (
    InstrumentConfig,
    compute_effective_position,
    load_instrument,
    resolve_target_voicing,
    resolve_voicing_electroacoustics,
)
from allomorph.pipeline.schema import (
    NamExportMetadata,
    NamSourceInstrumentMeta,
    NamSourceVoicingMeta,
    NamTargetVoicingMeta,
    NamTrainingMetadata,
)
from allomorph.trainer.callbacks import (
    compute_pareto_quality_score,
    evaluate_plateau_stopping_condition,
)
from allomorph.trainer.constants import (
    DEFAULT_ENGINE,
    DEFAULT_ETA_MIN,
    DEFAULT_LR_T_MAX,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    NATIVE_DEFAULT_BATCH_SIZE,
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
    NATIVE_DEFAULT_PRECISION,
    NATIVE_DEFAULT_SINE_AMPLITUDES,
    NATIVE_DEFAULT_SINE_FREQUENCIES,
    NATIVE_DEFAULT_SINE_REGULARIZER,
    NATIVE_DEFAULT_SLICE_LEN,
    NATIVE_DEFAULT_VAL_INTERVAL,
    NATIVE_DEFAULT_WEIGHT_DECAY,
)
from allomorph.trainer.native.loss import design_sharp_fir_highpass
from allomorph.trainer.native.models import AllomorphWaveNetA2
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    compute_file_sha256,
    compute_recipe_token,
    get_git_commit,
    is_stem_valid,
    resolve_tri_part_version,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIO_DIR = REPO_ROOT / "audio"
MODELS_DIR = REPO_ROOT / "models"


def load_audio_tensor(wav_path: str | Path) -> tuple[torch.Tensor, int]:
    """Loads an audio file directly into a 1D float32 PyTorch tensor using pedalboard."""
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    with AudioFile(str(p)) as f:
        sr = int(f.samplerate)
        audio = f.read(f.frames)
    mono = np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]
    return torch.from_numpy(mono.astype(np.float32)), sr


def train_voice_native(
    instrument: str | InstrumentConfig = "34in_standard_p",
    voice: str = "34in_standard_p:vintage",
    source_voicing: str | None = None,
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    reference_wav: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
    epochs: int = NATIVE_DEFAULT_MAX_EPOCHS,
    min_epochs: int = NATIVE_DEFAULT_MIN_EPOCHS,
    patience: int = NATIVE_DEFAULT_PATIENCE,
    min_delta: float = NATIVE_DEFAULT_MIN_DELTA,
    mse_weight: float = NATIVE_DEFAULT_MSE_WEIGHT,
    mrstft_weight: float = NATIVE_DEFAULT_MRSTFT_WEIGHT,
    lr_scheduler: str = NATIVE_DEFAULT_LR_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    lr_t_max: int = DEFAULT_LR_T_MAX,
    batch_size: int | str = NATIVE_DEFAULT_BATCH_SIZE,
    precision: str = NATIVE_DEFAULT_PRECISION,
    num_workers: int | str = DEFAULT_NUM_WORKERS,
    silent: bool = True,
    save_plot: bool = False,
    fast_dev_run: bool = False,
    basename: str | None = None,
    version_tag: str | None = "auto",
    no_manifest: bool = False,
    include_identity: bool = False,
    engine: str = DEFAULT_ENGINE,
    seed: int | None = DEFAULT_SEED,
    device: str | None = None,
    # Dedicated Native Engine Hyperparameters
    lr: float = NATIVE_DEFAULT_LR,
    weight_decay: float = NATIVE_DEFAULT_WEIGHT_DECAY,
    slice_len: int = NATIVE_DEFAULT_SLICE_LEN,
    lite_weight: float = NATIVE_DEFAULT_LITE_WEIGHT,
    val_interval: int = NATIVE_DEFAULT_VAL_INTERVAL,
    sine_regularizer: bool = NATIVE_DEFAULT_SINE_REGULARIZER,
    sine_frequencies: Sequence[float] = NATIVE_DEFAULT_SINE_FREQUENCIES,
    sine_amplitudes: Sequence[float] = NATIVE_DEFAULT_SINE_AMPLITUDES,
    fir_hp_weight: float = NATIVE_DEFAULT_FIR_HP_WEIGHT,
    fir_hp_fc: float = NATIVE_DEFAULT_FIR_HP_FC,
) -> bool:
    """Trains a Neural Amp Modeler (NAM) Architecture 2 model using Allomorph's Native PyTorch Engine."""
    from allomorph.trainer.core import (
        find_sweep_input,
        get_hardware_device_name,
        resolve_hardware_batch_size,
        resolve_hardware_precision,
        setup_headless_environment,
    )

    setup_headless_environment()

    # 1. Resolve Source Instrument Configuration
    if isinstance(instrument, str) and ":" in instrument:
        inst_part, parsed_src_voicing = instrument.split(":", 1)
        if source_voicing is None:
            source_voicing = parsed_src_voicing
        instrument = inst_part

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

    # 2. Resolve Target Instrument & Voicing
    tgt_inst, tgt_v = resolve_target_voicing(voice, instrument=inst_cfg)
    tgt_voicing_id = tgt_v.id or (voice.split(":", 1)[1] if ":" in voice else voice)
    voice_name = tgt_v.name

    inst_id = inst_cfg.id
    inst_name = inst_cfg.name
    inst_ver = getattr(inst_cfg, "version", 1)
    voice_ver = getattr(tgt_v, "version", 1)

    # 3. Resolve Source Voicing
    if source_voicing is not None:
        if source_voicing not in inst_cfg.voicings:
            raise KeyError(
                f"Source voicing '{source_voicing}' not found on instrument '{inst_cfg.id}'. "
                f"Available voicings: {list(inst_cfg.voicings.keys())}"
            )
        src_voicing_id = source_voicing
    elif tgt_voicing_id in inst_cfg.voicings:
        src_voicing_id = tgt_voicing_id
    elif inst_cfg.voicings:
        src_voicing_id = next(iter(inst_cfg.voicings.keys()))
    else:
        src_voicing_id = "default"

    src_vcfg = inst_cfg.voicings.get(src_voicing_id)
    src_voicing_name = src_vcfg.name if src_vcfg is not None else src_voicing_id

    # 4. Resolve Version Tag and Basename
    tri_part = resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
    actual_version_tag = (
        tri_part
        if (version_tag == "auto" or version_tag is True)
        else (version_tag if version_tag not in (None, "none", False) else None)
    )

    model_basename = basename or (
        f"{tgt_voicing_id}_{actual_version_tag}" if actual_version_tag else tgt_voicing_id
    )

    # 5. Resolve Output Model Directory (3-level standalone hierarchy vs flat pack/nam/)
    models_path = Path(models_dir)
    if models_path.name == "nam" or (
        len(models_path.parts) >= 3
        and models_path.parts[-3] == inst_id
        and models_path.parts[-2] == src_voicing_id
        and models_path.parts[-1] == tgt_inst.id
    ):
        inst_models_dir = models_path
    elif models_path.name == inst_id:
        inst_models_dir = models_path / src_voicing_id / tgt_inst.id
    else:
        inst_models_dir = models_path / inst_id / src_voicing_id / tgt_inst.id
    inst_models_dir.mkdir(parents=True, exist_ok=True)
    target_nam = inst_models_dir / f"{model_basename}.nam"

    # 6. Resolve Audio Input / Output Stems
    input_path = find_sweep_input(input_wav)

    if not output_wav:
        target_wet = AUDIO_DIR / "stems" / tgt_inst.id / f"{tgt_voicing_id}.wav"
        target_recipe_token = compute_recipe_token(
            instrument=tgt_inst,
            voicing=tgt_v,
            base_dry_path=input_path,
        )
        if not (
            target_wet.exists()
            and is_stem_valid(
                target_wet,
                expected_recipe_token=target_recipe_token,
                base_dry_path=input_path,
            )
        ):
            try:
                from allomorph.circuit import simulate_instrument_voicing

                if not silent:
                    print(
                        f"[Native Trainer] Target stem missing or stale, auto-simulating: {target_wet.name}"
                    )
                output_path = simulate_instrument_voicing(
                    instrument=tgt_inst, voicing=tgt_v, output_wav=target_wet
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as e:
                if not silent:
                    print(
                        f"[Native Trainer] Warning: Failed to auto-simulate {target_wet.name}: {e}"
                    )
                output_path = target_wet
        else:
            output_path = target_wet
    else:
        output_path = Path(output_wav)

    if reference_wav:
        reference_path = Path(reference_wav)
    elif input_wav and Path(input_wav).exists():
        reference_path = Path(input_wav)
    else:
        candidate_source = AUDIO_DIR / "stems" / inst_id / f"{src_voicing_id}.wav"
        src_v = inst_cfg.voicings.get(src_voicing_id)
        if src_v is None:
            src_v = next(iter(inst_cfg.voicings.values()))
        src_recipe_token = compute_recipe_token(
            instrument=inst_cfg,
            voicing=src_v,
            base_dry_path=input_path,
        )
        if candidate_source.exists() and is_stem_valid(
            candidate_source,
            expected_recipe_token=src_recipe_token,
            base_dry_path=input_path,
        ):
            reference_path = candidate_source
        else:
            try:
                from allomorph.circuit.forward import simulate_instrument_voicing

                if not silent:
                    print(
                        f"[Native Trainer] Dedicated source stem missing or stale, auto-generating: {candidate_source.name}"
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
                if not silent:
                    print(
                        f"[Native Trainer] Warning: Failed to auto-generate source voicing stem: {e}. Falling back to input sweep."
                    )
                reference_path = input_path or (AUDIO_DIR / "input.wav")

    if not include_identity and reference_path.resolve() == output_path.resolve():
        if not silent:
            print(
                f"[Native Trainer] Skipping identity pair for '{voice}' on {inst_id}: "
                f"source and target audio stems are identical ({reference_path.name}). "
                "Use --include-identity to force training."
            )
        return True

    if not reference_path.exists() or not output_path.exists():
        print(f"Error: Missing audio:\n  Source: {reference_path}\n  Target: {output_path}")
        return False

    # 7. Seed Execution
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 8. Resolve Hardware Device & Parameters
    if device is not None:
        hw_device = torch.device(device)
    elif torch.cuda.is_available():
        hw_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        hw_device = torch.device("mps")
    else:
        hw_device = torch.device("cpu")

    resolved_bs = resolve_hardware_batch_size(batch_size)
    resolved_precision = resolve_hardware_precision(precision)
    device_name = get_hardware_device_name(engine="native")

    if not silent:
        print("\n=======================================================")
        print(" Allomorph Native PyTorch A2 Trainer")
        print("=======================================================")
        print(f"  Target Voicing: {voice_name} ({voice})")
        print(f"  Source Audio:   {reference_path.name}")
        print(f"  Target Audio:   {output_path.name}")
        print(f"  Output Model:   {target_nam.name}")
        print(f"  Hardware:       {device_name} ({hw_device})")
        print(f"  Batch Size:     {resolved_bs} | Precision: {resolved_precision}")
        print(f"  Schedule:       {lr_scheduler} (max_epochs={epochs}, warmup={min_epochs})")
        print(f"  Loss Config:    MSE={mse_weight:.2f}, MRSTFT={mrstft_weight:.4f}")
        print("-------------------------------------------------------", flush=True)

    # 9. In-VRAM Zero-Copy Audio Loading
    x_full, sr_x = load_audio_tensor(reference_path)
    y_full, sr_y = load_audio_tensor(output_path)

    if sr_x != sr_y:
        raise ValueError(f"Sample rate mismatch: source={sr_x}Hz, target={sr_y}Hz")
    sample_rate = float(sr_x)

    n_samples = min(len(x_full), len(y_full))
    x_full = x_full[:n_samples]
    y_full = y_full[:n_samples]

    val_samples = min(432_000, max(48000, int(0.10 * n_samples)))
    train_samples = n_samples - val_samples

    # 9a. Transfer directly to accelerator VRAM in native precision
    autocast_dtype: torch.dtype = (
        torch.bfloat16 if resolved_precision == "bf16-mixed" else torch.float16
    )
    can_autocast = False
    if resolved_precision in ("bf16-mixed", "16-mixed"):
        try:
            with torch.autocast(device_type=hw_device.type, dtype=autocast_dtype):
                pass
            can_autocast = True
        except RuntimeError, TypeError, ValueError, OSError:
            can_autocast = False

    use_autocast = can_autocast and hw_device.type in ("cuda", "mps", "xla", "cpu")
    vram_dtype = (
        torch.bfloat16
        if (resolved_precision == "bf16-mixed" and use_autocast and hw_device.type != "cpu")
        else (
            torch.float16
            if (resolved_precision == "16-mixed" and use_autocast and hw_device.type != "cpu")
            else torch.float32
        )
    )

    x_train = x_full[:train_samples].to(device=hw_device, dtype=vram_dtype)
    y_train = y_full[:train_samples].to(device=hw_device, dtype=vram_dtype)
    x_val = x_full[train_samples:].to(device=hw_device, dtype=vram_dtype)
    y_val = y_full[train_samples:].to(device=hw_device, dtype=vram_dtype)

    # 9b. Calibrate Target RMS Normalization (Upstream NAM Reference Standard: -18.0 dBFS)
    target_rms_linear = 10.0 ** (-18.0 / 20.0)
    train_sum_sq = float(torch.sum(torch.square(y_train.float())).item())
    train_numel = y_train.numel()
    train_rms = math.sqrt(train_sum_sq / train_numel) if train_numel > 0 else 0.0
    if train_rms > 0.0 and math.isfinite(train_rms):
        y_scale_factor = target_rms_linear / train_rms
    else:
        y_scale_factor = 1.0

    if y_scale_factor != 1.0:
        y_train = y_train * y_scale_factor
        y_val = y_val * y_scale_factor

    # 9c. Pre-Synthesize Linear Pure-Sine Probes & Tilted Log Chirps into Unified Training Buffer (Parseval Linearity Enforcement)
    if sine_regularizer and not fast_dev_run:
        try:
            from allomorph.circuit.forward import simulate_voicing_dsp
            from allomorph.dsp.sweeps import _synth_log_chirp

            tgt_inst_reg, tgt_v_reg = tgt_inst, tgt_v
            src_v_reg = inst_cfg.voicings.get(src_voicing_id) if inst_cfg.voicings else None

            # Calculate broadband calibration gain ratio between audio stems and unnormalized DSP simulations
            # to guarantee 100% gain coherence between audio dataset and synthetic probes
            c_ref = _synth_log_chirp(
                dur=10.0,
                f_start=20.0,
                f_end=14000.0,
                amp=0.20,
                sample_rate=int(sample_rate),
                string_tilt=True,
                f_corner=2000.0,
            )
            c_src_ref = (
                simulate_voicing_dsp(
                    c_ref,
                    inst_cfg,
                    src_v_reg,
                    apply_saturation=False,
                    normalize="none",
                    apply_dither=False,
                )
                if src_v_reg is not None
                else c_ref
            )
            c_tgt_ref = simulate_voicing_dsp(
                c_ref,
                tgt_inst_reg,
                tgt_v_reg,
                apply_saturation=False,
                normalize="none",
                apply_dither=False,
            )
            rms_x_full = float(
                torch.sqrt(torch.mean(torch.square(x_full[:train_samples].float()))).item()
            )
            rms_y_full = float(
                torch.sqrt(torch.mean(torch.square(y_full[:train_samples].float()))).item()
            )
            stem_gain_ratio = rms_y_full / max(rms_x_full, 1e-9)

            chirp_src_rms = float(np.sqrt(np.mean(c_src_ref**2)))
            chirp_tgt_rms = float(np.sqrt(np.mean(c_tgt_ref**2)))
            raw_dsp_gain_ratio = chirp_tgt_rms / max(chirp_src_rms, 1e-9)

            probe_gain_correction = stem_gain_ratio / max(raw_dsp_gain_ratio, 1e-9)

            # 9c.1 Multi-Amplitude Continuous Tilted Log Chirps (Dense High-Frequency Linear Supervision)
            chirp_xs: list[np.ndarray] = []
            chirp_ys: list[np.ndarray] = []
            for c_amp in (0.40, 0.20, 0.08):
                if abs(c_amp - 0.20) < 1e-6:
                    c_src = c_src_ref
                    c_out = c_tgt_ref.copy()
                else:
                    c_raw = _synth_log_chirp(
                        dur=10.0,
                        f_start=20.0,
                        f_end=14000.0,
                        amp=c_amp,
                        sample_rate=int(sample_rate),
                        string_tilt=True,
                        f_corner=2000.0,
                    )
                    c_src = (
                        simulate_voicing_dsp(
                            c_raw,
                            inst_cfg,
                            src_v_reg,
                            apply_saturation=False,
                            normalize="none",
                            apply_dither=False,
                        )
                        if src_v_reg is not None
                        else c_raw
                    )
                    c_out = simulate_voicing_dsp(
                        c_raw,
                        tgt_inst_reg,
                        tgt_v_reg,
                        apply_saturation=False,
                        normalize="none",
                        apply_dither=False,
                    )
                c_out = c_out * probe_gain_correction
                if y_scale_factor != 1.0:
                    c_out = c_out * y_scale_factor
                chirp_xs.append(c_src.astype(np.float32))
                chirp_ys.append(c_out.astype(np.float32))

            # 9c.2 Pure-Sine Probes with Physical Acoustic & RLC Filtering
            t_probe = np.arange(int(sample_rate * 1.0)) / sample_rate
            fade_len = int(sample_rate * 0.05)
            fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
            fade_out = fade_in[::-1]
            window = np.ones(len(t_probe), dtype=np.float32)
            window[:fade_len] = fade_in
            window[-fade_len:] = fade_out
            sine_xs: list[np.ndarray] = []
            sine_ys: list[np.ndarray] = []

            for f0 in sine_frequencies:
                for amp in sine_amplitudes:
                    s_in = (amp * np.sin(2.0 * np.pi * f0 * t_probe) * window).astype(np.float32)
                    s_src = (
                        simulate_voicing_dsp(
                            s_in,
                            inst_cfg,
                            src_v_reg,
                            apply_saturation=False,
                            normalize="none",
                            apply_dither=False,
                        )
                        if src_v_reg is not None
                        else s_in
                    )
                    s_out = simulate_voicing_dsp(
                        s_in,
                        tgt_inst_reg,
                        tgt_v_reg,
                        apply_saturation=False,
                        normalize="none",
                        apply_dither=False,
                    )
                    s_out = s_out * probe_gain_correction
                    if y_scale_factor != 1.0:
                        s_out = s_out * y_scale_factor
                    sine_xs.append(s_src.astype(np.float32))
                    sine_ys.append(s_out.astype(np.float32))

            concat_x = np.concatenate(chirp_xs + sine_xs)
            concat_y = np.concatenate(chirp_ys + sine_ys)
            x_sines_t = torch.from_numpy(concat_x).to(device=hw_device, dtype=vram_dtype)
            y_sines_t = torch.from_numpy(concat_y).to(device=hw_device, dtype=vram_dtype)

            x_train = torch.cat([x_train, x_sines_t])
            y_train = torch.cat([y_train, y_sines_t])
            if not silent:
                corr_db = 20.0 * math.log10(max(probe_gain_correction, 1e-9))
                print(
                    f"  Linearity Reg:  Enabled (3 chirps + {len(sine_frequencies)} freqs x {len(sine_amplitudes)} amps, gain align: {corr_db:+.2f} dB)",
                    flush=True,
                )
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, TypeError, OSError) as e:
            if not silent:
                print(f"  Linearity Reg:  Bypassed ({e})", flush=True)

    # Prepend receptive field padding (6346 samples) for exact sample-by-sample alignment
    rf_pad = 6346
    total_train_samples = len(x_train)
    x_train_padded = torch.cat([torch.zeros(rf_pad, device=hw_device, dtype=vram_dtype), x_train])
    x_val_padded = torch.cat([torch.zeros(rf_pad, device=hw_device, dtype=vram_dtype), x_val])

    # 10. Model & Loss Initialization
    model = AllomorphWaveNetA2().to(hw_device)
    model.apply_weight_masks()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=True)
    mrstft = MultiResolutionSTFTLoss().to(hw_device) if mrstft_weight > 0.0 else None
    hp_kernel: torch.Tensor | None = None
    if fir_hp_weight > 0.0:
        hp_kernel = design_sharp_fir_highpass(fc=fir_hp_fc, fs=float(sample_rate), num_taps=255).to(
            device=hw_device, dtype=vram_dtype
        )

    # 11. Slicing Parameters
    ny = slice_len
    nx = ny + rf_pad  # ny + 6346
    num_blocks = max(1, (total_train_samples - ny) // ny)
    steps_per_epoch = max(1, num_blocks // resolved_bs)
    if fast_dev_run:
        epochs = 1
        steps_per_epoch = 1

    # Pre-allocate slice index offset grids in accelerator VRAM as int32
    arange_nx = torch.arange(nx, device=hw_device, dtype=torch.int32).unsqueeze(0)
    arange_ny = torch.arange(ny, device=hw_device, dtype=torch.int32).unsqueeze(0)

    val_chunks = val_samples // ny
    val_bs = resolved_bs
    val_steps = max(1, (val_chunks + val_bs - 1) // val_bs)
    all_val_indices = (
        (torch.arange(val_chunks, device=hw_device, dtype=torch.int32) * ny).unsqueeze(1)
        if val_chunks > 0
        else None
    )

    # State tracking
    best_pareto_score = float("inf")
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_esr = float("inf")
    best_val_esr_lite = float("inf")
    best_val_mrstft: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None

    val_loss_history: list[float] = []
    hp_high_history: list[float] = []
    min_mse_full = float("inf")
    stop_reason: str | None = None
    cur_esr_full: float = float("inf")
    cur_esr_lite: float = float("inf")
    mb_val: float = float("inf")
    pareto_score: float = float("inf")
    metrics_full: dict[str, float] = {}

    t0 = time.time()

    # 12. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()

        # Learning Rate Schedule (5-epoch linear warmup + decay)
        warmup_epochs = 5
        target_lr = lr
        if epoch <= warmup_epochs:
            cur_lr = target_lr * (epoch / warmup_epochs)
        elif lr_scheduler == "cosine":
            effective_t_max = (
                max(1, epochs - warmup_epochs) if lr_t_max == DEFAULT_LR_T_MAX else lr_t_max
            )
            t_cur = min(epoch - warmup_epochs, effective_t_max)
            cur_lr = eta_min + 0.5 * (target_lr - eta_min) * (
                1.0 + math.cos(math.pi * t_cur / effective_t_max)
            )
        else:
            cur_lr = target_lr * (0.994 ** (epoch - warmup_epochs))

        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        train_loss_acc = torch.tensor(0.0, device=hw_device)

        max_start = max(1, total_train_samples - ny)
        if num_blocks >= resolved_bs and max_start >= ny:
            rem = max(1, max_start - (num_blocks - 1) * ny)
            jitter = int(torch.randint(0, rem, (1,), device=hw_device).item()) if rem > 1 else 0
            perm_blocks = torch.randperm(num_blocks, device=hw_device, dtype=torch.int32)
            epoch_block_offsets: torch.Tensor | None = perm_blocks * ny + jitter
        else:
            epoch_block_offsets = None

        for step in range(steps_per_epoch):
            if epoch_block_offsets is not None and (step + 1) * resolved_bs <= len(
                epoch_block_offsets
            ):
                indices = epoch_block_offsets[
                    step * resolved_bs : (step + 1) * resolved_bs
                ].unsqueeze(1)
            else:
                indices = torch.randint(
                    0,
                    max(1, total_train_samples - ny),
                    (resolved_bs, 1),
                    device=hw_device,
                    dtype=torch.int32,
                )
            x_batch = x_train_padded[indices + arange_nx].unsqueeze(1)
            y_batch = y_train[indices + arange_ny]

            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type=hw_device.type, dtype=autocast_dtype):
                    pred = model(x_batch)
                    l_full = F.mse_loss(pred[:, 1, :], y_batch)
                    l_lite = F.mse_loss(pred[:, 0, :], y_batch)
                    total_step_loss = mse_weight * (l_full + lite_weight * l_lite)
                    if hp_kernel is not None and fir_hp_weight > 0.0:
                        p_hp_full = F.conv1d(pred[:, 1:2, :], hp_kernel, padding="valid")
                        p_hp_lite = F.conv1d(pred[:, 0:1, :], hp_kernel, padding="valid")
                        t_hp = F.conv1d(y_batch.unsqueeze(1), hp_kernel, padding="valid")
                        l_hp_full = F.mse_loss(p_hp_full, t_hp)
                        l_hp_lite = F.mse_loss(p_hp_lite, t_hp)
                        total_step_loss = total_step_loss + fir_hp_weight * (
                            l_hp_full + lite_weight * l_hp_lite
                        )
                    if mrstft is not None:
                        l_spec = mrstft(pred[:, 1:2, :].float(), y_batch.unsqueeze(1).float())
                        total_step_loss = total_step_loss + mrstft_weight * l_spec
            else:
                pred = model(x_batch)
                l_full = F.mse_loss(pred[:, 1, :], y_batch)
                l_lite = F.mse_loss(pred[:, 0, :], y_batch)
                total_step_loss = mse_weight * (l_full + lite_weight * l_lite)
                if hp_kernel is not None and fir_hp_weight > 0.0:
                    p_hp_full = F.conv1d(pred[:, 1:2, :], hp_kernel, padding="valid")
                    p_hp_lite = F.conv1d(pred[:, 0:1, :], hp_kernel, padding="valid")
                    t_hp = F.conv1d(y_batch.unsqueeze(1), hp_kernel, padding="valid")
                    l_hp_full = F.mse_loss(p_hp_full, t_hp)
                    l_hp_lite = F.mse_loss(p_hp_lite, t_hp)
                    total_step_loss = total_step_loss + fir_hp_weight * (
                        l_hp_full + lite_weight * l_hp_lite
                    )
                if mrstft is not None:
                    l_spec = mrstft(pred[:, 1:2, :].float(), y_batch.unsqueeze(1).float())
                    total_step_loss = total_step_loss + mrstft_weight * l_spec

            total_step_loss.backward()
            model.apply_weight_masks()
            optimizer.step()
            model.apply_weight_masks()

            train_loss_acc.add_(total_step_loss.detach())

        train_loss_epoch = float((train_loss_acc / steps_per_epoch).item())

        # 13. Batched Full-Track Validation Pass
        is_val_epoch = (
            (epoch % val_interval == 0)
            or (epoch == 1)
            or (epoch == epochs)
            or (epoch >= min_epochs)
        )
        if is_val_epoch:
            model.eval()
            val_autocast = (
                torch.autocast(device_type=hw_device.type, dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )
            with torch.no_grad(), val_autocast:
                if val_chunks > 0 and all_val_indices is not None:
                    val_err_full_acc = torch.tensor(0.0, device=hw_device)
                    val_err_lite_acc = torch.tensor(0.0, device=hw_device)
                    val_tgt_acc = torch.tensor(0.0, device=hw_device)
                    val_mrstft_acc = torch.tensor(0.0, device=hw_device)

                    for s in range(val_steps):
                        bs = min(val_bs, val_chunks - s * val_bs)
                        if bs <= 0:
                            break
                        c_indices = all_val_indices[s * val_bs : s * val_bs + bs]
                        c_bx = x_val_padded[c_indices + arange_nx].unsqueeze(1)
                        c_by = y_val[c_indices + arange_ny]

                        c_pred = model(c_bx)
                        c_pred_full = c_pred[:, 1, :]
                        c_pred_lite = c_pred[:, 0, :]
                        c_err_full = c_pred_full - c_by
                        c_err_lite = c_pred_lite - c_by

                        val_err_full_acc.add_(torch.sum(c_err_full**2))
                        val_err_lite_acc.add_(torch.sum(c_err_lite**2))
                        val_tgt_acc.add_(torch.sum(c_by**2))
                        if mrstft is not None:
                            val_mrstft_acc.add_(
                                mrstft(c_pred_full.unsqueeze(1).float(), c_by.unsqueeze(1).float())
                                * bs
                            )

                    cur_esr_full = float((val_err_full_acc / (val_tgt_acc + 1e-12)).item())
                    cur_esr_lite = float((val_err_lite_acc / (val_tgt_acc + 1e-12)).item())
                    total_eval_samples = float(val_chunks * ny)
                    val_mse_full = float((val_err_full_acc / total_eval_samples).item())
                    mse_cur = val_mse_full
                    val_mrstft = (
                        float((val_mrstft_acc / val_chunks).item()) if mrstft is not None else 0.0
                    )
                    mb_val = mse_weight * mse_cur + mrstft_weight * val_mrstft
                    metrics_full = {"mse": mse_cur, "mrstft": val_mrstft, "composite_loss": mb_val}
                else:
                    y_val_hat = model(x_val_padded.unsqueeze(0))
                    y_full_pred = y_val_hat[0, 1, :val_samples]
                    y_lite_pred = y_val_hat[0, 0, :val_samples]
                    mse_cur = float(F.mse_loss(y_full_pred, y_val).item())
                    err_full = y_full_pred - y_val
                    err_lite = y_lite_pred - y_val
                    target_energy = float(torch.sum(y_val**2).item()) + 1e-12
                    cur_esr_full = float(torch.sum(err_full**2).item()) / target_energy
                    cur_esr_lite = float(torch.sum(err_lite**2).item()) / target_energy
                    val_mrstft = (
                        float(
                            mrstft(
                                y_full_pred.unsqueeze(0).unsqueeze(0).float(),
                                y_val.unsqueeze(0).unsqueeze(0).float(),
                            ).item()
                        )
                        if mrstft is not None
                        else 0.0
                    )
                    mb_val = mse_weight * mse_cur + mrstft_weight * val_mrstft
                    metrics_full = {"mse": mse_cur, "mrstft": val_mrstft, "composite_loss": mb_val}

            # 14. Anti-Drift Pareto Early Stopping Evaluation
            if epoch >= min_epochs:
                if min_mse_full == float("inf"):
                    min_mse_full = mse_cur
                else:
                    min_mse_full = min(min_mse_full, mse_cur)

            if min_mse_full < float("inf") and min_mse_full > 0.0:
                mse_drift = max(0.0, (mse_cur - min_mse_full) / min_mse_full)
            else:
                mse_drift = 0.0

            val_loss_history.append(mb_val)
            hp_metric = metrics_full.get("mrstft", 0.0)
            hp_high_history.append(hp_metric)

            if min_mse_full < float("inf"):
                pareto_score = compute_pareto_quality_score(
                    mb_full_loss=mb_val,
                    mse_full=mse_cur,
                    min_mse_full=min_mse_full,
                    drift_tolerance=1.0,
                    drift_penalty_gain=1.0,
                )
            else:
                pareto_score = mb_val

            # Track global best Pareto checkpoint post-warmup
            if epoch >= warmup_epochs:
                is_new_best = pareto_score < best_pareto_score
                if is_new_best or best_pareto_score == float("inf"):
                    best_pareto_score = pareto_score
                    best_epoch = epoch
                    best_val_loss = mb_val
                    best_val_esr = cur_esr_full
                    best_val_esr_lite = cur_esr_lite
                    best_val_mrstft = metrics_full.get("mrstft") if mrstft is not None else None
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Compute trajectory slopes
            k_eval = 4
            if len(val_loss_history) >= k_eval:
                x_ax = np.arange(k_eval)
                val_slope = float(np.polyfit(x_ax, val_loss_history[-k_eval:], 1)[0])
                hp_slope = float(np.polyfit(x_ax, hp_high_history[-k_eval:], 1)[0])
            else:
                val_slope = 0.0
                hp_slope = 0.0

            if epoch >= min_epochs:
                should_stop, reason = evaluate_plateau_stopping_condition(
                    val_loss_history=val_loss_history,
                    val_loss_slope=val_slope,
                    epoch=epoch,
                    warmup_floor=min_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    hp_high_slope=hp_slope,
                    mse_drift_ratio=mse_drift,
                    drift_exit_threshold=1.50,
                )
            else:
                should_stop, reason = False, None
        else:
            should_stop, reason = False, None

        epoch_sec = time.time() - epoch_t0
        esr_full_db = 10.0 * math.log10(max(cur_esr_full, 1e-12))
        esr_lite_db = 10.0 * math.log10(max(cur_esr_lite, 1e-12))

        if not silent:
            is_best_str = " *" if epoch == best_epoch else ""
            if is_val_epoch:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.2f}s] "
                    f"Train: {train_loss_epoch:.4f} | "
                    f"Val Loss: {mb_val:.5f} (Q: {pareto_score:.5f}){is_best_str} | "
                    f"ESR Ch8: {cur_esr_full:.6f} ({esr_full_db:+.1f}dB) | "
                    f"ESR Ch3: {cur_esr_lite:.6f} ({esr_lite_db:+.1f}dB)",
                    flush=True,
                )
            else:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.2f}s] "
                    f"Train: {train_loss_epoch:.4f}",
                    flush=True,
                )

        if should_stop:
            stop_reason = reason
            if not silent:
                print(
                    f"\n[Native Trainer] Early stopping triggered at epoch {epoch:03d} "
                    f"({stop_reason}).",
                    flush=True,
                )
            break

    total_time = time.time() - t0

    # 15. Restore Best Pareto Checkpoint
    if best_state_dict is not None and best_epoch > 0:
        if not silent:
            print(
                f"\n[Native Trainer] Restoring optimal Pareto checkpoint from epoch {best_epoch:03d} "
                f"(Val Loss: {best_val_loss:.5f}, Pareto Q: {best_pareto_score:.5f})"
            )
        model.load_state_dict(best_state_dict)
    else:
        best_epoch = epoch
        best_val_loss = mb_val
        best_val_esr = cur_esr_full
        best_val_esr_lite = cur_esr_lite
        best_val_mrstft = metrics_full.get("mrstft") if mrstft is not None else None

    # 16. Export Compliant A2 .nam Model with Full Provenance Metadata
    if src_vcfg is not None:
        src_coils = resolve_voicing_electroacoustics(inst_cfg, src_vcfg).coils
        src_eff_pos_m = compute_effective_position(src_coils) if src_coils else 0.0
        src_eff_pos_mm = src_eff_pos_m * 1000.0
        src_tone_name = src_vcfg.tone_name
        src_harness = src_vcfg.harness
        src_controls: dict[str, float] = dict(src_vcfg.controls)
        src_switches: dict[str, str] = dict(src_vcfg.switches)
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
        src_tone_name = None
        src_harness = ""
        src_controls = {}
        src_switches = {}
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

    tgt_ea = resolve_voicing_electroacoustics(tgt_inst, tgt_v)
    target_coils = tgt_ea.coils
    target_eff_pos_m = compute_effective_position(target_coils) if target_coils else 0.0
    target_eff_pos_mm = target_eff_pos_m * 1000.0

    epochs_trained_val = int(best_epoch if best_epoch > 0 else epoch)
    raw_training_meta: dict[str, Any] = {
        "validation_spectral_loss": float(best_val_loss),
        "validation_spectral_loss_a2_full": float(best_val_loss),
        "validation_spectral_loss_a2_lite": float(best_val_esr_lite),
        "validation_esr": float(best_val_esr),
        "validation_esr_a2_full": float(best_val_esr),
        "validation_esr_a2_lite": float(best_val_esr_lite),
        "validation_esr_ch8": float(best_val_esr),
        "validation_esr_ch3": float(best_val_esr_lite),
        "epochs_trained": epochs_trained_val,
        "epochs": epochs,
        "lr": lr,
        "batch_size": resolved_bs,
        "precision": resolved_precision,
        "device_name": device_name,
        "engine": "native",
        "training_time_seconds": round(total_time, 2),
    }
    if stop_reason:
        raw_training_meta["stop_reason"] = stop_reason
    if best_val_mrstft is not None:
        raw_training_meta["mrstft_loss"] = float(best_val_mrstft)

    nam_meta = NamExportMetadata(
        training=NamTrainingMetadata.model_validate(raw_training_meta),
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
            scale_length_in=inst_cfg.scale_length_in or 34.0,
            scale_length_m=inst_cfg.scale_length_m,
            string_wave_speeds=inst_cfg.string_wave_speeds,
            voicing=src_voicing_meta,
        ),
        target_voicing=NamTargetVoicingMeta(
            id=tgt_voicing_id,
            name=voice_name,
            tone_name=tgt_v.tone_name,
            sensor_type="magnetic",
            resonant_frequency_hz=tgt_ea.fr,
            q_factor=tgt_ea.q,
            effective_position_m=target_eff_pos_m,
            effective_position_mm=target_eff_pos_mm,
            coils=target_coils,
        ),
    )

    model_title = f"{voice_name} [{inst_name}]"
    export_metadata: dict[str, Any] = {
        "name": model_title,
        "modeled_by": "Allomorph (Peter Nguyen <peter@phn.dev>)",
        "gear_make": inst_name,
        "gear_model": f"{src_voicing_name} -> {voice_name}",
        "gear_type": "preamp",
        "tone_type": "clean",
        "model_name": model_basename,
        "instrument_id": inst_id,
        "instrument_name": inst_name,
        "validation_spectral_loss": best_val_loss,
        "validation_esr": best_val_esr,
        "validation_esr_a2_full": best_val_esr,
        "validation_esr_a2_lite": best_val_esr_lite,
        "epochs_trained": epochs_trained_val,
        "training_time_seconds": round(total_time, 2),
        **nam_meta.model_dump(),
    }
    if stop_reason:
        export_metadata["stop_reason"] = stop_reason
    if best_val_mrstft is not None:
        export_metadata["validation_mrstft"] = best_val_mrstft

    # Rescale head_scale to undo target RMS normalization
    if y_scale_factor != 1.0:
        model.head_scale = model.head_scale * (1.0 / y_scale_factor)

    model.export_nam(
        out_path=target_nam,
        sample_rate=sample_rate,
        metadata=export_metadata,
    )

    if not no_manifest:
        updated_pack_manifest = False
        manifest_candidates = [
            target_nam.parent.parent / "manifest.json",
            target_nam.parent / "manifest.json",
        ]
        for m_path in manifest_candidates:
            if m_path.exists():
                try:
                    with open(m_path, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    if (
                        isinstance(m_data, dict)
                        and "models" in m_data
                        and isinstance(m_data["models"], dict)
                    ):
                        m_entry = m_data["models"].setdefault(target_nam.name, {})
                        m_entry.setdefault("filename", target_nam.name)
                        m_entry["sha256"] = compute_file_sha256(target_nam)
                        m_entry["dry_file"] = reference_path.name
                        m_entry["dry_sha256"] = compute_file_sha256(reference_path)
                        m_entry["wet_file"] = output_path.name
                        m_entry["wet_sha256"] = compute_file_sha256(output_path)
                        m_entry["validation_esr"] = float(best_val_esr)
                        m_entry["validation_esr_a2_full"] = float(best_val_esr)
                        m_entry["validation_esr_a2_lite"] = float(best_val_esr_lite)
                        m_entry["validation_esr_ch8"] = float(best_val_esr)
                        m_entry["validation_esr_ch3"] = float(best_val_esr_lite)
                        m_entry["epochs_trained"] = epochs_trained_val
                        m_entry["training_time_seconds"] = round(total_time, 2)
                        m_entry["batch_size"] = resolved_bs
                        m_entry["precision"] = resolved_precision
                        m_entry["engine"] = "native"
                        m_entry["device_name"] = (
                            torch.cuda.get_device_name(hw_device)
                            if hw_device.type == "cuda"
                            else str(hw_device)
                        )
                        if stop_reason:
                            m_entry["stop_reason"] = stop_reason
                        if best_val_mrstft is not None:
                            m_entry["mrstft_loss"] = float(best_val_mrstft)
                        with open(m_path, "w", encoding="utf-8") as mf:
                            json.dump(m_data, mf, indent=2)
                        updated_pack_manifest = True
                        break
                except OSError, json.JSONDecodeError:
                    pass

        if not updated_pack_manifest and target_nam.exists():
            write_manifest(
                output_dir=inst_models_dir,
                stage="train",
                files=[target_nam],
                version_tag=tri_part,
            )

    if not silent:
        full_db = 10.0 * math.log10(max(best_val_esr, 1e-12))
        lite_db = 10.0 * math.log10(max(best_val_esr_lite, 1e-12))
        print("\n=======================================================")
        print(f" Native Training Complete in {total_time:.2f}s ({total_time / 60.0:.1f}m)")
        print("=======================================================")
        print(f"  Exported File:  {target_nam.name}")
        print(f"  Best Epoch:     {best_epoch:03d} / {epoch:03d}")
        print(f"  Validation ESR: {best_val_esr:.6f} (A2 Full Ch8, {full_db:+.2f} dB)")
        print(f"  Lite Mode ESR:  {best_val_esr_lite:.6f} (A2 Lite Ch3, {lite_db:+.2f} dB)")
        if stop_reason:
            print(f"  Stop Condition: {stop_reason}")
        print("  Status: Ready for Darkglass Anagram Block 1 loading.")
        print("=======================================================\n")

    return True
