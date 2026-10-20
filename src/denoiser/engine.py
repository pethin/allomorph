"""In-VRAM Native PyTorch WaveNet A2 Trainer, Evaluator, and Audio Processor for the Allomorph Denoiser."""

from __future__ import annotations

import contextlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from allomorph.trainer.core import (
    get_hardware_device_name,
    resolve_hardware_batch_size,
    resolve_hardware_precision,
    setup_headless_environment,
)

setup_headless_environment()

import numpy as np
import torch

from allomorph.dsp.constants import CALIBRATION_PEAK_CEILING
from allomorph.dsp.io import read_wav, write_wav_24bit
from allomorph.trainer.callbacks import (
    compute_linear_slope,
    compute_pareto_quality_score,
    evaluate_loss_spike_condition,
    evaluate_plateau_stopping_condition,
)
from allomorph.trainer.constants import DEFAULT_ETA_MIN, DEFAULT_LR_T_MAX
from allomorph.trainer.native.models import (
    AllomorphWaveNetA2,
    apply_model_io_scale_compensation,
    select_warm_start_offset,
)
from denoiser.constants import (
    DEFAULT_DENOISER_OUTPUT_DIR,
    DEFAULT_EXCITATION_WAV,
    DENOISER_DEFAULT_EMA_DECAY,
    DENOISER_DEFAULT_FIR_HP_HIGH_FC,
    DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT,
    DENOISER_DEFAULT_FIR_HP_MID_FC,
    DENOISER_DEFAULT_FIR_HP_MID_WEIGHT,
    DENOISER_DEFAULT_HOMOTOPY_TAU_INIT,
    DENOISER_DEFAULT_LITE_WEIGHT,
    DENOISER_DEFAULT_LR,
    DENOISER_DEFAULT_MAX_EPOCHS,
    DENOISER_DEFAULT_MAX_SPIKE_RESTARTS,
    DENOISER_DEFAULT_MIN_DELTA,
    DENOISER_DEFAULT_MIN_EPOCHS,
    DENOISER_DEFAULT_MSE_WEIGHT,
    DENOISER_DEFAULT_PATIENCE,
    DENOISER_DEFAULT_SLICE_LEN,
    DENOISER_DEFAULT_SPIKE_RATIO_THRESHOLD,
    DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    DENOISER_DEFAULT_VAL_INTERVAL,
    DENOISER_DEFAULT_WEIGHT_DECAY,
    DENOISER_FIR_NUM_TAPS,
    DENOISER_GEAR_MAKE,
    DENOISER_GEAR_TYPE,
    DENOISER_MODELED_BY,
    DENOISER_PROFILES,
    DENOISER_TONE_TYPE,
    MAINS_DISPLAY_MAP,
    MAINS_SLUG_MAP,
    MainsMode,
    ProfileName,
    resolve_denoiser_display_name,
    resolve_denoiser_gear_model,
    resolve_denoiser_model_stem,
    resolve_mains_targets,
    resolve_profile_targets,
)
from denoiser.loss import DenoiserLoss
from denoiser.noise import (
    StochasticNoiseGenerator,
    build_clean_bass_corpus,
    synthesize_mains_hum_bank,
    synthesize_smps_and_hiss_banks,
)


def _resolve_torch_device(device_name: str) -> torch.device:
    """Resolves a CLI device string ('auto', 'cuda', 'mps', 'cpu') into a `torch.device`."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def generate_denoiser_storefront_description(
    output_dir: Path,
    trained_metadata: list[dict[str, Any]] | None = None,
) -> Path:
    """Generates `storefront_description.txt` (<= 10,000 chars) for the Faraday Tone3000 Pack."""
    output_dir.mkdir(parents=True, exist_ok=True)
    desc_path = output_dir / "storefront_description.txt"

    lines: list[str] = [
        "FARADAY: UNIVERSAL BASS DENOISER & HUM ELIMINATOR",
        "",
        "Studio- and stage-grade 50/60 Hz single-coil hum, dimmer buzz, and power-supply whine",
        "eliminator for any passive or active bass guitar.",
        "",
        "",
        "OVERVIEW",
        "",
        "Faraday is a pickup-agnostic Neural Amp Modeler Architecture 2 (.nam) utility pedal pack",
        "engineered for the Darkglass Anagram (Block 1) and desktop NAM plugins. Like a virtual",
        "copper Faraday shield wrapped around your instrument's electronics, Faraday eliminates:",
        "• 50 Hz and 60 Hz electromagnetic mains hum induced in single-coil Jazz, P90, and soapbar pickups",
        "• Upper-harmonic rectifier and stage LED/TRIAC dimmer buzz across the midrange",
        "• High-frequency switching power supply (SMPS), daisy-chain, and digital clock whine (2.5 kHz - 22 kHz)",
        "",
        "Unlike traditional hard noise gates that chop off decaying notes or notch filters that hollow",
        "out your fundamental low end, Faraday preserves 1:1 unity gain, pick and finger attack",
        "transients, and full-bodied sub-bass weight (20 Hz - 12 kHz) across any bass guitar.",
        "",
        "",
        "RECOMMENDED SIGNAL CHAIN",
        "",
        "[Any Passive or Active Bass]",
        "  -> [Block 1: Faraday Denoiser]     (Input / Front-End Noise Shield)",
        "  -> [Block 2: Compressor / Drive]   (Clean Sustain & Unmuddied Distortion)",
        "  -> [Block 3: Amp & Cab Block]      (SVT, B-15, Super Bassman, Modern Rig)",
        "  -> [FOH / Audio Interface / DAW]",
        "",
        "Why Block 1? Placing Faraday at the very beginning of your signal chain strips out 50/60 Hz",
        "buzz and ultrasonic switching noise before it hits your compressor, overdrive pedals, or",
        "high-gain amp blocks—preventing downstream gain stages from amplifying and intermodulating",
        "your noise floor.",
        "",
        "",
        "INCLUDED PROFILES & MAINS CONFIGURATIONS",
        "",
        "• Universal Gentle (-50 dBFS Gate Knee) [universal_gentle.nam]",
        "  Ultra-transparent studio profile for well-shielded humbuckers/split-coils, fretless bass,",
        "  and exposed solo passages requiring long, natural sustain tails.",
        "",
        "• Universal Balanced (-44 dBFS Gate Knee) [universal_balanced.nam — Recommended Default]",
        "  Everyday live and studio profile for P-Bass, Jazz Bass, StingRay, and active soapbars.",
        "  Balances deep 50/60 Hz hum and SMPS whine suppression with smooth sustain decay.",
        "",
        "• Universal Aggressive (-36 dBFS Gate Knee) [universal_aggressive.nam]",
        "  Maximum-suppression stage profile for unshielded vintage single-coils, dirty venue power,",
        "  fluorescent/LED stage dimmers, unisolated 9V power supplies, and high-gain distortion rigs.",
        "",
        "(Regional 60Hz and 50Hz dedicated grid variants follow the same Gentle / Balanced / Aggressive",
        "profiles when included in the pack.)",
        "",
        "",
        "HOW FARADAY WORKS",
        "",
        "• Dual-Tier WaveNet Architecture 2: Every .nam file embeds both A2 Standard (8-channel) and",
        "  A2 Lite (3-channel) submodels for flexible DSP allocation.",
        "• Smooth Envelope-Gated Suppression: Order-4 continuous rational gating around the knee",
        "  eliminates hard gate chatter, pumping, or sputter on decaying bass notes.",
        "• Musical Low-End Protection: Anchored on multi-voice clean bass spectra and high-level",
        "  50/60/100/120 Hz bass fundamentals so played E1/A1/B1 notes pass at 1:1 unity gain.",
        "• 14 kHz Ultrasonic Low-Pass Conditioning: 255-tap linear-phase Blackman conditioning",
        "  permanently blocks 14 kHz - 24 kHz SMPS switching carriers even while loud notes sustain.",
    ]

    if trained_metadata:
        lines.extend(
            [
                "",
                "",
                "PACK MODEL SUMMARY",
                "",
            ]
        )
        for item in trained_metadata:
            name = item.get("name", "Faraday")
            fname = item.get("nam_file", "")
            thresh = item.get("threshold_dbfs", -44.0)
            supp_db = item.get("silence_suppression_db_ch8", 0.0)
            esr = item.get("validation_esr_ch8", 0.0)
            lines.append(
                f"• {name} ({fname}): Gate Knee {thresh:.1f} dBFS | "
                f"Idle Noise Suppression {supp_db:+.1f} dB | Passband ESR {esr:.5f}"
            )

    text = "\n".join(lines).strip() + "\n"
    if len(text) > 10000:
        text = text[:9996] + "...\n"
    desc_path.write_text(text, encoding="utf-8")
    return desc_path


def train_single_denoiser_model(
    *,
    clean_corpus: np.ndarray,
    sample_rate: int,
    output_nam: Path,
    mains: MainsMode = "both",
    profile: ProfileName = "balanced",
    threshold_dbfs: float | None = None,
    noise_wav: Path | None = None,
    epochs: int = DENOISER_DEFAULT_MAX_EPOCHS,
    min_epochs: int = DENOISER_DEFAULT_MIN_EPOCHS,
    patience: int = DENOISER_DEFAULT_PATIENCE,
    min_delta: float = DENOISER_DEFAULT_MIN_DELTA,
    batch_size: int | str = "auto",
    precision: str = "auto",
    lr: float = DENOISER_DEFAULT_LR,
    weight_decay: float = DENOISER_DEFAULT_WEIGHT_DECAY,
    mse_weight: float = DENOISER_DEFAULT_MSE_WEIGHT,
    lite_weight: float = DENOISER_DEFAULT_LITE_WEIGHT,
    fir_hp_mid_weight: float = DENOISER_DEFAULT_FIR_HP_MID_WEIGHT,
    fir_hp_high_weight: float = DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT,
    fir_hp_mid_fc: float = DENOISER_DEFAULT_FIR_HP_MID_FC,
    fir_hp_high_fc: float = DENOISER_DEFAULT_FIR_HP_HIGH_FC,
    ultrasonic_lpf_fc: float = DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    val_interval: int = DENOISER_DEFAULT_VAL_INTERVAL,
    ema_decay: float = DENOISER_DEFAULT_EMA_DECAY,
    homotopy_tau_init: float = DENOISER_DEFAULT_HOMOTOPY_TAU_INIT,
    spike_ratio_threshold: float = DENOISER_DEFAULT_SPIKE_RATIO_THRESHOLD,
    max_spike_restarts: int = DENOISER_DEFAULT_MAX_SPIKE_RESTARTS,
    device_name: str = "auto",
    model_name: str | None = None,
    seed: int | None = 42,
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    """Trains a single dual-tier WaveNet A2 denoiser model on a pre-conditioned clean bass corpus."""
    profile_spec = DENOISER_PROFILES[profile]
    effective_threshold_dbfs = (
        float(threshold_dbfs) if threshold_dbfs is not None else profile_spec.threshold_dbfs
    )
    display_name = model_name or resolve_denoiser_display_name(mains, profile)
    gear_model = resolve_denoiser_gear_model(mains, profile)
    mains_slug = MAINS_SLUG_MAP[mains]

    hw_device = _resolve_torch_device(device_name)
    resolved_bs = resolve_hardware_batch_size(batch_size)
    resolved_precision = (
        resolve_hardware_precision(precision) if hw_device.type != "cpu" else "32-true"
    )
    hw_label = get_hardware_device_name(engine="native")

    print("=======================================================", flush=True)
    print(" Allomorph Universal Bass Denoiser Native A2 Trainer", flush=True)
    print("=======================================================", flush=True)
    print(f"  Model Name:     {display_name} ({output_nam.name})", flush=True)
    print(
        f"  Configuration:  Mains={MAINS_DISPLAY_MAP[mains]} ({mains}) | Profile={profile_spec.label} ({profile})",
        flush=True,
    )
    print(
        f"  Gate Knee:      {effective_threshold_dbfs:.1f} dBFS | Silence Weight={profile_spec.silence_weight:.0f}x (order={profile_spec.envelope_order})",
        flush=True,
    )
    print(
        f"  Ultrasonic LPF: {ultrasonic_lpf_fc:.0f} Hz (255-tap Blackman zero-phase)",
        flush=True,
    )
    if noise_wav is not None:
        print(f"  Hardware Noise: {noise_wav}", flush=True)
    print(f"  Output Model:   {output_nam}", flush=True)
    print(f"  Hardware:       {hw_label} ({hw_device.type})", flush=True)
    print(f"  Batch Size:     {resolved_bs} | Precision: {resolved_precision}", flush=True)
    print(f"  Schedule:       cosine (max_epochs={epochs}, warmup={min_epochs})", flush=True)
    print(
        f"  FIR HP Loss:    Mid={fir_hp_mid_weight}@{fir_hp_mid_fc:.0f}Hz | High={fir_hp_high_weight}@{fir_hp_high_fc:.0f}Hz",
        flush=True,
    )
    print("-------------------------------------------------------", flush=True)

    n_samples = len(clean_corpus)
    if n_samples < 65536:
        raise ValueError(f"Clean bass corpus too short for training ({n_samples} samples).")

    # Normalize clean corpus peak to CALIBRATION_PEAK_CEILING (-1.0 dBFS) with 1:1 I/O unity scale
    peak_val = float(np.max(np.abs(clean_corpus)))
    if peak_val > 1e-9:
        corpus_scale = float(CALIBRATION_PEAK_CEILING / peak_val)
    else:
        corpus_scale = 1.0
    clean_norm = (clean_corpus * np.float32(corpus_scale)).astype(np.float32)

    # Reversible I/O scale factor (1.0x unity so training dBFS == hardware dBFS)
    io_scale_factor = 1.0

    val_samples = min(432_000, max(48000, int(0.10 * n_samples)))
    train_samples = n_samples - val_samples

    x_train_clean_np = clean_norm[:train_samples]
    x_val_clean_np = clean_norm[train_samples:n_samples]

    # Select highest-energy active broadband window for Step-0 Wiener-Hopf identity warm-start
    cal_len = min(32768, train_samples)
    cal_offset = select_warm_start_offset(x_train_clean_np, window_len=cal_len)

    # Resolve accelerator autocast and VRAM buffer dtype
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

    y_train_clean = torch.from_numpy(x_train_clean_np).to(device=hw_device, dtype=vram_dtype)
    x_val_clean = torch.from_numpy(x_val_clean_np).to(device=hw_device, dtype=vram_dtype)

    rf_pad = 6346  # WaveNet A2 receptive field padding (samples)
    ny = DENOISER_DEFAULT_SLICE_LEN  # 8,192 samples
    nx = ny + rf_pad  # 14,538 samples

    x_train_clean_padded = torch.cat(
        [torch.zeros(rf_pad, device=hw_device, dtype=vram_dtype), y_train_clean]
    )

    arange_nx = torch.arange(nx, device=hw_device, dtype=torch.int32).unsqueeze(0)
    arange_ny = torch.arange(ny, device=hw_device, dtype=torch.int32).unsqueeze(0)

    # Build in-VRAM StochasticNoiseGenerator & deterministic validation pair
    base_seed = 42 if seed is None else int(seed)
    noise_gen = StochasticNoiseGenerator(
        n_samples=train_samples,
        mains=mains,
        profile=profile,
        device=hw_device,
        dtype=vram_dtype,
        sample_rate=sample_rate,
        threshold_dbfs_override=effective_threshold_dbfs,
        noise_wav=noise_wav,
        seed=base_seed,
    )
    x_val_noisy_padded, y_val_clean = noise_gen.build_deterministic_validation_pair(
        x_val_clean, rf_pad=rf_pad
    )

    num_blocks = max(1, (train_samples - ny) // ny)
    steps_per_epoch = max(1, num_blocks // resolved_bs)
    if fast_dev_run:
        epochs = 1
        min_epochs = 1
        steps_per_epoch = 1

    val_chunks = val_samples // ny
    val_bs = resolved_bs
    val_steps = max(1, (val_chunks + val_bs - 1) // val_bs)
    all_val_indices = (
        (torch.arange(val_chunks, device=hw_device, dtype=torch.int32) * ny).unsqueeze(1)
        if val_chunks > 0
        else None
    )

    criterion = DenoiserLoss(
        threshold_dbfs=effective_threshold_dbfs,
        silence_weight=profile_spec.silence_weight,
        envelope_order=profile_spec.envelope_order,
        mse_weight=mse_weight,
        lite_weight=lite_weight,
        fir_hp_mid_fc=fir_hp_mid_fc,
        fir_hp_high_fc=fir_hp_high_fc,
        fir_hp_mid_weight=fir_hp_mid_weight,
        fir_hp_high_weight=fir_hp_high_weight,
        fs=float(sample_rate),
        num_taps=DENOISER_FIR_NUM_TAPS,
        device=hw_device,
        dtype=vram_dtype,
    )

    if seed is not None:
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)

    start_train_time = time.time()
    warmup_epochs = 5
    homotopy_end_epoch = max(1, int(0.75 * min_epochs))

    max_attempts = 1 + max(0, int(max_spike_restarts))
    spike_restarts = 0

    model = AllomorphWaveNetA2().to(hw_device)
    best_pareto_score = float("inf")
    best_val_loss = float("inf")
    best_val_esr = float("inf")
    best_val_esr_lite = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    stop_reason: str | None = None
    cur_esr_full = float("inf")
    cur_esr_lite = float("inf")
    mb_val = float("inf")
    pareto_score = float("inf")
    epoch = 1

    for attempt in range(max_attempts):
        if attempt > 0:
            attempt_seed = base_seed + attempt * 10007
            torch.manual_seed(attempt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(attempt_seed)

        # Initialize Model & Closed-Form Identity Wiener Warm-Start (y_cal == x_cal)
        model = AllomorphWaveNetA2().to(hw_device)
        model.mask_weights()

        if train_samples >= 1024:
            model.warm_start_head(
                x_train_clean_padded[cal_offset : cal_offset + cal_len + rf_pad].float(),
                y_train_clean[cal_offset : cal_offset + cal_len].float(),
            )

        model_params: list[torch.Tensor] = list(model.parameters())
        ema_params: list[torch.Tensor] = (
            [p.detach().clone() for p in model_params] if ema_decay > 0.0 else []
        )
        global_step: int = 0

        attempt_lr = lr * (0.85**attempt)
        use_foreach = hw_device.type in ("cuda", "mps")
        optimizer = torch.optim.Adam(
            model_params,
            lr=attempt_lr,
            weight_decay=weight_decay,
            foreach=use_foreach,
        )

        best_pareto_score = float("inf")
        best_val_loss = float("inf")
        best_val_esr = float("inf")
        best_val_esr_lite = float("inf")
        best_epoch = 0
        best_state_dict = None

        val_loss_history: list[float] = []
        hp_high_history: list[float] = []
        min_mse_full = float("inf")
        min_train_loss = float("inf")
        min_val_loss_seen = float("inf")
        stop_reason = None
        cur_esr_full = float("inf")
        cur_esr_lite = float("inf")
        mb_val = float("inf")
        pareto_score = float("inf")
        restarted_on_spike = False

        for epoch in range(1, epochs + 1):
            epoch_t0 = time.time()
            model.train()

            # C-infinity SiLU-gated activation homotopy schedule (tau: 0.02 -> 0.0)
            if homotopy_tau_init > 0.0 and epoch <= homotopy_end_epoch and not fast_dev_run:
                prog = (epoch - 1) / max(1, homotopy_end_epoch)
                model.activation_tau = 0.5 * homotopy_tau_init * (1.0 + math.cos(math.pi * prog))
            else:
                model.activation_tau = 0.0

            # Learning Rate Schedule (5-epoch linear warmup + cosine decay)
            if epoch <= warmup_epochs:
                cur_lr = attempt_lr * (epoch / warmup_epochs)
            else:
                effective_t_max = max(1, min(epochs - warmup_epochs, DEFAULT_LR_T_MAX))
                t_cur = min(epoch - warmup_epochs, effective_t_max)
                cur_lr = DEFAULT_ETA_MIN + 0.5 * (attempt_lr - DEFAULT_ETA_MIN) * (
                    1.0 + math.cos(math.pi * t_cur / effective_t_max)
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            train_loss_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
            max_start = max(1, train_samples - ny)

            if num_blocks >= resolved_bs and max_start >= ny:
                rem = max(1, max_start - (num_blocks - 1) * ny)
                jitter = int(torch.randint(0, rem, (1,), device=hw_device).item()) if rem > 1 else 0
                perm_blocks = torch.randperm(num_blocks, device=hw_device, dtype=torch.int32)
                epoch_block_offsets: torch.Tensor | None = perm_blocks * ny + jitter
            else:
                epoch_block_offsets = None

            train_autocast = (
                torch.autocast(device_type=hw_device.type, dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )

            for step in range(1, steps_per_epoch + 1):
                if epoch_block_offsets is not None and step * resolved_bs <= len(
                    epoch_block_offsets
                ):
                    clean_indices = epoch_block_offsets[
                        (step - 1) * resolved_bs : step * resolved_bs
                    ].unsqueeze(1)
                else:
                    clean_indices = torch.randint(
                        0,
                        max_start,
                        (resolved_bs, 1),
                        device=hw_device,
                        dtype=torch.int32,
                    )

                x_noisy_batch, y_clean_batch = noise_gen.sample_training_batch(
                    x_clean_padded=x_train_clean_padded,
                    y_clean=y_train_clean,
                    clean_indices=clean_indices,
                    arange_nx=arange_nx,
                    arange_ny=arange_ny,
                    rf_pad=rf_pad,
                )

                optimizer.zero_grad(set_to_none=True)

                with train_autocast:
                    pred = model(x_noisy_batch)
                    total_step_loss, _, _ = criterion(pred, y_clean_batch)

                total_step_loss.backward()
                model.mask_gradients()
                optimizer.step()
                model.mask_weights()

                global_step += 1
                if ema_decay > 0.0 and ema_params:
                    step_ema_decay = min(ema_decay, (1.0 + global_step) / (10.0 + global_step))
                    with torch.no_grad():
                        torch._foreach_lerp_(ema_params, model_params, 1.0 - step_ema_decay)

                train_loss_acc.add_(total_step_loss.detach())

            avg_train_loss = float((train_loss_acc / steps_per_epoch).item())

            # Pre-validation unlucky training spike check (CL-37)
            is_train_spike, spike_desc = evaluate_loss_spike_condition(
                train_loss_epoch=avg_train_loss,
                min_train_loss=min_train_loss,
                spike_ratio_threshold=spike_ratio_threshold,
            )
            if is_train_spike and attempt < max_attempts - 1 and not fast_dev_run:
                spike_restarts += 1
                next_lr = lr * (0.85 ** (attempt + 1))
                print(
                    f"  [Denoiser Trainer] Unlucky loss spike detected at epoch {epoch:03d} "
                    f"({spike_desc}). Restarting training (attempt {attempt + 2}/{max_attempts}, "
                    f"lr={next_lr:.2e})...",
                    flush=True,
                )
                restarted_on_spike = True
                break

            if math.isfinite(avg_train_loss) and avg_train_loss > 0.0:
                min_train_loss = min(min_train_loss, avg_train_loss)

            # Validation Pass (with in-VRAM EMA Shadow Evaluation)
            is_val_epoch = (
                (epoch % val_interval == 0)
                or (epoch == 1)
                or (epoch == epochs)
                or (epoch >= min_epochs)
            )

            if is_val_epoch:
                model.eval()
                use_ema_val = ema_decay > 0.0 and bool(ema_params) and epoch >= warmup_epochs
                live_backup: list[torch.Tensor] = []
                if use_ema_val:
                    with torch.no_grad():
                        live_backup = [p.detach().clone() for p in model_params]
                        torch._foreach_copy_(model_params, ema_params)

                val_autocast = (
                    torch.autocast(device_type=hw_device.type, dtype=autocast_dtype)
                    if use_autocast
                    else contextlib.nullcontext()
                )
                with torch.no_grad(), val_autocast:
                    if val_chunks > 0 and all_val_indices is not None:
                        val_loss_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        val_err_full_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        val_err_lite_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        val_tgt_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        val_hp_high_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        actual_steps = 0

                        for s in range(val_steps):
                            bs = min(val_bs, val_chunks - s * val_bs)
                            if bs <= 0:
                                break
                            c_indices = all_val_indices[s * val_bs : s * val_bs + bs]
                            c_bx = x_val_noisy_padded[c_indices + arange_nx].unsqueeze(1)
                            c_by = y_val_clean[c_indices + arange_ny]

                            c_pred = model(c_bx)
                            c_loss, _, c_hp_high = criterion(c_pred, c_by)
                            c_err = (c_pred - c_by.unsqueeze(1)).float()

                            val_loss_acc.add_(c_loss.float())
                            val_err_lite_acc.add_(torch.sum(torch.square(c_err[:, 0, :])))
                            val_err_full_acc.add_(torch.sum(torch.square(c_err[:, 1, :])))
                            val_tgt_acc.add_(torch.sum(torch.square(c_by.float())))
                            val_hp_high_acc.add_(c_hp_high.float())
                            actual_steps += 1

                        cur_esr_full = float((val_err_full_acc / (val_tgt_acc + 1e-12)).item())
                        cur_esr_lite = float((val_err_lite_acc / (val_tgt_acc + 1e-12)).item())
                        total_eval_samples = float(val_chunks * ny)
                        mse_cur = float((val_err_full_acc / total_eval_samples).item())
                        mb_val = float((val_loss_acc / max(1, actual_steps)).item())
                        val_hp_high = float((val_hp_high_acc / max(1, actual_steps)).item())
                    else:
                        y_val_hat = model(x_val_noisy_padded.unsqueeze(0))
                        c_loss, mse_t, hp_high_t = criterion(
                            y_val_hat[:, :, :val_samples], y_val_clean.unsqueeze(0)
                        )
                        err_full = (y_val_hat[0, 1, :val_samples] - y_val_clean).float()
                        err_lite = (y_val_hat[0, 0, :val_samples] - y_val_clean).float()
                        target_energy = (
                            float(torch.sum(torch.square(y_val_clean.float())).item()) + 1e-12
                        )
                        cur_esr_full = (
                            float(torch.sum(torch.square(err_full)).item()) / target_energy
                        )
                        cur_esr_lite = (
                            float(torch.sum(torch.square(err_lite)).item()) / target_energy
                        )
                        mse_cur = float(mse_t.item())
                        mb_val = float(c_loss.item())
                        val_hp_high = float(hp_high_t.item())

                # Post-validation unlucky spike check (CL-37)
                is_val_spike, val_spike_desc = evaluate_loss_spike_condition(
                    train_loss_epoch=avg_train_loss,
                    min_train_loss=min_train_loss,
                    val_loss_epoch=mb_val,
                    min_val_loss=min_val_loss_seen,
                    spike_ratio_threshold=spike_ratio_threshold,
                )
                if is_val_spike and attempt < max_attempts - 1 and not fast_dev_run:
                    if use_ema_val and live_backup:
                        with torch.no_grad():
                            torch._foreach_copy_(model_params, live_backup)
                    spike_restarts += 1
                    next_lr = lr * (0.85 ** (attempt + 1))
                    print(
                        f"  [Denoiser Trainer] Unlucky validation spike detected at epoch {epoch:03d} "
                        f"({val_spike_desc}). Restarting training (attempt {attempt + 2}/{max_attempts}, "
                        f"lr={next_lr:.2e})...",
                        flush=True,
                    )
                    restarted_on_spike = True
                    break

                if math.isfinite(mb_val) and mb_val > 0.0:
                    min_val_loss_seen = min(min_val_loss_seen, mb_val)

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
                hp_high_history.append(val_hp_high)

                if min_mse_full < float("inf"):
                    pareto_score = compute_pareto_quality_score(
                        mb_full_loss=mb_val,
                        mse_full=mse_cur,
                        min_mse_full=min_mse_full,
                        drift_tolerance=0.15,
                        drift_penalty_gain=2.5,
                    )
                else:
                    pareto_score = mb_val

                if (epoch >= warmup_epochs or epochs <= warmup_epochs) and (
                    pareto_score < best_pareto_score or best_pareto_score == float("inf")
                ):
                    best_pareto_score = pareto_score
                    best_epoch = epoch
                    best_val_loss = mb_val
                    best_val_esr = cur_esr_full
                    best_val_esr_lite = cur_esr_lite
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if use_ema_val and live_backup:
                    with torch.no_grad():
                        torch._foreach_copy_(model_params, live_backup)

                k_eval = max(patience, 4)
                val_slope = (
                    compute_linear_slope(val_loss_history[-k_eval:])
                    if len(val_loss_history) >= 2
                    else 0.0
                )
                hp_slope = (
                    compute_linear_slope(hp_high_history[-k_eval:])
                    if len(hp_high_history) >= 2
                    else 0.0
                )

                if epoch >= min_epochs:
                    should_stop, stop_reason = evaluate_plateau_stopping_condition(
                        val_loss_history=val_loss_history,
                        val_loss_slope=val_slope,
                        epoch=epoch,
                        warmup_floor=min_epochs,
                        patience=patience,
                        min_delta=min_delta,
                        hp_high_slope=hp_slope,
                        mse_drift_ratio=mse_drift,
                        drift_exit_threshold=0.25,
                    )
                else:
                    should_stop, stop_reason = False, None

                epoch_sec = time.time() - epoch_t0
                is_best_str = " *" if epoch == best_epoch else ""
                esr_full_db = 10.0 * math.log10(max(cur_esr_full, 1e-12))
                esr_lite_db = 10.0 * math.log10(max(cur_esr_lite, 1e-12))
                print(
                    f"  Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.2f}s] "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val Loss: {mb_val:.5f} (Q: {pareto_score:.5f}){is_best_str} | "
                    f"ESR Ch8: {cur_esr_full:.6f} ({esr_full_db:+.1f}dB) | "
                    f"ESR Ch3: {cur_esr_lite:.6f} ({esr_lite_db:+.1f}dB)",
                    flush=True,
                )
            else:
                should_stop, stop_reason = False, None
                epoch_sec = time.time() - epoch_t0
                print(
                    f"  Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.2f}s] Train: {avg_train_loss:.4f}",
                    flush=True,
                )

            if should_stop:
                print(
                    f"  [Denoiser Trainer] Early stopping triggered at epoch {epoch:03d} "
                    f"({stop_reason}).",
                    flush=True,
                )
                break

        if not restarted_on_spike:
            break

    total_train_dur = time.time() - start_train_time

    # Restore Best Pareto Checkpoint & Apply Unity I/O Scale Compensation
    model.activation_tau = 0.0
    model.eval()
    if best_state_dict is not None and best_epoch > 0:
        print(
            f"  [Denoiser Trainer] Restoring optimal Pareto checkpoint from epoch {best_epoch:03d} "
            f"(Val Loss: {best_val_loss:.5f}, Pareto Q: {best_pareto_score:.5f})",
            flush=True,
        )
        model.load_state_dict(best_state_dict)
    else:
        best_epoch = epoch
        best_val_loss = mb_val
        best_val_esr = cur_esr_full
        best_val_esr_lite = cur_esr_lite

    apply_model_io_scale_compensation(
        model,
        x_scale_factor=io_scale_factor,
        y_scale_factor=io_scale_factor,
    )

    # Measure idle silence suppression on the trailing 20% silence segment of validation
    silence_tail = max(4096, int(0.20 * val_samples))
    with torch.no_grad():
        idle_noisy_seg = x_val_noisy_padded[-(silence_tail + rf_pad) :].float().unsqueeze(0)
        idle_pred = model(idle_noisy_seg)
        idle_in_rms = float(
            torch.sqrt(torch.mean(torch.square(idle_noisy_seg[0, rf_pad:])) + 1e-16).item()
        )
        idle_out_full_rms = float(
            torch.sqrt(torch.mean(torch.square(idle_pred[0, 1, :])) + 1e-16).item()
        )
        idle_out_lite_rms = float(
            torch.sqrt(torch.mean(torch.square(idle_pred[0, 0, :])) + 1e-16).item()
        )
        supp_db_ch8 = 20.0 * math.log10(max(idle_out_full_rms, 1e-9) / max(idle_in_rms, 1e-9))
        supp_db_ch3 = 20.0 * math.log10(max(idle_out_lite_rms, 1e-9) / max(idle_in_rms, 1e-9))

    output_nam.parent.mkdir(parents=True, exist_ok=True)

    meta_dict: dict[str, Any] = {
        "name": display_name,
        "modeled_by": DENOISER_MODELED_BY,
        "gear_make": DENOISER_GEAR_MAKE,
        "gear_model": gear_model,
        "gear_type": DENOISER_GEAR_TYPE,
        "tone_type": DENOISER_TONE_TYPE,
        "mains": mains,
        "mains_slug": mains_slug,
        "profile": profile,
        "threshold_dbfs": round(effective_threshold_dbfs, 2),
        "noise_min_dbfs": round(noise_gen.noise_min_dbfs, 2),
        "noise_max_dbfs": round(noise_gen.noise_max_dbfs, 2),
        "silence_weight": round(profile_spec.silence_weight, 2),
        "ultrasonic_lpf_fc_hz": round(ultrasonic_lpf_fc, 1),
        "date": datetime.now(tz=UTC).isoformat(),
        "epochs": best_epoch,
        "epochs_trained": best_epoch,
        "val_mse": best_val_loss,
        "validation_spectral_loss": best_val_loss,
        "validation_esr": best_val_esr,
        "validation_esr_a2_full": best_val_esr,
        "validation_esr_a2_lite": best_val_esr_lite,
        "validation_esr_ch8": best_val_esr,
        "validation_esr_ch3": best_val_esr_lite,
        "silence_suppression_db_ch8": round(supp_db_ch8, 2),
        "silence_suppression_db_ch3": round(supp_db_ch3, 2),
        "training_time_seconds": round(total_train_dur, 2),
        "spike_restarts": spike_restarts,
        "input_scale_compensation": round(io_scale_factor, 6),
        "output_scale_compensation": round(1.0 / io_scale_factor, 6),
    }
    if noise_wav is not None:
        meta_dict["noise_wav"] = str(noise_wav.name)
    if stop_reason:
        meta_dict["stop_reason"] = stop_reason

    model.export_nam(
        out_path=output_nam,
        sample_rate=float(sample_rate),
        metadata=meta_dict,
    )

    sidecar_payload: dict[str, Any] = {
        "nam_file": output_nam.name,
        **meta_dict,
    }
    sidecar_out = output_nam.with_suffix(".json")
    sidecar_out.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")

    full_db = 10.0 * math.log10(max(best_val_esr, 1e-12))
    lite_db = 10.0 * math.log10(max(best_val_esr_lite, 1e-12))
    print("=======================================================", flush=True)
    print(
        f" Denoiser Training Complete in {total_train_dur:.2f}s ({total_train_dur / 60.0:.1f}m)",
        flush=True,
    )
    print("=======================================================", flush=True)
    print(f"  Exported File:  {output_nam.name}", flush=True)
    print(f"  Sidecar JSON:   {sidecar_out.name}", flush=True)
    print(f"  Best Epoch:     {best_epoch:03d} / {epoch:03d}", flush=True)
    print(
        f"  Validation ESR: {best_val_esr:.6f} (A2 Full Ch8, {full_db:+.2f} dB)",
        flush=True,
    )
    print(
        f"  Lite Mode ESR:  {best_val_esr_lite:.6f} (A2 Lite Ch3, {lite_db:+.2f} dB)",
        flush=True,
    )
    print(
        f"  Idle Gate Supp: {supp_db_ch8:+.2f} dB (Ch8) | {supp_db_ch3:+.2f} dB (Ch3)",
        flush=True,
    )
    print("  Status: Ready for Darkglass Anagram Block 1 loading.", flush=True)
    print("=======================================================\n", flush=True)
    return sidecar_payload


def train_denoiser_matrix(
    *,
    mains: str = "both",
    profile: str = "all",
    input_file: Path = DEFAULT_EXCITATION_WAV,
    output_dir: Path = DEFAULT_DENOISER_OUTPUT_DIR,
    noise_wav: Path | None = None,
    threshold_db: float | None = None,
    ultrasonic_lpf_fc: float = DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    epochs: int = DENOISER_DEFAULT_MAX_EPOCHS,
    min_epochs: int = DENOISER_DEFAULT_MIN_EPOCHS,
    patience: int = DENOISER_DEFAULT_PATIENCE,
    batch_size: int | str = "auto",
    precision: str = "auto",
    lr: float = DENOISER_DEFAULT_LR,
    weight_decay: float = DENOISER_DEFAULT_WEIGHT_DECAY,
    fir_hp_mid_weight: float = DENOISER_DEFAULT_FIR_HP_MID_WEIGHT,
    fir_hp_high_weight: float = DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT,
    val_interval: int = DENOISER_DEFAULT_VAL_INTERVAL,
    ema_decay: float = DENOISER_DEFAULT_EMA_DECAY,
    device: str = "auto",
    seed: int | None = 42,
    fast_dev_run: bool = False,
) -> list[Path]:
    """Trains one or more denoiser models across `--mains` and `--profile`, reusing the clean corpus."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Clean excitation WAV file '{input_file}' not found.")
    if noise_wav is not None and not noise_wav.is_file():
        raise FileNotFoundError(f"Hardware noise WAV file '{noise_wav}' not found.")

    mains_targets = resolve_mains_targets(mains)
    profile_targets = resolve_profile_targets(profile)

    output_dir.mkdir(parents=True, exist_ok=True)
    max_dur = 8.0 if fast_dev_run else None
    clean_corpus, sr = build_clean_bass_corpus(
        input_file,
        ultrasonic_lpf_fc=ultrasonic_lpf_fc,
        max_duration_sec=max_dur,
    )

    total_models = len(mains_targets) * len(profile_targets)
    exported_paths: list[Path] = []
    trained_meta: list[dict[str, Any]] = []

    idx = 0
    for m_mode in mains_targets:
        for p_name in profile_targets:
            idx += 1
            stem = resolve_denoiser_model_stem(m_mode, p_name)
            out_nam = output_dir / f"{stem}.nam"
            if total_models > 1:
                print(
                    f"[{idx}/{total_models}] Training Denoiser Target: {out_nam.name} "
                    f"(mains={m_mode}, profile={p_name})",
                    flush=True,
                )
            meta = train_single_denoiser_model(
                clean_corpus=clean_corpus,
                sample_rate=sr,
                output_nam=out_nam,
                mains=m_mode,
                profile=p_name,
                threshold_dbfs=threshold_db,
                noise_wav=noise_wav,
                epochs=epochs,
                min_epochs=min_epochs,
                patience=patience,
                batch_size=batch_size,
                precision=precision,
                lr=lr,
                weight_decay=weight_decay,
                fir_hp_mid_weight=fir_hp_mid_weight,
                fir_hp_high_weight=fir_hp_high_weight,
                ultrasonic_lpf_fc=ultrasonic_lpf_fc,
                val_interval=val_interval,
                ema_decay=ema_decay,
                device_name=device,
                seed=seed,
                fast_dev_run=fast_dev_run,
            )
            exported_paths.append(out_nam)
            trained_meta.append(meta)

    generate_denoiser_storefront_description(output_dir, trained_metadata=trained_meta)
    return exported_paths


def evaluate_denoiser_model(
    model_path: Path,
    *,
    mains: MainsMode = "both",
    noise_dbfs: float = -48.0,
    signal_dbfs: float = -16.0,
    duration_sec: float = 2.0,
    submodel: int = 1,
    device_name: str = "auto",
) -> dict[str, float | str]:
    """Evaluates a trained `.nam` denoiser model on synthetic 50/60 Hz hum + SMPS noise and bass notes."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Denoiser model '{model_path}' not found.")

    hw_device = _resolve_torch_device(device_name)
    model, _metadata, sr = AllomorphWaveNetA2.from_nam_file(model_path, device=hw_device)
    n = max(8192, round(duration_sec * sr))
    rf_pad = 6346

    # 1. Idle Noise Suppression Test (pure 50/60 Hz hum + SMPS whine at noise_dbfs)
    hum = synthesize_mains_hum_bank(n, mains=mains, sample_rate=sr, seed=77)
    smps, hiss = synthesize_smps_and_hiss_banks(n, sample_rate=sr, seed=88)
    raw_noise = 0.65 * hum + 0.60 * smps + 0.20 * hiss
    raw_rms = float(np.sqrt(np.mean(np.square(raw_noise, dtype=np.float64))))
    noise_amp = float(10.0 ** (noise_dbfs / 20.0))
    idle_noise = (raw_noise / max(raw_rms, 1e-12)) * noise_amp

    # 2. Active Passband Bass Test (multi-harmonic bass note at 82.4 Hz E2 + 164.8 Hz + 247.2 Hz)
    t = np.arange(n, dtype=np.float64) / float(sr)
    bass_wave = (
        0.75 * np.sin(2.0 * np.pi * 82.407 * t)
        + 0.20 * np.sin(2.0 * np.pi * 164.814 * t + 0.3)
        + 0.05 * np.sin(2.0 * np.pi * 247.221 * t + 0.9)
    )
    bass_rms = float(np.sqrt(np.mean(np.square(bass_wave))))
    sig_amp = float(10.0 ** (signal_dbfs / 20.0))
    clean_bass = ((bass_wave / max(bass_rms, 1e-12)) * sig_amp).astype(np.float32)

    noisy_bass = np.clip(clean_bass + idle_noise, -0.999, 0.999).astype(np.float32)

    ch_idx = 0 if submodel == 0 else 1
    with torch.no_grad():
        idle_in_t = torch.cat(
            [
                torch.zeros(rf_pad, device=hw_device, dtype=torch.float32),
                torch.from_numpy(idle_noise.astype(np.float32)).to(hw_device),
            ]
        ).unsqueeze(0)
        idle_out = model(idle_in_t)[0, ch_idx, :].cpu().numpy()

        clean_in_t = torch.cat(
            [
                torch.zeros(rf_pad, device=hw_device, dtype=torch.float32),
                torch.from_numpy(clean_bass).to(hw_device),
            ]
        ).unsqueeze(0)
        clean_out = model(clean_in_t)[0, ch_idx, :].cpu().numpy()

        noisy_in_t = torch.cat(
            [
                torch.zeros(rf_pad, device=hw_device, dtype=torch.float32),
                torch.from_numpy(noisy_bass).to(hw_device),
            ]
        ).unsqueeze(0)
        noisy_out = model(noisy_in_t)[0, ch_idx, :].cpu().numpy()

    idle_in_rms = float(np.sqrt(np.mean(np.square(idle_noise, dtype=np.float64))))
    idle_out_rms = float(np.sqrt(np.mean(np.square(idle_out, dtype=np.float64))))
    silence_suppression_db = 20.0 * math.log10(max(idle_out_rms, 1e-9) / max(idle_in_rms, 1e-9))

    clean_in_rms = float(np.sqrt(np.mean(np.square(clean_bass, dtype=np.float64))))
    clean_out_rms = float(np.sqrt(np.mean(np.square(clean_out, dtype=np.float64))))
    passband_gain_delta_db = 20.0 * math.log10(max(clean_out_rms, 1e-9) / max(clean_in_rms, 1e-9))
    passband_esr = float(
        np.sum(np.square(clean_out - clean_bass, dtype=np.float64))
        / (np.sum(np.square(clean_bass, dtype=np.float64)) + 1e-12)
    )

    in_err_energy = float(np.sum(np.square(noisy_bass - clean_bass, dtype=np.float64)))
    out_err_energy = float(np.sum(np.square(noisy_out - clean_bass, dtype=np.float64)))
    snr_improvement_db = 10.0 * math.log10(max(in_err_energy, 1e-12) / max(out_err_energy, 1e-12))

    metrics: dict[str, float | str] = {
        "model_file": model_path.name,
        "submodel": "A2 Full (Ch8)" if ch_idx == 1 else "A2 Lite (Ch3)",
        "silence_suppression_db": round(silence_suppression_db, 2),
        "passband_esr": round(passband_esr, 6),
        "passband_gain_delta_db": round(passband_gain_delta_db, 3),
        "snr_improvement_db": round(snr_improvement_db, 2),
    }

    print("=======================================================", flush=True)
    print(f" Denoiser Evaluation: {model_path.name} ({metrics['submodel']})", flush=True)
    print("=======================================================", flush=True)
    print(
        f"  Idle Noise Suppression: {silence_suppression_db:+.2f} dB (at {noise_dbfs:.1f} dBFS)",
        flush=True,
    )
    print(f"  Active Passband ESR:    {passband_esr:.6f}", flush=True)
    print(f"  Passband Gain Delta:    {passband_gain_delta_db:+.3f} dB", flush=True)
    print(f"  Active SNR Change:      {snr_improvement_db:+.2f} dB", flush=True)
    print("=======================================================", flush=True)
    return metrics


def denoise_audio_file(
    input_wav: Path,
    output_wav: Path,
    *,
    model_path: Path = DEFAULT_DENOISER_OUTPUT_DIR / "universal_balanced.nam",
    submodel: int = 1,
    chunk_samples: int = 65536,
    device_name: str = "auto",
) -> Path:
    """Applies a trained `.nam` denoiser model to an audio file with stateful causal buffering."""
    if not input_wav.is_file():
        raise FileNotFoundError(f"Input WAV '{input_wav}' not found.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Denoiser model '{model_path}' not found.")

    hw_device = _resolve_torch_device(device_name)
    model, _, _ = AllomorphWaveNetA2.from_nam_file(model_path, device=hw_device)
    raw, sr = read_wav(input_wav, dtype=np.float32)
    if raw.ndim > 1:
        raw = np.mean(raw, axis=0)

    rf_pad = 6346
    ch_idx = 0 if submodel == 0 else 1
    n_total = len(raw)
    out_audio = np.empty(n_total, dtype=np.float32)

    padded = np.concatenate([np.zeros(rf_pad, dtype=np.float32), raw.astype(np.float32)])
    with torch.no_grad():
        for start in range(0, n_total, chunk_samples):
            end = min(n_total, start + chunk_samples)
            win = padded[start : end + rf_pad]
            win_t = torch.from_numpy(win).to(device=hw_device, dtype=torch.float32).unsqueeze(0)
            pred = model(win_t)[0, ch_idx, :].cpu().numpy()
            out_audio[start:end] = pred

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    write_wav_24bit(output_wav, out_audio, sample_rate=sr)
    return output_wav
