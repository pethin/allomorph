"""Allomorph Reamper - Re-Amped Rig NAM Architecture 2 Native Trainer.

Trains production-grade Neural Amp Modeler (NAM) Architecture 2 (.nam) models
from dry string excitation (reamp/input.wav or audio/input.wav) to hardware re-amped
wet audio stems from a Darkglass Anagram using the signal chains in reamp/signal_chains.md
and Allomorph's high-throughput In-VRAM PyTorch Native Engine.

Specifically customized for non-linear amplifier heads and speaker cabinets:
  - Disables linear pickup regularizer (no synthetic pickup sines) to preserve authentic tube/FET saturation
  - Rebalances high-pass loss weighting (w=0.75, fc=1000 Hz) for steep speaker cabinet roll-offs
  - Performs 2-stage coarse-to-fine envelope + impulse cross-correlation alignment (up to 1000 ms, polarity-invariant)
  - Executes in native bfloat16 mixed precision with fused Conv1D error filtering and zero per-step PCIe host syncs
  - Uses in-VRAM EMA shadow weights (0.995), Anti-Drift Pareto early stopping, and CL-37 spike restarts
  - Maps stems to Darkglass Anagram signal chains from reamp/signal_chains.md and exports dual-tier A2 (.nam) models + JSON sidecars.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

# Must run before importing torch so ROCm MIOpen / Apple Silicon MPS env vars take effect
from allomorph.trainer.core import (
    get_hardware_device_name,
    resolve_hardware_batch_size,
    resolve_hardware_precision,
    setup_headless_environment,
)

setup_headless_environment()

import numpy as np
import torch
import torch.nn.functional as F
from cyclopts import Parameter

from allomorph.dsp.constants import CALIBRATION_PEAK_CEILING, FS, TARGET_RMS_DBFS
from allomorph.dsp.io import read_wav
from allomorph.trainer.callbacks import (
    compute_linear_slope,
    compute_pareto_quality_score,
    evaluate_loss_spike_condition,
    evaluate_plateau_stopping_condition,
)
from allomorph.trainer.constants import (
    DEFAULT_ETA_MIN,
    DEFAULT_LR_T_MAX,
    NATIVE_DEFAULT_EMA_DECAY,
    NATIVE_DEFAULT_HOMOTOPY_TAU_INIT,
    NATIVE_DEFAULT_LITE_WEIGHT,
    NATIVE_DEFAULT_LR,
    NATIVE_DEFAULT_MAX_EPOCHS,
    NATIVE_DEFAULT_MAX_SPIKE_RESTARTS,
    NATIVE_DEFAULT_MIN_DELTA,
    NATIVE_DEFAULT_MIN_EPOCHS,
    NATIVE_DEFAULT_MSE_WEIGHT,
    NATIVE_DEFAULT_PATIENCE,
    NATIVE_DEFAULT_SLICE_LEN,
    NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD,
    NATIVE_DEFAULT_VAL_INTERVAL,
    NATIVE_DEFAULT_WEIGHT_DECAY,
)
from allomorph.trainer.native.loss import design_sharp_fir_highpass
from allomorph.trainer.native.models import (
    AllomorphWaveNetA2,
    apply_model_io_scale_compensation,
    select_warm_start_offset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AnagramSignalChain:
    """Metadata specification for a Darkglass Anagram amp + cab + mic signal chain."""

    index: int
    rig_name: str
    style: str
    amp_block: str
    cab_block: str
    mic: str
    mic_position: str
    rationale: str
    gear_make: str
    gear_model: str
    tone_type: str
    aliases: tuple[str, ...]


CANONICAL_SIGNAL_CHAINS: tuple[AnagramSignalChain, ...] = (
    AnagramSignalChain(
        index=1,
        rig_name="Ampeg SVT-CL + Heritage 8×10",
        style="Classic/Modern Rock",
        amp_block="Peggy Classic",
        cab_block="Vintage Peggy 8×10",
        mic="Dynamic 421",
        mic_position="Cone On Axis",
        rationale=(
            "The true studio rock standard. Moving off the dust cap onto Cone On Axis gives "
            "Dynamic 421 balanced lows and classic Ampeg midrange growl even at 100% wet."
        ),
        gear_make="Ampeg",
        gear_model="SVT-CL + Heritage 8x10 (Anagram Peggy Classic)",
        tone_type="overdrive",
        aliases=(
            "peggy_classic_8x10",
            "peggy_classic",
            "svt_cl_8x10",
            "ampeg_svt_cl",
            "vintage_peggy_8x10",
        ),
    ),
    AnagramSignalChain(
        index=2,
        rig_name="'70s SVT Blueline + 8×10",
        style="Aggressive Pick Rock / Punk",
        amp_block="Peggy Bass",
        cab_block="Peggy 8×10",
        mic="Dynamic 421",
        mic_position="Cap Off Axis",
        rationale=(
            "Gives you rawer upper-mid pick bite than Cone On Axis, while the off-axis angle "
            "prevents harsh paper-cone fizz."
        ),
        gear_make="Ampeg",
        gear_model="'70s SVT Blueline + 8x10 (Anagram Peggy Bass)",
        tone_type="overdrive",
        aliases=(
            "peggy_blueline_8x10",
            "peggy_bass_8x10",
            "peggy_bass",
            "svt_blueline_8x10",
            "svt_blueline",
            "blueline_8x10",
        ),
    ),
    AnagramSignalChain(
        index=3,
        rig_name="Ampeg B-15 + B-15 1×15",
        style="Motown / Soul / Studio #1",
        amp_block="Peggy FlipTop",
        cab_block="Peggy Fliptop 1×15",
        mic="Condenser 67",
        mic_position="Cone On Axis",
        rationale=(
            "Historically recorded 100% mic-only. Condenser 67 at Cone On Axis nails the warm, "
            'pillowy 15" fundamental and smooth vintage top end.'
        ),
        gear_make="Ampeg",
        gear_model="B-15 Portaflex + 1x15 (Anagram Peggy FlipTop)",
        tone_type="clean",
        aliases=(
            "peggy_fliptop_1x15",
            "peggy_fliptop",
            "ampeg_b15_1x15",
            "ampeg_b15",
            "b15_1x15",
            "fliptop_1x15",
        ),
    ),
    AnagramSignalChain(
        index=4,
        rig_name="Fender Super Bassman + 4×10",
        style="Classic Rock / Indie / Blues",
        amp_block="Super California",
        cab_block="California Bass 4×10",
        mic="Condenser 67",
        mic_position="Cone On Axis",
        rationale=(
            "Warms up the Fender's natural mid-scoop and rounds off the bright neodymium speakers "
            "(the Fender Bassman 410 Neo) while keeping deep 100% wet lows."
        ),
        gear_make="Fender",
        gear_model="Super Bassman + 4x10 (Anagram Super California)",
        tone_type="clean",
        aliases=(
            "super_california_4x10",
            "super_california",
            "california_bass_4x10",
            "fender_super_bassman_4x10",
            "super_bassman_4x10",
            "super_bassman",
        ),
    ),
    AnagramSignalChain(
        index=5,
        rig_name="GK 800RB + 4×10",
        style="'80s/'90s Alt-Rock / Funk — Substitute Cab",
        amp_block="Kallinen Kruukkeri",
        cab_block="Peggy 4×10",
        mic="Dynamic 52",
        mic_position="Cone On Axis",
        rationale=(
            "Since there is no GK cab on the Anagram, a ported 4×10 with Dynamic 52 (Beta 52A) "
            "gives the punchy lows and snappy upper-mid presence GK players rely on."
        ),
        gear_make="Gallien-Krueger",
        gear_model="800RB + 4x10 (Anagram Kallinen Kruukkeri)",
        tone_type="crunch",
        aliases=(
            "kallinen_kruukkeri_4x10",
            "kallinen_kruukkeri",
            "kruukkeri_4x10",
            "gk_800rb_4x10",
            "gk_800rb",
            "800rb_4x10",
        ),
    ),
    AnagramSignalChain(
        index=6,
        rig_name="Mesa Big Block 750 + Mesa 4×10",
        style="Hard Rock / Modern Metal",
        amp_block="Modern ISOBRICK750 OD",
        cab_block="Modern Bass 4×10",
        mic="Dynamic 52",
        mic_position="Cap Off Axis",
        rationale=(
            "Matches the Mesa head to the Mesa PowerHouse 4×10. Dynamic 52 keeps the sub-bass "
            "huge at 100% wet while scooping out low-mid mud from the overdrive."
        ),
        gear_make="Mesa/Boogie",
        gear_model="Big Block 750 + PowerHouse 4x10 (Anagram Modern ISOBRICK750 OD)",
        tone_type="hi_gain",
        aliases=(
            "modern_isobrick750_4x10",
            "modern_isobrick750_od",
            "isobrick750_4x10",
            "isobrick_4x10",
            "mesa_big_block_750_4x10",
            "mesa_big_block_4x10",
            "modern_bass_4x10",
        ),
    ),
    AnagramSignalChain(
        index=7,
        rig_name="Orange Terror Bass + Orange 4×10",
        style="Stoner / Doom / Indie",
        amp_block="Fruity Horror Bass 500",
        cab_block="Fruity Bass 4×10",
        mic="Dynamic 7",
        mic_position="Cone On Axis",
        rationale=(
            "Moving to Cone On Axis with Dynamic 7 (SM7B) gives thick low-mids and a smooth, "
            "forgiving top end that keeps Orange overdrive creamy instead of fizzy."
        ),
        gear_make="Orange",
        gear_model="Terror Bass 500 + OBC410 4x10 (Anagram Fruity Horror Bass 500)",
        tone_type="overdrive",
        aliases=(
            "fruity_horror_4x10",
            "fruity_horror_bass_500",
            "fruity_bass_4x10",
            "orange_terror_bass_4x10",
            "orange_terror_4x10",
            "terror_bass_4x10",
        ),
    ),
    AnagramSignalChain(
        index=8,
        rig_name="SWR SM-900 + SWR Goliath 4×10",
        style="'90s Hi-Fi / Slap / Fusion",
        amp_block="Swirl-900",
        cab_block="SWIRL 4×10",
        mic="Condenser 103",
        mic_position="Cone On Axis",
        rationale=(
            "Matches the SWR head and Goliath cab. Condenser 103 (TLM 103) delivers the extended "
            "hi-fi highs and deep lows that define the '90s SWR slap sound."
        ),
        gear_make="SWR",
        gear_model="SM-900 + Goliath 4x10 (Anagram Swirl-900)",
        tone_type="clean",
        aliases=(
            "swirl_900_4x10",
            "swirl_900",
            "swirl_4x10",
            "swr_sm900_4x10",
            "swr_goliath_4x10",
            "sm900_4x10",
        ),
    ),
    AnagramSignalChain(
        index=9,
        rig_name="Markbass Little Mark + 4×10",
        style="Jazz / Fusion / Pop / Funk",
        amp_block="Black and Yellow",
        cab_block="Black & Yellow 4×10",
        mic="Condenser 414",
        mic_position="Cone On Axis",
        rationale=(
            "Matches the Markbass head and 104HR cab. Condenser 414 (C414) has a flat, neutral "
            "midrange that lets the dry, punchy Markbass character come through unaltered."
        ),
        gear_make="Markbass",
        gear_model="Little Mark + Standard 104HR 4x10 (Anagram Black and Yellow)",
        tone_type="clean",
        aliases=(
            "black_and_yellow_4x10",
            "black_and_yellow",
            "markbass_little_mark_4x10",
            "markbass_4x10",
            "little_mark_4x10",
        ),
    ),
    AnagramSignalChain(
        index=10,
        rig_name="EBS TD650 + Neo 4×10",
        style="Scandinavian Metal / Prog — Substitute Cab",
        amp_block="SWE TC650",
        cab_block="Darkglass Neo 4×10",
        mic="Dynamic 112",
        mic_position="Cone On Axis",
        rationale=(
            "Since there is no EBS cab on the unit, the neodymium Darkglass Neo 4×10 paired with "
            "Dynamic 112 (D112) captures that tight 100–150 Hz Scandinavian rock/metal punch."
        ),
        gear_make="EBS",
        gear_model="TD650 + Darkglass Neo 4x10 (Anagram SWE TC650)",
        tone_type="overdrive",
        aliases=(
            "swe_tc650_4x10",
            "swe_tc650",
            "darkglass_neo_4x10",
            "ebs_td650_4x10",
            "ebs_td650",
            "td650_4x10",
        ),
    ),
)


def _slugify(text: str) -> str:
    """Normalizes a string into lowercase alphanumeric underscore tokens (converting × to x)."""
    cleaned = text.lower().replace("×", "x").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")


def load_signal_chains(
    md_path: Path = REPO_ROOT / "reamp" / "signal_chains.md",
) -> list[AnagramSignalChain]:
    """Loads Darkglass Anagram signal chain definitions, enriching from reamp/signal_chains.md if present."""
    chains_by_idx: dict[int, AnagramSignalChain] = {c.index: c for c in CANONICAL_SIGNAL_CHAINS}
    if not md_path.is_file():
        return list(chains_by_idx.values())

    try:
        raw_lines = md_path.read_text(encoding="utf-8").splitlines()
        for line in raw_lines:
            stripped = line.strip()
            if not stripped.startswith("|") or stripped.startswith("|---"):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) < 6 or "Iconic Real-World Rig" in cells[0]:
                continue

            m_idx = re.search(r"\*\*(\d+)\.\s*([^*]+)\*\*", cells[0])
            if not m_idx:
                continue
            idx = int(m_idx.group(1))
            rig_name = m_idx.group(2).strip()
            m_style = re.search(r"\*\(([^)]+)\)\*", cells[0])
            style = (
                m_style.group(1).strip()
                if m_style
                else (chains_by_idx[idx].style if idx in chains_by_idx else "")
            )

            def _extract_code(cell: str) -> str:
                m_code = re.search(r"`([^`]+)`", cell)
                return m_code.group(1).strip() if m_code else re.sub(r"[*`]", "", cell).strip()

            amp_block = _extract_code(cells[1])
            cab_block = _extract_code(cells[2])
            mic = _extract_code(cells[3])
            mic_position = _extract_code(cells[4])
            rationale = re.sub(r"[*`]", "", cells[5]).strip()

            base = chains_by_idx.get(idx)
            if base is not None:
                chains_by_idx[idx] = AnagramSignalChain(
                    index=idx,
                    rig_name=rig_name,
                    style=style,
                    amp_block=amp_block,
                    cab_block=cab_block,
                    mic=mic,
                    mic_position=mic_position,
                    rationale=rationale,
                    gear_make=base.gear_make,
                    gear_model=base.gear_model,
                    tone_type=base.tone_type,
                    aliases=base.aliases,
                )
    except OSError:
        pass

    return [chains_by_idx[k] for k in sorted(chains_by_idx)]


def resolve_signal_chain_for_stem(
    stem_name: str,
    chains: list[AnagramSignalChain] | None = None,
) -> AnagramSignalChain | None:
    """Matches a re-amped WAV stem filename to its corresponding AnagramSignalChain."""
    catalog = chains if chains is not None else load_signal_chains()
    slug = _slugify(Path(stem_name).stem)

    # 1. Exact or substring match against canonical aliases
    for chain in catalog:
        for alias in chain.aliases:
            if slug == alias or alias in slug:
                return chain

    # 2. Match against slugified amp_block, cab_block, or rig_name
    for chain in catalog:
        amp_slug = _slugify(chain.amp_block)
        rig_slug = _slugify(chain.rig_name)
        if (amp_slug and amp_slug in slug) or (rig_slug and rig_slug in slug):
            return chain

    # 3. Leading index prefix match (e.g. "01_...", "1_...")
    m_num = re.match(r"^0*(\d+)_", slug)
    if m_num:
        idx = int(m_num.group(1))
        for chain in catalog:
            if chain.index == idx:
                return chain

    return None


def verify_or_align_stems(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int = FS,
    max_search_ms: float = 1000.0,
    causal_guard_samples: int = 4,
) -> tuple[int, np.ndarray]:
    """Two-stage coarse-to-fine latency alignment between dry excitation and re-amped wet stem.

    Stage 1 (Envelope FFT Cross-Correlation):
        Cross-correlates rectified first-difference transient envelopes (|diff(x)| vs |diff(y)|)
        over up to 60 seconds using O(N log N) FFTs. This is strictly invariant to amp phase
        inversion (polarity -1), heavy non-linear tube saturation, speaker cabinet low-pass
        filtering, and the 0.500s (24,000-sample) dual calibration blip ambiguity.

    Stage 2 (Sub-Millisecond Impulse Refinement + Causal Guard):
        Refines the lag within +/- 64 samples around the coarse envelope peak using direct
        waveform cross-correlation on the leading calibration blips (or full search window),
        then backs off by `causal_guard_samples` (default 4 samples) from the impulse peak so
        the entire rising edge of the amp+cab impulse response lies causally inside the
        WaveNet receptive field (n >= 0).

    Returns:
        Tuple of (delay_samples, aligned_wet) truncated to the valid overlapping audio region.
    """
    if len(dry) == 0 or len(wet) == 0:
        return 0, wet[: len(dry)]

    max_delay = min(int(sample_rate * (max_search_ms / 1000.0)), max(0, len(wet) - 16))
    if max_delay <= 0 or len(wet) <= 16:
        return 0, wet[: len(dry)]

    # Use up to 60s of audio where dry and shifted wet both have valid samples
    avail_dry = max(0, min(len(dry), len(wet) - max_delay))
    if avail_dry < min(len(dry), int(sample_rate * 1.5)):
        max_delay = max(0, min(max_delay, len(wet) // 2))
        avail_dry = max(0, min(len(dry), len(wet) - max_delay))

    search_len = min(avail_dry, int(sample_rate * 60.0))
    if search_len <= 16 or len(wet) < search_len + max_delay:
        return 0, wet[: len(dry)]

    # Stage 1: Rectified transient-derivative envelope FFT cross-correlation
    rx = np.abs(np.diff(dry[:search_len].astype(np.float32)))
    ry = np.abs(np.diff(wet[: search_len + max_delay].astype(np.float32)))
    rx -= float(np.mean(rx))
    ry -= float(np.mean(ry))

    n_fft = 1 << (len(rx) + len(ry) - 1).bit_length()
    cc_env = np.fft.irfft(
        np.fft.rfft(ry, n=n_fft) * np.conj(np.fft.rfft(rx, n=n_fft)),
        n=n_fft,
    )[: max_delay + 1]
    coarse_delay = int(np.argmax(cc_env))

    # Stage 2: Fine waveform cross-correlation around coarse_delay (+/- 64 samples)
    fine_radius = 64
    w_lo = max(0, coarse_delay - fine_radius)
    w_hi = min(max_delay, coarse_delay + fine_radius)

    # Prefer the first 1.5s calibration blips if they contain energy; otherwise use up to 6s
    blip_window = min(search_len, int(sample_rate * 1.5))
    if float(np.max(np.abs(dry[:blip_window]))) < 1e-3:
        blip_window = min(search_len, int(sample_rate * 6.0))

    xb = dry[:blip_window].astype(np.float32)
    yb = wet[w_lo : blip_window + w_hi].astype(np.float32)

    if len(yb) > len(xb) and float(np.max(np.abs(yb))) > 1e-6:
        corr_fine = np.correlate(yb, xb, mode="valid")
        peak_offset = int(np.argmax(np.abs(corr_fine)))
        peak_delay = w_lo + peak_offset
    else:
        peak_delay = coarse_delay

    # Apply causal guard offset so impulse leading edge is causal (n >= 0)
    if peak_delay >= causal_guard_samples:
        delay = peak_delay - causal_guard_samples
    else:
        delay = peak_delay

    # Slice valid overlapping region (do not zero-pad over active dry validation audio)
    aligned = wet[delay : delay + len(dry)]
    return delay, aligned


def resolve_stem_send_gain_db(
    wet_path: Path,
    explicit_send_gain_db: float | None = None,
) -> tuple[float, str]:
    """Resolves the hardware reamp USB send gain (in dB) for a captured wet stem.

    Priority:
        1. Explicit `--send-gain` CLI override (`explicit_send_gain_db`).
        2. JSON sidecar `<stem>.json` written automatically by `reamper capture`.
        3. Unity gain (`0.0 dB`) fallback when no sidecar is present.
    """
    if explicit_send_gain_db is not None:
        return float(explicit_send_gain_db), "cli"

    sidecar_path = wet_path.with_suffix(".json")
    if sidecar_path.is_file():
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw_gain = payload.get("send_gain_db")
                if isinstance(raw_gain, int | float) and math.isfinite(float(raw_gain)):
                    return float(raw_gain), f"sidecar ({sidecar_path.name})"
        except OSError, ValueError, TypeError:
            pass

    return 0.0, "default (0.00 dB)"


def train_single_reamp_pair(
    dry_path: Path,
    wet_path: Path,
    output_nam: Path,
    epochs: int = NATIVE_DEFAULT_MAX_EPOCHS,
    min_epochs: int = NATIVE_DEFAULT_MIN_EPOCHS,
    patience: int = NATIVE_DEFAULT_PATIENCE,
    min_delta: float = NATIVE_DEFAULT_MIN_DELTA,
    batch_size: int | str = "auto",
    precision: str = "auto",
    lr: float = NATIVE_DEFAULT_LR,
    weight_decay: float = NATIVE_DEFAULT_WEIGHT_DECAY,
    mse_weight: float = NATIVE_DEFAULT_MSE_WEIGHT,
    fir_hp_weight: float = 0.75,
    fir_hp_fc: float = 1000.0,
    lite_weight: float = NATIVE_DEFAULT_LITE_WEIGHT,
    val_interval: int = NATIVE_DEFAULT_VAL_INTERVAL,
    ema_decay: float = NATIVE_DEFAULT_EMA_DECAY,
    homotopy_tau_init: float = NATIVE_DEFAULT_HOMOTOPY_TAU_INIT,
    spike_ratio_threshold: float = NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD,
    max_spike_restarts: int = NATIVE_DEFAULT_MAX_SPIKE_RESTARTS,
    auto_align: bool = True,
    device_name: str = "auto",
    model_name: str | None = None,
    seed: int | None = 42,
    fast_dev_run: bool = False,
    signal_chain: AnagramSignalChain | None = None,
    send_gain_db: float | None = None,
) -> bool:
    """Trains a dual-tier WaveNet A2 model on a single dry -> re-amped wet audio pair."""
    resolved_chain = signal_chain or resolve_signal_chain_for_stem(wet_path.name)
    if resolved_chain is not None:
        clean_name = model_name or resolved_chain.rig_name.replace("×", "x")
    else:
        clean_name = model_name or wet_path.stem.replace("_", " ").title()

    resolved_send_gain_db, send_gain_source = resolve_stem_send_gain_db(
        wet_path,
        explicit_send_gain_db=send_gain_db,
    )
    send_gain_linear = float(10.0 ** (resolved_send_gain_db / 20.0))

    # 1. Resolve Hardware Device, Batch Size & Precision
    if device_name == "auto":
        if torch.cuda.is_available():
            hw_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw_device = torch.device("mps")
        else:
            hw_device = torch.device("cpu")
    else:
        hw_device = torch.device(device_name)

    resolved_bs = resolve_hardware_batch_size(batch_size)
    resolved_precision = (
        resolve_hardware_precision(precision) if hw_device.type != "cpu" else "32-true"
    )
    hw_label = get_hardware_device_name(engine="native")

    print("=======================================================", flush=True)
    print(" Allomorph Reamper Native PyTorch A2 Trainer", flush=True)
    print("=======================================================", flush=True)
    print(f"  Target Voicing: {clean_name}", flush=True)
    if resolved_chain is not None:
        print(
            f"  Anagram Blocks: Amp='{resolved_chain.amp_block}' | Cab='{resolved_chain.cab_block}' | "
            f"Mic='{resolved_chain.mic}' ({resolved_chain.mic_position})",
            flush=True,
        )
    print(f"  Source Audio:   {dry_path.name}", flush=True)
    print(f"  Target Audio:   {wet_path.name}", flush=True)
    print(
        f"  Reamp Send Gain:{resolved_send_gain_db:+.2f} dB ({send_gain_linear:.4f}x via {send_gain_source})",
        flush=True,
    )
    print(f"  Output Model:   {output_nam}", flush=True)
    print(f"  Hardware:       {hw_label} ({hw_device.type})", flush=True)
    print(f"  Batch Size:     {resolved_bs} | Precision: {resolved_precision}", flush=True)
    print(f"  Schedule:       cosine (max_epochs={epochs}, warmup={min_epochs})", flush=True)
    print(
        f"  Loss Config:    MSE={mse_weight}, FIR_HP={fir_hp_weight} (fc={fir_hp_fc}Hz)",
        flush=True,
    )

    # 2. Ingest Audio Stems
    dry_raw, sr_dry = read_wav(dry_path, dtype=np.float32)
    wet_raw, sr_wet = read_wav(wet_path, dtype=np.float32)

    if dry_raw.ndim > 1:
        dry_raw = np.mean(dry_raw, axis=0)
    if wet_raw.ndim > 1:
        wet_raw = np.mean(wet_raw, axis=0)

    if sr_dry != sr_wet:
        raise ValueError(f"Sample rate mismatch: dry={sr_dry} Hz, wet={sr_wet} Hz")
    if sr_dry != FS:
        raise ValueError(f"Sample rate must be {FS} Hz (got {sr_dry} Hz)")

    # 3. Latency Verification & Alignment
    delay_samples = 0
    if auto_align:
        delay_samples, wet_aligned = verify_or_align_stems(
            dry_raw,
            wet_raw,
            sample_rate=sr_dry,
        )
        delay_ms = (delay_samples / sr_dry) * 1000.0
        if delay_samples > 0:
            print(
                f"  Alignment:      Detected {delay_samples} samples ({delay_ms:.2f} ms) delay — shifted",
                flush=True,
            )
        else:
            print("  Alignment:      Zero latency confirmed (0 samples)", flush=True)
    else:
        wet_aligned = wet_raw[: len(dry_raw)]

    n_samples = min(len(dry_raw), len(wet_aligned))
    if n_samples < 65536:
        raise ValueError(f"Aligned audio stem too short for training ({n_samples} samples).")

    # Split Train (90%) / Validation (10%, up to 432,000 samples = 9s)
    val_samples = min(432_000, max(48000, int(0.10 * n_samples)))
    train_samples = n_samples - val_samples

    # 4. Reconstruct Hardware Send Signal (x_sent) & Reversible Training Conditioning Scales
    if abs(resolved_send_gain_db) > 1e-9:
        x_sent = np.clip(
            dry_raw[:n_samples].astype(np.float64) * send_gain_linear,
            -CALIBRATION_PEAK_CEILING,
            CALIBRATION_PEAK_CEILING,
        ).astype(np.float32)
    else:
        x_sent = dry_raw[:n_samples].astype(np.float32, copy=True)
    y_sent = wet_aligned[:n_samples].astype(np.float32, copy=True)

    target_x_rms = float(10.0 ** (TARGET_RMS_DBFS / 20.0))
    x_sent_rms = float(np.sqrt(np.mean(np.square(x_sent, dtype=np.float64))))
    x_sent_peak = float(np.max(np.abs(x_sent))) if len(x_sent) > 0 else 0.0
    if x_sent_rms > 1e-9 and math.isfinite(x_sent_rms):
        raw_x_scale = target_x_rms / x_sent_rms
        if x_sent_peak * raw_x_scale > CALIBRATION_PEAK_CEILING and x_sent_peak > 1e-9:
            raw_x_scale = float(CALIBRATION_PEAK_CEILING / x_sent_peak)
        x_scale_factor = float(np.float32(raw_x_scale))
    else:
        x_scale_factor = 1.0

    # Calibrate Target Training RMS (Upstream NAM Reference Standard: -18.0 dBFS on train split)
    target_y_rms = 10.0 ** (-18.0 / 20.0)
    y_train_rms = float(np.sqrt(np.mean(np.square(y_sent[:train_samples], dtype=np.float64))))
    if y_train_rms > 1e-9 and math.isfinite(y_train_rms):
        y_scale_factor = float(np.float32(target_y_rms / y_train_rms))
    else:
        y_scale_factor = 1.0

    x_norm = (x_sent * np.float32(x_scale_factor)).astype(np.float32)
    y_norm = (y_sent * np.float32(y_scale_factor)).astype(np.float32)

    print(
        f"  I/O Scale Fold: Input={20.0 * math.log10(max(x_scale_factor, 1e-12)):+.2f} dB "
        f"({x_scale_factor:.4f}x) | Output={20.0 * math.log10(max(1.0 / y_scale_factor, 1e-12)):+.2f} dB "
        f"({1.0 / y_scale_factor:.4f}x)",
        flush=True,
    )
    print("-------------------------------------------------------", flush=True)

    x_train_np = x_norm[:train_samples]
    y_train_np = y_norm[:train_samples]
    x_val_np = x_norm[train_samples:n_samples]
    y_val_np = y_norm[train_samples:n_samples]

    # Select active broadband window for Step-0 Wiener-Hopf head warm-start
    cal_len = min(32768, train_samples)
    cal_offset = select_warm_start_offset(x_train_np, window_len=cal_len)

    # 5. Transfer Directly to Accelerator VRAM in Native Precision
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

    x_train = torch.from_numpy(x_train_np).to(device=hw_device, dtype=vram_dtype)
    y_train = torch.from_numpy(y_train_np).to(device=hw_device, dtype=vram_dtype)
    x_val = torch.from_numpy(x_val_np).to(device=hw_device, dtype=vram_dtype)
    y_val = torch.from_numpy(y_val_np).to(device=hw_device, dtype=vram_dtype)

    rf_pad = 6346  # WaveNet A2 receptive field padding (samples)
    ny = NATIVE_DEFAULT_SLICE_LEN  # 8,192 samples
    nx = ny + rf_pad  # 14,538 samples

    x_train_padded = torch.cat([torch.zeros(rf_pad, device=hw_device, dtype=vram_dtype), x_train])
    x_val_padded = torch.cat([torch.zeros(rf_pad, device=hw_device, dtype=vram_dtype), x_val])

    # Pre-allocated 2D int32 index grids in VRAM
    arange_nx = torch.arange(nx, device=hw_device, dtype=torch.int32).unsqueeze(0)
    arange_ny = torch.arange(ny, device=hw_device, dtype=torch.int32).unsqueeze(0)

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

    # 6. 255-Tap Blackman FIR High-Pass Loss Filter for Amp+Cab
    hp_kernel: torch.Tensor | None = None
    if fir_hp_weight > 0.0:
        hp_kernel = design_sharp_fir_highpass(fc=fir_hp_fc, fs=float(sr_dry), num_taps=255).to(
            device=hw_device, dtype=vram_dtype
        )

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    start_train_time = time.time()
    warmup_epochs = 5
    homotopy_end_epoch = max(1, int(0.75 * min_epochs))

    # 7. Spike-Resilient Training & Restart Loop (CL-37)
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
        if attempt > 0 and seed is not None:
            attempt_seed = int(seed) + attempt * 10007
            torch.manual_seed(attempt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(attempt_seed)

        # Initialize Model & Closed-Form Wiener Warm-Start on Active Excitation Window
        model = AllomorphWaveNetA2().to(hw_device)
        model.mask_weights()

        if train_samples >= 1024:
            model.warm_start_head(
                x_train_padded[cal_offset : cal_offset + cal_len + rf_pad].float(),
                y_train[cal_offset : cal_offset + cal_len].float(),
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

        # Reset state per attempt
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

        # 8. Training Loop
        for epoch in range(1, epochs + 1):
            epoch_t0 = time.time()
            model.train()

            # C-infinity SiLU-gated activation homotopy schedule (tau: 0.02 -> 0.0)
            if homotopy_tau_init > 0.0 and epoch <= homotopy_end_epoch and not fast_dev_run:
                prog = (epoch - 1) / max(1, homotopy_end_epoch)
                model.activation_tau = 0.5 * homotopy_tau_init * (1.0 + math.cos(math.pi * prog))
            else:
                model.activation_tau = 0.0

            # Learning Rate Schedule (5-epoch linear warmup + cosine decay to 1e-6)
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
                    indices = epoch_block_offsets[
                        (step - 1) * resolved_bs : step * resolved_bs
                    ].unsqueeze(1)
                else:
                    indices = torch.randint(
                        0,
                        max_start,
                        (resolved_bs, 1),
                        device=hw_device,
                        dtype=torch.int32,
                    )

                x_batch = x_train_padded[indices + arange_nx].unsqueeze(1)
                y_batch = y_train[indices + arange_ny]

                optimizer.zero_grad(set_to_none=True)

                with train_autocast:
                    pred = model(x_batch)
                    err = pred - y_batch.unsqueeze(1)
                    l_lite = torch.mean(torch.square(err[:, 0, :]))
                    l_full = torch.mean(torch.square(err[:, 1, :]))
                    total_step_loss = mse_weight * (l_full + lite_weight * l_lite)

                    if hp_kernel is not None and fir_hp_weight > 0.0:
                        # Single fused Conv1D GEMM pass across both submodel error channels
                        err_hp = F.conv1d(
                            err.reshape(-1, 1, ny),
                            hp_kernel,
                            padding="valid",
                        ).view(x_batch.shape[0], 2, -1)
                        l_hp_lite = torch.mean(torch.square(err_hp[:, 0, :]))
                        l_hp_full = torch.mean(torch.square(err_hp[:, 1, :]))
                        total_step_loss = total_step_loss + fir_hp_weight * (
                            l_hp_full + lite_weight * l_hp_lite
                        )

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

            # 8a. Pre-validation unlucky training spike check (CL-37)
            is_train_spike, spike_desc = evaluate_loss_spike_condition(
                train_loss_epoch=avg_train_loss,
                min_train_loss=min_train_loss,
                spike_ratio_threshold=spike_ratio_threshold,
            )
            if is_train_spike and attempt < max_attempts - 1 and not fast_dev_run:
                spike_restarts += 1
                next_lr = lr * (0.85 ** (attempt + 1))
                print(
                    f"  [Native Trainer] Unlucky loss spike detected at epoch {epoch:03d} "
                    f"({spike_desc}). Restarting training (attempt {attempt + 2}/{max_attempts}, "
                    f"lr={next_lr:.2e})...",
                    flush=True,
                )
                restarted_on_spike = True
                break

            if math.isfinite(avg_train_loss) and avg_train_loss > 0.0:
                min_train_loss = min(min_train_loss, avg_train_loss)

            # 8b. Batched Vectorized Validation Pass (with in-VRAM EMA Shadow Evaluation)
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
                        val_err_full_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        val_err_lite_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        val_tgt_acc = torch.zeros((), device=hw_device, dtype=torch.float32)
                        val_hp_full_acc = torch.zeros((), device=hw_device, dtype=torch.float32)

                        for s in range(val_steps):
                            bs = min(val_bs, val_chunks - s * val_bs)
                            if bs <= 0:
                                break
                            c_indices = all_val_indices[s * val_bs : s * val_bs + bs]
                            c_bx = x_val_padded[c_indices + arange_nx].unsqueeze(1)
                            c_by = y_val[c_indices + arange_ny]

                            c_pred = model(c_bx)
                            c_err = c_pred - c_by.unsqueeze(1)
                            c_err_lite = c_err[:, 0, :]
                            c_err_full = c_err[:, 1, :]

                            val_err_full_acc.add_(torch.sum(torch.square(c_err_full.float())))
                            val_err_lite_acc.add_(torch.sum(torch.square(c_err_lite.float())))
                            val_tgt_acc.add_(torch.sum(torch.square(c_by.float())))

                            if hp_kernel is not None and fir_hp_weight > 0.0 and ny >= 255:
                                c_hp_full = F.conv1d(
                                    c_err_full.unsqueeze(1),
                                    hp_kernel,
                                    padding="valid",
                                )
                                val_hp_full_acc.add_(torch.sum(torch.square(c_hp_full.float())))

                        cur_esr_full = float((val_err_full_acc / (val_tgt_acc + 1e-12)).item())
                        cur_esr_lite = float((val_err_lite_acc / (val_tgt_acc + 1e-12)).item())
                        total_eval_samples = float(val_chunks * ny)
                        mse_cur = float((val_err_full_acc / total_eval_samples).item())
                        if hp_kernel is not None and fir_hp_weight > 0.0 and ny >= 255:
                            total_hp_samples = float(val_chunks * (ny - 254))
                            val_hp_full = float((val_hp_full_acc / total_hp_samples).item())
                        else:
                            val_hp_full = 0.0
                        mb_val = mse_weight * mse_cur + fir_hp_weight * val_hp_full
                    else:
                        y_val_hat = model(x_val_padded.unsqueeze(0))
                        y_full_pred = y_val_hat[0, 1, :val_samples]
                        y_lite_pred = y_val_hat[0, 0, :val_samples]
                        err_full = (y_full_pred - y_val).float()
                        err_lite = (y_lite_pred - y_val).float()
                        mse_cur = float(torch.mean(torch.square(err_full)).item())
                        target_energy = float(torch.sum(torch.square(y_val.float())).item()) + 1e-12
                        cur_esr_full = (
                            float(torch.sum(torch.square(err_full)).item()) / target_energy
                        )
                        cur_esr_lite = (
                            float(torch.sum(torch.square(err_lite)).item()) / target_energy
                        )
                        val_hp_full = 0.0
                        mb_val = mse_weight * mse_cur

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
                        f"  [Native Trainer] Unlucky validation spike detected at epoch {epoch:03d} "
                        f"({val_spike_desc}). Restarting training (attempt {attempt + 2}/{max_attempts}, "
                        f"lr={next_lr:.2e})...",
                        flush=True,
                    )
                    restarted_on_spike = True
                    break

                if math.isfinite(mb_val) and mb_val > 0.0:
                    min_val_loss_seen = min(min_val_loss_seen, mb_val)

                # Anti-Drift Pareto Early Stopping Evaluation
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
                hp_high_history.append(val_hp_full)

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
                    f"  [Native Trainer] Early stopping triggered at epoch {epoch:03d} "
                    f"({stop_reason}).",
                    flush=True,
                )
                break

        if not restarted_on_spike:
            break

    total_train_dur = time.time() - start_train_time

    # 9. Restore Best Pareto Checkpoint & Fold I/O Scale Compensation for 1:1 Hardware Levels
    model.activation_tau = 0.0
    model.eval()
    if best_state_dict is not None and best_epoch > 0:
        print(
            f"  [Native Trainer] Restoring optimal Pareto checkpoint from epoch {best_epoch:03d} "
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
        x_scale_factor=x_scale_factor,
        y_scale_factor=y_scale_factor,
    )

    output_nam.parent.mkdir(parents=True, exist_ok=True)

    if resolved_chain is not None:
        gear_make = resolved_chain.gear_make
        gear_model = resolved_chain.gear_model
        tone_type = resolved_chain.tone_type
        chain_meta: dict[str, Any] = {
            "signal_chain_index": resolved_chain.index,
            "real_world_rig": resolved_chain.rig_name,
            "style": resolved_chain.style,
            "anagram_amp_block": resolved_chain.amp_block,
            "anagram_cab_block": resolved_chain.cab_block,
            "anagram_mic": resolved_chain.mic,
            "anagram_mic_position": resolved_chain.mic_position,
            "rationale": resolved_chain.rationale,
        }
    else:
        gear_make = "Darkglass Anagram Reamp"
        gear_model = wet_path.stem
        tone_type = "overdrive"
        chain_meta = {}

    meta_dict: dict[str, Any] = {
        "name": clean_name,
        "modeled_by": "Allomorph Native Engine",
        "gear_make": gear_make,
        "gear_model": gear_model,
        "gear_type": "rig",
        "tone_type": tone_type,
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
        "training_time_seconds": round(total_train_dur, 2),
        "delay_samples": delay_samples,
        "spike_restarts": spike_restarts,
        "send_gain_db": round(resolved_send_gain_db, 4),
        "send_gain_source": send_gain_source,
        "input_scale_compensation": round(x_scale_factor, 6),
        "output_scale_compensation": round(1.0 / y_scale_factor, 6),
        **chain_meta,
    }
    if stop_reason:
        meta_dict["stop_reason"] = stop_reason

    model.export_nam(
        out_path=output_nam,
        sample_rate=float(sr_dry),
        metadata=meta_dict,
    )

    sidecar_out = output_nam.with_suffix(".json")
    sidecar_out.write_text(
        json.dumps(
            {
                "nam_file": output_nam.name,
                "source_wav": dry_path.name,
                "target_wav": wet_path.name,
                **meta_dict,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    full_db = 10.0 * math.log10(max(best_val_esr, 1e-12))
    lite_db = 10.0 * math.log10(max(best_val_esr_lite, 1e-12))
    print("=======================================================", flush=True)
    print(
        f" Native Training Complete in {total_train_dur:.2f}s ({total_train_dur / 60.0:.1f}m)",
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
    if spike_restarts > 0:
        print(f"  Spike Restarts: {spike_restarts}", flush=True)
    if stop_reason:
        print(f"  Stop Condition: {stop_reason}", flush=True)
    print("  Status: Ready for Darkglass Anagram Block 1 loading.", flush=True)
    print("=======================================================\n", flush=True)
    return True


def train_reamp(
    reamp_target: Annotated[
        Path,
        Parameter(
            name=["--reamp", "-r"],
            help="Path to a re-amped wet WAV file or directory containing re-amped WAVs.",
        ),
    ] = REPO_ROOT / "reamp",
    *,
    input_file: Annotated[
        Path,
        Parameter(
            name=["--input", "-i"],
            help="Path to dry excitation input WAV file.",
        ),
    ] = REPO_ROOT / "reamp" / "input.wav",
    output_dir: Annotated[
        Path,
        Parameter(
            name=["--output-dir", "-o"],
            help="Directory to save exported .nam models.",
        ),
    ] = REPO_ROOT / "models" / "reamp",
    signal_chains_file: Annotated[
        Path,
        Parameter(
            name=["--signal-chains"],
            help="Path to Darkglass Anagram signal chains markdown table.",
        ),
    ] = REPO_ROOT / "reamp" / "signal_chains.md",
    send_gain: Annotated[
        float | None,
        Parameter(
            name=["--send-gain", "-g"],
            help=(
                "Explicit USB send gain in dB applied during re-amping. "
                "If omitted, auto-reads <stem>.json sidecar written by reamper capture "
                "(or defaults to 0.0 dB)."
            ),
        ),
    ] = None,
    epochs: Annotated[
        int,
        Parameter(name=["--epochs"], help="Maximum training epochs budget."),
    ] = NATIVE_DEFAULT_MAX_EPOCHS,
    min_epochs: Annotated[
        int,
        Parameter(name=["--min-epochs"], help="Minimum warmup epochs before early stopping."),
    ] = NATIVE_DEFAULT_MIN_EPOCHS,
    patience: Annotated[
        int,
        Parameter(name=["--patience"], help="Early stopping plateau patience epochs."),
    ] = NATIVE_DEFAULT_PATIENCE,
    batch_size: Annotated[
        int | str,
        Parameter(name=["--batch-size"], help="Training batch size ('auto' or integer)."),
    ] = "auto",
    precision: Annotated[
        str,
        Parameter(
            name=["--precision"],
            help="Training precision ('auto', 'bf16-mixed', '16-mixed', '32-true').",
        ),
    ] = "auto",
    lr: Annotated[
        float,
        Parameter(name=["--lr"], help="Initial learning rate."),
    ] = NATIVE_DEFAULT_LR,
    weight_decay: Annotated[
        float,
        Parameter(name=["--weight-decay"], help="Adam L2 weight decay regularization."),
    ] = NATIVE_DEFAULT_WEIGHT_DECAY,
    fir_hp_weight: Annotated[
        float,
        Parameter(name=["--fir-hp-weight"], help="High-pass loss weight (tuned for cab roll-off)."),
    ] = 0.75,
    fir_hp_fc: Annotated[
        float,
        Parameter(name=["--fir-hp-fc"], help="High-pass cutoff frequency in Hz."),
    ] = 1000.0,
    val_interval: Annotated[
        int,
        Parameter(name=["--val-interval"], help="Validation interval in epochs during warmup."),
    ] = NATIVE_DEFAULT_VAL_INTERVAL,
    ema_decay: Annotated[
        float,
        Parameter(name=["--ema-decay"], help="Exponential moving average weight decay."),
    ] = NATIVE_DEFAULT_EMA_DECAY,
    no_align: Annotated[
        bool,
        Parameter(name=["--no-align"], help="Skip latency verification / alignment check."),
    ] = False,
    device: Annotated[
        str,
        Parameter(name=["--device"], help="Target compute device ('auto', 'mps', 'cuda', 'cpu')."),
    ] = "auto",
    seed: Annotated[
        int | None,
        Parameter(name=["--seed"], help="Random seed for deterministic training."),
    ] = 42,
    fast_dev_run: Annotated[
        bool,
        Parameter(name=["--fast-dev-run"], help="Run a single-step 1-epoch smoke test."),
    ] = False,
) -> None:
    """Trains production-grade NAM Architecture 2 models from re-amped audio stems."""
    if not input_file.exists() and input_file == REPO_ROOT / "reamp" / "input.wav":
        fallback_input = REPO_ROOT / "audio" / "input.wav"
        if fallback_input.exists():
            input_file = fallback_input
    if not input_file.exists():
        raise FileNotFoundError(f"Dry excitation file '{input_file}' not found.")

    input_resolved = input_file.resolve()
    targets: list[Path] = []
    if reamp_target.is_file() and reamp_target.suffix.lower() == ".wav":
        targets.append(reamp_target)
    elif reamp_target.is_dir():
        targets.extend(sorted(reamp_target.glob("*.wav")))
        # Exclude dry excitation input.wav and raw stems if save_raw was used
        targets = [
            t
            for t in targets
            if not t.stem.endswith("_raw")
            and t.stem.lower() != "input"
            and t.resolve() != input_resolved
        ]
    else:
        raise FileNotFoundError(f"Target re-amp file or directory '{reamp_target}' not found.")

    if not targets:
        print(f"No .wav files found to train in '{reamp_target}'.", flush=True)
        return

    chains = load_signal_chains(signal_chains_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================", flush=True)
    print(" Allomorph Reamper - Re-Amped Rig NAM Native Trainer", flush=True)
    print("=======================================================", flush=True)
    print(f"  Excitation Input: {input_file}", flush=True)
    print(f"  Signal Chains:    {len(chains)} configured ({signal_chains_file.name})", flush=True)
    print(f"  Targets Found:    {len(targets)} WAV stem(s)", flush=True)
    for idx, t in enumerate(targets, 1):
        matched = resolve_signal_chain_for_stem(t.name, chains=chains)
        stem_gain_db, stem_gain_src = resolve_stem_send_gain_db(t, explicit_send_gain_db=send_gain)
        chain_desc = (
            f" -> [#{matched.index} {matched.rig_name} | {matched.amp_block} + {matched.cab_block}]"
            if matched is not None
            else ""
        )
        print(
            f"    [{idx}] {t.name}{chain_desc} (send_gain={stem_gain_db:+.2f} dB via {stem_gain_src})",
            flush=True,
        )
    print(f"  Model Output Dir: {output_dir}", flush=True)
    print("=======================================================\n", flush=True)

    for idx, target_wav in enumerate(targets, 1):
        model_out = output_dir / f"{target_wav.stem}.nam"
        matched_chain = resolve_signal_chain_for_stem(target_wav.name, chains=chains)
        if len(targets) > 1:
            print(f"[{idx}/{len(targets)}] Training Model: {model_out.name}", flush=True)
        train_single_reamp_pair(
            dry_path=input_file,
            wet_path=target_wav,
            output_nam=model_out,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            batch_size=batch_size,
            precision=precision,
            lr=lr,
            weight_decay=weight_decay,
            fir_hp_weight=fir_hp_weight,
            fir_hp_fc=fir_hp_fc,
            val_interval=val_interval,
            ema_decay=ema_decay,
            auto_align=not no_align,
            device_name=device,
            seed=seed,
            fast_dev_run=fast_dev_run,
            signal_chain=matched_chain,
            send_gain_db=send_gain,
        )
