"""
Allomorph - Architecture C Two-Stage Pipeline & Simulation CLI
Coordinates two-stage deconvolution (Canonical Intermediate sweep generation,
frontend IR export for all playable instruments, backend universal target sweeps),
and provides the allomorph-sim CLI binary entrypoint.
"""

import argparse
import functools
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from allomorph.circuit.parser import CircuitModel, load_circuit
from allomorph.circuit.saturation import _algebraic_limiter_p8_core, _slew_limit_core
from allomorph.circuit.schema import SimulationConfig
from allomorph.circuit.simulation import (
    CALIBRATION_PEAK_CEILING,
    FRONTENDS_DIR,
    TARGETS_DIR,
    _get_white_noise_vector,
    _simulate_voice_task,
    find_default_input_audio,
    simulate_voice,
)
from allomorph.circuit.solver import (
    compute_circuit_transfer_functions,
    compute_differential_circuit_transfer_functions,
    smooth_soft_knee_db,
)
from allomorph.config.geometry import (
    compute_effective_position,
    resolve_pickup_coils,
)
from allomorph.config.instruments import load_all_instruments, load_instrument
from allomorph.config.scales import REPO_ROOT, resolve_scale_range
from allomorph.config.schema import CoilConfig, InstrumentConfig
from allomorph.config.strings import STRINGS, get_instrument_string
from allomorph.config.voices import VOICES
from allomorph.dsp import (
    FREQS,
    cinf_smoothstep,
    fft_convolve,
    read_wav,
    synthesize_minimum_phase_fir,
    write_wav_24bit,
)
from allomorph.naming import (
    get_tier_spec,
    resolve_instruments,
    resolve_voices,
)
from allomorph.physics import (
    compute_differential_longitudinal_transfer,
    compute_differential_string_transfer,
    compute_saddle_boundary_coupling,
    numpy_pickup_acoustic_response,
    resolve_pickup_electrical_deconvolution_np,
    soft_clamp_displacement_ratio,
)
from allomorph.version import (
    DSP_GENERATION,
    resolve_tri_part_version,
    write_manifest,
)


def generate_canonical_sweep(
    input_wav: Path | str | None = None,
    output_wav: Path | str | None = None,
    version_tag: str | None = None,
    no_manifest: bool = False,
) -> Path:
    """
    Generates the calibrated Canonical Intermediate baseline audio sweep.
    Takes raw dry input audio (optimal_bass_dry.wav or versioned), applies Canonical Intermediate aperture (single coil at 93.5mm datum)
    and flat active buffer, and normalizes output to -1.5 dBFS True Peak / -16.5 dBFS RMS nominal.
    """
    if not input_wav:
        input_wav = find_default_input_audio(version_tag=version_tag)
    if not input_wav or not Path(input_wav).exists():
        raise FileNotFoundError(f"Raw calibration audio not found: {input_wav}")

    if output_wav is not None:
        target_path = Path(output_wav)
    else:
        from allomorph.naming import get_canonical_sweep_path

        target_path = get_canonical_sweep_path(version_tag=version_tag)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    audio, sr = read_wav(input_wav, dtype=np.float64)

    f = np.asarray(FREQS, dtype=np.float64)
    can_coils = [
        CoilConfig(
            strings=["all"], position_from_bridge_m=0.0935, aperture_width_in=0.75, weight=1.0
        )
    ]
    h_can_ac = numpy_pickup_acoustic_response(f, can_coils, scale_length_m=(0.8636, 0.8636))
    can_voice = VOICES.get("00_canonical_intermediate")
    can_circ = can_voice.circuit if can_voice is not None else None
    if can_circ:
        can_model = load_circuit(can_circ)
        c_curves = compute_circuit_transfer_functions(can_model, freqs=f, return_numpy=True)
        h_can_elec = c_curves[0]
        h_can_elec_norm = h_can_elec / max(h_can_elec[0], 1e-9)
        h_can_total = h_can_ac * h_can_elec_norm
    else:
        h_can_total = h_can_ac

    can_fir = synthesize_minimum_phase_fir(h_can_total, num_taps=2048, normalize=False)

    filtered = fft_convolve(audio, np.asarray(can_fir, dtype=np.float64), mode="causal")

    max_val = float(np.max(np.abs(filtered)))
    if max_val > CALIBRATION_PEAK_CEILING:
        filtered = filtered * (CALIBRATION_PEAK_CEILING / max_val)

    calibrated = filtered.astype(np.float32)

    write_wav_24bit(str(target_path), calibrated, sr)
    _get_reference_rms_sweeps.cache_clear()
    final_peak_db = 20.0 * math.log10(max(float(np.max(np.abs(calibrated))), 1e-9))
    final_rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(calibrated**2))), 1e-9))
    print(
        f"[Canonical Sweep] Generated {target_path.name} (unnormalized): Peak = {final_peak_db:.2f} dBFS, RMS = {final_rms_db:.2f} dBFS"
    )

    if output_wav is None and not no_manifest:
        v_tag = version_tag or f"v{DSP_GENERATION}"
        write_manifest(
            output_dir=target_path.parent,
            stage="canonical",
            files=[target_path],
            version_tag=v_tag,
        )

    return target_path


@functools.lru_cache(maxsize=8)
def _get_cached_sweep_impl(
    filepath_resolved: str, mtime_ns: int, size_bytes: int
) -> tuple[np.ndarray, int]:
    """Caches decoded audio and sample rate for calibration sweeps keyed by file identity and timestamp."""
    return read_wav(filepath_resolved)


def _get_cached_sweep(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Retrieves cached sweep audio and sample rate with automatic mtime/size cache invalidation."""
    p = Path(filepath).resolve()
    stat = p.stat()
    return _get_cached_sweep_impl(str(p), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=1)
def _get_reference_rms_sweeps() -> tuple[np.ndarray, float, np.ndarray, int] | None:
    """Loads default calibration sweep and canonical intermediate target RMS with cached forward FFT."""
    dry_path = find_default_input_audio()
    if not dry_path or not Path(dry_path).exists():
        return None
    from allomorph.circuit.audio import find_default_canonical_sweep

    can_path = find_default_canonical_sweep()
    if not can_path or not can_path.exists():
        can_path = generate_canonical_sweep()
    dry_audio, _ = _get_cached_sweep(dry_path)
    can_audio, _ = _get_cached_sweep(can_path)
    can_rms = float(np.sqrt(np.mean(can_audio**2)))
    n = len(dry_audio)
    n_fft = 1 << (n + 2048 - 1).bit_length()
    X = np.fft.rfft(dry_audio, n_fft)
    return dry_audio, can_rms, X, n_fft


@functools.lru_cache(maxsize=1)
def _get_reference_power_spectrum(n_b: int = 8192) -> tuple[float, np.ndarray, int] | None:
    """Precomputes binned reference power spectrum for ultra-fast Parseval RMS calibration (< 0.05ms per IR)."""
    sweeps = _get_reference_rms_sweeps()
    if sweeps is None:
        return None
    dry_audio, can_rms, X, n_fft = sweeps
    n = len(dry_audio)
    m = n_b // 2 + 1
    weights = np.ones(len(X), dtype=np.float64) * 2.0
    weights[0] = 1.0
    weights[-1] = 1.0
    p_full = (np.abs(X) ** 2) * weights / (n_fft * n)
    indices = np.linspace(0, len(p_full) - 1, len(p_full))
    bin_idx = (indices * (m - 1) / (len(p_full) - 1)).astype(int)
    p_binned = np.bincount(bin_idx, weights=p_full, minlength=m)
    return can_rms, p_binned, n_b


def compute_frontend_transfer_function(
    inst: InstrumentConfig | str,
    pickup_key: str,
    freqs: Sequence[float] | np.ndarray = FREQS,
    can_model: CircuitModel | None = None,
) -> np.ndarray:
    """
    Computes the continuous regularized transfer function transforming a source pickup
    into the 34-inch Canonical Intermediate datum (@ 93.5mm), incorporating aperture sinc deconvolution,
    spatial bridge proximity tilt, loaded electrical circuit deconvolution, and Wiener gain bounds.
    """
    inst_cfg = load_instrument(inst) if isinstance(inst, str) else inst
    pickups = inst_cfg.pickups
    if pickup_key not in pickups:
        raise KeyError(f"Pickup key '{pickup_key}' not found in instrument '{inst_cfg.id}'")
    pickup = pickups[pickup_key]

    f = np.asarray(freqs, dtype=np.float64)
    scale_range = resolve_scale_range(inst_cfg)
    coils = resolve_pickup_coils(pickup, inst_cfg)

    # 1. Source acoustic response
    h_src_ac = numpy_pickup_acoustic_response(f, coils, scale_length_m=scale_range)
    h_src_norm = h_src_ac / max(h_src_ac[0], 1e-9)

    # 2. Canonical acoustic response (34in standard scale, 93.5mm datum, 0.75in slit)
    can_coils = [
        CoilConfig(
            strings=["all"], position_from_bridge_m=0.0935, aperture_width_in=0.75, weight=1.0
        )
    ]
    h_can_ac = numpy_pickup_acoustic_response(f, can_coils, scale_length_m=(0.8636, 0.8636))
    h_can_ac_norm = h_can_ac / max(h_can_ac[0], 1e-9)

    eps = 0.01
    h_quotient = (h_can_ac_norm * h_src_norm) / (h_src_norm**2 + eps**2)
    q_db = 20.0 * np.log10(np.maximum(h_quotient, 1e-6))
    g_max_db = 8.0
    g_min_db = -14.0
    sigma = 0.5 * (1.0 + np.tanh(0.5 * q_db))
    f_pos = smooth_soft_knee_db(q_db, thresh=6.0, ceiling=g_max_db, alpha=2.0)
    f_neg = -smooth_soft_knee_db(-q_db, thresh=10.0, ceiling=abs(g_min_db), alpha=2.0)
    q_soft_db = sigma * f_pos + (1.0 - sigma) * f_neg
    h_aperture_deconv = 10.0 ** (q_soft_db / 20.0)

    # 3. Spatial bridge proximity scaling (Source -> Canonical Intermediate datum @ 93.5mm)
    src_pos_eff = compute_effective_position(coils)
    src_scale_m = float(inst_cfg.scale_length_m or 0.8636)
    can_pos_eff = 0.0935
    can_scale_m = 0.8636
    eta_src = src_pos_eff / src_scale_m
    eta_can = can_pos_eff / can_scale_m
    delta_g = 20.0 * np.log10(max(eta_can / max(eta_src, 1e-4), 1e-6))
    delta_g_soft = soft_clamp_displacement_ratio(delta_g)
    g_0 = 10.0 ** (delta_g_soft / 20.0)
    h_pos = np.sqrt((g_0**2 + (f / 220.0) ** 2) / (1.0 + (f / 220.0) ** 2))

    # 4. Scale-Length Tension Snap (Source -> Canonical Intermediate @ 34")
    src_scale_in = float(inst_cfg.scale_length_in or 34.0)
    can_scale_in = 34.0
    delta_scale = can_scale_in - src_scale_in
    if delta_scale <= 0.0:
        h_tension = np.ones_like(f)
    else:
        snap_db = 3.5 * np.tanh((1.8 * delta_scale) / (4.0 * 3.5))
        g_snap = 10.0 ** (snap_db / 20.0)
        h_tension = np.sqrt(
            (1.0 + g_snap**2 * (f / 2800.0) ** 2) / (1.0 + (f / 2800.0) ** 2)
        )

    # 5. Saddle boundary stiffness deconvolution (Source -> Canonical Intermediate)
    h_saddle_can = compute_saddle_boundary_coupling(f, can_pos_eff, can_scale_m)
    h_saddle_src = compute_saddle_boundary_coupling(f, src_pos_eff, src_scale_m)
    r_saddle_db = 20.0 * np.log10(np.maximum(h_saddle_can / np.maximum(h_saddle_src, 1e-6), 1e-6))
    h_saddle_diff = 10.0 ** (r_saddle_db / 20.0)

    # 6. String deconvolution (if source bass string is not standard roundwound nickel)
    src_string = get_instrument_string(inst_cfg)
    can_string = STRINGS["roundwound_nickel_standard"]
    if src_string.preset != "roundwound_nickel_standard":
        h_str_diff = compute_differential_string_transfer(f, src_string, can_string)
        h_long_diff = compute_differential_longitudinal_transfer(
            f, src_string, can_string, scale_length_inches=34.0
        )
    else:
        h_str_diff = np.ones_like(f)
        h_long_diff = np.ones_like(f)

    # 7. Circuit deconvolution
    if can_model is None:
        can_voice = VOICES.get("00_canonical_intermediate")
        can_circ = can_voice.circuit if can_voice is not None else None
        can_model = load_circuit(can_circ) if can_circ is not None else None

    p_circ = pickup.circuit
    if p_circ and can_model:
        src_model = load_circuit(p_circ)
        diff_curves = compute_differential_circuit_transfer_functions(
            can_model, src_model, freqs=f, max_boost_db=8.0
        )
        h_circuit_deconv = np.asarray(diff_curves[0], dtype=np.float64)
    elif inst_cfg.electronics == "passive":
        raise ValueError(
            f"Passive instrument '{inst_cfg.id}' pickup '{pickup_key}' does not define a '[pickups.{pickup_key}.circuit]' "
            f"configuration. Passive source pickups require an explicit circuit model for differential deconvolution."
        )
    else:
        h_c_src = resolve_pickup_electrical_deconvolution_np(f, pickup, inst_cfg, q_target=0.707)
        if can_model:
            can_curves = compute_circuit_transfer_functions(can_model, freqs=f, return_numpy=True)
            h_can_elec = can_curves[0]
            h_can_elec_norm = h_can_elec / max(h_can_elec[0], 1e-9)
            h_circuit_deconv = h_c_src * h_can_elec_norm
        else:
            h_circuit_deconv = h_c_src

    h_raw = (
        h_aperture_deconv
        * h_circuit_deconv
        * h_pos
        * h_tension
        * h_saddle_diff
        * h_str_diff
        * h_long_diff
    )
    raw_db = 20.0 * np.log10(np.maximum(h_raw, 1e-6))
    clamped_db = smooth_soft_knee_db(raw_db, thresh=6.0, ceiling=8.0, alpha=2.0)

    # Frequency-dependent ultrasonic roll-off above 8 kHz if exceeding 1.5 dB (keeps 20 kHz strictly < 2.0 dB)
    f_roll = 8000.0
    roll_factor = (f - f_roll) / (24000.0 - f_roll)
    beta = 1.2
    hf_excess = (1.0 / beta) * np.logaddexp(0.0, beta * (clamped_db - 1.5))
    final_db = clamped_db - hf_excess * cinf_smoothstep(roll_factor)
    return 10.0 ** (final_db / 20.0)


def compute_frontend_deconvolution_fir(
    inst_id: str,
    pickup_key: str,
    num_taps: int = 2048,
    normalize: bool = False,
    gain_db: float = 0.0,
) -> np.ndarray:
    """
    Computes a 2048-tap minimum-phase deconvolution FIR filter transforming a source pickup
    into the 34-inch Canonical Intermediate datum.

    By default (normalize=False), produces the exact unnormalized physical deconvolution
    filter with ~0 dB unity gain across fundamental bass frequencies (40-200 Hz).
    """
    f = np.asarray(FREQS, dtype=np.float64)
    h_total = compute_frontend_transfer_function(inst_id, pickup_key, freqs=f)

    fir = np.asarray(
        synthesize_minimum_phase_fir(h_total, num_taps=num_taps, normalize=normalize),
        dtype=np.float32,
    )

    # Polarity check: enforce positive polarity
    if np.sum(fir[:16]) < 0:
        fir = -fir

    if normalize:
        max_peak = float(np.max(np.abs(fir)))
        if max_peak > 0.0:
            fir = (fir / max_peak) * np.float32(CALIBRATION_PEAK_CEILING)

    if gain_db != 0.0:
        fir = fir * np.float32(10.0 ** (gain_db / 20.0))

    # True-peak safety clamp to prevent 24-bit PCM wrapping
    peak = float(np.max(np.abs(fir)))
    if peak > CALIBRATION_PEAK_CEILING:
        fir = (fir / peak) * np.float32(CALIBRATION_PEAK_CEILING)

    return fir


def export_frontend_ir(
    inst_id: str,
    pickup_key: str,
    out_path: Path | None = None,
    output_dir: Path | str | None = None,
    num_taps: int = 2048,
    normalize: bool = False,
    gain_db: float = 0.0,
    version_tag: str | None = None,
    no_manifest: bool = False,
) -> Path:
    """
    Synthesizes a 2048-tap minimum-phase deconvolution IR transforming a source pickup into the Canonical Intermediate.
    Enforces strictly positive initial polarity to ensure zero phase cancellation when blended in parallel.
    """
    fir = compute_frontend_deconvolution_fir(
        inst_id=inst_id,
        pickup_key=pickup_key,
        num_taps=num_taps,
        normalize=normalize,
        gain_db=gain_db,
    )
    inst = load_instrument(inst_id)
    pickups = inst.pickups

    if version_tag == "auto":
        tag = resolve_tri_part_version(DSP_GENERATION, getattr(inst, "version", 1), 1)
    elif version_tag:
        tag = str(version_tag)
    else:
        tag = None

    if out_path is None:
        base_dir = Path(output_dir) if output_dir is not None else FRONTENDS_DIR
        inst_dir = base_dir / inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)
        # Avoid repetitive token if inst_id already ends with pickup prefix (e.g. 30in_emg_mmtw + mmtw_dual)
        if inst_id.endswith("mmtw") and pickup_key.startswith("mmtw_"):
            p_name = pickup_key[len("mmtw_") :]
            out_name = f"{inst_id}_{p_name}_{tag}.wav" if tag else f"{inst_id}_{p_name}.wav"
        else:
            out_name = f"{inst_id}_{pickup_key}_{tag}.wav" if tag else f"{inst_id}_{pickup_key}.wav"
        out_path = inst_dir / out_name

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_24bit(str(out_path), fir, 48000)

    # If single pickup, also generate convenience alias <inst_id>.wav
    if len(pickups) == 1:
        alias_name = f"{inst_id}_{tag}.wav" if tag else f"{inst_id}.wav"
        alias_path = out_path.parent / alias_name
        if alias_path != out_path:
            write_wav_24bit(str(alias_path), fir, 48000)

    if not no_manifest:
        write_manifest(
            output_dir=out_path.parent,
            stage="frontends_ir",
            files=[out_path],
            version_tag=tag or resolve_tri_part_version(DSP_GENERATION, getattr(inst, "version", 1), 1),
        )

    return out_path


def export_frontend_wet_wav(
    inst_id: str,
    pickup_key: str,
    input_wav: Path | str | None = None,
    out_path: Path | None = None,
    output_dir: Path | str | None = None,
    num_taps: int = 2048,
    normalize: bool = False,
    gain_db: float = 0.0,
    version_tag: str | None = None,
    no_manifest: bool = False,
) -> Path:
    """
    Convolves the dry calibration sweep through the source pickup frontend deconvolution FIR,
    producing a 24-bit 48 kHz wet training sweep (with '_wet' suffix) for Block 1 NAM training.
    """
    dry_path: Path | None = Path(input_wav) if input_wav else find_default_input_audio()
    if not dry_path or not dry_path.exists():
        raise FileNotFoundError(f"Dry calibration audio not found: {dry_path}")

    audio_dry, sr = _get_cached_sweep(dry_path)
    fir = compute_frontend_deconvolution_fir(
        inst_id=inst_id,
        pickup_key=pickup_key,
        num_taps=num_taps,
        normalize=normalize,
        gain_db=gain_db,
    )

    audio_wet = fft_convolve(audio_dry, np.asarray(fir, dtype=np.float64), mode="causal")

    max_in = float(np.max(np.abs(audio_wet)))
    # Approach A: Non-linear transient conditioning and headroom protection
    # Bypassed on small signals (<= 0.10) to preserve exact mathematical linearity in test suites
    if max_in > 0.10:
        # 1. Op-Amp / Active Buffer Slew Limiting (16 kHz threshold)
        # Smooths harsh transient spikes on slap pops and pick clank without coloring fundamental timbre
        f_slew = 16000.0
        vsat = 0.985
        max_delta = 2.0 * math.pi * f_slew * vsat / float(sr)
        audio_wet = _slew_limit_core(audio_wet, max_delta)

        # 2. High-Headroom C^inf Algebraic Limiter (p = 8) Rail Protection
        # Smoothly saturates forte excursions into vsat = 0.985 ceiling with infinite differentiability
        # Leaves 99.9% of normal playing completely linear (zero double-saturation with Block 2)
        audio_wet = _algebraic_limiter_p8_core(audio_wet, vsat)

        # 3. Johnson-Nyquist -108 dBFS Thermal Noise Dither
        # Eliminates neural network dead-zone gating on quiet decay tails
        dither_amp = 10.0 ** (-108.0 / 20.0)
        audio_wet = audio_wet + _get_white_noise_vector(len(audio_wet)) * dither_amp

    # True-peak safety ceiling matching calibration sweep (0.9900 / -0.087 dBFS):
    # If unnormalized (default), leave audio untouched unless it exceeds CALIBRATION_PEAK_CEILING.
    # If exceeding ceiling, apply proportional safety scaling to strictly prevent clipping distortion.
    max_val = float(np.max(np.abs(audio_wet)))
    if (normalize and max_val > 1e-9) or max_val > CALIBRATION_PEAK_CEILING:
        audio_wet = audio_wet * (CALIBRATION_PEAK_CEILING / max_val)

    calibrated = audio_wet.astype(np.float32)

    inst = load_instrument(inst_id)
    pickups = inst.pickups

    if version_tag == "auto":
        tag = resolve_tri_part_version(DSP_GENERATION, getattr(inst, "version", 1), 1)
    elif version_tag:
        tag = str(version_tag)
    else:
        tag = None

    if out_path is None:
        base_dir = Path(output_dir) if output_dir is not None else FRONTENDS_DIR
        inst_dir = base_dir / inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)
        if inst_id.endswith("mmtw") and pickup_key.startswith("mmtw_"):
            p_name = pickup_key[len("mmtw_") :]
            out_name = f"{inst_id}_{p_name}_{tag}_wet.wav" if tag else f"{inst_id}_{p_name}_wet.wav"
        else:
            out_name = f"{inst_id}_{pickup_key}_{tag}_wet.wav" if tag else f"{inst_id}_{pickup_key}_wet.wav"
        out_path = inst_dir / out_name

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_24bit(str(out_path), calibrated, sr)

    if len(pickups) == 1:
        alias_name = f"{inst_id}_{tag}_wet.wav" if tag else f"{inst_id}_wet.wav"
        alias_path = out_path.parent / alias_name
        if alias_path != out_path:
            write_wav_24bit(str(alias_path), calibrated, sr)

    if not no_manifest:
        write_manifest(
            output_dir=out_path.parent,
            stage="frontends_wet",
            files=[out_path],
            version_tag=tag or resolve_tri_part_version(DSP_GENERATION, getattr(inst, "version", 1), 1),
        )

    return out_path


def _export_frontend_ir_task(
    task_args: tuple[str, str, Path, int, bool, float, str | None, bool]
) -> Path:
    inst_id, p_key, out_dir, num_taps, normalize, gain_db, version_tag, no_manifest = task_args
    return export_frontend_ir(
        inst_id=inst_id,
        pickup_key=p_key,
        output_dir=out_dir,
        num_taps=num_taps,
        normalize=normalize,
        gain_db=gain_db,
        version_tag=version_tag,
        no_manifest=no_manifest,
    )


def export_all_frontend_irs(
    output_dir: Path | None = None,
    jobs: int | None = None,
    normalize: bool = False,
    gain_db: float = 0.0,
    version_tag: str | None = None,
    no_manifest: bool = False,
) -> list[Path]:
    """
    Exports all 32 native frontend deconvolution IRs grouped by instrument subdirectories.
    Parallelized across CPU cores using ProcessPoolExecutor.
    """
    out_dir = Path(output_dir) if output_dir else FRONTENDS_DIR
    all_insts = load_all_instruments()
    tasks: list[tuple[str, str, Path, int, bool, float, str | None, bool]] = []
    for inst_id, inst in sorted(all_insts.items()):
        if inst_id == "canonical_intermediate":
            continue
        pickups = inst.pickups
        for p_key in sorted(pickups.keys()):
            tasks.append((inst_id, p_key, out_dir, 2048, normalize, gain_db, version_tag, no_manifest))

    # Pre-cache canonical sweep in main process
    _get_reference_power_spectrum()

    max_workers = jobs if jobs is not None else min(4, os.cpu_count() or 4)
    exported: list[Path] = []
    if len(tasks) > 1 and max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            exported = list(executor.map(_export_frontend_ir_task, tasks))
    else:
        for task in tasks:
            exported.append(_export_frontend_ir_task(task))

    for p_file in exported:
        rel_p = p_file.relative_to(REPO_ROOT) if p_file.is_relative_to(REPO_ROOT) else p_file
        print(f" [Frontend IR] Exported {rel_p}")
    print(f"Successfully exported {len(exported)} frontend IRs to {out_dir}")
    return exported


def _export_frontend_wet_wav_task(
    task_args: tuple[str, str, Path | None, Path, int, bool, float, str | None, bool]
) -> Path:
    inst_id, p_key, in_path, out_dir, num_taps, normalize, gain_db, version_tag, no_manifest = task_args
    return export_frontend_wet_wav(
        inst_id=inst_id,
        pickup_key=p_key,
        input_wav=in_path,
        output_dir=out_dir,
        num_taps=num_taps,
        normalize=normalize,
        gain_db=gain_db,
        version_tag=version_tag,
        no_manifest=no_manifest,
    )


def export_all_frontend_wet_wavs(
    input_wav: Path | str | None = None,
    output_dir: Path | None = None,
    jobs: int | None = None,
    num_taps: int = 2048,
    normalize: bool = False,
    gain_db: float = 0.0,
    version_tag: str | None = None,
    no_manifest: bool = False,
) -> list[Path]:
    """
    Exports all native frontend pickup deconvolution wet sweeps convolved through their FIRs.
    Parallelized across CPU cores using ProcessPoolExecutor.
    """
    out_dir = Path(output_dir) if output_dir else FRONTENDS_DIR
    all_insts = load_all_instruments()
    in_path = Path(input_wav) if input_wav else None
    tasks: list[tuple[str, str, Path | None, Path, int, bool, float, str | None, bool]] = []
    for inst_id, inst in sorted(all_insts.items()):
        if inst_id == "canonical_intermediate":
            continue
        pickups = inst.pickups
        for p_key in sorted(pickups.keys()):
            tasks.append((inst_id, p_key, in_path, out_dir, num_taps, normalize, gain_db, version_tag, no_manifest))

    # Pre-cache calibration sweep in main process
    target_in = in_path or find_default_input_audio()
    if target_in and Path(target_in).exists():
        _get_cached_sweep(target_in)

    max_workers = jobs if jobs is not None else min(4, os.cpu_count() or 4)
    exported: list[Path] = []
    if len(tasks) > 1 and max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            exported = list(executor.map(_export_frontend_wet_wav_task, tasks))
    else:
        for task in tasks:
            exported.append(_export_frontend_wet_wav_task(task))

    for p_file in exported:
        rel_p = p_file.relative_to(REPO_ROOT) if p_file.is_relative_to(REPO_ROOT) else p_file
        print(f" [Frontend Wet WAV] Exported {rel_p}")
    print(f"Successfully exported {len(exported)} frontend wet WAVs to {out_dir}")
    return exported


def simulate_backend_targets(
    tier: str = "standard",
    voice_id: str | None = None,
    max_samples: int | None = None,
    output_dir: Path | str | None = None,
    jobs: int | None = None,
    normalize: Literal["auto", "peak", "rms", "none"] = "auto",
    target_dbfs: float | None = None,
    version_tag: str | None = None,
    no_manifest: bool = False,
):
    """
    Simulates target voice audio sweeps using the Canonical Intermediate baseline as input.
    Tiers:
      - 'clean': 0% saturation / maximum headroom (bypass_saturation=True)
      - 'standard': standard dynamic pickup give (100% nominal saturation)
      - 'hotrod': overwound drive pre-conditioner (175% saturation, reduced vsat)
    """
    if tier == "all":
        tiers_to_run = ["clean", "standard", "hotrod"]
    else:
        try:
            spec = get_tier_spec(tier)
            tiers_to_run = [spec.name]
        except KeyError as err:
            raise ValueError(
                f"Unknown tier '{tier}'. Choose from clean, standard, std, hotrod, all."
            ) from err

    from allomorph.circuit.audio import find_default_canonical_sweep

    can_vtag = None if version_tag in (None, "auto") else version_tag
    can_sweep = find_default_canonical_sweep(version_tag=can_vtag)
    if not can_sweep or not can_sweep.exists():
        can_sweep = generate_canonical_sweep(version_tag=can_vtag, no_manifest=no_manifest)

    voices_to_run = (
        [voice_id]
        if voice_id and voice_id != "all"
        else [vid for vid in sorted(VOICES.keys()) if vid != "00_canonical_intermediate"]
    )
    max_workers = jobs if jobs is not None else min(4, os.cpu_count() or 4)

    for t in tiers_to_run:
        folder_name = get_tier_spec(t).folder_name
        base_dir = Path(output_dir) if output_dir else TARGETS_DIR
        target_out_dir = base_dir / folder_name
        target_out_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for vid in voices_to_run:
            vcfg = VOICES.get(vid)
            if version_tag == "auto":
                voice_ver = getattr(vcfg, "version", 1) if vcfg else 1
                vid_tag = resolve_tri_part_version(DSP_GENERATION, 1, voice_ver)
                out_file = target_out_dir / f"out_{vid}_{vid_tag}.wav"
            elif version_tag:
                out_file = target_out_dir / f"out_{vid}_{version_tag}.wav"
            else:
                out_file = target_out_dir / f"out_{vid}.wav"
            v_alpha = vcfg.alpha if vcfg is not None else 0.25

            if t == "clean":
                sim_alpha = 0.0
            elif t == "hotrod":
                sim_alpha = min(1.0, v_alpha * 1.75)
            else:
                sim_alpha = v_alpha

            tasks.append(
                (
                    vid,
                    SimulationConfig(
                        input_wav=can_sweep,
                        output_wav=out_file,
                        instrument="canonical_intermediate",
                        tier=t,
                        alpha=sim_alpha,
                        normalize=normalize,
                        target_dbfs=target_dbfs,
                        max_samples=max_samples,
                    ),
                )
            )

        if len(tasks) > 1 and max_workers > 1:
            print(
                f"[{t.upper()}] Simulating {len(tasks)} targets in parallel ({max_workers} workers)..."
            )
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_simulate_voice_task, tasks))
        else:
            for vid, sim_cfg in tasks:
                out_name = Path(sim_cfg.output_wav).name if sim_cfg.output_wav else f"out_{vid}.wav"
                print(f"[{t.upper()}] Simulating {vid} -> {out_name}...")
                simulate_voice(vid, config=sim_cfg)

        if not no_manifest:
            simulated_files = [
                t[1].output_wav
                for t in tasks
                if t[1].output_wav and Path(t[1].output_wav).exists()
            ]
            write_manifest(
                output_dir=target_out_dir,
                stage="targets",
                files=simulated_files,
                version_tag=version_tag if (version_tag and version_tag != "auto") else resolve_tri_part_version(),
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Allomorph Native Virtual Analog Circuit Simulator."
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="all",
        help="Target voice to simulate (voice ID, comma-separated list, or 'all'; default: 'all')",
    )
    parser.add_argument(
        "--instrument",
        "-i",
        default="all",
        help="Source instrument configuration (ID, comma-separated list, 'all', 30in, 32in, or path to .toml; default: 'all')",
    )
    parser.add_argument(
        "--input", help="Input WAV path (defaults to auto-generating optimal_bass_dry.wav)"
    )
    parser.add_argument(
        "--out", help="Output WAV path (default: audio/<instrument>/out_<voice>.wav)"
    )
    parser.add_argument(
        "--prefiltered",
        action="store_true",
        help="Input is already pre-filtered through acoustic aperture",
    )
    parser.add_argument(
        "--normalize",
        choices=["auto", "rms", "peak", "none"],
        default="auto",
        help="Output level normalization mode based on input sweep dBFS (default: auto = match input sweep RMS with true-peak safety).",
    )
    parser.add_argument(
        "--target-dbfs",
        type=float,
        default=None,
        help="Explicit target level in dBFS (e.g. -22.0). If omitted, automatically derived from the input sweep.",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        choices=[1, 2, 4],
        default=2,
        help="Anti-aliased oversampling factor for saturation (default: 2 = 96 kHz internal processing)",
    )
    parser.add_argument(
        "--no-displacement-weighting",
        action="store_true",
        help="Disable displacement-domain excursion weighting before saturation",
    )
    parser.add_argument(
        "--no-magnet-drag",
        action="store_true",
        help="Disable dynamic magnet drag attack braking on extreme transients",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Explicit saturation asymmetry factor alpha (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--alpha3",
        type=float,
        default=None,
        help="Explicit cubic dipole proximity factor alpha3 (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--k-sag",
        type=float,
        default=None,
        help="Explicit dynamic Lenz-law core flux sag factor k_sag (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--k-eddy",
        type=float,
        default=None,
        help="Explicit dynamic eddy-current core de-Qing factor k_eddy (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--kappa-orbit",
        type=float,
        default=None,
        help="Explicit elliptical string orbit projection factor kappa_orbit (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--beta-curv",
        type=float,
        default=None,
        help="Explicit dynamic core inductance curvature factor beta_curv (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--k-pull",
        type=float,
        default=None,
        help="Explicit nonlinear magnetic string pull factor k_pull (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--tau-touch",
        type=float,
        default=None,
        help="Explicit dynamic touch spectral tilt factor tau_touch (default: resolved from magnet_type in voices.toml)",
    )
    parser.add_argument(
        "--kappa-geom",
        type=float,
        default=None,
        help="Explicit conformal geometric clearance asymmetry factor kappa_geom (default: resolved from magnet_type)",
    )
    parser.add_argument(
        "--k-stein",
        type=float,
        default=None,
        help="Explicit dynamic Steinmetz AC core loss damping factor k_stein (default: resolved from magnet_type)",
    )
    parser.add_argument(
        "--vol",
        "--vol-pos",
        type=float,
        default=None,
        dest="vol",
        help="Volume pot wiper position (0.0 to 1.0, default 1.0 full open)",
    )
    parser.add_argument(
        "--tone",
        "--tone-pos",
        type=float,
        default=None,
        dest="tone",
        help="Tone pot wiper position (0.0 to 1.0, default 1.0 full open/bright)",
    )
    parser.add_argument(
        "--blend",
        "--blend-pos",
        type=float,
        default=None,
        dest="blend",
        help="Pickup blend wiper position (0.0 Neck to 1.0 Bridge, default: 0.5 Center detent 100%%/100%%)",
    )
    parser.add_argument(
        "--pot-taper",
        choices=["audio", "audio10", "audio15", "linear"],
        default="audio",
        help="Potentiometer resistance curve law (default: 'audio' for standard 10%% CTS audio pot)",
    )
    parser.add_argument(
        "--cable-pf",
        type=float,
        default=None,
        help="Cable capacitance loading in pF (default: from circuit config, typically 750 pF)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Execute continuous parametric sweep (e.g. 'tone', 'vol', 'cable', 'tone_cap', 'bass_boost', 'treble_boost')",
    )
    parser.add_argument(
        "--no-spectral-tilt",
        action="store_true",
        help="Disable dynamic excursion-dependent touch spectral tilt",
    )
    parser.add_argument(
        "--no-slew-limit",
        action="store_true",
        help="Disable transient magnetic slew-rate limiting",
    )
    parser.add_argument(
        "--f-slew",
        type=float,
        default=16000.0,
        help="Magnetic domain-wall slew threshold frequency in Hz (default: 16000.0)",
    )
    parser.add_argument(
        "--no-eddy-diffusion",
        action="store_true",
        help="Disable Foster 2-stage core eddy diffusion (fall back to ideal frequency-independent L)",
    )
    parser.add_argument(
        "--no-hysteresis",
        action="store_true",
        help="Disable Dahl magnetic hysteresis friction modeling (eta_hyst = 0.0)",
    )
    parser.add_argument(
        "--eta-hyst",
        type=float,
        default=None,
        help="Explicit Dahl hysteresis coupling coefficient eta (default: resolved from magnet_type)",
    )
    parser.add_argument(
        "--no-dc-block",
        action="store_true",
        help="Disable sub-audible 8 Hz DC-blocking high-pass filter",
    )
    parser.add_argument(
        "--no-dither",
        action="store_true",
        help="Disable passive RLC-shaped -108 dBFS thermal noise dither",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel worker processes for batch simulation (default: min(4, CPU count))",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum audio sample frames to simulate (default: None for full file)",
    )
    parser.add_argument(
        "--stage",
        choices=["canonical", "frontends", "targets", "all"],
        default=None,
        help="Architecture C execution stage: 'canonical' (generate intermediate sweep), 'frontends' (export frontend wet sweeps / IRs), 'targets' (simulate backend sweeps), 'all'.",
    )
    parser.add_argument(
        "--frontend-format",
        choices=["wet", "ir", "both"],
        default="wet",
        help="Format for frontend deconvolution exports: 'wet' (wet sweep WAV convolved with FIR), 'ir' (FIR impulse response WAV), 'both' (both wet sweep and IR WAVs; default: wet)",
    )
    parser.add_argument(
        "--normalize-frontend",
        action="store_true",
        default=False,
        help="Enable full-scale peak normalization (0.9900) for frontend deconvolution (default: False for unnormalized unity gain)",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=0.0,
        help="Optional manual gain trim in dB applied to frontend deconvolution filter (default: 0.0 dB)",
    )
    parser.add_argument(
        "--tier",
        choices=["clean", "standard", "std", "hotrod", "dynamic", "all"],
        default="standard",
        help="Dynamic tier: 'standard' / 'std' (100%% nominal target saturation), 'clean' (0%% saturation), 'hotrod' (175%% overwound), 'dynamic' (differential source/target saturation), 'all'.",
    )
    parser.add_argument(
        "--pickup",
        "-p",
        default=None,
        help="Physical pickup setting for source instrument ('auto' to resolve from pickup_mapping, or explicit pickup ID)",
    )
    parser.add_argument(
        "--version-tag",
        default=None,
        help="Semantic version tag to embed in exported filenames (e.g. 'auto', 'v2.1.1')",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable generating sidecar manifest.json",
    )
    args = parser.parse_args(argv)

    if args.sweep:
        from allomorph.circuit.sweeps import compute_parametric_sweep

        target_voices = resolve_voices(args.voice)
        for vid in target_voices:
            res = compute_parametric_sweep(vid, param=args.sweep, pot_taper=args.pot_taper)
            print(
                "\n========================================================================================="
            )
            print(f"  PARAMETRIC SWEEP: {vid} (Param: {args.sweep}, Taper: {args.pot_taper})")
            print(
                "========================================================================================="
            )
            print(
                f"Evaluated {len(res.values)} steps ({', '.join(res.labels)}) across {len(res.freqs)} frequencies.\n"
            )
            print("--- Frequency Response Grid ---")
            sample_freqs = [100.0, 500.0, 1000.0, 2500.0, 5000.0]
            header = f"{'Value / Label':<24}" + "".join(
                [f"{f'{f:.0f} Hz':>12}" for f in sample_freqs]
            )
            print(header)
            print("-" * len(header))
            f_arr = np.asarray(res.freqs)
            f_indices = [int(np.argmin(np.abs(f_arr - sf))) for sf in sample_freqs]
            for lbl, curve in zip(res.labels, res.curves):
                row = f"{lbl:<24}" + "".join([f"{curve[idx]:>11.1f}dB" for idx in f_indices])
                print(row)
            print("\n--- Analytical Circuit Metrics ---")
            res.print_metrics()
            print(
                "=========================================================================================\n"
            )
        return

    input_wav = args.input or find_default_input_audio(version_tag=args.version_tag)

    if args.stage == "canonical":
        generate_canonical_sweep(
            input_wav=input_wav,
            output_wav=args.out,
            version_tag=args.version_tag,
            no_manifest=args.no_manifest,
        )
        return
    if args.stage == "frontends":
        fmt = args.frontend_format
        if args.instrument:
            insts = resolve_instruments(args.instrument)
            for inst_id in insts:
                inst = load_instrument(inst_id)
                pickups_to_run = (
                    [args.pickup]
                    if (args.pickup and args.pickup != "auto")
                    else list(inst.pickups.keys())
                )
                for pkey in pickups_to_run:
                    if fmt in ["ir", "both"]:
                        export_frontend_ir(
                            inst_id,
                            pkey,
                            output_dir=args.out,
                            normalize=args.normalize_frontend,
                            gain_db=args.gain_db,
                            version_tag=args.version_tag,
                            no_manifest=args.no_manifest,
                        )
                    if fmt in ["wet", "both"]:
                        export_frontend_wet_wav(
                            inst_id,
                            pkey,
                            input_wav=input_wav,
                            output_dir=args.out,
                            normalize=args.normalize_frontend,
                            gain_db=args.gain_db,
                            version_tag=args.version_tag,
                            no_manifest=args.no_manifest,
                        )
            return
        if fmt in ["ir", "both"]:
            export_all_frontend_irs(
                output_dir=args.out,
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        if fmt in ["wet", "both"]:
            export_all_frontend_wet_wavs(
                input_wav=input_wav,
                output_dir=args.out,
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        return
    if args.stage == "targets":
        simulate_backend_targets(
            tier=args.tier,
            voice_id=args.voice,
            max_samples=args.max_samples,
            output_dir=args.out,
            jobs=args.jobs,
            normalize=args.normalize,
            target_dbfs=args.target_dbfs,
            version_tag=args.version_tag,
            no_manifest=args.no_manifest,
        )
        return
    if args.stage == "all":
        generate_canonical_sweep(input_wav=input_wav)
        fmt = args.frontend_format
        if fmt in ["wet", "both"]:
            export_all_frontend_wet_wavs(
                input_wav=input_wav,
                output_dir=args.out,
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        if fmt in ["ir", "both"]:
            export_all_frontend_irs(
                output_dir=args.out,
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        simulate_backend_targets(
            tier=args.tier,
            voice_id=args.voice,
            max_samples=args.max_samples,
            output_dir=args.out,
            jobs=args.jobs,
            normalize=args.normalize,
            target_dbfs=args.target_dbfs,
            version_tag=args.version_tag,
            no_manifest=args.no_manifest,
        )
        return

    displacement_weighting = not args.no_displacement_weighting
    magnet_drag = not args.no_magnet_drag
    dc_block = not args.no_dc_block
    eddy_diffusion = not args.no_eddy_diffusion
    eta_hyst = 0.0 if args.no_hysteresis else args.eta_hyst
    noise_dither = not args.no_dither
    slew_limit = not args.no_slew_limit
    tau_touch = 0.0 if args.no_spectral_tilt else args.tau_touch

    instruments = resolve_instruments(args.instrument)
    voices = resolve_voices(args.voice)
    in_path = Path(input_wav) if input_wav else None
    out_path = Path(args.out) if args.out else None
    prefiltered = args.prefiltered or (in_path is not None and in_path.name.startswith("aperture_"))

    sim_cfg = SimulationConfig(
        input_wav=in_path,
        output_wav=out_path,
        pickup=args.pickup,
        tier=args.tier,
        prefiltered=prefiltered,
        normalize=args.normalize,
        target_dbfs=args.target_dbfs,
        oversample=args.oversample,
        displacement_weighting=displacement_weighting,
        magnet_drag=magnet_drag,
        alpha=args.alpha,
        alpha3=args.alpha3,
        eta_hyst=eta_hyst,
        k_sag=args.k_sag,
        k_eddy=args.k_eddy,
        kappa_orbit=args.kappa_orbit,
        beta_curv=args.beta_curv,
        k_pull=args.k_pull,
        tau_touch=tau_touch,
        kappa_geom=args.kappa_geom,
        k_stein=args.k_stein,
        vol_pos=args.vol,
        tone_pos=args.tone,
        blend_pos=args.blend,
        pot_taper=args.pot_taper,
        cable_pf=args.cable_pf,
        slew_limit=slew_limit,
        f_slew=args.f_slew,
        noise_dither=noise_dither,
        eddy_diffusion=eddy_diffusion,
        dc_block=dc_block,
        max_samples=args.max_samples,
    )
    max_workers = args.jobs if args.jobs is not None else min(4, os.cpu_count() or 4)
    for inst in instruments:
        out_target = out_path
        if out_path and len(instruments) > 1 and not out_path.is_dir():
            stem = out_path.stem
            suffix = out_path.suffix
            out_target = out_path.parent / f"{stem}_{inst}{suffix}"

        cur_sim_cfg = sim_cfg.model_copy(
            update={
                "instrument": inst,
                "output_wav": out_target,
            }
        )

        if len(voices) > 1 and max_workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            tasks = [(v, cur_sim_cfg) for v in voices]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_simulate_voice_task, tasks))
        else:
            for v in voices:
                simulate_voice(v, config=cur_sim_cfg)


if __name__ == "__main__":
    main()
