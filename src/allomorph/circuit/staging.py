"""
Allomorph - Virtual Analog Circuit Simulator & Source Instrument Dry CLI
Provides audio caching, source instrument dry synthesis, and the allomorph-sim CLI binary entrypoint.
"""

import argparse
import functools
import math
from pathlib import Path

import numpy as np
import pedalboard

from allomorph.circuit.audio import find_default_input_audio
from allomorph.circuit.forward import (
    CALIBRATION_PEAK_CEILING,
    simulate_instrument_voicing,
)
from allomorph.circuit.parser import MAGNET_PROPERTIES, load_circuit
from allomorph.circuit.saturation import apply_oversampled_saturation
from allomorph.circuit.solver import (
    apply_magnet_properties_to_model,
    compute_circuit_transfer_functions,
)
from allomorph.config.geometry import compute_effective_position, resolve_pickup_coils
from allomorph.config.instruments import load_instrument
from allomorph.config.scales import REPO_ROOT, resolve_scale_range
from allomorph.dsp import (
    FREQS,
    fft_convolve,
    read_wav,
    synthesize_minimum_phase_fir,
    write_wav_24bit,
)
from allomorph.naming import (
    get_instrument_pickup_basename,
    resolve_instruments,
    resolve_voices,
)
from allomorph.physics import MEAN_BASS_F0
from allomorph.physics.aperture import (
    UNIVERSAL_DATUM_POS_M,
    compute_displacement_proximity_shelf,
    compute_pickup_isolation_leveling,
    compute_saddle_boundary_coupling,
    numpy_pickup_acoustic_response,
)
from allomorph.version import (
    DSP_GENERATION,
    compute_file_sha256,
    resolve_tri_part_version,
    write_manifest,
)

AUDIO_DIR = REPO_ROOT / "audio"
WET_AUDIO_DIR = AUDIO_DIR / "wet"


def export_instrument_pickup_wav(
    inst_id: str = "30in",
    pickup_key: str | None = None,
    input_wav: Path | str | None = None,
    output_dir: Path | str | None = None,
    version_tag: str | None = "auto",
    no_manifest: bool = False,
    max_samples: int | None = None,
) -> Path:
    """Synthesizes the dedicated wet training audio for a source instrument pickup.

    Convolves the raw string excitation (optimal_bass_dry.wav) with the source
    pickup's acoustic aperture, loaded circuit response, and source magnetic non-linear dynamics:
        H_src(f) = H_src,ac(f) * H_src,elec(f)
    Writes the dedicated pickup stem to:
        audio/wet/<inst_id>/<pickup>.wav
    """
    inst = load_instrument(inst_id)
    eff_pickup = pickup_key or inst.default_pickup
    if not eff_pickup or eff_pickup not in inst.pickups:
        matched = [
            k for k in inst.pickups if eff_pickup and (k.endswith(eff_pickup) or eff_pickup in k)
        ]
        if matched:
            eff_pickup = matched[0]
        else:
            raise ValueError(
                f"Instrument '{inst.id}' does not define pickup '{eff_pickup}'. "
                f"Available pickups: {list(inst.pickups.keys())}"
            )
    src_pickup = inst.pickups[eff_pickup]

    if input_wav is None:
        input_wav = find_default_input_audio(version_tag=version_tag)
    if not input_wav or not Path(input_wav).exists():
        raise FileNotFoundError(f"Raw dry calibration audio not found: {input_wav}")

    raw_audio, sr = read_wav(input_wav, max_samples=max_samples, dtype=np.float64)
    input_mono = raw_audio[0] if raw_audio.ndim > 1 else raw_audio

    f = np.asarray(FREQS, dtype=np.float64)
    scale_range = resolve_scale_range(inst)
    scale_m = (scale_range[0] + scale_range[1]) / 2.0

    circ_model = None
    curves = None
    if src_pickup.circuit:
        circ_model = load_circuit(src_pickup.circuit)
        apply_magnet_properties_to_model(circ_model, src_pickup, eddy_diffusion=True)
        curves = compute_circuit_transfer_functions(circ_model, freqs=f, return_numpy=True)

    is_composite = (
        src_pickup.type == "composite"
        or bool(src_pickup.components)
        or (curves is not None and len(curves) > 1)
    )

    single_positions = [
        float(p.position_from_bridge_m)
        for p in inst.pickups.values()
        if not p.components and p.position_from_bridge_m is not None
    ]
    max_p_pos = max(single_positions, default=UNIVERSAL_DATUM_POS_M)
    ref_pos = max(max_p_pos, UNIVERSAL_DATUM_POS_M)

    if is_composite and src_pickup.components:
        N = 8192
        f_bins = np.fft.rfftfreq(N, 1.0 / 48000.0)
        c_mean = 2.0 * scale_m * MEAN_BASS_F0

        branch_sub_pickups = []
        for comp in src_pickup.components:
            if comp.pickup and comp.pickup in inst.pickups:
                branch_sub_pickups.append(
                    (inst.pickups[comp.pickup], float(comp.weight), float(comp.polarity))
                )

        branch_coils_list = []
        branch_positions = []
        for sp, w_comp, pol_comp in branch_sub_pickups:
            b_coils = resolve_pickup_coils(sp, inst)
            branch_coils_list.append((b_coils, w_comp, pol_comp))
            branch_positions.append(compute_effective_position(b_coils))

        pos_max = max(branch_positions) if branch_positions else 0.0
        H_channels = []
        peaks = []

        for i, (b_coils, w_comp, pol_comp) in enumerate(branch_coils_list):
            c_curve_raw = (
                curves[i]
                if (curves is not None and i < len(curves))
                else (curves[0] if curves is not None else np.ones_like(FREQS))
            )
            c_curve = np.interp(f_bins, FREQS, np.asarray(c_curve_raw, dtype=np.float64))

            weight_fac = 1.0 if (curves is not None and len(curves) > 1) else w_comp
            ac_raw = numpy_pickup_acoustic_response(f_bins, b_coils, scale_length_m=scale_range) * (
                weight_fac * pol_comp
            )

            b_pos = branch_positions[i]
            h_pos = compute_displacement_proximity_shelf(f_bins, b_pos, scale_m=scale_m)
            k_iso = compute_pickup_isolation_leveling(
                b_pos, scale_m=scale_m, ref_pos_m=ref_pos
            )
            ac = ac_raw * h_pos * k_iso

            min_pos = min((c.position_from_bridge_m for c in b_coils), default=0.10)
            if min_pos < 0.075:
                h_saddle = compute_saddle_boundary_coupling(f_bins, min_pos)
                ac = ac * np.asarray(h_saddle, dtype=np.float64)

            fir_ac = synthesize_minimum_phase_fir(ac, num_taps=2048, normalize=False)
            tau_i = (pos_max - b_pos) / c_mean if len(branch_coils_list) > 1 else 0.0
            delay_samples = round(tau_i * 48000.0)
            if 0 < delay_samples < 2048:
                fir_ac = [0.0] * delay_samples + fir_ac[: 2048 - delay_samples]
            peaks.append(int(np.argmax(np.abs(fir_ac))))

            fir_circ = synthesize_minimum_phase_fir(c_curve, num_taps=2048, normalize=False)
            H_channels.append(np.fft.rfft(fir_ac, N) * np.fft.rfft(fir_circ, N))

        H_channels_arr = np.array(H_channels)
        delta_samples = max(peaks) - min(peaks) if len(peaks) > 1 else 0

        if len(H_channels_arr) > 1 and delta_samples > 0:
            P_coh = np.abs(np.sum(H_channels_arr, axis=0)) ** 2
            P_incoh = np.sum(np.abs(H_channels_arr) ** 2, axis=0)
            delta_tau = delta_samples / 48000.0
            f_notch = 1.0 / (2.0 * delta_tau)
            f_mid = 1.35 * f_notch
            f_sigma = max(0.35 * f_notch, 1.0)
            gamma = 0.5 * (1.0 - np.tanh((f_bins - f_mid) / f_sigma))
            mag_spectrum = np.sqrt(gamma * P_coh + (1.0 - gamma) * P_incoh)
        elif len(H_channels_arr) > 1:
            mag_spectrum = np.abs(np.sum(H_channels_arr, axis=0))
        else:
            mag_spectrum = np.abs(H_channels_arr[0])

        h_total = np.interp(f, f_bins, mag_spectrum)
    else:
        coils = resolve_pickup_coils(src_pickup, inst)
        eff_pos = compute_effective_position(coils)
        h_ac = numpy_pickup_acoustic_response(f, coils, scale_length_m=scale_range)

        h_pos = compute_displacement_proximity_shelf(f, eff_pos, scale_m=scale_m)
        k_iso = compute_pickup_isolation_leveling(
            eff_pos, scale_m=scale_m, ref_pos_m=ref_pos
        )
        h_ac = h_ac * h_pos * k_iso

        min_pos = min((c.position_from_bridge_m for c in coils), default=0.10)
        if min_pos < 0.075:
            h_saddle = compute_saddle_boundary_coupling(f, min_pos)
            h_ac = h_ac * np.asarray(h_saddle, dtype=np.float64)

        if curves is not None:
            h_elec = np.asarray(curves[0], dtype=np.float64)
            h_total = h_ac * h_elec
        else:
            h_total = h_ac

    h_fir = synthesize_minimum_phase_fir(h_total, num_taps=2048, normalize=False)
    filtered = fft_convolve(input_mono, np.asarray(h_fir, dtype=np.float64), mode="causal")[
        : len(input_mono)
    ]

    # Authentic non-linear dynamics for magnetic source pickup
    mag_type = src_pickup.magnet_type or (
        "active" if getattr(inst, "electronics", "") == "active" else "alnico_v"
    )
    props = MAGNET_PROPERTIES.get(mag_type, MAGNET_PROPERTIES["alnico_v"])
    if circ_model is not None and circ_model.vsat is not None:
        vsat_eff = float(circ_model.vsat)
    else:
        vsat_eff = float(props.vsat)

    if src_pickup.alpha is not None:
        alpha_eff = float(src_pickup.alpha)
    else:
        alpha_eff = float(props.alpha)

    filtered = apply_oversampled_saturation(
        filtered.astype(np.float32),
        vsat=vsat_eff,
        alpha=alpha_eff,
        alpha3=float(props.alpha3),
        eta_hyst=float(props.eta_hyst),
        k_sag=float(props.k_sag),
        k_eddy=float(props.k_eddy),
        kappa_orbit=float(props.kappa_orbit),
        beta_curv=float(props.beta_curv),
        k_pull=float(props.k_pull),
        tau_touch=float(props.tau_touch),
        kappa_geom=float(props.kappa_geom),
        k_stein=float(props.k_stein),
        k_emf=float(props.k_emf),
        lambda_L=float(props.lambda_L),
        slew_limit=True,
        f_slew=16000.0,
        oversample=2,
        displacement_weighting=True,
        magnet_drag=True,
    ).astype(np.float64)

    # Sub-audible 8 Hz DC-blocking filter
    hp = pedalboard.HighpassFilter(cutoff_frequency_hz=8.0)
    filtered = hp(filtered.astype(np.float32)[np.newaxis, :], sr)[0].astype(np.float64)
    filtered = filtered - float(np.mean(filtered))

    # Level matching & peak ceiling
    in_rms = float(np.sqrt(np.mean(input_mono**2)))
    out_rms = float(np.sqrt(np.mean(filtered**2)))
    if in_rms > 1e-9 and out_rms > 1e-9:
        filtered = filtered * (in_rms / out_rms)

    max_val = float(np.max(np.abs(filtered)))
    if max_val > CALIBRATION_PEAK_CEILING:
        filtered = filtered * (CALIBRATION_PEAK_CEILING / max_val)

    wet_audio = filtered.astype(np.float32)

    if output_dir is not None:
        p_out = Path(output_dir)
        if p_out.suffix.lower() == ".wav":
            dest_dir = p_out.parent
            primary_path = p_out
        else:
            dest_dir = p_out
            basename = get_instrument_pickup_basename(inst.id, eff_pickup)
            primary_path = dest_dir / f"{basename}.wav"
    else:
        dest_dir = WET_AUDIO_DIR / inst.id
        basename = get_instrument_pickup_basename(inst.id, eff_pickup)
        primary_path = dest_dir / f"{basename}.wav"

    dest_dir.mkdir(parents=True, exist_ok=True)
    write_wav_24bit(str(primary_path), wet_audio, sr)

    final_peak_db = 20.0 * math.log10(max(float(np.max(np.abs(wet_audio))), 1e-9))
    final_rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(wet_audio**2))), 1e-9))
    print(
        f"[Pickup Audio] Exported {primary_path.name} in {dest_dir}/: Peak = {final_peak_db:.2f} dBFS, RMS = {final_rms_db:.2f} dBFS"
    )

    if not no_manifest:
        inst_ver = getattr(inst, "version", 1)
        voice_ver = 1
        for v in inst.voicings.values():
            if v.pickup == eff_pickup:
                voice_ver = getattr(v, "version", 1)
                break
        v_tag = (
            resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
            if (version_tag is None or version_tag == "auto")
            else version_tag
        )
        base_dry_sha = compute_file_sha256(input_wav)
        write_manifest(
            output_dir=dest_dir,
            stage="pickup",
            files=[primary_path],
            version_tag=v_tag,
            base_dry_sha256=base_dry_sha,
            base_dry_file=Path(input_wav).name,
            instrument_version=inst_ver,
            voicing_version=voice_ver,
        )

    return primary_path


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
        "--out", help="Output WAV path (default: audio/wet/<instrument>/<voice>.wav)"
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
        help="Explicit saturation asymmetry factor alpha (default: resolved from magnet_type in instrument configuration)",
    )
    parser.add_argument(
        "--alpha3",
        type=float,
        default=None,
        help="Explicit cubic dipole proximity factor alpha3 (default: resolved from magnet_type in instrument configuration)",
    )
    parser.add_argument(
        "--k-sag",
        type=float,
        default=None,
        help="Explicit dynamic Lenz-law core flux sag factor k_sag (default: resolved from magnet_type in instrument configuration)",
    )
    parser.add_argument(
        "--k-eddy",
        type=float,
        default=None,
        help="Explicit dynamic eddy-current core de-Qing factor k_eddy (default: resolved from magnet_type in instrument configuration)",
    )
    parser.add_argument(
        "--kappa-orbit",
        type=float,
        default=None,
        help="Explicit elliptical string orbit projection factor kappa_orbit (default: resolved from magnet_type in instrument configuration)",
    )
    parser.add_argument(
        "--beta-curv",
        type=float,
        default=None,
        help="Explicit dynamic core inductance curvature factor beta_curv (default: resolved from magnet_type in instrument configuration)",
    )
    parser.add_argument(
        "--k-pull",
        type=float,
        default=None,
        help="Explicit nonlinear magnetic string pull factor k_pull (default: resolved from magnet_type in instrument configuration)",
    )
    parser.add_argument(
        "--tau-touch",
        type=float,
        default=None,
        help="Explicit dynamic touch spectral tilt factor tau_touch (default: resolved from magnet_type in instrument configuration)",
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
        choices=["sim", "all"],
        default=None,
        help="Execution stage: 'sim' (direct forward simulation of instrument voicings), 'all'.",
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
    dc_block = not args.no_dc_block
    noise_dither = not args.no_dither

    instruments = resolve_instruments(args.instrument)
    voices = resolve_voices(args.voice)
    in_path = Path(input_wav) if input_wav else None
    out_path = Path(args.out) if args.out else None

    for inst in instruments:
        out_target = out_path
        if out_path and len(instruments) > 1 and not out_path.is_dir():
            stem = out_path.stem
            suffix = out_path.suffix
            out_target = out_path.parent / f"{stem}_{inst}{suffix}"

        for v in voices:
            simulate_instrument_voicing(
                instrument=inst,
                voicing=v,
                input_wav=in_path,
                output_wav=out_target if (out_target and len(voices) == 1) else None,
                max_samples=args.max_samples,
                apply_dither=noise_dither,
                apply_saturation=True,
                vol_pos=args.vol,
                tone_pos=args.tone,
                blend_pos=args.blend,
                cable_pf=args.cable_pf,
                dc_block=dc_block,
            )


if __name__ == "__main__":
    main()
