"""
Allomorph - Virtual Analog Circuit Simulator CLI.
Provides the allomorph-sim CLI entrypoint for simulating digital twin instrument voicings and sweeps.
"""

import argparse
from pathlib import Path

import numpy as np

from allomorph.circuit.audio import find_default_input_audio
from allomorph.circuit.forward import simulate_instrument_voicing
from allomorph.naming import resolve_instruments, resolve_voices


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
                normalize=args.normalize,
                target_dbfs=args.target_dbfs,
            )


if __name__ == "__main__":
    main()
