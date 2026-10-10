"""
Allomorph Pipeline - Batch Concurrent Circuit Simulation
Parallel batch runner coordinating multiple circuit simulations with ProcessPoolExecutor.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from allomorph.config.schema import InstrumentConfig
from allomorph.config.voices import VOICES
from allomorph.pipeline.stages import run_circuit_simulation


def _run_circuit_simulation_task(
    task_args: tuple[str, str, str | Path | None, int | None, str | Path | None],
) -> tuple[str, bool]:
    """Top-level picklable task runner for multiprocessing."""
    voice, instrument, input_wav, max_samples, output_wav = task_args
    success = run_circuit_simulation(
        voice=voice,
        instrument=instrument,
        input_wav=input_wav,
        output_wav=output_wav,
        max_samples=max_samples,
    )
    return voice, success


def run_spice_batch(
    voices: Sequence[str] | None = None,
    instrument: str = "30in",
    input_wav: str | Path | None = None,
    backend: str = "native",
    jobs: int | None = None,
    max_samples: int | None = None,
    output_dir: str | Path | None = None,
) -> bool:
    """Executes batch simulation of specified voice circuit models with multi-process concurrency."""
    if backend != "native":
        raise ValueError(
            f"Unsupported backend '{backend}'. The legacy LTspice pipeline has been removed; "
            "Allomorph uses the built-in native Apple Silicon WAV SPICE engine."
        )
    target_voices = voices if voices else list(VOICES.keys())
    max_workers = jobs if jobs is not None else min(4, os.cpu_count() or 4)
    samples_str = str(max_samples) if max_samples is not None else "full"

    if len(target_voices) > 1 and max_workers > 1:
        print(
            f"\n[Stage 3] Executing circuit simulations in parallel ({len(target_voices)} voices, {max_workers} workers, Max Samples: {samples_str})..."
        )
        tasks = [
            (
                v,
                instrument,
                input_wav,
                max_samples,
                Path(output_dir) / f"out_{v}.wav" if output_dir is not None else None,
            )
            for v in target_voices
        ]
        failed = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_circuit_simulation_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                try:
                    v, success = future.result()
                    if not success:
                        failed.append(v)
                        print(f"  [{completed}/{len(target_voices)}] Voice simulation FAILED: {v}")
                    else:
                        print(
                            f"  [{completed}/{len(target_voices)}] Voice simulation finished: {v}"
                        )
                except (OSError, RuntimeError, ValueError, KeyError) as e:
                    failed.append(f"unknown (error: {e})")
                    print(
                        f"  [{completed}/{len(target_voices)}] Voice simulation worker error: {e}"
                    )
        if failed:
            print(f"Warning: {len(failed)} voice simulations failed: {', '.join(failed)}")
            return False
        print("Batch circuit simulation finished.")
        return True
    else:
        mode_desc = "sequentially" if len(target_voices) > 1 else "single voice"
        print(
            f"\n[Stage 3] Executing circuit simulation {mode_desc} ({len(target_voices)} voice{'s' if len(target_voices) > 1 else ''}, Max Samples: {samples_str})..."
        )
        all_ok = True
        for idx, voice in enumerate(target_voices, 1):
            print(f"\n[{idx}/{len(target_voices)}] Circuit simulation: {voice}...")
            out_wav = Path(output_dir) / f"out_{voice}.wav" if output_dir is not None else None
            ok = run_circuit_simulation(
                voice,
                instrument=instrument,
                input_wav=input_wav,
                output_wav=out_wav,
                max_samples=max_samples,
            )
            if not ok:
                all_ok = False
        print("Batch circuit simulation finished.")
        return all_ok


def simulate_all_instrument_voicings(
    instrument: InstrumentConfig | str | Path | None = None,
    input_wav: Path | str | None = None,
    max_samples: int | None = None,
    jobs: int | None = None,
    force: bool = False,
) -> list[Path]:
    """Simulates all native voicings for a specified instrument (or all instruments in catalog).
    Outputs files to audio/wet/<inst_id>/<voicing_id>.wav.
    """
    from allomorph.circuit.forward import simulate_instrument_voicing
    from allomorph.config.instruments import INSTRUMENTS, load_instrument
    from allomorph.config.preamps import PREAMPS
    from allomorph.config.schema import InstrumentConfig, VoicingConfig
    from allomorph.config.strings import STRINGS

    if instrument is not None:
        target_insts = [
            load_instrument(instrument)
            if not isinstance(instrument, InstrumentConfig)
            else instrument
        ]
    else:
        target_insts = list(INSTRUMENTS.values())

    exported_paths: list[Path] = []
    tasks: list[tuple[InstrumentConfig, VoicingConfig]] = []
    for inst in target_insts:
        for voicing in inst.voicings.values():
            tasks.append((inst, voicing))

    def _sim(item: tuple[InstrumentConfig, VoicingConfig]) -> Path:
        i, v = item
        return simulate_instrument_voicing(
            instrument=i,
            voicing=v,
            input_wav=input_wav,
            max_samples=max_samples,
            force=force,
            preamps=PREAMPS,
            strings=STRINGS,
        )

    eff_jobs = jobs if jobs is not None and jobs > 0 else 1
    if eff_jobs > 1 and len(tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=eff_jobs) as executor:
            exported_paths = list(executor.map(_sim, tasks))
    else:
        exported_paths = [_sim(t) for t in tasks]

    return exported_paths


def simulate_voice(
    voice: str,
    output_wav: Path | str | None = None,
    input_wav: Path | str | None = None,
    instrument: InstrumentConfig | str | Path | None = None,
    normalize: str = "auto",
    target_dbfs: float | None = None,
    max_samples: int | None = None,
    config: object = None,
    pickup: str | None = None,
    force: bool = False,
    **kwargs: object,
) -> bool:
    """Simulates an instrument voicing digital twin.
    Resolves targets using the catalog and dispatches to simulate_instrument_voicing.
    """
    from allomorph.circuit.forward import simulate_instrument_voicing
    from allomorph.circuit.parser import MAGNET_PROPERTIES
    from allomorph.config.instruments import load_instrument, resolve_target_voicing
    from allomorph.config.preamps import PREAMPS
    from allomorph.config.schema import InstrumentConfig
    from allomorph.config.strings import STRINGS
    from allomorph.physics.aperture import is_voice_matching_source

    if config is not None:
        inst_target = getattr(config, "instrument", None) or instrument
        out_target = getattr(config, "output_wav", None) or output_wav
        in_target = getattr(config, "input_wav", None) or input_wav
        max_s = getattr(config, "max_samples", None) or max_samples
        skip_id = bool(getattr(config, "skip_identity", False))
        norm = str(getattr(config, "normalize", normalize))
        tgt_db = getattr(config, "target_dbfs", target_dbfs)
        force_val = bool(getattr(config, "force", force))
    else:
        inst_target = instrument
        out_target = output_wav
        in_target = input_wav
        max_s = max_samples
        skip_id = bool(kwargs.get("skip_identity", False))
        norm = normalize
        tgt_db = target_dbfs
        force_val = force

    if inst_target is not None:
        inst_obj = (
            load_instrument(inst_target)
            if not isinstance(inst_target, InstrumentConfig)
            else inst_target
        )
        if pickup and pickup != "auto" and pickup not in inst_obj.pickups:
            raise KeyError(
                f"Pickup '{pickup}' not found on instrument '{inst_obj.id}'. "
                f"Available pickups: {list(inst_obj.pickups.keys())}"
            )
        if inst_obj.electronics == "passive":
            p_key = pickup or (next(iter(inst_obj.pickups.keys())) if inst_obj.pickups else "p")
            if p_key in inst_obj.pickups and not inst_obj.harnesses:
                raise ValueError(
                    f"Passive instrument '{inst_obj.id}' pickup '{p_key}' "
                    f"does not define a control harness. Passive source pickups require an explicit "
                    f"circuit harness for forward circuit simulation."
                )

        t_inst, v_cfg = resolve_target_voicing(voice, instrument=inst_obj)

        from allomorph.config.instruments import is_identity_voicing
        from allomorph.config.voices import VOICES

        if voice in VOICES:
            is_id = is_voice_matching_source(inst_obj, VOICES[voice])
        else:
            is_id = is_identity_voicing(inst_obj, voice, t_inst, v_cfg)
        if skip_id and is_id:
            if out_target and Path(out_target).exists():
                Path(out_target).unlink()
            return False

        apply_sat = bool(kwargs.get("apply_saturation", True))
        p_key = pickup or (next(iter(inst_obj.pickups.keys())) if inst_obj.pickups else None)
        src_p = inst_obj.pickups.get(p_key) if p_key else None
        src_mag = (src_p.magnet_type if src_p else None) or (
            "active" if inst_obj.electronics == "active" else "alnico_v"
        )
        src_props = MAGNET_PROPERTIES.get(src_mag, MAGNET_PROPERTIES["alnico_v"])

        from allomorph.config.voices import resolve_voicing_active_pickups

        tgt_act = resolve_voicing_active_pickups(t_inst, v_cfg)
        tgt_p = t_inst.pickups.get(tgt_act[0]) if tgt_act else None
        tgt_mag = (tgt_p.magnet_type if tgt_p else None) or "alnico_v"
        tgt_props = MAGNET_PROPERTIES.get(tgt_mag, MAGNET_PROPERTIES["alnico_v"])
        if is_id or (
            float(tgt_props.alpha) <= float(src_props.alpha)
            and float(tgt_props.vsat) >= float(src_props.vsat)
        ):
            apply_sat = False
    else:
        t_inst, v_cfg = resolve_target_voicing(voice)
        apply_sat = bool(kwargs.get("apply_saturation", True))

    out = simulate_instrument_voicing(
        instrument=t_inst,
        voicing=v_cfg,
        input_wav=in_target,
        output_wav=out_target,
        max_samples=max_s,
        apply_saturation=apply_sat,
        normalize=norm,
        target_dbfs=tgt_db,
        force=force_val,
        preamps=PREAMPS,
        strings=STRINGS,
    )
    return out.exists()
