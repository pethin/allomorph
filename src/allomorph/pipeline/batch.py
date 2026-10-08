"""
Allomorph Pipeline - Batch Concurrent Circuit Simulation
Parallel batch runner coordinating multiple circuit simulations with ProcessPoolExecutor.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
