"""
Allomorph Pipeline - Execution Stages
Individual execution stages for visualization, audio pre-filtering,
circuit simulation, and neural model training.
"""

import subprocess
import sys
from pathlib import Path

from allomorph.circuit.forward import simulate_instrument_voicing

REPO_ROOT = Path(__file__).resolve().parents[3]

DOCS_DIR = REPO_ROOT / "docs"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def run_visualization(instrument: str = "all"):
    """Generates the interactive Altair visualization charts and master portal."""
    inst_desc = "all instruments" if instrument == "all" else f"Instrument: {instrument}"
    print(f"\n[Stage 1] Generating interactive Altair visualization ({inst_desc})...")
    script = SCRIPTS_DIR / "analyze_voices.py"
    if instrument == "all":
        cmd = [sys.executable, str(script), "--all"]
    else:
        cmd = [sys.executable, str(script), "--instrument", instrument]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if res.returncode != 0:
        print(f"Warning: Visualization generation returned non-zero code {res.returncode}")
    else:
        resp_dir = DOCS_DIR / "frequency_responses"
        print(f"Interactive charts generated in {resp_dir}/")
        print(f"Master interactive portal updated at {DOCS_DIR / 'frequency_responses.html'}")


def run_circuit_simulation(
    voice: str,
    instrument: str = "30in",
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    backend: str = "native",
    max_samples: int | None = None,
) -> bool:
    """Executes circuit simulation for a single target voice netlist using the native Apple Silicon WAV SPICE engine."""
    if backend != "native":
        raise ValueError(
            f"Unsupported backend '{backend}'. The legacy LTspice pipeline has been removed; "
            "Allomorph uses the built-in native Apple Silicon WAV SPICE engine."
        )
    try:
        out = simulate_instrument_voicing(
            instrument=instrument,
            voicing=voice,
            input_wav=input_wav,
            output_wav=output_wav,
            max_samples=max_samples,
        )
        return out.exists()
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        print(f"Error during native circuit simulation: {e}")
        return False


def run_training(
    instrument: str = "30in",
    voice: str = "precision_active",
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    reference_wav: str | Path | None = None,
    models_dir: str | Path | None = None,
    epochs: int = 35,
    min_epochs: int = 5,
    patience: int = 5,
    min_delta: float = 2.0e-6,
    pre_emph_weight: float = 0.25,
    pre_emph_coef: float = 0.85,
    mrstft_weight: float = 0.0010,
    lr_scheduler: str = "cosine",
    eta_min: float = 1e-5,
    lr_t_max: int = 35,
    fast_dev_run: bool = False,
    basename: str | None = None,
    batch_size: int | str = "auto",
    precision: str = "auto",
    num_workers: int | str = "auto",
    version_tag: str | None = "auto",
    no_manifest: bool = False,
    include_identity: bool = False,
) -> bool:
    """Trains a Neural Amp Modeler (NAM) Architecture 2 model locally under the studio reference standard."""
    from allomorph.trainer import train_voice

    print(
        f"\n[Training] Training Neural Amp Modeler Architecture 2 Slimmable model for {voice} (Instrument: {instrument})..."
    )
    return train_voice(
        instrument=instrument,
        voice=voice,
        input_wav=input_wav,
        output_wav=output_wav,
        reference_wav=reference_wav,
        models_dir=models_dir or (REPO_ROOT / "models"),
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        min_delta=min_delta,
        pre_emph_weight=pre_emph_weight,
        pre_emph_coef=pre_emph_coef,
        mrstft_weight=mrstft_weight,
        lr_scheduler=lr_scheduler,
        eta_min=eta_min,
        lr_t_max=lr_t_max,
        batch_size=batch_size,
        precision=precision,
        num_workers=num_workers,
        fast_dev_run=fast_dev_run,
        basename=basename,
        version_tag=version_tag,
        no_manifest=no_manifest,
        include_identity=include_identity,
    )


def run_tone_pack_training(
    pack: str,
    voice: str = "all",
    overwrite: bool = False,
    epochs: int = 35,
    min_epochs: int = 5,
    patience: int = 5,
    min_delta: float = 2.0e-6,
    batch_size: int | str = "auto",
    precision: str = "auto",
    num_workers: int | str = "auto",
    lr_scheduler: str = "cosine",
    eta_min: float = 1e-5,
    lr_t_max: int = 35,
    fast_dev_run: bool = False,
) -> Path:
    """Executes local NAM Architecture 2 model training for all wet stems in a Tone3000 pack."""
    from allomorph.pipeline.pack import train_tone_pack

    return train_tone_pack(
        pack=pack,
        voice=voice,
        overwrite=overwrite,
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        min_delta=min_delta,
        batch_size=batch_size,
        precision=precision,
        num_workers=num_workers,
        lr_scheduler=lr_scheduler,
        eta_min=eta_min,
        lr_t_max=lr_t_max,
        fast_dev_run=fast_dev_run,
    )
