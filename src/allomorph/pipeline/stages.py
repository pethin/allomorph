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
    models_dir: str | Path | None = None,
    epochs: int = 400,
    min_epochs: int = 5,
    goal_esr: float | None = 0.0002,
    fast_dev_run: bool = False,
    basename: str | None = None,
    batch_size: int = 16,
    goal_esr_lite: float | None = 0.00250,
    goal_delta_esr_lite: float | None = 0.080,
    goal_delta_mrstft_lite: float | None = 0.75,
    max_mrstft_ceiling_lite: float = 0.450,
    version_tag: str | None = "auto",
    no_manifest: bool = False,
):
    """Trains a Neural Amp Modeler (NAM) Architecture 2 model locally under the studio reference standard."""
    print(
        f"\n[Training] Training Neural Amp Modeler Architecture 2 Slimmable model for {voice} (Instrument: {instrument})..."
    )
    script = SCRIPTS_DIR / "train_nam.py"
    cmd = [
        sys.executable,
        str(script),
        "--instrument",
        instrument,
        "--voice",
        voice,
        "--epochs",
        str(epochs),
        "--min-epochs",
        str(min_epochs),
        "--batch-size",
        str(batch_size),
    ]
    if output_wav:
        cmd.extend(["--output", str(output_wav)])
    if models_dir:
        cmd.extend(["--models-dir", str(models_dir)])
    if basename:
        cmd.extend(["--basename", basename])
    if goal_esr is not None and goal_esr > 0:
        cmd.extend(["--goal-esr", str(goal_esr)])
    else:
        cmd.append("--no-goal-esr")
    if goal_esr_lite is not None:
        cmd.extend(["--goal-esr-lite", str(goal_esr_lite)])
    if goal_delta_esr_lite is not None:
        cmd.extend(["--goal-delta-esr-lite", str(goal_delta_esr_lite)])
    if goal_delta_mrstft_lite is not None:
        cmd.extend(["--goal-delta-mrstft-lite", str(goal_delta_mrstft_lite)])
    if max_mrstft_ceiling_lite is not None:
        cmd.extend(["--max-mrstft-ceiling-lite", str(max_mrstft_ceiling_lite)])
    if input_wav:
        cmd.extend(["--input", str(input_wav)])
    if fast_dev_run:
        cmd.append("--fast-dev-run")
    if version_tag:
        cmd.extend(["--version-tag", str(version_tag)])
    if no_manifest:
        cmd.append("--no-manifest")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if res.returncode != 0:
        print(f"Notice: Model training exited with code {res.returncode}")


def run_tone_pack_training(
    pack: str,
    voice: str = "all",
    overwrite: bool = False,
    epochs: int = 400,
    min_epochs: int = 5,
    goal_esr: float | None = 0.00020,
    goal_delta_esr: float | None = 0.020,
    goal_delta_mrstft: float | None = 0.50,
    max_mrstft_ceiling: float = 0.320,
    goal_esr_lite: float | None = 0.00250,
    goal_delta_esr_lite: float | None = 0.080,
    goal_delta_mrstft_lite: float | None = 0.75,
    max_mrstft_ceiling_lite: float = 0.450,
    consecutive_patience: int = 3,
    patience: int = 12,
    batch_size: int = 16,
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
        goal_esr=goal_esr,
        goal_delta_esr=goal_delta_esr,
        goal_delta_mrstft=goal_delta_mrstft,
        max_mrstft_ceiling=max_mrstft_ceiling,
        goal_esr_lite=goal_esr_lite,
        goal_delta_esr_lite=goal_delta_esr_lite,
        goal_delta_mrstft_lite=goal_delta_mrstft_lite,
        max_mrstft_ceiling_lite=max_mrstft_ceiling_lite,
        consecutive_patience=consecutive_patience,
        patience=patience,
        batch_size=batch_size,
        lr_scheduler=lr_scheduler,
        eta_min=eta_min,
        lr_t_max=lr_t_max,
        fast_dev_run=fast_dev_run,
    )
