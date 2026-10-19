"""Cyclopts CLI for the Allomorph Universal Bass Denoiser (`denoiser`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from denoiser.constants import (
    DEFAULT_DENOISER_OUTPUT_DIR,
    DEFAULT_EXCITATION_WAV,
    DENOISER_DEFAULT_EMA_DECAY,
    DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT,
    DENOISER_DEFAULT_FIR_HP_MID_WEIGHT,
    DENOISER_DEFAULT_LR,
    DENOISER_DEFAULT_MAX_EPOCHS,
    DENOISER_DEFAULT_MIN_EPOCHS,
    DENOISER_DEFAULT_PATIENCE,
    DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    DENOISER_DEFAULT_VAL_INTERVAL,
    DENOISER_DEFAULT_WEIGHT_DECAY,
    resolve_mains_targets,
)
from denoiser.engine import (
    denoise_audio_file,
    evaluate_denoiser_model,
    train_denoiser_matrix,
)
from reamper.capture import capture_idle_noise

app = App(
    name="denoiser",
    help="Allomorph Universal Bass Denoiser — Native NAM Architecture 2 Trainer & Utility CLI.",
    version="1.0.0",
)


@app.default
@app.command(name="train")
def train_cmd(
    *,
    mains: Annotated[
        str,
        Parameter(
            name=["--mains", "-m"],
            help="Mains hum grid target: 'both' (50+60 Hz universal), '60', '50', or 'all'.",
        ),
    ] = "both",
    profile: Annotated[
        str,
        Parameter(
            name=["--profile", "-p"],
            help="Gate knee & suppression profile: 'gentle', 'balanced', 'aggressive', or 'all'.",
        ),
    ] = "all",
    input_file: Annotated[
        Path,
        Parameter(
            name=["--input", "-i"],
            help="Clean dry bass excitation WAV file used to build the multi-voice corpus.",
        ),
    ] = DEFAULT_EXCITATION_WAV,
    output_dir: Annotated[
        Path,
        Parameter(
            name=["--output-dir", "-o"],
            help="Output directory for exported .nam models, .json sidecars, and storefront_description.txt.",
        ),
    ] = DEFAULT_DENOISER_OUTPUT_DIR,
    noise_wav: Annotated[
        Path | None,
        Parameter(
            name=["--noise-wav", "-n"],
            help="Optional captured real-world hardware idle noise WAV to blend into the stochastic noise bank.",
        ),
    ] = None,
    threshold_db: Annotated[
        float | None,
        Parameter(
            name=["--threshold-db", "-t"],
            help="Explicit gate knee threshold override in dBFS (overrides profile default).",
        ),
    ] = None,
    ultrasonic_lpf_fc: Annotated[
        float,
        Parameter(
            name=["--ultrasonic-lpf-fc"],
            help="255-tap Blackman zero-phase ultrasonic low-pass cutoff in Hz (strips >fc SMPS whine).",
        ),
    ] = DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    epochs: Annotated[
        int,
        Parameter(name=["--epochs"], help="Maximum training epochs budget."),
    ] = DENOISER_DEFAULT_MAX_EPOCHS,
    min_epochs: Annotated[
        int,
        Parameter(name=["--min-epochs"], help="Minimum warmup epochs before early stopping."),
    ] = DENOISER_DEFAULT_MIN_EPOCHS,
    patience: Annotated[
        int,
        Parameter(name=["--patience"], help="Early stopping plateau patience epochs."),
    ] = DENOISER_DEFAULT_PATIENCE,
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
    ] = DENOISER_DEFAULT_LR,
    weight_decay: Annotated[
        float,
        Parameter(name=["--weight-decay"], help="Adam L2 weight decay regularization."),
    ] = DENOISER_DEFAULT_WEIGHT_DECAY,
    fir_hp_mid_weight: Annotated[
        float,
        Parameter(name=["--fir-hp-mid-weight"], help="1.2 kHz Blackman FIR high-pass loss weight."),
    ] = DENOISER_DEFAULT_FIR_HP_MID_WEIGHT,
    fir_hp_high_weight: Annotated[
        float,
        Parameter(
            name=["--fir-hp-high-weight"], help="5.5 kHz Blackman FIR high-pass loss weight."
        ),
    ] = DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT,
    val_interval: Annotated[
        int,
        Parameter(name=["--val-interval"], help="Validation interval in epochs during warmup."),
    ] = DENOISER_DEFAULT_VAL_INTERVAL,
    ema_decay: Annotated[
        float,
        Parameter(name=["--ema-decay"], help="Exponential moving average weight decay."),
    ] = DENOISER_DEFAULT_EMA_DECAY,
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
    """Trains universal bass denoiser NAM Architecture 2 (.nam) models across mains grids and profiles."""
    train_denoiser_matrix(
        mains=mains,
        profile=profile,
        input_file=input_file,
        output_dir=output_dir,
        noise_wav=noise_wav,
        threshold_db=threshold_db,
        ultrasonic_lpf_fc=ultrasonic_lpf_fc,
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        batch_size=batch_size,
        precision=precision,
        lr=lr,
        weight_decay=weight_decay,
        fir_hp_mid_weight=fir_hp_mid_weight,
        fir_hp_high_weight=fir_hp_high_weight,
        val_interval=val_interval,
        ema_decay=ema_decay,
        device=device,
        seed=seed,
        fast_dev_run=fast_dev_run,
    )


@app.command(name="evaluate")
def evaluate_cmd(
    model_path: Annotated[
        Path,
        Parameter(
            name=["--model", "-m"],
            help="Path to the trained denoiser .nam file to evaluate.",
        ),
    ] = DEFAULT_DENOISER_OUTPUT_DIR / "universal_balanced.nam",
    *,
    mains: Annotated[
        str,
        Parameter(
            name=["--mains"],
            help="Mains hum test grid ('both', '60', or '50').",
        ),
    ] = "both",
    noise_dbfs: Annotated[
        float,
        Parameter(name=["--noise-dbfs"], help="Idle noise test RMS level in dBFS."),
    ] = -48.0,
    signal_dbfs: Annotated[
        float,
        Parameter(name=["--signal-dbfs"], help="Active bass test RMS level in dBFS."),
    ] = -16.0,
    duration: Annotated[
        float,
        Parameter(name=["--duration", "-d"], help="Evaluation test signal duration in seconds."),
    ] = 2.0,
    submodel: Annotated[
        int,
        Parameter(name=["--submodel"], help="Submodel index (1 = A2 Full Ch8, 0 = A2 Lite Ch3)."),
    ] = 1,
    device: Annotated[
        str,
        Parameter(name=["--device"], help="Target compute device ('auto', 'mps', 'cuda', 'cpu')."),
    ] = "auto",
) -> None:
    """Evaluates idle hum/SMPS suppression and passband transparency of a trained denoiser .nam model."""
    mains_mode = resolve_mains_targets(mains)[0]
    evaluate_denoiser_model(
        model_path=model_path,
        mains=mains_mode,
        noise_dbfs=noise_dbfs,
        signal_dbfs=signal_dbfs,
        duration_sec=duration,
        submodel=submodel,
        device_name=device,
    )


@app.command(name="apply")
def apply_cmd(
    input_wav: Annotated[
        Path,
        Parameter(
            name=["--input", "-i"],
            help="Path to noisy input bass WAV file.",
        ),
    ],
    output_wav: Annotated[
        Path,
        Parameter(
            name=["--output", "-o"],
            help="Path to save the denoised output WAV file.",
        ),
    ],
    *,
    model_path: Annotated[
        Path,
        Parameter(
            name=["--model", "-m"],
            help="Path to the trained denoiser .nam file.",
        ),
    ] = DEFAULT_DENOISER_OUTPUT_DIR / "universal_balanced.nam",
    submodel: Annotated[
        int,
        Parameter(name=["--submodel"], help="Submodel index (1 = A2 Full Ch8, 0 = A2 Lite Ch3)."),
    ] = 1,
    device: Annotated[
        str,
        Parameter(name=["--device"], help="Target compute device ('auto', 'mps', 'cuda', 'cpu')."),
    ] = "auto",
) -> None:
    """Applies a trained .nam denoiser model to a WAV audio file."""
    out = denoise_audio_file(
        input_wav=input_wav,
        output_wav=output_wav,
        model_path=model_path,
        submodel=submodel,
        device_name=device,
    )
    print(f"Denoised WAV written to: {out}", flush=True)


app.command(capture_idle_noise, name=["capture-noise"])


def main() -> None:
    """Entry point for the `denoiser` CLI."""
    app()
