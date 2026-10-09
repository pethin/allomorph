"""
Tests for Apple Silicon Native MLX Architecture 2 Trainer Engine.
Verifies MLX detection, engine resolution, model topology, receptive field,
loss parity against PyTorch auraloss, and .nam export parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from allomorph.trainer import (
    get_hardware_device_name,
    is_mlx_available,
    resolve_trainer_engine,
)


def test_mlx_detection_and_engine_resolution():
    """Verifies dual-engine resolution rules and fail-fast platform enforcement."""
    mlx_avail = is_mlx_available()
    assert isinstance(mlx_avail, bool)

    # Auto resolution
    resolved_auto = resolve_trainer_engine("auto")
    if mlx_avail:
        assert resolved_auto == "mlx"
    else:
        assert resolved_auto == "torch"

    # Explicit torch always works
    assert resolve_trainer_engine("torch") == "torch"

    # Invalid engine raises ValueError
    with pytest.raises(ValueError, match="Invalid engine"):
        resolve_trainer_engine("invalid_engine")

    # When MLX is unavailable, requesting 'mlx' raises RuntimeError
    with (
        patch("allomorph.trainer.core.is_mlx_available", return_value=False),
        pytest.raises(RuntimeError, match="MLX engine requested, but Apple Silicon MLX"),
    ):
        resolve_trainer_engine("mlx")

    # Hardware device name
    if mlx_avail:
        assert get_hardware_device_name(engine="mlx") == "Apple Silicon (MLX Metal)"
    assert len(get_hardware_device_name(engine="torch")) > 0


@pytest.mark.skipif(not is_mlx_available(), reason="Requires Apple Silicon MLX Metal")
def test_mlx_model_shapes_and_receptive_field():
    """Verifies PackedWaveNet layer counts, receptive field, and submodel parameter counts."""
    # pyrefly: ignore [missing-import]
    import mlx.core as mx

    from allomorph.trainer.engine_mlx import (
        DEFAULT_NY,
        RECEPTIVE_FIELD,
        MLXPackedWaveNet,
    )

    model = MLXPackedWaveNet()
    assert len(model.layers) == 23
    assert RECEPTIVE_FIELD == 6347

    # Verify parameter counts match NAM's canonical A2 submodels
    sub0_weights = model.get_submodel_flat_weights(submodel_index=0)
    sub1_weights = model.get_submodel_flat_weights(submodel_index=1)
    assert len(sub0_weights) == 1871, f"Sub0 expected 1871 parameters, got {len(sub0_weights)}"
    assert len(sub1_weights) == 12146, f"Sub1 expected 12146 parameters, got {len(sub1_weights)}"

    # Forward pass: valid convolution with ny=8192
    ny = DEFAULT_NY
    seq_in_len = RECEPTIVE_FIELD + ny - 1  # 14,538
    x_in = mx.zeros((1, seq_in_len))
    y_out = model(x_in, pad_start=False)

    # Output shape: (Batch, ny, 2) where channels are (Lite, Full)
    assert y_out.shape == (1, ny, 2)


@pytest.mark.skipif(not is_mlx_available(), reason="Requires Apple Silicon MLX Metal")
def test_mlx_mrstft_loss():
    """Verifies MLX 1D Convolution STFT filterbank and MRSTFT loss computation."""
    # pyrefly: ignore [missing-import]
    import mlx.core as mx

    from allomorph.trainer.engine_mlx import MLXMRSTFTLoss, esr_loss, pre_emphasis_loss

    mrstft = MLXMRSTFTLoss()
    assert len(mrstft.fft_sizes) == 3

    # Generate synthetic target and perturbed prediction
    np.random.seed(42)
    sig_len = 8192
    t = np.linspace(0, 1, sig_len, dtype=np.float32)
    y_target = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    y_pred = y_target + 0.05 * np.random.randn(sig_len).astype(np.float32)

    from typing import Any, cast

    batch_y = mx.array(cast(Any, y_target[None, :]))
    batch_pred = mx.array(cast(Any, y_pred[None, :]))

    # Compare against PyTorch MultiResolutionSTFTLoss for numerical parity
    loss_val = float(mrstft(batch_y, batch_pred))

    import torch
    from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss

    torch_fn = MultiResolutionSTFTLoss()
    torch_loss = float(torch_fn(torch.from_numpy(y_pred[None, None, :]), torch.from_numpy(y_target[None, None, :])))
    assert abs(loss_val - torch_loss) < 0.05, f"MLX MRSTFT ({loss_val}) differs from PyTorch ({torch_loss})"

    # Identity should yield very small loss
    loss_ident = float(mrstft(batch_y, batch_y))
    assert loss_ident < 1e-4

    # ESR loss
    esr = float(esr_loss(batch_y, batch_pred))
    assert 0.0 < esr < 0.1

    # Pre-emphasis loss
    pre = float(pre_emphasis_loss(batch_y, batch_pred, coef=0.85))
    y_pre = y_target[1:] - 0.85 * y_target[:-1]
    pred_pre = y_pred[1:] - 0.85 * y_pred[:-1]
    expected_pre = float(np.sum((y_pre - pred_pre) ** 2) / (np.sum(y_pre**2) + 1e-12))
    assert abs(pre - expected_pre) < 1e-4


@pytest.mark.skipif(not is_mlx_available(), reason="Requires Apple Silicon MLX Metal")
def test_mlx_training_fast_dev_run_export(tmp_path: Path):
    """Executes a 1-epoch fast_dev_run using MLX and verifies .nam container and parity."""
    from pedalboard.io import AudioFile

    from allomorph.trainer import train_voice

    # Synthesize small test audio files (1.0s at 48kHz = 48,000 samples)
    sr = 48000
    n_samples = 48000
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    dry = (0.5 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wet = (0.4 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    dry_path = tmp_path / "dry.wav"
    wet_path = tmp_path / "wet.wav"

    with AudioFile(str(dry_path), "w", samplerate=sr, num_channels=1) as f:
        f.write(dry[None, :])
    with AudioFile(str(wet_path), "w", samplerate=sr, num_channels=1) as f:
        f.write(wet[None, :])

    models_dir = tmp_path / "models"

    ok = train_voice(
        instrument="30in",
        voice="precision_active",
        input_wav=dry_path,
        output_wav=wet_path,
        reference_wav=dry_path,
        models_dir=models_dir,
        epochs=1,
        min_epochs=1,
        fast_dev_run=True,
        engine="mlx",
        seed=123,
    )
    assert ok is True

    # Find the exported .nam file
    nam_files = list(models_dir.rglob("*.nam"))
    assert len(nam_files) == 1
    nam_path = nam_files[0]

    with open(nam_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "0.7.0"
    assert data["architecture"] == "SlimmableContainer"
    submodels = data["config"]["submodels"]
    assert len(submodels) == 2
    assert submodels[0]["max_value"] == 0.5
    assert submodels[0]["model"]["architecture"] == "WaveNet"
    assert submodels[0]["model"]["config"]["layers"][0]["channels"] == 3
    assert submodels[1]["max_value"] == 1.0
    assert submodels[1]["model"]["architecture"] == "WaveNet"
    assert submodels[1]["model"]["config"]["layers"][0]["channels"] == 8

    # Verify weight arrays are populated
    sub0_w = submodels[0]["model"]["weights"]
    sub1_w = submodels[1]["model"]["weights"]
    assert len(sub0_w) == 1871
    assert len(sub1_w) == 12146

    # Verify metadata
    assert "metadata" in data
    assert "other_metadata" in data["metadata"]
    other_meta = data["metadata"]["other_metadata"]
    assert other_meta["target_voice"]["id"] == "precision_active"
    assert "training" in other_meta
    assert other_meta["training"]["validation_esr_a2_full"] is not None
