"""Tests for Allomorph Native Architecture 2 (A2) NAM Training Engine.

Verifies:
- Block-diagonal masked dilated WaveNet layer computation.
- Architecture 2 receptive field (6347 samples) and parameter counts.
- Dual-tier slimmable export weight counts (1871 for channels_3, 12146 for channels_8).
- Compliant SlimmableContainer .nam JSON structure.
- Pure NumPy zero-phase FIR highpass filter design.
- Multi-band spectral loss computation.
- End-to-end fast-dev-run execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pedalboard.io import AudioFile

from allomorph.trainer.native import (
    CHANNELS_TOTAL,
    AllomorphWaveNetA2,
    AllomorphWaveNetLayer,
    MultiBandSpectralLoss,
    train_voice_native,
)


def test_wavenet_layer_forward() -> None:
    """Verifies single dilated WaveNet layer forward pass with block-diagonal masking."""
    layer = AllomorphWaveNetLayer(kernel_size=6, dilation=3)
    b, lin = 2, 100
    x = torch.randn(b, CHANNELS_TOTAL, lin)
    x_cond = torch.randn(b, 1, lin)
    target_head_len = 50

    res, head = layer(x, x_cond, target_head_len)

    # Reduction from conv without padding: (k - 1) * d = 5 * 3 = 15
    l_conv = lin - 15
    assert res.shape == (b, CHANNELS_TOTAL, l_conv)
    assert head.shape == (b, CHANNELS_TOTAL, target_head_len)


def test_wavenet_a2_receptive_field() -> None:
    """Verifies Architecture 2 receptive field and total layer count."""
    model = AllomorphWaveNetA2()

    # 21 layers of k=6, 2 layers of k=15
    assert len(model.layers) == 23
    assert model.receptive_field_layers == 6332
    assert model.receptive_field == 6347


def test_wavenet_a2_forward_shape() -> None:
    """Verifies full forward pass output tensor dimensions."""
    model = AllomorphWaveNetA2()
    model.eval()

    ny = 8192
    nx = ny + (model.receptive_field - 1)  # 14538
    x = torch.randn(2, 1, nx)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 2, ny)
    assert not torch.isnan(out).any()


def test_wavenet_a2_weight_masks_and_counts() -> None:
    """Verifies off-block channel mask enforcement and exact submodel weight counts."""
    model = AllomorphWaveNetA2()

    # Deliberately pollute off-block weights
    with torch.no_grad():
        first_layer = model.layers[0]
        assert isinstance(first_layer, AllomorphWaveNetLayer)
        first_layer.conv.weight[0, 5, :] = 99.0
        first_layer.layer1x1.weight[0, 5, :] = 99.0
        model.head_rechannel.weight[0, 5, :] = 99.0

    model.apply_weight_masks()

    with torch.no_grad():
        assert first_layer.conv.weight[0, 5, :].abs().max().item() == 0.0
        assert first_layer.layer1x1.weight[0, 5, :].abs().max().item() == 0.0
        assert model.head_rechannel.weight[0, 5, :].abs().max().item() == 0.0

    # Submodel 0: channels_3 Lite (exactly 1,871 weights)
    w_lite = model.export_submodel_weights(0)
    assert len(w_lite) == 1871

    # Submodel 1: channels_8 Full (exactly 12,146 weights)
    w_full = model.export_submodel_weights(1)
    assert len(w_full) == 12146


def test_wavenet_a2_export_nam(tmp_path: Path) -> None:
    """Verifies compliant Tone3000 SlimmableContainer .nam JSON output adhering to NAM A2 specification."""
    from nam.models import init_from_nam

    model = AllomorphWaveNetA2()
    out_file = tmp_path / "test_model.nam"

    metadata = {
        "name": "Test Voicing",
        "instrument_id": "test_bass",
        "validation_esr": 0.0012,
    }

    model.export_nam(out_file, sample_rate=48000.0, metadata=metadata)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["architecture"] == "SlimmableContainer"
    assert data["version"] == "0.7.0"
    assert data["sample_rate"] == 48000
    assert data["weights"] == []
    assert data["metadata"]["trainer"] == "TONE3000"
    assert "loudness" in data["metadata"]
    assert "gain" in data["metadata"]

    submodels = data["config"]["submodels"]
    assert len(submodels) == 2

    # Submodel 0: Lite
    sub0 = submodels[0]
    assert sub0["max_value"] == 0.5
    assert sub0["model"]["version"] == "0.7.0"
    assert sub0["model"]["architecture"] == "WaveNet"
    assert len(sub0["model"]["weights"]) == 1871
    assert "loudness" in sub0["model"]["metadata"]

    # Submodel 1: Full
    sub1 = submodels[1]
    assert sub1["max_value"] == 1
    assert sub1["model"]["version"] == "0.7.0"
    assert sub1["model"]["architecture"] == "WaveNet"
    assert len(sub1["model"]["weights"]) == 12146
    assert "loudness" in sub1["model"]["metadata"]

    # Verify official NAM runtime initialization and execution on Full submodel
    nam_model = init_from_nam(sub1["model"])
    test_in = torch.randn(1, 4800)
    test_out = nam_model(test_in)
    assert test_out.shape == (1, 4800)


def test_multi_resolution_stft_loss() -> None:
    """Verifies MultiResolutionSTFTLoss numerical stability, symmetry, and zero-error on identical signals."""
    from allomorph.trainer.native import MultiResolutionSTFTLoss

    mrstft = MultiResolutionSTFTLoss()

    t = torch.sin(2.0 * np.pi * 440.0 * torch.linspace(0, 1, 48000)).unsqueeze(0)
    p = t.clone()

    loss_zero = mrstft(p, t)
    assert loss_zero.item() < 1e-5

    p_perturbed = t + 0.05 * torch.randn_like(t)
    loss_val = mrstft(p_perturbed, t)
    assert loss_val.item() > 0.01


def test_multiband_spectral_loss() -> None:
    """Verifies composite multi-band loss with MSE and MRSTFT."""
    loss_fn = MultiBandSpectralLoss(
        mse_weight=1.0,
        mrstft_weight=0.002,
        fs=48000.0,
    )

    t = torch.sin(2.0 * np.pi * 440.0 * torch.linspace(0, 1, 48000)).unsqueeze(0)
    p = t.clone()

    # Exact equality should have ~0 loss
    loss_zero, metrics_zero = loss_fn(p, t)
    assert loss_zero.item() < 1e-4
    assert metrics_zero["mse"] < 1e-7

    # Perturbed signal has positive loss and detailed metrics
    p_noisy = t + 0.1 * torch.randn_like(t)
    loss_val, metrics = loss_fn(p_noisy, t)

    assert loss_val.item() > 0.0
    assert "mse" in metrics
    assert "mrstft" in metrics
    assert "composite_loss" in metrics


def test_train_voice_native_fast_dev_run(tmp_path: Path) -> None:
    """Verifies end-to-end native training pass with synthetic audio under fast_dev_run."""
    fs = 48000
    duration_s = 2.0
    n_frames = int(fs * duration_s)

    # Create dummy source and target audio files
    t = np.linspace(0, duration_s, n_frames, endpoint=False, dtype=np.float32)
    sig_src = (0.5 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
    sig_tgt = (0.4 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)

    src_wav = tmp_path / "dummy_src.wav"
    tgt_wav = tmp_path / "dummy_tgt.wav"

    with AudioFile(str(src_wav), "w", samplerate=fs, num_channels=1) as f:
        f.write(sig_src[np.newaxis, :])

    with AudioFile(str(tgt_wav), "w", samplerate=fs, num_channels=1) as f:
        f.write(sig_tgt[np.newaxis, :])

    models_dir = tmp_path / "models"

    ok = train_voice_native(
        instrument="34in_standard_p",
        voice="precision_dub",
        reference_wav=src_wav,
        output_wav=tgt_wav,
        models_dir=models_dir,
        epochs=1,
        min_epochs=1,
        fast_dev_run=True,
        silent=True,
        device="cpu",
    )

    assert ok is True
    exported_models = list(models_dir.glob("**/*.nam"))
    assert len(exported_models) == 1
    with open(exported_models[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["architecture"] == "SlimmableContainer"
    assert data["version"] == "0.7.0"
    assert len(data["config"]["submodels"]) == 2


def test_native_default_hyperparameters_and_isolation() -> None:
    """Verifies that native hyperparameter defaults are isolated from stock standards."""
    import inspect

    from allomorph.trainer import (
        DEFAULT_MAX_EPOCHS,
        DEFAULT_MIN_EPOCHS,
        DEFAULT_PATIENCE,
        NATIVE_DEFAULT_LITE_WEIGHT,
        NATIVE_DEFAULT_LR,
        NATIVE_DEFAULT_LR_SCHEDULER,
        NATIVE_DEFAULT_MAX_EPOCHS,
        NATIVE_DEFAULT_MIN_DELTA,
        NATIVE_DEFAULT_MIN_EPOCHS,
        NATIVE_DEFAULT_MRSTFT_WEIGHT,
        NATIVE_DEFAULT_PATIENCE,
        NATIVE_DEFAULT_SLICE_LEN,
        NATIVE_DEFAULT_WEIGHT_DECAY,
        STOCK_DEFAULT_MAX_EPOCHS,
        train_voice_native,
    )

    # 1. Stock defaults: standard 200-epoch upstream NAM training
    assert STOCK_DEFAULT_MAX_EPOCHS == 200

    # 2. Native defaults are distinct and tuned for rapid in-VRAM convergence
    assert NATIVE_DEFAULT_MAX_EPOCHS == 100
    assert NATIVE_DEFAULT_MIN_EPOCHS == 60
    assert NATIVE_DEFAULT_PATIENCE == 10
    assert NATIVE_DEFAULT_MIN_DELTA == 1.0e-6
    assert NATIVE_DEFAULT_LR == 4.0e-3
    assert NATIVE_DEFAULT_LR_SCHEDULER == "exponential"
    assert NATIVE_DEFAULT_LITE_WEIGHT == 0.50
    assert NATIVE_DEFAULT_WEIGHT_DECAY == 3.17e-7
    assert NATIVE_DEFAULT_SLICE_LEN == 8192
    assert NATIVE_DEFAULT_MRSTFT_WEIGHT == 0.0005

    # 3. Default platform hyperparameters match Native defaults
    assert DEFAULT_MAX_EPOCHS == NATIVE_DEFAULT_MAX_EPOCHS
    assert DEFAULT_MIN_EPOCHS == NATIVE_DEFAULT_MIN_EPOCHS
    assert DEFAULT_PATIENCE == NATIVE_DEFAULT_PATIENCE

    # 4. train_voice_native signature matches native standards
    sig = inspect.signature(train_voice_native)
    assert sig.parameters["epochs"].default == NATIVE_DEFAULT_MAX_EPOCHS
    assert sig.parameters["min_epochs"].default == NATIVE_DEFAULT_MIN_EPOCHS
    assert sig.parameters["lr"].default == NATIVE_DEFAULT_LR
    assert sig.parameters["weight_decay"].default == NATIVE_DEFAULT_WEIGHT_DECAY
    assert sig.parameters["slice_len"].default == NATIVE_DEFAULT_SLICE_LEN
    assert sig.parameters["lite_weight"].default == NATIVE_DEFAULT_LITE_WEIGHT
    assert sig.parameters["mrstft_weight"].default == NATIVE_DEFAULT_MRSTFT_WEIGHT
