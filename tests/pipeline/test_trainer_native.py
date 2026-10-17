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
    """Verifies off-block channel weight and gradient mask enforcement and exact submodel weight counts."""
    model = AllomorphWaveNetA2()

    # Deliberately pollute off-block weights
    with torch.no_grad():
        first_layer = model.layers[0]
        assert isinstance(first_layer, AllomorphWaveNetLayer)
        first_layer.conv.weight[0, 5, :] = 99.0
        first_layer.layer1x1.weight[0, 5, :] = 99.0
        model.head_rechannel.weight[0, 5, :] = 99.0

    model.mask_weights()

    with torch.no_grad():
        assert first_layer.conv.weight[0, 5, :].abs().max().item() == 0.0
        assert first_layer.layer1x1.weight[0, 5, :].abs().max().item() == 0.0
        assert model.head_rechannel.weight[0, 5, :].abs().max().item() == 0.0

    # Verify gradient masking on off-block entries
    x = torch.randn(1, 1, model.receptive_field + 64)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()
    with torch.no_grad():
        assert first_layer.conv.weight.grad is not None
        first_layer.conv.weight.grad[0, 5, :] = 42.0
    model.mask_gradients()
    with torch.no_grad():
        assert first_layer.conv.weight.grad is not None
        assert first_layer.conv.weight.grad[0, 5, :].abs().max().item() == 0.0

    # Submodel 0: channels_3 Lite (exactly 1,871 weights)
    w_lite = model.export_submodel_weights(0)
    assert len(w_lite) == 1871

    # Submodel 1: channels_8 Full (exactly 12,146 weights)
    w_full = model.export_submodel_weights(1)
    assert len(w_full) == 12146


def test_wavenet_a2_activation_homotopy() -> None:
    """Verifies C-infinity SiLU-gated LeakyReLU activation homotopy and exact eval() convergence."""
    model = AllomorphWaveNetA2()
    x = torch.randn(1, 1, model.receptive_field + 128)

    # 1. Exact baseline at tau = 0.0
    model.train()
    model.activation_tau = 0.0
    with torch.no_grad():
        out_exact = model(x)

    # 2. Asymptotic convergence as tau -> 0+
    model.activation_tau = 1e-4
    with torch.no_grad():
        out_tiny_tau = model(x)
    assert torch.allclose(out_exact, out_tiny_tau, atol=1e-4)

    # 3. Eval mode strictly enforces exact LeakyReLU(0.01) even if activation_tau > 0
    model.activation_tau = 0.02
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert torch.allclose(out_exact, out_eval, atol=1e-6)


def test_wavenet_a2_warm_start_head() -> None:
    """Verifies closed-form Tikhonov-regularized warm-start reduces initial linear error and preserves masks."""
    torch.manual_seed(42)
    model = AllomorphWaveNetA2()
    model.eval()

    ny = 2048
    rf_pad = model.receptive_field - 1
    t = torch.linspace(0.0, 1.0, ny)
    sig_in = 0.3 * torch.sin(2.0 * np.pi * 110.0 * t) + 0.15 * torch.sin(2.0 * np.pi * 440.0 * t)
    sig_tgt = 0.25 * torch.sin(2.0 * np.pi * 110.0 * t + 0.2) + 0.08 * torch.sin(
        2.0 * np.pi * 440.0 * t - 0.3
    )
    x_cal = torch.cat([torch.zeros(rf_pad), sig_in])

    with torch.no_grad():
        pred_before = model(x_cal.unsqueeze(0).unsqueeze(0))
        mse_before_full = float(torch.mean((pred_before[0, 1] - sig_tgt) ** 2).item())
        mse_before_lite = float(torch.mean((pred_before[0, 0] - sig_tgt) ** 2).item())

    model.warm_start_head(x_cal, sig_tgt)

    with torch.no_grad():
        pred_after = model(x_cal.unsqueeze(0).unsqueeze(0))
        mse_after_full = float(torch.mean((pred_after[0, 1] - sig_tgt) ** 2).item())
        mse_after_lite = float(torch.mean((pred_after[0, 0] - sig_tgt) ** 2).item())

        # Off-block head_rechannel weights must remain strictly zero
        assert model.head_rechannel.weight[0, 3:, :].abs().max().item() == 0.0
        assert model.head_rechannel.weight[1, :3, :].abs().max().item() == 0.0

    assert mse_after_full < mse_before_full
    assert mse_after_lite < mse_before_lite


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
    """Verifies composite multi-band loss with broadband MSE and 255-tap Blackman FIR high-pass MSE."""
    loss_fn = MultiBandSpectralLoss(
        mse_weight=1.0,
        fir_hp_weight=2.5,
        fir_hp_fc=1200.0,
        fs=48000.0,
    )

    t = torch.sin(2.0 * np.pi * 440.0 * torch.linspace(0, 1, 48000)).unsqueeze(0)
    p = t.clone()

    # Exact equality should have ~0 loss
    loss_zero, metrics_zero = loss_fn(p, t)
    assert loss_zero.item() < 1e-7
    assert metrics_zero["mse"] < 1e-7
    assert metrics_zero["fir_hp"] < 1e-7

    # Perturbed signal has positive loss and detailed metrics
    p_noisy = t + 0.1 * torch.randn_like(t)
    loss_val, metrics = loss_fn(p_noisy, t)

    assert loss_val.item() > 0.0
    assert "mse" in metrics
    assert "fir_hp" in metrics
    assert "composite_loss" in metrics
    assert metrics["fir_hp"] > 0.0


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
        instrument="30in_emg_mmtw:dual_mode",
        voice="34in_standard_p:slab_dub",
        reference_wav=src_wav,
        output_wav=tgt_wav,
        models_dir=models_dir,
        epochs=1,
        min_epochs=1,
        mse_weight=1.0,
        fast_dev_run=True,
        silent=True,
        device="cpu",
    )

    assert ok is True
    exported_models = list(models_dir.glob("**/*.nam"))
    assert len(exported_models) == 1
    # Verify 3-level standalone directory structure: models/<src_inst>/<src_voicing>/<tgt_inst>/
    rel_parts = exported_models[0].relative_to(models_dir).parts
    assert rel_parts[:3] == ("30in_emg_mmtw", "dual_mode", "34in_standard_p")

    with open(exported_models[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["architecture"] == "SlimmableContainer"
    assert data["version"] == "0.7.0"
    assert len(data["config"]["submodels"]) == 2

    # Finding 4.1: Verify full NamExportMetadata provenance is preserved in exported .nam
    from allomorph.pipeline.schema import NamExportMetadata

    export_meta = NamExportMetadata.model_validate(data["metadata"])
    assert export_meta.source_instrument.id == "30in_emg_mmtw"
    assert export_meta.source_instrument.voicing.id == "dual_mode"
    assert export_meta.target_voicing.id == "slab_dub"
    assert (
        export_meta.training.validation_esr is not None
        and export_meta.training.validation_esr > 0.0
    )

    # Finding 4.4: Verify standalone manifest.json is written in target instrument directory
    manifest_file = exported_models[0].parent / "manifest.json"
    assert manifest_file.exists()


def test_native_default_hyperparameters_and_isolation() -> None:
    """Verifies that native hyperparameter defaults are isolated from stock standards."""
    import inspect

    from allomorph.trainer import (
        DEFAULT_MAX_EPOCHS,
        DEFAULT_MIN_EPOCHS,
        DEFAULT_PATIENCE,
        NATIVE_DEFAULT_EMA_DECAY,
        NATIVE_DEFAULT_FIR_HP_FC,
        NATIVE_DEFAULT_FIR_HP_WEIGHT,
        NATIVE_DEFAULT_HOMOTOPY_TAU_INIT,
        NATIVE_DEFAULT_LITE_WEIGHT,
        NATIVE_DEFAULT_LR,
        NATIVE_DEFAULT_LR_SCHEDULER,
        NATIVE_DEFAULT_MAX_EPOCHS,
        NATIVE_DEFAULT_MAX_SPIKE_RESTARTS,
        NATIVE_DEFAULT_MIN_DELTA,
        NATIVE_DEFAULT_MIN_EPOCHS,
        NATIVE_DEFAULT_PATIENCE,
        NATIVE_DEFAULT_SINE_AMPLITUDES,
        NATIVE_DEFAULT_SINE_FREQUENCIES,
        NATIVE_DEFAULT_SINE_REGULARIZER,
        NATIVE_DEFAULT_SLICE_LEN,
        NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD,
        NATIVE_DEFAULT_VAL_INTERVAL,
        NATIVE_DEFAULT_WARM_START,
        NATIVE_DEFAULT_WEIGHT_DECAY,
        STOCK_DEFAULT_MAX_EPOCHS,
        train_voice_native,
    )

    # 1. Stock defaults: standard 200-epoch upstream NAM training
    assert STOCK_DEFAULT_MAX_EPOCHS == 200

    # 2. Native defaults are distinct and tuned for rapid in-VRAM convergence
    assert NATIVE_DEFAULT_MAX_EPOCHS == 120
    assert NATIVE_DEFAULT_MIN_EPOCHS == 100
    assert NATIVE_DEFAULT_PATIENCE == 15
    assert NATIVE_DEFAULT_MIN_DELTA == 1.0e-6
    assert NATIVE_DEFAULT_LR == 4.0e-3
    assert NATIVE_DEFAULT_LR_SCHEDULER == "cosine"
    assert NATIVE_DEFAULT_LITE_WEIGHT == 0.50
    assert NATIVE_DEFAULT_WEIGHT_DECAY == 3.17e-7
    assert NATIVE_DEFAULT_SLICE_LEN == 8192
    assert NATIVE_DEFAULT_FIR_HP_WEIGHT == 2.5
    assert NATIVE_DEFAULT_FIR_HP_FC == 1200.0
    assert NATIVE_DEFAULT_VAL_INTERVAL == 5
    assert NATIVE_DEFAULT_SINE_REGULARIZER is True
    assert NATIVE_DEFAULT_SINE_AMPLITUDES == (0.25, 0.10, 0.05, 0.03, 0.008)
    assert NATIVE_DEFAULT_WARM_START is True
    assert NATIVE_DEFAULT_HOMOTOPY_TAU_INIT == 0.02
    assert NATIVE_DEFAULT_EMA_DECAY == 0.995
    assert NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD == 500.0
    assert NATIVE_DEFAULT_MAX_SPIKE_RESTARTS == 3

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
    assert sig.parameters["val_interval"].default == NATIVE_DEFAULT_VAL_INTERVAL
    assert sig.parameters["fir_hp_weight"].default == NATIVE_DEFAULT_FIR_HP_WEIGHT
    assert sig.parameters["fir_hp_fc"].default == NATIVE_DEFAULT_FIR_HP_FC
    assert sig.parameters["sine_regularizer"].default == NATIVE_DEFAULT_SINE_REGULARIZER
    assert sig.parameters["sine_frequencies"].default == NATIVE_DEFAULT_SINE_FREQUENCIES
    assert sig.parameters["sine_amplitudes"].default == NATIVE_DEFAULT_SINE_AMPLITUDES
    assert sig.parameters["warm_start"].default == NATIVE_DEFAULT_WARM_START
    assert sig.parameters["homotopy_tau_init"].default == NATIVE_DEFAULT_HOMOTOPY_TAU_INIT
    assert sig.parameters["ema_decay"].default == NATIVE_DEFAULT_EMA_DECAY
    assert sig.parameters["spike_ratio_threshold"].default == NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD
    assert sig.parameters["max_spike_restarts"].default == NATIVE_DEFAULT_MAX_SPIKE_RESTARTS


def test_normalize_training_stem_calibrates_unnormalized_stems() -> None:
    """Verifies normalize_training_stem calibrates arbitrary unnormalized stems to TARGET_RMS_DBFS (-20.50 dBFS RMS)."""
    import math

    from allomorph.dsp import TARGET_RMS_DBFS, compute_true_peak
    from allomorph.dsp.conditioning import CALIBRATION_PEAK_CEILING
    from allomorph.trainer import normalize_training_stem

    sr = 48000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float64)
    quiet_sub_stem = 0.02 * np.sin(2.0 * np.pi * 80.0 * t)
    loud_clank_stem = 0.60 * np.sin(2.0 * np.pi * 3000.0 * t)

    norm_quiet = normalize_training_stem(quiet_sub_stem, sample_rate=sr)
    norm_loud = normalize_training_stem(loud_clank_stem, sample_rate=sr)

    rms_quiet_dbfs = 20.0 * math.log10(float(np.sqrt(np.mean(norm_quiet**2))))
    rms_loud_dbfs = 20.0 * math.log10(float(np.sqrt(np.mean(norm_loud**2))))

    assert abs(rms_quiet_dbfs - TARGET_RMS_DBFS) < 1e-3
    assert abs(rms_loud_dbfs - TARGET_RMS_DBFS) < 1e-3
    assert compute_true_peak(norm_quiet, sample_rate=sr) <= CALIBRATION_PEAK_CEILING + 1e-4
    assert compute_true_peak(norm_loud, sample_rate=sr) <= CALIBRATION_PEAK_CEILING + 1e-4


def test_evaluate_loss_spike_condition() -> None:
    """Verifies pure loss spike detection logic across normal progression, transient wobbles, catastrophic explosions, and NaNs."""
    from allomorph.trainer import evaluate_loss_spike_condition

    # 1. Normal progression or small fluctuation below min_abs_loss_floor (0.02) -> no spike
    is_spike, reason = evaluate_loss_spike_condition(
        train_loss_epoch=0.0046,
        min_train_loss=0.00001,
        val_loss_epoch=0.00002,
        min_val_loss=0.00001,
    )
    assert is_spike is False
    assert reason is None

    # 2. Catastrophic train loss explosion (e.g. 0.0000 -> 0.0343 / 14.8920 as in Rickenbacker Vintage)
    is_spike_t, reason_t = evaluate_loss_spike_condition(
        train_loss_epoch=0.0343,
        min_train_loss=0.00004,
    )
    assert is_spike_t is True
    assert reason_t is not None and "train loss spiked" in reason_t

    # 3. Catastrophic validation loss explosion (e.g. 0.00001 -> 0.29012)
    is_spike_v, reason_v = evaluate_loss_spike_condition(
        train_loss_epoch=0.0002,
        min_train_loss=0.0001,
        val_loss_epoch=0.29012,
        min_val_loss=0.00001,
    )
    assert is_spike_v is True
    assert reason_v is not None and "val loss spiked" in reason_v

    # 4. Non-finite NaN / Inf loss
    is_spike_nan, reason_nan = evaluate_loss_spike_condition(
        train_loss_epoch=float("nan"),
        min_train_loss=0.001,
    )
    assert is_spike_nan is True
    assert reason_nan is not None and "non-finite" in reason_nan


def test_train_voice_native_restarts_on_unlucky_spike(tmp_path: Path, monkeypatch: object) -> None:
    """Verifies that train_voice_native detects an unlucky loss spike and automatically restarts training."""
    import pytest

    mp = pytest.MonkeyPatch()
    fs = 48000
    duration_s = 1.5
    n_frames = int(fs * duration_s)

    t = np.linspace(0, duration_s, n_frames, endpoint=False, dtype=np.float32)
    sig_src = (0.5 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
    sig_tgt = (0.4 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)

    src_wav = tmp_path / "spike_src.wav"
    tgt_wav = tmp_path / "spike_tgt.wav"

    with AudioFile(str(src_wav), "w", samplerate=fs, num_channels=1) as f:
        f.write(sig_src[np.newaxis, :])
    with AudioFile(str(tgt_wav), "w", samplerate=fs, num_channels=1) as f:
        f.write(sig_tgt[np.newaxis, :])

    models_dir = tmp_path / "models_spike"
    orig_forward = AllomorphWaveNetA2.forward
    forward_calls = 0

    def spiky_forward(self: AllomorphWaveNetA2, x: torch.Tensor) -> torch.Tensor:
        nonlocal forward_calls
        out = orig_forward(self, x)
        if self.training:
            forward_calls += 1
            # Inject an unlucky 100x output spike on the 2nd training forward call (Epoch 2 of Attempt 1)
            if forward_calls == 2:
                return out * 100.0 + 10.0
        return out

    mp.setattr(AllomorphWaveNetA2, "forward", spiky_forward)
    try:
        ok = train_voice_native(
            instrument="30in_emg_mmtw:dual_mode",
            voice="34in_standard_p:slab_dub",
            reference_wav=src_wav,
            output_wav=tgt_wav,
            models_dir=models_dir,
            epochs=3,
            min_epochs=2,
            slice_len=4096,
            sine_regularizer=False,
            warm_start=False,
            fast_dev_run=False,
            silent=True,
            seed=42,
            device="cpu",
            max_spike_restarts=2,
        )
    finally:
        mp.undo()

    assert ok is True
    exported = list(models_dir.glob("**/*.nam"))
    assert len(exported) == 1
    with open(exported[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["training"]["spike_restarts"] == 1
