"""Unit tests for `denoiser.loss` (C-infinity silence envelope weighting and fused dual-band Blackman FIR high-pass loss)."""

from __future__ import annotations

import pytest
import torch

from denoiser.loss import (
    DenoiserLoss,
    build_fused_dual_hp_kernels,
    compute_cinf_envelope_weights,
)


def test_cinf_envelope_weights_monotonicity_and_smooth_gradients() -> None:
    """compute_cinf_envelope_weights must equal 1 + silence_weight at silence, approach 1.0 on loud notes, and be C-infinity differentiable."""
    silence = torch.zeros((2, 1024), dtype=torch.float32)
    loud = torch.ones((2, 1024), dtype=torch.float32) * 0.35

    w_sil = compute_cinf_envelope_weights(
        silence, threshold_dbfs=-44.0, silence_weight=100.0, order=4
    )
    w_loud = compute_cinf_envelope_weights(
        loud, threshold_dbfs=-44.0, silence_weight=100.0, order=4
    )

    assert torch.allclose(w_sil, torch.full_like(w_sil, 101.0), atol=1e-4)
    assert float(torch.max(w_loud).item()) < 1.001
    assert float(torch.min(w_loud).item()) >= 1.0

    # Verify smooth monotonic transition across ramp
    ramp = torch.linspace(0.0, 0.10, 2048, dtype=torch.float32).unsqueeze(0)
    w_ramp = compute_cinf_envelope_weights(
        ramp, threshold_dbfs=-44.0, silence_weight=100.0, order=4
    )
    diffs = torch.diff(w_ramp[0])
    assert float(torch.max(diffs).item()) <= 1e-5


def test_fused_dual_hp_kernels_and_denoiser_loss_penalizes_smps_and_silence_noise() -> None:
    """DenoiserLoss must fuse 1.2 kHz and 5.5 kHz Blackman HP kernels and heavily penalize idle hum/SMPS noise."""
    kernels = build_fused_dual_hp_kernels(fc_mid=1200.0, fc_high=5500.0, fs=48000.0, num_taps=255)
    assert kernels.shape == (2, 1, 255)

    loss_fn = DenoiserLoss(
        threshold_dbfs=-44.0,
        silence_weight=100.0,
        mse_weight=20.0,
        fir_hp_mid_weight=1.5,
        fir_hp_high_weight=3.0,
    )

    bs = 2
    ny = 2048
    y_silence = torch.zeros((bs, ny), dtype=torch.float32)
    y_loud = torch.ones((bs, ny), dtype=torch.float32) * 0.25

    # Identical small residual error (0.002) during silence vs during loud bass note
    t = torch.arange(ny, dtype=torch.float32) / 48000.0
    smps_err = (0.002 * torch.sin(2.0 * 3.14159265 * 10000.0 * t)).view(1, 1, ny).expand(bs, 2, ny)

    pred_on_silence = (y_silence.unsqueeze(1) + smps_err).requires_grad_(True)
    pred_on_loud = (y_loud.unsqueeze(1) + smps_err).requires_grad_(True)

    loss_sil, mse_sil, hp_sil = loss_fn(pred_on_silence, y_silence)
    loss_loud, mse_loud, _hp_loud = loss_fn(pred_on_loud, y_loud)

    # Raw MSE and HP error are identical within float32 precision, while total loss on silence is >30x higher
    assert float(mse_sil.item()) == pytest.approx(float(mse_loud.item()), rel=1e-4)
    assert float(hp_sil.item()) > 0.0
    assert float(loss_sil.item()) > 30.0 * float(loss_loud.item())

    loss_sil.backward()
    assert pred_on_silence.grad is not None
    assert torch.all(torch.isfinite(pred_on_silence.grad))
