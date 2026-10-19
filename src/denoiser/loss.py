"""C-infinity envelope-gated silence loss and fused dual-band Blackman FIR high-pass loss for the Allomorph Denoiser."""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from torch import nn

from allomorph.dsp.constants import FS
from allomorph.trainer.native.loss import design_sharp_fir_highpass
from denoiser.constants import (
    DENOISER_DEFAULT_FIR_HP_HIGH_FC,
    DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT,
    DENOISER_DEFAULT_FIR_HP_MID_FC,
    DENOISER_DEFAULT_FIR_HP_MID_WEIGHT,
    DENOISER_DEFAULT_LITE_WEIGHT,
    DENOISER_DEFAULT_MSE_WEIGHT,
    DENOISER_FIR_NUM_TAPS,
)


def compute_cinf_envelope_weights(
    y_clean: torch.Tensor,
    *,
    threshold_dbfs: float = -44.0,
    silence_weight: float = 100.0,
    order: int = 4,
    window_samples: int = 257,
) -> torch.Tensor:
    """Computes a C-infinity smooth rational silence-gate weight envelope from the clean target.

    Given clean target `y_clean` of shape `(B, T)`, estimates the local RMS envelope `E(y)`
    over `window_samples` (~5.35 ms at 48 kHz) and evaluates the C-infinity rational curve:
        w_env(y) = 1.0 + silence_weight / (1.0 + (E(y) / tau_gate)^(2 * order))

    Properties:
      - At digital silence (`E(y) = 0`), `w_env = 1.0 + silence_weight`, heavily penalizing any
        residual 50/60 Hz hum or SMPS whine leaking through between notes.
      - Above the gate knee (`E(y) >> tau_gate`), `w_env -> 1.0` smoothly with zero piecewise
        `torch.where` kinks, preserving natural bass note sustain and pick/finger transients.
    """
    if silence_weight <= 0.0:
        return torch.ones_like(y_clean, dtype=torch.float32)

    tau_gate = float(10.0 ** (threshold_dbfs / 20.0))
    y_sq = torch.square(y_clean.float()).unsqueeze(1)  # (B, 1, T)
    pad = window_samples // 2
    local_ms = F.avg_pool1d(
        F.pad(y_sq, (pad, pad), mode="replicate"),
        kernel_size=window_samples,
        stride=1,
    ).squeeze(1)  # (B, T)

    # C-infinity rational transition: (E^2 / tau_gate^2)^order == (E / tau_gate)^(2 * order)
    tau_sq = max(tau_gate * tau_gate, 1e-16)
    ratio_pow = torch.pow(local_ms / tau_sq, float(order))
    weights = 1.0 + float(silence_weight) / (1.0 + ratio_pow)
    return weights


def build_fused_dual_hp_kernels(
    *,
    fc_mid: float = DENOISER_DEFAULT_FIR_HP_MID_FC,
    fc_high: float = DENOISER_DEFAULT_FIR_HP_HIGH_FC,
    fs: float = float(FS),
    num_taps: int = DENOISER_FIR_NUM_TAPS,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Designs a stacked 2-channel `(2, 1, num_taps)` Blackman FIR high-pass kernel tensor.

    Channel 0 (`fc_mid = 1200 Hz`) targets upper-order 50/60 Hz dimmer buzz harmonics.
    Channel 1 (`fc_high = 5500 Hz`) targets SMPS switching regulators, charge-pump clock whine,
    and broadband treble hiss.
    """
    k_mid = design_sharp_fir_highpass(fc=fc_mid, fs=fs, num_taps=num_taps)  # (1, 1, K)
    k_high = design_sharp_fir_highpass(fc=fc_high, fs=fs, num_taps=num_taps)  # (1, 1, K)
    stacked = torch.cat([k_mid, k_high], dim=0)  # (2, 1, K)
    if device is not None:
        return stacked.to(device=device, dtype=dtype)
    return stacked.to(dtype=dtype)


class DenoiserLoss(nn.Module):
    """Fused time-domain + C-infinity silence envelope + dual-band FIR high-pass loss for WaveNet A2."""

    dual_hp_kernels: torch.Tensor

    def __init__(
        self,
        *,
        threshold_dbfs: float = -44.0,
        silence_weight: float = 100.0,
        envelope_order: int = 4,
        mse_weight: float = DENOISER_DEFAULT_MSE_WEIGHT,
        lite_weight: float = DENOISER_DEFAULT_LITE_WEIGHT,
        fir_hp_mid_fc: float = DENOISER_DEFAULT_FIR_HP_MID_FC,
        fir_hp_high_fc: float = DENOISER_DEFAULT_FIR_HP_HIGH_FC,
        fir_hp_mid_weight: float = DENOISER_DEFAULT_FIR_HP_MID_WEIGHT,
        fir_hp_high_weight: float = DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT,
        fs: float = float(FS),
        num_taps: int = DENOISER_FIR_NUM_TAPS,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.threshold_dbfs = float(threshold_dbfs)
        self.silence_weight = float(silence_weight)
        self.envelope_order = int(envelope_order)
        self.mse_weight = float(mse_weight)
        self.lite_weight = float(lite_weight)
        self.fir_hp_mid_weight = float(fir_hp_mid_weight)
        self.fir_hp_high_weight = float(fir_hp_high_weight)
        self.num_taps = int(num_taps)

        kernels = build_fused_dual_hp_kernels(
            fc_mid=fir_hp_mid_fc,
            fc_high=fir_hp_high_fc,
            fs=fs,
            num_taps=num_taps,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("dual_hp_kernels", kernels)

    @override
    def forward(
        self,
        pred: torch.Tensor,
        y_clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the composite denoiser loss on dual-tier WaveNet A2 predictions.

        Args:
            pred: Model prediction tensor of shape `(B, 2, T)` (`[:, 0, :]` = Ch3 Lite, `[:, 1, :]` = Ch8 Full).
            y_clean: Clean target tensor of shape `(B, T)`.

        Returns:
            Tuple of `(total_loss, mse_full_detached, hp_high_full_detached)` as 0D GPU tensors.
        """
        bs, _, ny = pred.shape
        err = pred - y_clean.unsqueeze(1)  # (B, 2, T)
        err_sq = torch.square(err.float())

        # 1. C-infinity envelope-weighted MSE (boosts penalty in silent & decay regions)
        w_env = compute_cinf_envelope_weights(
            y_clean,
            threshold_dbfs=self.threshold_dbfs,
            silence_weight=self.silence_weight,
            order=self.envelope_order,
        )  # (B, T)

        l_lite_env = torch.mean(w_env * err_sq[:, 0, :])
        l_full_env = torch.mean(w_env * err_sq[:, 1, :])
        mse_full_raw = torch.mean(err_sq[:, 1, :])

        total_loss = self.mse_weight * (l_full_env + self.lite_weight * l_lite_env)

        # 2. Single fused Conv1D GEMM pass across both submodels and both Blackman FIR HP filters
        if (self.fir_hp_mid_weight > 0.0 or self.fir_hp_high_weight > 0.0) and ny >= self.num_taps:
            # Reshape (B, 2, T) -> (2*B, 1, T), convolve with (2, 1, K) -> (2*B, 2, T - K + 1)
            err_hp = F.conv1d(
                err.reshape(bs * 2, 1, ny),
                self.dual_hp_kernels,
                padding="valid",
            ).view(bs, 2, 2, -1)
            err_hp_sq = torch.square(err_hp.float())

            # Submodel dim=1 (0: Lite, 1: Full), Filter band dim=2 (0: Mid 1.2kHz, 1: High 5.5kHz)
            hp_mid_lite = torch.mean(err_hp_sq[:, 0, 0, :])
            hp_mid_full = torch.mean(err_hp_sq[:, 1, 0, :])
            hp_high_lite = torch.mean(err_hp_sq[:, 0, 1, :])
            hp_high_full = torch.mean(err_hp_sq[:, 1, 1, :])

            total_loss = (
                total_loss
                + self.fir_hp_mid_weight * (hp_mid_full + self.lite_weight * hp_mid_lite)
                + self.fir_hp_high_weight * (hp_high_full + self.lite_weight * hp_high_lite)
            )
        else:
            hp_high_full = torch.zeros((), device=pred.device, dtype=torch.float32)

        return total_loss, mse_full_raw.detach(), hp_high_full.detach()
