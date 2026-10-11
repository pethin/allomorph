"""First-principles spectral and time-domain loss functions for NAM training.

Houses the Self-Normalizing Multi-Band High-Pass ESR Loss and FIR filter design.
"""

from __future__ import annotations

from typing import override

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def design_fir_highpass(num_taps: int, cutoff_hz: float, fs: float = 48000.0) -> np.ndarray:
    """Designs a zero-phase linear-phase FIR high-pass filter matching scipy bit-for-bit using pure NumPy.

    Constructs windowed-sinc impulse response with symmetric Hamming window and unity passband scaling at Nyquist.

    Args:
        num_taps: Number of filter taps (must be odd for Type I zero-phase FIR).
        cutoff_hz: High-pass corner frequency in Hz.
        fs: Sampling rate in Hz (default: 48000.0).

    Returns:
        1D float32 array of normalized impulse response coefficients.
    """
    nyq = 0.5 * fs
    left = cutoff_hz / nyq
    alpha = 0.5 * (num_taps - 1)
    m = np.arange(num_taps, dtype=np.float64) - alpha
    h = np.sinc(m) - left * np.sinc(left * m)
    win = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(num_taps) / (num_taps - 1))
    h = h * win
    s = np.sum(h * np.cos(np.pi * m))
    h = h / s
    return h.astype(np.float32)


class NormalizedHighPassESRLoss(nn.Module):
    """Zero-phase linear-phase FIR high-pass ESR loss with batch energy normalization.

    Neutralizes physical 22,000x string harmonic energy roll-off above 3-7 kHz by computing
    high-pass error energy normalized by high-pass target energy.
    """

    kernel: torch.Tensor

    def __init__(
        self,
        cutoff_hz: float,
        num_taps: int = 31,
        fs: float = 48000.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.cutoff_hz = float(cutoff_hz)
        self.num_taps = int(num_taps)
        self.fs = float(fs)
        self.eps = float(eps)

        taps = design_fir_highpass(self.num_taps, self.cutoff_hz, fs=self.fs)
        kernel = torch.from_numpy(taps).view(1, 1, -1)
        self.register_buffer("kernel", kernel)

    @override
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes energy-normalized high-pass ESR.

        Args:
            pred: Predicted waveform tensor (B, L) or (B, 1, L).
            target: Target waveform tensor (B, L) or (B, 1, L).

        Returns:
            Scalar tensor representing normalized high-pass ESR.
        """
        if pred.ndim == 1:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 2:
            target = target.unsqueeze(1)

        pad = (self.num_taps - 1) // 2
        kernel = self.kernel.to(dtype=pred.dtype)
        pred_hp = F.conv1d(pred, kernel, padding=pad)
        target_hp = F.conv1d(target, kernel, padding=pad)

        error_hp = pred_hp - target_hp
        error_energy = torch.sum(error_hp**2)
        target_energy = torch.sum(target_hp**2)

        return error_energy / (target_energy + self.eps)


class MultiBandSpectralLoss(nn.Module):
    """Composite Multi-Band Loss combining MSE anchoring and normalized high-pass ESR."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        hp_mid_weight: float = 0.50,
        hp_mid_cutoff: float = 3000.0,
        hp_high_weight: float = 0.50,
        hp_high_cutoff: float = 7000.0,
        fs: float = 48000.0,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.hp_mid_weight = float(hp_mid_weight)
        self.hp_high_weight = float(hp_high_weight)

        self.hp_mid_loss = (
            NormalizedHighPassESRLoss(cutoff_hz=hp_mid_cutoff, fs=fs)
            if self.hp_mid_weight > 0.0
            else None
        )
        self.hp_high_loss = (
            NormalizedHighPassESRLoss(cutoff_hz=hp_high_cutoff, fs=fs)
            if self.hp_high_weight > 0.0
            else None
        )

    @override
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, return_metrics: bool = True
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes multi-band loss and returns scalar loss and component breakdown."""
        if pred.ndim == 1:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 2:
            target = target.unsqueeze(1)

        mse = F.mse_loss(pred, target)
        loss = self.mse_weight * mse

        if not return_metrics:
            if self.hp_mid_loss is not None:
                hp_mid = self.hp_mid_loss(pred, target)
                loss = loss + self.hp_mid_weight * hp_mid
            if self.hp_high_loss is not None:
                hp_high = self.hp_high_loss(pred, target)
                loss = loss + self.hp_high_weight * hp_high
            return loss, {}

        metrics: dict[str, float] = {
            "mse": float(mse.detach().cpu().item()),
        }

        if self.hp_mid_loss is not None:
            hp_mid = self.hp_mid_loss(pred, target)
            loss = loss + self.hp_mid_weight * hp_mid
            metrics["hp_mid_esr"] = float(hp_mid.detach().cpu().item())

        if self.hp_high_loss is not None:
            hp_high = self.hp_high_loss(pred, target)
            loss = loss + self.hp_high_weight * hp_high
            metrics["hp_high_esr"] = float(hp_high.detach().cpu().item())

        metrics["composite_loss"] = float(loss.detach().cpu().item())
        return loss, metrics
