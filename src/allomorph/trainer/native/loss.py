"""First-principles spectral and time-domain loss functions for NAM Architecture 2 training.

Houses MultiResolutionSTFTLoss and composite MultiBandSpectralLoss.
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from torch import nn

from allomorph.trainer.constants import (
    NATIVE_DEFAULT_MRSTFT_FFT_SIZES,
    NATIVE_DEFAULT_MRSTFT_HOP_SIZES,
    NATIVE_DEFAULT_MRSTFT_WEIGHT,
    NATIVE_DEFAULT_MRSTFT_WIN_LENGTHS,
    NATIVE_DEFAULT_MSE_WEIGHT,
)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss combining spectral convergence and log-magnitude distance across multiple FFT window scales.

    Standard audio ML loss (Yamamoto et al. 2019) providing uniform spectral gradient pressure across the audible bandwidth
    without any small-signal denominator instability. Uses C-infinity power clamping (min=1e-8) for zero-gradient silence.
    """

    fft_sizes: tuple[int, ...]
    hop_sizes: tuple[int, ...]
    win_lengths: tuple[int, ...]
    w_sc: float
    w_log_mag: float
    eps: float

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = NATIVE_DEFAULT_MRSTFT_FFT_SIZES,
        hop_sizes: tuple[int, ...] = NATIVE_DEFAULT_MRSTFT_HOP_SIZES,
        win_lengths: tuple[int, ...] = NATIVE_DEFAULT_MRSTFT_WIN_LENGTHS,
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.w_sc = float(w_sc)
        self.w_log_mag = float(w_log_mag)
        self.eps = float(eps)

        for idx, w_len in enumerate(win_lengths):
            self.register_buffer(f"window_{idx}", torch.hann_window(w_len))

    @override
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        elif pred.ndim == 1:
            pred = pred.unsqueeze(0)

        if target.ndim == 3:
            target = target.squeeze(1)
        elif target.ndim == 1:
            target = target.unsqueeze(0)

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for idx, (fft_size, hop_size, win_length) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths, strict=False)
        ):
            raw_win = getattr(self, f"window_{idx}")
            if not isinstance(raw_win, torch.Tensor):
                raw_win = torch.hann_window(win_length, device=pred.device, dtype=pred.dtype)
            window = raw_win.to(device=pred.device, dtype=pred.dtype)

            pred_stft = torch.stft(
                pred,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
            target_stft = torch.stft(
                target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            pred_power = pred_stft.real**2 + pred_stft.imag**2
            target_power = target_stft.real**2 + target_stft.imag**2
            pred_mag = torch.sqrt(torch.clamp(pred_power, min=self.eps))
            target_mag = torch.sqrt(torch.clamp(target_power, min=self.eps))

            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + self.eps
            )
            log_mag_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))

            total_loss = total_loss + (self.w_sc * sc_loss + self.w_log_mag * log_mag_loss)

        return total_loss / len(self.fft_sizes)


class MultiBandSpectralLoss(nn.Module):
    """Composite spectral loss combining full-band time-domain MSE and multi-resolution STFT."""

    mse_weight: float
    mrstft_weight: float
    mrstft_loss: MultiResolutionSTFTLoss | None

    def __init__(
        self,
        mse_weight: float = NATIVE_DEFAULT_MSE_WEIGHT,
        mrstft_weight: float = NATIVE_DEFAULT_MRSTFT_WEIGHT,
        mrstft_fft_sizes: tuple[int, ...] = NATIVE_DEFAULT_MRSTFT_FFT_SIZES,
        mrstft_hop_sizes: tuple[int, ...] = NATIVE_DEFAULT_MRSTFT_HOP_SIZES,
        mrstft_win_lengths: tuple[int, ...] = NATIVE_DEFAULT_MRSTFT_WIN_LENGTHS,
        fs: float = 48000.0,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.mrstft_weight = float(mrstft_weight)
        self.mrstft_loss = (
            MultiResolutionSTFTLoss(
                fft_sizes=mrstft_fft_sizes,
                hop_sizes=mrstft_hop_sizes,
                win_lengths=mrstft_win_lengths,
            )
            if self.mrstft_weight > 0.0
            else None
        )

    @override
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, return_metrics: bool = True
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes composite loss and returns scalar loss and component breakdown."""
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

        if self.mrstft_loss is not None:
            mr_l = self.mrstft_loss(pred, target)
            loss = loss + self.mrstft_weight * mr_l
        else:
            mr_l = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if not return_metrics:
            return loss, {}

        metrics: dict[str, float] = {
            "mse": float(mse.detach().cpu().item()),
            "mrstft": float(mr_l.detach().cpu().item()),
            "composite_loss": float(loss.detach().cpu().item()),
        }
        return loss, metrics
