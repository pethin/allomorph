"""First-principles spectral and time-domain loss functions for NAM Architecture 2 training.

Houses MultiResolutionSTFTLoss and composite MultiBandSpectralLoss.
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from torch import nn

from allomorph.trainer.constants import (
    NATIVE_DEFAULT_FIR_HP_FC,
    NATIVE_DEFAULT_FIR_HP_WEIGHT,
    NATIVE_DEFAULT_MSE_WEIGHT,
)


def design_sharp_fir_highpass(fc: float, fs: float, num_taps: int = 255) -> torch.Tensor:
    """Designs a sharp linear-phase FIR high-pass filter with >108 dB stopband rejection.

    Uses a Blackman-windowed sinc high-pass impulse response, normalized to peak passband
    magnitude of 1.00 via FFT. Returns a 1x1xT 3D tensor suitable for F.conv1d.
    """
    import numpy as np

    fc_norm = fc / fs
    n = np.arange(num_taps) - (num_taps - 1) / 2
    h_lp = 2 * fc_norm * np.sinc(2 * fc_norm * n)
    h_hp = -h_lp
    center = (num_taps - 1) // 2
    h_hp[center] += 1.0
    w = np.blackman(num_taps)
    h_hp = (h_hp * w).astype(np.float32)
    H_max = float(np.max(np.abs(np.fft.rfft(h_hp, n=4096))))
    h_hp = (h_hp / H_max).astype(np.float32)
    return torch.from_numpy(h_hp).unsqueeze(0).unsqueeze(0)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss combining spectral convergence and log-magnitude distance across multiple FFT window scales.

    Standard audio ML loss (Yamamoto et al. 2019) for offline spectral evaluation.
    Uses C-infinity power clamping (min=1e-8) for zero-gradient silence.
    """

    fft_sizes: tuple[int, ...]
    hop_sizes: tuple[int, ...]
    win_lengths: tuple[int, ...]
    w_sc: float
    w_log_mag: float
    eps: float

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (1024, 2048, 512),
        hop_sizes: tuple[int, ...] = (120, 240, 50),
        win_lengths: tuple[int, ...] = (600, 1200, 240),
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
    """Composite loss combining full-band time-domain MSE and 255-tap Blackman FIR high-pass MSE."""

    mse_weight: float
    fir_hp_weight: float
    fir_hp_fc: float
    num_taps: int
    hp_kernel: torch.Tensor

    def __init__(
        self,
        mse_weight: float = NATIVE_DEFAULT_MSE_WEIGHT,
        fir_hp_weight: float = NATIVE_DEFAULT_FIR_HP_WEIGHT,
        fir_hp_fc: float = NATIVE_DEFAULT_FIR_HP_FC,
        num_taps: int = 255,
        fs: float = 48000.0,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.fir_hp_weight = float(fir_hp_weight)
        self.fir_hp_fc = float(fir_hp_fc)
        self.num_taps = int(num_taps)
        self.register_buffer(
            "hp_kernel",
            design_sharp_fir_highpass(fc=self.fir_hp_fc, fs=float(fs), num_taps=self.num_taps),
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

        if self.fir_hp_weight > 0.0 and pred.shape[-1] >= self.num_taps:
            kernel = self.hp_kernel.to(device=pred.device, dtype=pred.dtype)
            pred_hp = F.conv1d(pred, kernel, padding="valid")
            target_hp = F.conv1d(target, kernel, padding="valid")
            fir_hp = F.mse_loss(pred_hp, target_hp)
            loss = loss + self.fir_hp_weight * fir_hp
        else:
            fir_hp = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if not return_metrics:
            return loss, {}

        metrics: dict[str, float] = {
            "mse": float(mse.detach().cpu().item()),
            "fir_hp": float(fir_hp.detach().cpu().item()),
            "composite_loss": float(loss.detach().cpu().item()),
        }
        return loss, metrics


__all__ = [
    "MultiBandSpectralLoss",
    "MultiResolutionSTFTLoss",
    "design_sharp_fir_highpass",
]
