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


class PreEmphasisMSELoss(nn.Module):
    """Linear pre-emphasis MSE loss.

    Filters predicted and target waveforms with a first-order differentiator FIR (1 - alpha * z^-1),
    boosting high frequencies by up to +26 dB linearly, amplifying high-frequency gradients without
    any per-slice denominator division or small-signal blow-up.
    """

    kernel: torch.Tensor

    def __init__(self, coef: float = 0.90) -> None:
        super().__init__()
        self.coef = float(coef)
        kernel = torch.tensor([1.0, -self.coef], dtype=torch.float32).view(1, 1, 2)
        self.register_buffer("kernel", kernel)

    @override
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 1:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 2:
            target = target.unsqueeze(1)

        kernel = self.kernel.to(dtype=pred.dtype)
        pred_pad = F.pad(pred, (1, 0))
        target_pad = F.pad(target, (1, 0))
        pred_pre = F.conv1d(pred_pad, kernel)
        target_pre = F.conv1d(target_pad, kernel)

        return F.mse_loss(pred_pre, target_pre)


class NormalizedHighPassESRLoss(nn.Module):
    """Zero-phase linear-phase FIR high-pass ESR loss with safe batch energy normalization.

    Neutralizes physical 22,000x string harmonic energy roll-off above 3-7 kHz by computing
    high-pass error energy normalized by high-pass target energy anchored by min_energy.
    """

    kernel: torch.Tensor

    def __init__(
        self,
        cutoff_hz: float,
        num_taps: int = 31,
        fs: float = 48000.0,
        eps: float = 1e-8,
        min_energy: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.cutoff_hz = float(cutoff_hz)
        self.num_taps = int(num_taps)
        self.fs = float(fs)
        self.eps = float(eps)
        self.min_energy = float(min_energy)

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
        denom = torch.clamp(target_energy + self.eps, min=self.min_energy)

        return error_energy / denom


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss combining spectral convergence and log-magnitude distance across multiple FFT window scales.

    Standard audio ML loss (Yamamoto et al. 2019) providing uniform spectral gradient pressure across the audible bandwidth
    without any small-signal denominator instability.
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
    """Composite Multi-Band Loss combining MSE anchoring, linear pre-emphasis, MRSTFT, and safe high-pass ESR."""

    mse_weight: float
    pre_emph_weight: float
    pre_emph_coef: float
    mrstft_weight: float
    hp_mid_weight: float
    hp_high_weight: float
    pre_emph_loss: PreEmphasisMSELoss | None
    mrstft_loss: MultiResolutionSTFTLoss | None
    hp_mid_loss: NormalizedHighPassESRLoss | None
    hp_high_loss: NormalizedHighPassESRLoss | None

    def __init__(
        self,
        mse_weight: float = 1.0,
        pre_emph_weight: float = 0.50,
        pre_emph_coef: float = 0.90,
        mrstft_weight: float = 0.001,
        mrstft_fft_sizes: tuple[int, ...] = (1024, 2048, 512),
        mrstft_hop_sizes: tuple[int, ...] = (120, 240, 50),
        mrstft_win_lengths: tuple[int, ...] = (600, 1200, 240),
        hp_mid_weight: float = 0.00,
        hp_mid_cutoff: float = 3000.0,
        hp_high_weight: float = 0.00,
        hp_high_cutoff: float = 7000.0,
        hp_min_energy: float = 1.0e-2,
        fs: float = 48000.0,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.pre_emph_weight = float(pre_emph_weight)
        self.pre_emph_coef = float(pre_emph_coef)
        self.mrstft_weight = float(mrstft_weight)
        self.hp_mid_weight = float(hp_mid_weight)
        self.hp_high_weight = float(hp_high_weight)

        self.pre_emph_loss = (
            PreEmphasisMSELoss(coef=self.pre_emph_coef) if self.pre_emph_weight > 0.0 else None
        )
        self.mrstft_loss = (
            MultiResolutionSTFTLoss(
                fft_sizes=mrstft_fft_sizes,
                hop_sizes=mrstft_hop_sizes,
                win_lengths=mrstft_win_lengths,
            )
            if self.mrstft_weight > 0.0
            else None
        )
        self.hp_mid_loss = (
            NormalizedHighPassESRLoss(cutoff_hz=hp_mid_cutoff, fs=fs, min_energy=hp_min_energy)
            if self.hp_mid_weight > 0.0
            else None
        )
        self.hp_high_loss = (
            NormalizedHighPassESRLoss(cutoff_hz=hp_high_cutoff, fs=fs, min_energy=hp_min_energy)
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
            if self.pre_emph_loss is not None:
                loss = loss + self.pre_emph_weight * self.pre_emph_loss(pred, target)
            if self.mrstft_loss is not None:
                loss = loss + self.mrstft_weight * self.mrstft_loss(pred, target)
            if self.hp_mid_loss is not None:
                loss = loss + self.hp_mid_weight * self.hp_mid_loss(pred, target)
            if self.hp_high_loss is not None:
                loss = loss + self.hp_high_weight * self.hp_high_loss(pred, target)
            return loss, {}

        metrics: dict[str, float] = {
            "mse": float(mse.detach().cpu().item()),
        }

        if self.pre_emph_loss is not None:
            pre_l = self.pre_emph_loss(pred, target)
            loss = loss + self.pre_emph_weight * pre_l
            metrics["pre_emph_mse"] = float(pre_l.detach().cpu().item())

        if self.mrstft_loss is not None:
            mr_l = self.mrstft_loss(pred, target)
            loss = loss + self.mrstft_weight * mr_l
            metrics["mrstft"] = float(mr_l.detach().cpu().item())

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
