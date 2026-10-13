"""Native Architecture 2 (A2) WaveNet implementation for Allomorph.

Implements the official slimmable WaveNet architecture (channels_3 Lite + channels_8 Full)
and native export to Darkglass Anagram Block 1 and open-source Neural Amp Modeler (.nam)
JSON format without third-party framework dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, override

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from allomorph.trainer.constants import NAM_MODEL_VERSION

# Architecture 2 Standards
KERNEL_SIZES: list[int] = [
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    15,
    15,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
]
DILATIONS: list[int] = [
    1,
    3,
    7,
    17,
    41,
    101,
    239,
    1,
    3,
    7,
    17,
    41,
    101,
    239,
    1,
    13,
    1,
    3,
    7,
    17,
    41,
    101,
    239,
]
HEAD_KERNEL_SIZE: int = 16
HEAD_SCALE: float = 0.01

CHANNELS_LITE: int = 3
CHANNELS_FULL: int = 8
CHANNELS_TOTAL: int = CHANNELS_LITE + CHANNELS_FULL  # 11 channels


class AllomorphWaveNetLayer(nn.Module):
    """Single dilated residual WaveNet layer with block-diagonal channel masking."""

    conv: nn.Conv1d
    input_mixer: nn.Conv1d
    layer1x1: nn.Conv1d
    mask_conv: torch.Tensor
    mask_1x1: torch.Tensor

    def __init__(self, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(
            CHANNELS_TOTAL, CHANNELS_TOTAL, kernel_size, dilation=dilation, bias=True
        )
        self.input_mixer = nn.Conv1d(1, CHANNELS_TOTAL, 1, bias=False)
        self.layer1x1 = nn.Conv1d(CHANNELS_TOTAL, CHANNELS_TOTAL, 1, bias=True)

        # Block-diagonal masks isolating Lite (0:3) and Full (3:11) channels
        mask_conv = torch.zeros_like(self.conv.weight)
        mask_conv[:CHANNELS_LITE, :CHANNELS_LITE, :] = 1.0
        mask_conv[CHANNELS_LITE:, CHANNELS_LITE:, :] = 1.0
        self.register_buffer("mask_conv", mask_conv)

        mask_1x1 = torch.zeros_like(self.layer1x1.weight)
        mask_1x1[:CHANNELS_LITE, :CHANNELS_LITE, :] = 1.0
        mask_1x1[CHANNELS_LITE:, CHANNELS_LITE:, :] = 1.0
        self.register_buffer("mask_1x1", mask_1x1)

        with torch.no_grad():
            self.conv.weight.mul_(self.mask_conv)
            self.layer1x1.weight.mul_(self.mask_1x1)

    @override
    def forward(
        self, x: torch.Tensor, x_cond: torch.Tensor, target_head_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward layer computation.

        Args:
            x: Current residual tensor of shape (B, 11, L_in).
            x_cond: Input conditioning tensor of shape (B, 1, L_cond).
            target_head_len: Truncated sequence length for head accumulation.

        Returns:
            Tuple of (residual_output, head_contribution).
        """
        zconv = self.conv(x)

        l_conv = zconv.shape[2]
        cond_slice = x_cond[:, :, -l_conv:]
        zmix = self.input_mixer(cond_slice)

        z = F.leaky_relu(zconv + zmix, negative_slope=0.01)

        z1x1 = self.layer1x1(z)

        res = x[:, :, -l_conv:] + z1x1
        head_term = z[:, :, -target_head_len:]
        return res, head_term


class AllomorphWaveNetA2(nn.Module):
    """Full slimmable Architecture 2 WaveNet model.

    Trains both channels_3 (A2 Lite) and channels_8 (A2 Full) concurrently in a single
    block-masked forward pass and exports bit-exact .nam JSON models.
    """

    rechannel: nn.Conv1d
    layers: nn.ModuleList
    head_rechannel: nn.Conv1d
    mask_head: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.receptive_field_layers = 1 + sum(
            (k - 1) * d for k, d in zip(KERNEL_SIZES, DILATIONS, strict=True)
        )
        self.receptive_field = self.receptive_field_layers + (HEAD_KERNEL_SIZE - 1)
        self.head_scale = HEAD_SCALE

        self.rechannel = nn.Conv1d(1, CHANNELS_TOTAL, 1, bias=False)
        self.layers = nn.ModuleList(
            [AllomorphWaveNetLayer(k, d) for k, d in zip(KERNEL_SIZES, DILATIONS, strict=True)]
        )
        self.head_rechannel = nn.Conv1d(CHANNELS_TOTAL, 2, HEAD_KERNEL_SIZE, bias=True)

        # Output channel 0 routes to Lite (0:3), channel 1 routes to Full (3:11)
        mask_head = torch.zeros_like(self.head_rechannel.weight)
        mask_head[0, :CHANNELS_LITE, :] = 1.0
        mask_head[1, CHANNELS_LITE:, :] = 1.0
        self.register_buffer("mask_head", mask_head)

        with torch.no_grad():
            self.head_rechannel.weight.mul_(self.mask_head)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass across both slimmable submodels.

        Args:
            x: Input waveform tensor of shape (B, 1, L) or (B, L).

        Returns:
            Output tensor of shape (B, 2, L_out) where channel 0 is A2 Lite (channels_3)
            and channel 1 is A2 Full (channels_8). L_out = L - (receptive_field - 1).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        l_in = x.shape[2]
        target_head_len = l_in - (self.receptive_field_layers - 1)

        x_cond = x
        x_curr = self.rechannel(x)

        head_sum = torch.zeros(
            (x.shape[0], CHANNELS_TOTAL, target_head_len),
            device=x.device,
            dtype=x.dtype,
        )

        for layer in self.layers:
            layer_mod = cast(AllomorphWaveNetLayer, layer)
            x_curr, head_term = layer_mod(x_curr, x_cond, target_head_len)
            head_sum = head_sum + head_term

        out = self.head_rechannel(head_sum)
        return out * self.head_scale

    _cached_mask_weights: list[torch.Tensor] | None = None
    _cached_mask_masks: list[torch.Tensor] | None = None

    def _get_mask_tensors(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if (
            self._cached_mask_weights is None
            or len(self._cached_mask_weights) == 0
            or self._cached_mask_weights[0].device != self.head_rechannel.weight.device
        ):
            w_list: list[torch.Tensor] = []
            m_list: list[torch.Tensor] = []
            for layer in self.layers:
                layer_mod = cast(AllomorphWaveNetLayer, layer)
                w_list.extend([layer_mod.conv.weight, layer_mod.layer1x1.weight])
                m_list.extend([layer_mod.mask_conv, layer_mod.mask_1x1])
            w_list.append(self.head_rechannel.weight)
            m_list.append(self.mask_head)
            self._cached_mask_weights = w_list
            self._cached_mask_masks = m_list
        assert self._cached_mask_weights is not None
        assert self._cached_mask_masks is not None
        return self._cached_mask_weights, self._cached_mask_masks

    def apply_weight_masks(self) -> None:
        """Enforces zeroing of off-block cross-chassis weights and gradients."""
        weights, masks = self._get_mask_tensors()
        with torch.no_grad():
            torch._foreach_mul_(weights, masks)
            grads = [w.grad for w in weights if w.grad is not None]
            if grads:
                grad_masks = [m for w, m in zip(weights, masks, strict=True) if w.grad is not None]
                torch._foreach_mul_(grads, grad_masks)

    def export_submodel_weights(self, submodel_idx: int) -> list[float]:
        """Flattens sliced submodel weights into the 1D array expected by NAM C++ core.

        Args:
            submodel_idx: 0 for channels_3 Lite (1,871 weights), 1 for channels_8 Full (12,146 weights).

        Returns:
            List of floats representing the flattened weight sequence.
        """
        self.apply_weight_masks()
        weights: list[float] = []
        if submodel_idx == 0:
            c_slice = slice(0, CHANNELS_LITE)
            h_idx = 0
        else:
            c_slice = slice(CHANNELS_LITE, CHANNELS_TOTAL)
            h_idx = 1

        # 1. Rechannel
        weights.extend(self.rechannel.weight[c_slice, :, :].detach().cpu().flatten().tolist())

        # 2. Layers
        for layer in self.layers:
            layer_mod = cast(AllomorphWaveNetLayer, layer)
            conv_w = layer_mod.conv.weight
            weights.extend(conv_w[c_slice, c_slice, :].detach().cpu().flatten().tolist())
            if layer_mod.conv.bias is not None:
                conv_b = cast(torch.Tensor, layer_mod.conv.bias)
                weights.extend(conv_b[c_slice].detach().cpu().flatten().tolist())

            in_w = layer_mod.input_mixer.weight
            weights.extend(in_w[c_slice, :, :].detach().cpu().flatten().tolist())
            if layer_mod.input_mixer.bias is not None:
                in_b = cast(torch.Tensor, layer_mod.input_mixer.bias)
                weights.extend(in_b[c_slice].detach().cpu().flatten().tolist())

            l1x1_w = layer_mod.layer1x1.weight
            weights.extend(l1x1_w[c_slice, c_slice, :].detach().cpu().flatten().tolist())
            if layer_mod.layer1x1.bias is not None:
                l1x1_b = cast(torch.Tensor, layer_mod.layer1x1.bias)
                weights.extend(l1x1_b[c_slice].detach().cpu().flatten().tolist())

        # 3. Head rechannel
        hr_w = self.head_rechannel.weight
        weights.extend(hr_w[h_idx : h_idx + 1, c_slice, :].detach().cpu().flatten().tolist())
        if self.head_rechannel.bias is not None:
            hr_b = cast(torch.Tensor, self.head_rechannel.bias)
            weights.extend(hr_b[h_idx : h_idx + 1].detach().cpu().flatten().tolist())

        # 4. Head scale
        weights.append(float(self.head_scale))
        return weights

    def export_submodel_config(self, submodel_idx: int) -> dict[str, Any]:
        """Builds standard NAM C++ WaveNet layer config dictionary for a submodel."""
        channels = CHANNELS_LITE if submodel_idx == 0 else CHANNELS_FULL
        layer_config = {
            "input_size": 1,
            "condition_size": 1,
            "head": {
                "out_channels": 1,
                "kernel_size": HEAD_KERNEL_SIZE,
                "bias": True,
            },
            "channels": channels,
            "kernel_sizes": KERNEL_SIZES,
            "dilations": DILATIONS,
            "activation": [{"type": "LeakyReLU", "negative_slope": 0.01} for _ in KERNEL_SIZES],
            "bottleneck": channels,
            "head1x1": {"active": False, "out_channels": 1, "groups": 1},
            "layer1x1": {"active": True, "groups": 1},
            "groups_input": 1,
            "groups_input_mixin": 1,
            "conv_pre_film": {"active": False, "shift": True, "groups": 1},
            "conv_post_film": {"active": False, "shift": True, "groups": 1},
            "input_mixin_pre_film": {"active": False, "shift": True, "groups": 1},
            "input_mixin_post_film": {"active": False, "shift": True, "groups": 1},
            "activation_pre_film": {"active": False, "shift": True, "groups": 1},
            "activation_post_film": {"active": False, "shift": True, "groups": 1},
            "layer1x1_post_film": {"active": False, "shift": True, "groups": 1},
            "head1x1_post_film": {"active": False, "shift": True, "groups": 1},
            "gating_mode": ["none" for _ in KERNEL_SIZES],
            "secondary_activation": [None for _ in KERNEL_SIZES],
            "slimmable": None,
        }
        return {
            "layers": [layer_config],
            "head": None,
            "head_scale": float(self.head_scale),
        }

    def compute_metadata_loudness_and_gain(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calculates standard NAM nominal loudness (dB) and compression gain [0, 1] for Lite and Full tiers.

        Uses upstream NAM standard calibration input if available, returning default
        nominal values (-18.0 dB, 0.50 gain) on failure.
        """
        default_res = ((-18.0, 0.5), (-18.0, 0.5))
        try:
            from pedalboard.io import AudioFile

            spec = importlib.util.find_spec("nam")
            if not spec or not spec.origin:
                return default_res
            res_path = Path(spec.origin).parent / "models" / "_resources" / "loudness_input.wav"
            if not res_path.exists():
                return default_res

            with AudioFile(str(res_path)) as f:
                x_np = f.read(f.frames)

            dev = next(self.parameters()).device
            x = torch.from_numpy(x_np[0:1]).to(dev).float()
            was_training = self.training
            self.eval()
            with torch.no_grad():
                y = self(x)
                results: list[tuple[float, float]] = []
                x_gains = np.linspace(0.0, 1.0, 11)

                for ch in (0, 1):
                    y_ch = y[:, ch, :]
                    rms = torch.sqrt(torch.mean(torch.square(y_ch))).item()
                    loudness = 20.0 * math.log10(max(rms, 1e-12))

                    y_rms_list: list[float] = []
                    for g in x_gains:
                        yg = self(float(g) * x)
                        y_rms_list.append(torch.sqrt(torch.mean(torch.square(yg[:, ch, :]))).item())
                    y_rms = np.array(y_rms_list)
                    max_gain = float(y_rms[-1] * len(y_rms))
                    min_gain = 0.5 * max_gain
                    gain_range = max_gain - min_gain
                    gain = (
                        float(np.clip((float(y_rms.sum()) - min_gain) / gain_range, 0.0, 1.0))
                        if abs(gain_range) > 1e-12
                        else 0.5
                    )
                    results.append((round(loudness, 4), round(gain, 4)))

            if was_training:
                self.train()
            return results[0], results[1]
        except ImportError, RuntimeError, OSError, ValueError, KeyError:
            return default_res

    def export_nam(
        self,
        out_path: str | Path,
        sample_rate: float = 48000.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Exports the dual-tier model directly to a Tone3000-compliant SlimmableContainer .nam JSON file.

        Conforms strictly to Neural Amp Modeler Architecture 2 container specifications (v0.7.0)
        and Tone3000 storefront validation criteria.

        Args:
            out_path: Output file path (.nam).
            sample_rate: Audio clock rate (default: 48000.0).
            metadata: User and training metadata dictionary.

        Returns:
            Complete container dictionary as written to disk.
        """
        t = datetime.now(tz=UTC)
        date_dict = {
            "year": t.year,
            "month": t.month,
            "day": t.day,
            "hour": t.hour,
            "minute": t.minute,
            "second": t.second,
        }

        (loudness_lite, gain_lite), (loudness_full, gain_full) = (
            self.compute_metadata_loudness_and_gain()
        )

        sub0 = {
            "version": NAM_MODEL_VERSION,
            "metadata": {
                "date": date_dict,
                "loudness": loudness_lite,
                "gain": gain_lite,
            },
            "architecture": "WaveNet",
            "config": self.export_submodel_config(0),
            "weights": self.export_submodel_weights(0),
            "sample_rate": int(sample_rate),
        }

        sub1 = {
            "version": NAM_MODEL_VERSION,
            "metadata": {
                "date": date_dict,
                "loudness": loudness_full,
                "gain": gain_full,
            },
            "architecture": "WaveNet",
            "config": self.export_submodel_config(1),
            "weights": self.export_submodel_weights(1),
            "sample_rate": int(sample_rate),
        }

        out_stem = Path(out_path).stem
        meta_in = metadata or {}
        top_meta: dict[str, Any] = {
            "date": date_dict,
            "loudness": loudness_full,
            "gain": gain_full,
            "name": meta_in.get("name", out_stem),
            "modeled_by": meta_in.get("modeled_by", "bassedpotato"),
            "gear_type": meta_in.get("gear_type", "other"),
            "gear_make": meta_in.get("gear_make", "Fender Precision Bass"),
            "gear_model": meta_in.get("gear_model", "Fender Precision Bass"),
            "tone_type": meta_in.get("tone_type", "clean"),
            "trainer": "TONE3000",
        }

        empty_weights: list[float] = []
        container: dict[str, Any] = {
            "version": NAM_MODEL_VERSION,
            "metadata": top_meta,
            "architecture": "SlimmableContainer",
            "config": {
                "submodels": [
                    {"max_value": 0.5, "model": sub0},
                    {"max_value": 1, "model": sub1},
                ]
            },
            "weights": empty_weights,
            "sample_rate": int(sample_rate),
        }

        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(container, f, separators=(",", ":"))
        return container
