"""Native Architecture 2 (A2) NAM Engine.

Custom PyTorch neural training pipeline featuring:
- Zero-copy in-VRAM tensor slicing.
- Block-diagonal masked dual-tier slimmable WaveNet architecture (A2 Lite + Full).
- Multi-resolution STFT and time-domain MSE composite spectral loss.
- Anti-drift Pareto quality tracking and in-memory checkpoint restoration.
- Native export to compliant Darkglass Anagram Block 1 .nam JSON.
"""

from allomorph.trainer.native.engine import train_voice_native
from allomorph.trainer.native.loss import (
    MultiBandSpectralLoss,
    MultiResolutionSTFTLoss,
)
from allomorph.trainer.native.models import (
    CHANNELS_FULL,
    CHANNELS_LITE,
    CHANNELS_TOTAL,
    AllomorphWaveNetA2,
    AllomorphWaveNetLayer,
)

__all__ = [
    "CHANNELS_FULL",
    "CHANNELS_LITE",
    "CHANNELS_TOTAL",
    "AllomorphWaveNetA2",
    "AllomorphWaveNetLayer",
    "MultiBandSpectralLoss",
    "MultiResolutionSTFTLoss",
    "train_voice_native",
]
