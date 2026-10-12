"""Stock Neural Amp Modeler (NAM) Architecture 2 Engine wrapper.

Leverages PyTorch Lightning and upstream neural-amp-modeler training routines.
"""

from allomorph.trainer.stock.engine import (
    configure_a2_architecture,
    train_voice_stock,
)

__all__ = [
    "configure_a2_architecture",
    "train_voice_stock",
]
