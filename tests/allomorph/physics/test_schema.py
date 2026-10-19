"""
Unit tests for physics schemas (wave-speed continuum points and string mechanics).
"""

import pytest
from pydantic import ValidationError

from allomorph.physics.schema import WaveSpeedContinuumPoint


def test_wave_speed_continuum_point_validation():
    """Verify WaveSpeedContinuumPoint field constraints, register literals, and attribute access."""
    pt = WaveSpeedContinuumPoint(
        f0=40.0,
        v0=75.0,
        scale_m=0.8636,
        register="lower",
        weight=1.0 / 24,
    )
    assert pt.f0 == 40.0
    assert pt.v0 == 75.0
    assert pt.scale_m == 0.8636
    assert pt.register == "lower"
    with pytest.raises(TypeError):
        _ = pt["f0"]  # type: ignore[index]

    # Rejection of invalid register
    with pytest.raises(ValidationError):
        WaveSpeedContinuumPoint(
            f0=40.0,
            v0=75.0,
            scale_m=0.8636,
            register="middle",  # type: ignore[arg-type]
        )

    # Rejection of non-positive frequency
    with pytest.raises(ValidationError):
        WaveSpeedContinuumPoint(
            f0=0.0,
            v0=75.0,
            scale_m=0.8636,
            register="lower",
        )
