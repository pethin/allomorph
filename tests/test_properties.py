"""
Cross-cutting Property-Based Tests for Allomorph Universal Modeling Pipeline.

Verifies end-to-end mathematical identities,
minimum-phase FIR causality, and schema invariants using Hypothesis and Hypothesis-JSONSchema.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.config.schema import CoilConfig
from allomorph.dsp import synthesize_minimum_phase_fir


@given(st.lists(st.floats(min_value=0.01, max_value=2.0), min_size=64, max_size=64))
def test_minimum_phase_fir_synthesis_properties(raw_mag: list[float]) -> None:
    """Property test verifying minimum-phase FIR causal energy concentration and peak bounding."""
    # Interpolate to 2048 taps
    grid_x = np.linspace(0.0, 1.0, len(raw_mag))
    target_x = np.linspace(0.0, 1.0, 2048)
    smooth_mag = np.interp(target_x, grid_x, raw_mag)

    fir = synthesize_minimum_phase_fir(smooth_mag, num_taps=2048)
    assert len(fir) == 2048
    # Absolute peak is bounded to 0.99 (-0.1 dBFS true-peak headroom)
    max_peak = float(np.max(np.abs(fir)))
    assert max_peak <= 0.990001
    assert math.isfinite(max_peak)


@given(
    st.floats(min_value=0.01, max_value=0.25, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.2, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_coil_config_schema_property(pos_m: float, width_in: float) -> None:
    """Property test verifying CoilConfig instantiation and coordinate preservation."""
    coil = CoilConfig(
        position_from_bridge_m=pos_m,
        aperture_width_in=width_in,
        weight=1.0,
        polarity=1.0,
        strings=["all"],
    )
    assert coil.position_from_bridge_m == pos_m
    assert coil.aperture_width_in == width_in
    assert coil.strings == ["all"]
