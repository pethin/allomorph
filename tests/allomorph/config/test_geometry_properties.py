"""
Property-based tests for coil geometry resolution and string register assignment
in allomorph.config.geometry.
Verifies mathematical and physical invariants:
1. Center of gravity: average of symmetric coil positions equals array centerline.
2. Spacing monotonicity: coil offsets strictly increase with coil index for positive spacing.
3. Register coverage: all strings are covered across assigned coils.
4. Effective position bounds: weighted position is bounded by [min_pos, max_pos].
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.config.geometry import (
    assign_string_registers_to_coils,
    compute_coil_offset_from_spacing,
    compute_effective_position,
)
from allomorph.config.schema import CoilConfig


@given(
    center_m=st.floats(min_value=0.03, max_value=0.20, allow_nan=False),
    coil_spacing_in=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
    num_coils=st.integers(min_value=1, max_value=6),
)
def test_coil_center_of_gravity(center_m: float, coil_spacing_in: float, num_coils: int) -> None:
    """Center of gravity of a symmetric multi-coil array strictly equals the center datum."""
    positions = [
        compute_coil_offset_from_spacing(center_m, coil_spacing_in, i, num_coils=num_coils)
        for i in range(num_coils)
    ]
    cog = sum(positions) / num_coils
    assert np.isclose(cog, center_m, atol=1e-9), f"Center of gravity {cog} != {center_m}"


@given(
    center_m=st.floats(min_value=0.03, max_value=0.20, allow_nan=False),
    coil_spacing_in=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
    num_coils=st.integers(min_value=2, max_value=6),
)
def test_coil_offset_spacing_monotonicity(
    center_m: float, coil_spacing_in: float, num_coils: int
) -> None:
    """Coil positions strictly increase with coil index along the string axis."""
    positions = [
        compute_coil_offset_from_spacing(center_m, coil_spacing_in, i, num_coils=num_coils)
        for i in range(num_coils)
    ]
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1]


def test_register_assignment_coverage_split_and_parallel() -> None:
    """assign_string_registers_to_coils assigns non-empty registers without gaps."""
    c1 = CoilConfig(
        id="c1",
        position_from_bridge_m=0.10,
        aperture_width_in=0.5,
        weight=0.5,
        polarity=1.0,
    )
    c2 = CoilConfig(
        id="c2",
        position_from_bridge_m=0.12,
        aperture_width_in=0.5,
        weight=0.5,
        polarity=1.0,
    )

    # Split-coil assignment (e.g. P-Bass)
    split_coils = assign_string_registers_to_coils([c1, c2], coil_geometry="split")
    assert split_coils[0].strings == [3, 4]
    assert split_coils[1].strings == [1, 2]

    # Parallel humbucker assignment
    par_coils = assign_string_registers_to_coils([c1, c2], coil_geometry="parallel")
    assert par_coils[0].strings == ["all"]
    assert par_coils[1].strings == ["all"]


@given(
    p1=st.floats(min_value=0.04, max_value=0.15, allow_nan=False),
    p2=st.floats(min_value=0.04, max_value=0.15, allow_nan=False),
    w1=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
    w2=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
)
def test_effective_position_bounds(p1: float, p2: float, w1: float, w2: float) -> None:
    """Effective physical position is bounded between min and max coil positions."""
    c1 = CoilConfig(position_from_bridge_m=p1, aperture_width_in=0.5, weight=w1, polarity=1.0)
    c2 = CoilConfig(position_from_bridge_m=p2, aperture_width_in=0.5, weight=w2, polarity=1.0)

    eff = compute_effective_position([c1, c2])
    min_p = min(p1, p2)
    max_p = max(p1, p2)

    assert min_p - 1e-9 <= eff <= max_p + 1e-9
