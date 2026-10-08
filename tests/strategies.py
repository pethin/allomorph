"""
Reusable Hypothesis strategies and domain-specific generators for Allomorph.

Provides strictly-typed strategies conforming to pyrefly strict typing for:
- 1D Audio buffers and impulse responses (NumPy arrays)
- Audio-band frequencies and sorted frequency grids
- Potentiometer wiper rotations [0.0, 1.0]
- Fractional displacement excursion ratios ΔG (dB)
- Analog rail saturation ceilings vsat and signal levels
- Bassist lexicon tokens and Tone3000 model names
"""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst


def st_audio_buffers(
    min_len: int = 1,
    max_len: int = 256,
    min_val: float = -1.0,
    max_val: float = 1.0,
    dtype: type = np.float32,
) -> st.SearchStrategy[np.ndarray]:
    """Generate 1D NumPy floating-point audio buffers with finite values."""
    if dtype == np.float32:
        # Guarantee bounds are exactly representable as IEEE-754 single precision
        eff_min = float(np.float32(min_val))
        eff_max = float(np.float32(max_val))
        width = 32
    else:
        eff_min = float(min_val)
        eff_max = float(max_val)
        width = 64

    return npst.arrays(
        dtype=dtype,
        shape=st.integers(min_value=min_len, max_value=max_len),
        elements=st.floats(
            min_value=eff_min,
            max_value=eff_max,
            allow_nan=False,
            allow_infinity=False,
            width=width,
        ),
    )


def st_frequencies(
    min_f: float = 10.0,
    max_f: float = 24000.0,
) -> st.SearchStrategy[float]:
    """Generate realistic positive audio frequencies in Hz."""
    return st.floats(
        min_value=min_f,
        max_value=max_f,
        allow_nan=False,
        allow_infinity=False,
    )


def st_frequency_arrays(
    min_len: int = 2,
    max_len: int = 64,
    min_f: float = 20.0,
    max_f: float = 20000.0,
) -> st.SearchStrategy[np.ndarray]:
    """Generate sorted arrays of unique positive audio frequencies."""
    return (
        st.lists(
            st.floats(min_value=min_f, max_value=max_f, allow_nan=False, allow_infinity=False),
            min_size=min_len,
            max_size=max_len,
            unique=True,
        )
        .map(sorted)
        .map(lambda lst: np.array(lst, dtype=np.float64))
    )


def st_pot_wipers() -> st.SearchStrategy[float]:
    """Generate potentiometer rotation fractions alpha in [0.0, 1.0]."""
    return st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )


def st_displacement_ratios_db(
    min_db: float = -100.0,
    max_db: float = 100.0,
) -> st.SearchStrategy[float]:
    """Generate fractional displacement excursion ratios ΔG in dB."""
    return st.floats(
        min_value=min_db,
        max_value=max_db,
        allow_nan=False,
        allow_infinity=False,
    )


def st_rail_voltages(
    min_vsat: float = 0.1,
    max_vsat: float = 2.0,
) -> st.SearchStrategy[float]:
    """Generate analog rail saturation threshold voltages vsat > 0."""
    return st.floats(
        min_value=min_vsat,
        max_value=max_vsat,
        allow_nan=False,
        allow_infinity=False,
    )


def st_signals_for_rail(
    min_x: float = -10.0,
    max_x: float = 10.0,
) -> st.SearchStrategy[float]:
    """Generate input signal instantaneous amplitudes for rail limit testing."""
    return st.floats(
        min_value=min_x,
        max_value=max_x,
        allow_nan=False,
        allow_infinity=False,
    )


def st_naming_tokens() -> st.SearchStrategy[str]:
    """Generate strings for tone names and position tags, including path separators."""
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po", "Pd"),
            blacklist_characters=("\x00", "\n", "\r"),
        ),
        min_size=1,
        max_size=70,
    )
