"""
Tests verifying that differential prefilter FIRs and difference curves
remain finite, stable, bounded, and well-behaved.
"""

import numpy as np

from allomorph.config import (
    VOICES,
    load_instrument,
)
from allomorph.physics.prefilter import compute_voice_prefilter_firs
from allomorph.visualizer.dataframe import build_voice_dataframe


def test_differential_prefilter_firs_stability():
    """Validates that differential prefilter FIRs remain stable, finite, and well-behaved."""
    test_cases = [
        ("34in_preamp_soapbar", "neck", "precision_vintage"),
        ("34in_preamp_soapbar", "bridge", "jazz_bridge_open"),
        ("34in_preamp_soapbar", "pair_parallel", "jazz_pair_open"),
        ("34in_standard_p", "split_p", "precision_vintage"),
        ("34in_active_emg", "bridge", "stingray_parallel"),
    ]

    for inst_id, pkey, vid in test_cases:
        inst = load_instrument(inst_id)
        firs = compute_voice_prefilter_firs(
            vid, instrument=inst, num_taps=2048, src_pickup_key=pkey
        )
        assert len(firs) > 0
        for fir in firs:
            arr = np.asarray(fir, dtype=np.float64)
            assert len(arr) == 2048
            assert not np.any(np.isnan(arr))
            assert not np.any(np.isinf(arr))
            # Must have non-zero energy
            assert np.max(np.abs(arr)) > 1e-4


def test_difference_dataframe_invariance():
    """Validates that direct difference visualizer curves remain finite and bounded."""
    inst = load_instrument("34in_preamp_soapbar")
    df_neck = build_voice_dataframe(
        "precision_vintage",
        VOICES["precision_vintage"],
        instrument=inst,
        mode="difference",
        src_pickup_key="neck",
    )
    mags = df_neck["magnitude_db"].to_numpy()
    assert len(mags) == 600
    assert not np.any(np.isnan(mags))
    assert np.max(mags) < 25.0
    assert np.min(mags) > -100.0
