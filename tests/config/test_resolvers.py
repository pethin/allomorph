"""Deterministic unit tests for config/resolvers.py decomposition primitives."""

from __future__ import annotations

import numpy as np

from allomorph.config.instruments import load_instrument
from allomorph.config.resolvers import (
    determine_pickup_resonance_and_q,
    find_voicing_by_slug_or_id,
    parse_target_voicing_token,
    resolve_voicing_electroacoustics,
)
from allomorph.config.schema import (
    VoicingConfig,
)
from allomorph.dsp.constants import FREQS


def test_determine_pickup_resonance_and_q_synthetic_rlc():
    """Verify sub-bin interpolation and analytical Q extraction on synthetic 2nd-order RLC."""
    f = np.logspace(np.log10(20), np.log10(20000), 1000)
    w = 2.0 * np.pi * f
    w0 = 2.0 * np.pi * 3500.0
    q_true = 1.8
    # Theoretical peak for 2nd-order lowpass: f_peak = f0 * sqrt(1 - 1/(2*Q^2)) = 3218.6 Hz
    h = w0**2 / (-(w**2) + 1j * w * w0 / q_true + w0**2)
    fr, q = determine_pickup_resonance_and_q(np.abs(h), f)
    assert fr is not None and 3210.0 <= fr <= 3225.0
    assert q is not None and 1.4 <= q <= 1.6


def test_determine_pickup_resonance_and_q_flat_response():
    """Verify flat response returns (None, None)."""
    f = np.asarray(FREQS, dtype=np.float64)
    h_flat = np.ones_like(f)
    fr, q = determine_pickup_resonance_and_q(h_flat, f)
    assert fr is None
    assert q is None


def test_determine_pickup_resonance_and_q_below_threshold():
    """Verify bumps below +0.20 dB threshold are rejected as flat."""
    f = np.asarray(FREQS, dtype=np.float64)
    h_small_bump = np.ones_like(f)
    mid_idx = len(f) // 2
    # Add +0.10 dB bump (approx 1.0116 lin)
    h_small_bump[mid_idx] = 1.0116
    fr, q = determine_pickup_resonance_and_q(h_small_bump, f)
    assert fr is None
    assert q is None


def test_determine_pickup_resonance_and_q_monotonic_cutoff():
    """Verify monotonic 1st-order lowpass (boundary peak at DC) returns (None, None)."""
    f = np.asarray(FREQS, dtype=np.float64)
    h_lp = 1.0 / (1.0 + (f / 1000.0) ** 2)
    fr, q = determine_pickup_resonance_and_q(h_lp, f)
    assert fr is None
    assert q is None


def test_determine_pickup_resonance_and_q_none_and_empty():
    """Verify None and empty inputs return (None, None) gracefully."""
    assert determine_pickup_resonance_and_q(None) == (None, None)
    assert determine_pickup_resonance_and_q([]) == (None, None)
    assert determine_pickup_resonance_and_q([1.0, 2.0]) == (None, None)


def test_parse_target_voicing_token():
    # Colon syntax
    inst_id, vid = parse_target_voicing_token(
        "34in_active_stingray:stingray_open", default_inst_id="my_inst"
    )
    assert inst_id == "34in_active_stingray"
    assert vid == "stingray_open"

    # Fallback to default
    inst_id2, vid2 = parse_target_voicing_token("custom_tone", default_inst_id="my_inst")
    assert inst_id2 == "my_inst"
    assert vid2 == "custom_tone"


def test_find_voicing_by_slug_or_id():
    from tests.conftest import make_generic_instrument_config

    inst = make_generic_instrument_config()
    v1 = VoicingConfig(
        id="v1", name="Vintage Split", tone_name="Precision Vintage", harness="passive"
    )
    v2 = VoicingConfig(
        id="v2", name="Bridge Growl", tone_name="Jazz Bridge Growl", harness="passive"
    )
    inst.voicings["v1"] = v1
    inst.voicings["v2"] = v2

    # Match strictly by exact voicing_id
    assert find_voicing_by_slug_or_id(inst, "v1") is v1
    assert find_voicing_by_slug_or_id(inst, "v2") is v2

    # Tone name slugs or non-existent IDs return None
    assert find_voicing_by_slug_or_id(inst, "precision_vintage") is None
    assert find_voicing_by_slug_or_id(inst, "non_existent") is None


def test_resolve_voicing_electroacoustics():
    inst = load_instrument("34in_standard_p")
    ea = resolve_voicing_electroacoustics(inst, inst.voicings["vintage_open"])
    assert len(ea.coils) == 2
    assert ea.fr is None or ea.fr > 0.0
    assert ea.magnet_type == "alnico_v"
