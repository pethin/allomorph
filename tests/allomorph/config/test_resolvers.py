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
    from tests.allomorph.conftest import make_generic_instrument_config

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
    import pytest

    # 1. Passive Alnico V Split-P: alpha=0.26, vsat=1.00
    inst_p = load_instrument("34in_standard_p")
    ea_p = resolve_voicing_electroacoustics(inst_p, inst_p.voicings["vintage_open"])
    assert len(ea_p.coils) == 2
    assert ea_p.fr is None or ea_p.fr > 0.0
    assert ea_p.magnet_type == "alnico_v"
    assert ea_p.alpha == pytest.approx(0.26)
    assert ea_p.vsat == pytest.approx(1.00)

    # 2. Active EMG Classic 9V vs X-Series 18V op-amp bounding: min(mag_vsat, opamp_vsat)
    inst_emg = load_instrument("34in_active_emg")
    ea_classic = resolve_voicing_electroacoustics(inst_emg, inst_emg.voicings["bridge_solo"])
    assert ea_classic.magnet_type == "ceramic"
    assert ea_classic.alpha == pytest.approx(0.12)
    assert ea_classic.vsat == pytest.approx(0.50)

    inst_mmtw = load_instrument("30in_emg_mmtw")
    ea_x = resolve_voicing_electroacoustics(inst_mmtw, inst_mmtw.voicings["dual_mode"])
    assert ea_x.magnet_type == "ceramic_steel"
    assert ea_x.alpha == pytest.approx(0.15)
    assert ea_x.vsat == pytest.approx(1.28)  # min(1.28, 2.40) == 1.28

    # 3. Multi-pickup heterogeneous blend (Alnico V Split-P + Ceramic MM): transmission-weighted mean
    inst_pmm = load_instrument("34in_active_pmm").model_copy(deep=True)
    object.__setattr__(inst_pmm.pickups["mm"], "magnet_type", "ceramic")
    ea_pmm = resolve_voicing_electroacoustics(inst_pmm, inst_pmm.voicings["pmm_parallel"])
    assert 0.12 < ea_pmm.alpha < 0.26
    assert 1.00 < ea_pmm.vsat < 1.35

    # 4. Non-magnetic piezo bridge force sensor: magnet_type is None, vsat=2.00, alpha=0.08
    inst_up = load_instrument("41in_upright_bass")
    ea_up = resolve_voicing_electroacoustics(inst_up, inst_up.voicings["bridge_piezo"])
    assert ea_up.magnet_type is None
    assert ea_up.vsat == pytest.approx(2.00)
    assert ea_up.alpha == pytest.approx(0.08)

    # 5. Fail-fast KeyError on unrecognized magnet_type
    inst_bad = inst_p.model_copy(deep=True)
    object.__setattr__(inst_bad.pickups["split_p"], "magnet_type", "unobtainium")
    with pytest.raises(KeyError, match="Unrecognized magnet type"):
        resolve_voicing_electroacoustics(inst_bad, inst_bad.voicings["vintage_open"])
