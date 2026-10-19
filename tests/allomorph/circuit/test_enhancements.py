"""
Tests for post-LTspice architectural enhancements:
1. Audio / logarithmic potentiometer tapers (10% CTS / 15% Bourns).
2. Continuous MN dual-pickup blend / pan potentiometer modeling.
3. Analytical circuit metric extraction (f_res, loaded Q, bandwidth, insertion loss, HF slope).
"""

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from allomorph.config.schema import InstrumentConfig

from allomorph.circuit import (
    compute_parametric_sweep,
    eval_pot_taper,
    solve_mna_harness,
)
from allomorph.dsp import FREQS


def test_eval_pot_taper_boundaries_and_monotonicity():
    """Verify pot taper boundary conditions, smoothness, and strict monotonicity."""
    # Boundary conditions
    for taper in ["audio", "audio10", "audio15", "linear"]:
        assert eval_pot_taper(0.0, taper) == pytest.approx(0.0, abs=1e-7)
        assert eval_pot_taper(1.0, taper) == pytest.approx(1.0, abs=1e-7)

    # 50% rotation values
    assert eval_pot_taper(0.5, "audio") == pytest.approx(0.10, abs=1e-4)
    assert eval_pot_taper(0.5, "audio10") == pytest.approx(0.10, abs=1e-4)
    assert eval_pot_taper(0.5, "audio15") == pytest.approx(0.15, abs=1e-4)
    assert eval_pot_taper(0.5, "linear") == pytest.approx(0.50, abs=1e-4)

    # Strict monotonicity and positive slope
    theta_vals = np.linspace(0.0, 1.0, 101)
    for taper in ["audio", "audio15", "linear"]:
        res_vals = [eval_pot_taper(th, taper) for th in theta_vals]
        diffs = np.diff(res_vals)
        assert np.all(diffs > 0.0), f"Taper '{taper}' is not strictly monotonic"

    with pytest.raises(ValueError, match="Unknown pot taper"):
        eval_pot_taper(0.5, "unknown_taper")


def test_mn_blend_potentiometer_behavior(generic_dual_pickup_instrument: InstrumentConfig):
    """Verify dual-pickup blend: balanced at center detent, attenuation away from center."""
    inst = generic_dual_pickup_instrument
    voicing = inst.voicings["blend_controls"]
    harness = inst.harnesses[voicing.harness]

    # Center detent (0.5): both pickups active and balanced within 1 dB
    v_center = voicing.model_copy(deep=True)
    v_center.controls["neck_vol"] = 1.0
    v_center.controls["bridge_vol"] = 1.0
    tr_center = solve_mna_harness(inst, harness, v_center, freqs=FREQS)
    assert "neck" in tr_center and "bridge" in tr_center
    mag_n_center = 20.0 * math.log10(max(float(tr_center["neck"][100]), 1e-6))
    mag_b_center = 20.0 * math.log10(max(float(tr_center["bridge"][100]), 1e-6))
    assert abs(mag_n_center - mag_b_center) < 1.0

    # Full Neck (0.0): Bridge must be heavily attenuated
    v_neck = voicing.model_copy(deep=True)
    v_neck.controls["neck_vol"] = 1.0
    v_neck.controls["bridge_vol"] = 0.0
    tr_neck = solve_mna_harness(inst, harness, v_neck, freqs=FREQS)
    assert tr_neck["bridge"][100] < 0.05  # Bridge attenuated
    assert tr_neck["neck"][100] > 0.50  # Neck active

    # Full Bridge (1.0): Neck must be heavily attenuated
    v_bridge = voicing.model_copy(deep=True)
    v_bridge.controls["neck_vol"] = 0.0
    v_bridge.controls["bridge_vol"] = 1.0
    tr_bridge = solve_mna_harness(inst, harness, v_bridge, freqs=FREQS)
    assert tr_bridge["neck"][100] < 0.05  # Neck attenuated
    assert tr_bridge["bridge"][100] > 0.50  # Bridge active


def test_compute_parametric_sweep_blend(generic_dual_pickup_instrument: InstrumentConfig):
    """Verify continuous blend parametric sweep executes and produces valid results."""
    inst = generic_dual_pickup_instrument
    voicing = inst.voicings["blend_controls"]
    res = compute_parametric_sweep((inst, voicing), param="blend")
    assert res.param == "blend"
    assert len(res.values) == 5
    assert len(res.curves) == 5
    assert "Center (100%/100%)" in res.labels
    assert "Neck 100%" in res.labels
    assert "Bridge 100%" in res.labels

    # Verify Polars DataFrame export
    df = res.to_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert "frequency" in df.columns
    assert "magnitude_db" in df.columns
    assert "label" in df.columns
    assert len(df) == len(res.freqs) * 5


def test_analytical_circuit_metrics_extraction(generic_instrument_config: InstrumentConfig):
    """Verify ParametricSweepResult.metrics() extracts accurate resonant peak, Q, bandwidth, and slope."""
    inst = generic_instrument_config
    voicing = inst.voicings["generic_voice"]
    res = compute_parametric_sweep((inst, voicing), param="tone")
    df_metrics = res.metrics()

    assert isinstance(df_metrics, pl.DataFrame)
    assert "f_res_hz" in df_metrics.columns
    assert "peak_db" in df_metrics.columns
    assert "insertion_loss_db" in df_metrics.columns
    assert "q_loaded" in df_metrics.columns
    assert "bandwidth_hz" in df_metrics.columns
    assert "cutoff_3db_hz" in df_metrics.columns
    assert "hf_slope_db_oct" in df_metrics.columns

    # Full open tone (Tone 100%)
    recs = res.metrics_records()
    rec_100 = recs[-1]
    assert rec_100.label == "Tone 100%"
    assert rec_100.f_res_hz is not None
    assert rec_100.q_loaded is not None
    assert rec_100.hf_slope_db_oct is not None

    # Summary table formatting
    table_str = res.summary_table()
    assert "f_res (Hz)" in table_str
    assert "Q loaded" in table_str
    assert "Tone 100%" in table_str
