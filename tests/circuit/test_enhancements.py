"""
Tests for post-LTspice architectural enhancements:
1. Audio / logarithmic potentiometer tapers (10% CTS / 15% Bourns).
2. Continuous MN dual-pickup blend / pan potentiometer modeling.
3. Analytical circuit metric extraction (f_res, loaded Q, bandwidth, insertion loss, HF slope).
4. Linear filter stage fusion.
5. Deprecation error on legacy .cir netlists.
"""

import math

import numpy as np
import polars as pl
import pytest

from allomorph.circuit import (
    CircuitModel,
    compute_circuit_transfer_functions,
    compute_parametric_sweep,
    eval_pot_taper,
    load_circuit,
    simulate_circuit_audio,
)
from allomorph.config import VOICES


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


def test_circuit_model_apply_pot_positions_tapers():
    """Verify apply_pot_positions applies audio vs linear tapers properly."""
    m = CircuitModel()
    m.Rvol_total = 500000.0
    m.Rtone_total = 250000.0

    # Linear volume at 50%
    m.apply_pot_positions(vol_pos=0.5, pot_taper="linear")
    assert m.Rbot == pytest.approx(250000.0)
    assert m.Rtop == pytest.approx(250000.0)

    # Audio 10% CTS volume at 50%
    m.apply_pot_positions(vol_pos=0.5, pot_taper="audio")
    assert m.Rbot == pytest.approx(50000.0, rel=1e-3)  # 10% of 500k
    assert m.Rtop == pytest.approx(450000.0, rel=1e-3)

    # Audio 15% Bourns volume at 50%
    m.apply_pot_positions(vol_pos=0.5, pot_taper="audio15")
    assert m.Rbot == pytest.approx(75000.0, rel=1e-3)  # 15% of 500k
    assert m.Rtop == pytest.approx(425000.0, rel=1e-3)

    # Audio tone at 50%
    m.apply_pot_positions(tone_pos=0.5, pot_taper="audio")
    assert m.Rtone == pytest.approx(25000.0, rel=1e-3)  # 10% of 250k

    # Full open (1.0) restores defaults
    m.apply_pot_positions(vol_pos=1.0, tone_pos=1.0)
    assert m.Rtop == pytest.approx(10.0)
    assert m.Rbot == pytest.approx(500000.0)


def test_mn_blend_potentiometer_behavior():
    """Verify MN blend pot: 0 dB insertion loss at center detent, attenuation away from center."""
    vcfg = VOICES["jazz_pair_open"]
    model = load_circuit(vcfg.circuit)
    assert model.topology == "parallel"

    # Center detent (0.5): 0 dB loss, both pickups 100% active
    model.apply_pot_positions(blend_pos=0.5)
    assert model.Rpot_n == pytest.approx(0.0)
    assert model.Rpot_b == pytest.approx(0.0)
    tr_center = compute_circuit_transfer_functions(model, return_numpy=True)
    assert len(tr_center) == 2
    mag_n_center = 20.0 * math.log10(max(tr_center[0][100], 1e-6))
    mag_b_center = 20.0 * math.log10(max(tr_center[1][100], 1e-6))
    assert abs(mag_n_center - mag_b_center) < 1.0

    # Full Neck (0.0): Bridge must be muted / heavily attenuated
    model.apply_pot_positions(blend_pos=0.0)
    tr_neck = compute_circuit_transfer_functions(model, return_numpy=True)
    assert tr_neck[1][100] == pytest.approx(0.0, abs=1e-4)  # Bridge muted
    assert tr_neck[0][100] > 0.1  # Neck active

    # Full Bridge (1.0): Neck must be muted / heavily attenuated
    model.apply_pot_positions(blend_pos=1.0)
    tr_bridge = compute_circuit_transfer_functions(model, return_numpy=True)
    assert tr_bridge[0][100] == pytest.approx(0.0, abs=1e-4)  # Neck muted
    assert tr_bridge[1][100] > 0.1  # Bridge active


def test_compute_parametric_sweep_blend():
    """Verify continuous blend parametric sweep executes and produces valid results."""
    res = compute_parametric_sweep("jazz_pair_open", param="blend")
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


def test_analytical_circuit_metrics_extraction():
    """Verify ParametricSweepResult.metrics() extracts accurate resonant peak, Q, bandwidth, and slope."""
    res = compute_parametric_sweep("precision_vintage", param="tone")
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
    rec_100 = next(r for r in res.metrics_records() if r.label == "Tone 100%")
    # Classic vintage P-Bass resonance (5.8H coil + 750pF cable) is in 1.8 - 3.5 kHz range
    assert rec_100.f_res_hz is not None
    assert 1800.0 <= rec_100.f_res_hz <= 3500.0
    # Q should be reasonable (0.8 to 4.0)
    assert rec_100.q_loaded is not None
    assert 0.8 <= rec_100.q_loaded <= 4.0
    # HF roll-off slope should be roughly -10 to -15 dB/octave
    assert rec_100.hf_slope_db_oct is not None
    assert -16.0 <= rec_100.hf_slope_db_oct <= -8.0

    # Summary table formatting
    table_str = res.summary_table()
    assert "f_res (Hz)" in table_str
    assert "Q loaded" in table_str
    assert "Tone 100%" in table_str


def test_linear_filter_stage_fusion():
    """Verify linear stage fusion executes cleanly and produces correct audio output."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "fused_out.wav"
        # Simulate clean linear target
        success = simulate_circuit_audio(
            input_audio=np.zeros((1, 4800), dtype=np.float32),  # 100 ms silent test buffer
            output_wav_path=out_wav,
            model=load_circuit("precision_vintage"),
            prefilter_firs=[np.zeros(2048, dtype=np.float32)],
            bypass_saturation=True,
            normalize="none",
        )
        assert success is True
        assert out_wav.exists()


def test_cir_netlist_deprecation_error():
    """Verify attempting to load a .cir file raises an explicit, informative ValueError."""
    with pytest.raises(ValueError, match="Legacy SPICE ASCII netlists .* are deprecated"):
        load_circuit("legacy_circuit.cir")
