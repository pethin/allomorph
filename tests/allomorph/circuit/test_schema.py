"""
Unit tests for circuit schemas (potentiometer controls, dynamic metallurgy, simulation).
"""

import numpy as np
import pytest
from pydantic import ValidationError

from allomorph.base import parse_spice_unit
from allomorph.circuit.schema import (
    CircuitMetricsRecord,
    MagnetPropertiesConfig,
    SaturationConfig,
)
from allomorph.circuit.sweeps import ParametricSweepResult
from allomorph.config.schema import ControlElementConfig


def test_spice_float_validation():
    """Verify SpiceFloat engineering notation parsing."""
    assert parse_spice_unit("500k") == 500000.0
    assert parse_spice_unit("47nF") == pytest.approx(4.7e-8)
    assert parse_spice_unit("1.0Meg") == 1e6
    assert parse_spice_unit("750pF") == pytest.approx(7.5e-10)
    assert parse_spice_unit(100.0) == 100.0
    assert parse_spice_unit(None) is None
    assert parse_spice_unit("") is None
    assert parse_spice_unit("   ") is None
    assert parse_spice_unit("250ohm") == 250.0
    assert parse_spice_unit("100 ohm") == 100.0
    with pytest.raises(ValueError):
        parse_spice_unit("F")
    with pytest.raises(TypeError, match="Invalid SPICE value"):
        parse_spice_unit([1, 2])

    # ControlElementConfig accepts parsed engineering notation
    ctrl = ControlElementConfig.model_validate(
        {
            "name": "Volume",
            "type": "pot",
            "resistance": parse_spice_unit("250k"),
            "cap": parse_spice_unit("47nF"),
        }
    )
    assert ctrl.resistance == 250000.0
    assert ctrl.cap == pytest.approx(47e-9)


def test_magnet_properties_config_and_diff():
    """Verify MagnetPropertiesConfig and differential calculation."""
    alnico = MagnetPropertiesConfig(alpha=0.26, alpha3=0.10, k_sag=0.08, vsat=0.50)
    ceramic = MagnetPropertiesConfig(alpha=0.12, alpha3=0.04, k_sag=0.03, vsat=0.70)

    # Differential softening (alnico target vs ceramic source)
    diff = alnico.diff(ceramic)
    assert diff.alpha == pytest.approx(0.14)
    assert diff.alpha3 == pytest.approx(0.06)
    assert diff.k_sag == pytest.approx(0.05)
    assert diff.vsat == 0.50  # Target vsat preserved

    # Zero negative clipping
    inv_diff = ceramic.diff(alnico)
    assert inv_diff.alpha == 0.0
    assert inv_diff.alpha3 == 0.0
    assert inv_diff.k_sag == 0.0


def test_saturation_config_validation():
    """Verify SaturationConfig validation, defaults, and boundary checks."""
    sat = SaturationConfig()
    assert sat.vsat == 0.50
    assert sat.alpha == 0.20
    assert sat.alpha3 == 0.08
    assert sat.oversample == 2
    assert sat.slew_limit is True
    assert sat.f_slew == 16000.0

    # Custom valid config
    custom = SaturationConfig(
        vsat=0.35,
        alpha=0.15,
        alpha3=0.05,
        eta_hyst=0.12,
        k_sag=0.04,
        k_eddy=0.05,
        oversample=4,
        slew_limit=False,
    )
    assert custom.vsat == 0.35
    assert custom.oversample == 4
    assert custom.slew_limit is False

    # Negative vsat rejected
    with pytest.raises(ValidationError):
        SaturationConfig(vsat=-0.5)

    # Invalid oversample factor
    with pytest.raises(ValidationError):
        SaturationConfig(oversample=3)  # type: ignore[arg-type]


def test_parametric_sweep_result_validation():
    """Verify ParametricSweepResult validation, invariants, and DataFrame conversion."""
    freqs = np.array([100.0, 1000.0, 5000.0, 10000.0])
    c0 = np.array([-1.0, 2.0, -3.0, -12.0])
    c1 = np.array([-1.0, 3.5, 0.5, -8.0])

    res = ParametricSweepResult(
        param="tone",
        values=[0.0, 1.0],
        freqs=freqs,
        curves=[c0, c1],
        labels=["Tone 0%", "Tone 100%"],
        voice_id="precision_vintage",
    )
    assert len(res.curves) == 2
    assert len(res.values) == 2

    # Polars DataFrame conversion
    df = res.to_dataframe(include_voice_id=True)
    assert df.shape == (8, 6)
    assert "frequency" in df.columns
    assert "magnitude_db" in df.columns
    assert "voice_id" in df.columns

    # Linear curves property
    lin = res.curves_linear
    assert len(lin) == 2
    assert lin[0][1] == pytest.approx(10.0 ** (2.0 / 20.0))

    # Mismatched lengths should fail validation
    with pytest.raises(ValidationError):
        ParametricSweepResult(
            param="tone",
            values=[0.0, 1.0],
            freqs=freqs,
            curves=[c0],  # Mismatched length (1 vs 2)
            labels=["Tone 0%", "Tone 100%"],
        )


def test_orphaned_circuit_models_eliminated():
    """Verify that legacy HarnessControls and SimulationConfig are removed from circuit schema."""
    import allomorph.circuit.schema as circ_schema

    assert not hasattr(circ_schema, "HarnessControls")
    assert not hasattr(circ_schema, "SimulationConfig")


def test_circuit_metrics_record_validation():
    """Verify CircuitMetricsRecord schema validation, fields, and dict export."""
    rec = CircuitMetricsRecord(
        param="vol_pos",
        param_value=0.5,
        label="Volume 50%",
        f_res_hz=3250.5,
        peak_db=4.2,
        insertion_loss_db=-0.8,
        peak_boost_db=5.0,
        q_loaded=1.85,
        bandwidth_hz=1757.0,
        cutoff_3db_hz=4500.0,
        hf_slope_db_oct=-12.0,
    )
    assert rec.param == "vol_pos"
    assert rec.param_value == 0.5
    assert rec.f_res_hz == 3250.5

    data = rec.model_dump()
    assert data["label"] == "Volume 50%"
    assert data["q_loaded"] == 1.85

    # Rejection of missing required fields
    with pytest.raises(ValidationError):
        CircuitMetricsRecord.model_validate(
            {
                "param": "vol_pos",
            }
        )
