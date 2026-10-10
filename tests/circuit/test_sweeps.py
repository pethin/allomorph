"""
Tests for instantaneous continuous parametric circuit sweeps in Allomorph.
Verifies tone pot sweep, volume pot sweep, cable capacitance sweep,
active EQ sweeps, Polars DataFrame generation, and performance benchmark (< 180 ms for 100 MNA solves).
"""

import time

import numpy as np
import polars as pl
import pytest

from allomorph.circuit.forward import resolve_target_voicing
from allomorph.circuit.sweeps import (
    compute_parametric_sweep,
)
from allomorph.dsp import FREQS


def test_tone_pot_sweep_treble_cut():
    """Tone pot sweep (0.0 -> 1.0) must show progressive treble cut (> 20 dB at 5 kHz)."""
    res = compute_parametric_sweep("precision_vintage", param="tone", values=[0.0, 0.25, 0.5, 0.75, 1.0])

    assert len(res.curves) == 5
    f_arr = np.asarray(res.freqs)
    idx_5k = int(np.argmin(np.abs(f_arr - 5000.0)))

    # Tone 0.0 (full roll-off) vs Tone 1.0 (full open)
    mag_closed = res.curves[0][idx_5k]
    mag_open = res.curves[-1][idx_5k]

    treble_cut_db = mag_open - mag_closed
    assert treble_cut_db > 20.0, f"Expected > 20 dB treble cut at 5 kHz, got {treble_cut_db:.2f} dB"

    # Verify monotonic treble increase with tone wiper position
    mags_5k = [c[idx_5k] for c in res.curves]
    for i in range(len(mags_5k) - 1):
        assert mags_5k[i] <= mags_5k[i + 1] + 1e-6


def test_volume_pot_sweep_attenuation():
    """Volume pot sweep (0.0 -> 1.0) must show cable loading attenuation."""
    res = compute_parametric_sweep("precision_vintage", param="vol", values=[0.0, 0.25, 0.5, 0.75, 1.0])

    assert len(res.curves) == 5
    f_arr = np.asarray(res.freqs)
    idx_1k = int(np.argmin(np.abs(f_arr - 1000.0)))

    # Vol 0.0 (grounded) vs Vol 1.0 (full)
    mag_zero = res.curves[0][idx_1k]
    mag_full = res.curves[-1][idx_1k]

    assert mag_zero < -60.0  # Fully attenuated
    assert mag_full > -3.0  # Nominal unattenuated passband
    assert mag_full > mag_zero + 50.0


def test_cable_capacitance_resonance_downshift():
    """Cable capacitance sweep (500 to 1500 pF) must show resonance downshifting."""
    res = compute_parametric_sweep("precision_vintage", param="cable", values=[500.0, 750.0, 1000.0, 1500.0])

    assert len(res.curves) == 4
    f_arr = np.asarray(res.freqs)

    # Filter frequencies between 500 Hz and 5000 Hz to locate resonant peaks
    mask = (f_arr >= 500.0) & (f_arr <= 5000.0)
    sub_f = f_arr[mask]

    peak_freqs = []
    for curve in res.curves:
        sub_curve = curve[mask]
        peak_idx = int(np.argmax(sub_curve))
        peak_freqs.append(sub_f[peak_idx])

    # Increasing capacitance must downshift resonance monotonically
    for i in range(len(peak_freqs) - 1):
        assert peak_freqs[i] >= peak_freqs[i + 1], (
            f"Expected downward resonance shift, got {peak_freqs[i]} -> {peak_freqs[i + 1]}"
        )

    # Treble at 4 kHz must roll off monotonically with higher cable capacitance
    idx_4k = int(np.argmin(np.abs(f_arr - 4000.0)))
    mags_4k = [c[idx_4k] for c in res.curves]
    for i in range(len(mags_4k) - 1):
        assert mags_4k[i] > mags_4k[i + 1]


def test_active_preamp_boost_sweep():
    """Active preamp bass boost sweep must increase low-frequency gain."""
    res = compute_parametric_sweep("jazz_pair_active", param="bass_boost", values=[0.0, 6.0, 12.0])

    assert len(res.curves) == 3
    f_arr = np.asarray(res.freqs)
    idx_40 = int(np.argmin(np.abs(f_arr - 40.0)))

    mag_0 = res.curves[0][idx_40]
    mag_12 = res.curves[-1][idx_40]
    boost = mag_12 - mag_0
    assert boost > 8.0, f"Expected active bass boost at 40 Hz, got {boost:.2f} dB"


def test_sweep_performance_benchmark():
    """100-step parametric MNA sweep must execute in < 180 ms."""
    values = np.linspace(0.0, 1.0, 100)

    # Warm-up run
    compute_parametric_sweep("precision_vintage", param="tone", values=[0.0, 1.0])

    runs = []
    res = None
    for _ in range(5):
        t0 = time.perf_counter()
        res = compute_parametric_sweep("precision_vintage", param="tone", values=values)
        runs.append((time.perf_counter() - t0) * 1000.0)

    best_ms = float(min(runs))
    assert res is not None
    assert len(res.curves) == 100
    assert best_ms < 180.0, (
        f"Expected < 180 ms for 100 MNA steps, best was {best_ms:.2f} ms (runs: {runs})"
    )


def test_to_dataframe_schema():
    """to_dataframe() must return a Polars DataFrame with the expected columns."""
    res = compute_parametric_sweep("precision_vintage", param="tone", values=[0.0, 0.5, 1.0])

    df = res.to_dataframe()
    assert isinstance(df, pl.DataFrame)
    expected_cols = ["frequency", "magnitude_db", "param", "param_value", "label"]
    assert df.columns == expected_cols
    assert len(df) == len(FREQS) * 3

    # Test with include_voice_id=True
    res.voice_id = "precision_vintage"
    df_voice = res.to_dataframe(include_voice_id=True)
    assert "voice_id" in df_voice.columns


def test_circuit_state_restoration():
    """Parametric sweep must not permanently alter the instrument voicing state."""
    inst, voicing = resolve_target_voicing("precision_vintage")
    orig_tone = voicing.controls.get("tone")

    compute_parametric_sweep((inst, voicing), param="tone", values=[0.1, 0.5, 0.9])
    assert voicing.controls.get("tone") == orig_tone

    compute_parametric_sweep((inst, voicing), param="cable", values=[250.0, 1200.0])
    assert voicing.controls.get("tone") == orig_tone


def test_curves_linear():
    """Verify curves_linear property properly converts dB curves to linear scale."""
    res = compute_parametric_sweep("precision_vintage", param="tone", values=[0.0, 1.0])
    lin_curves = res.curves_linear
    assert len(lin_curves) == 2
    for c_db, c_lin in zip(res.curves, lin_curves):
        expected_lin = 10.0 ** (np.asarray(c_db) / 20.0)
        assert np.allclose(c_lin, expected_lin)


def test_metrics_and_summary_table(capsys: pytest.CaptureFixture[str]):
    """Verify analytical metrics computation and summary table printing."""
    res = compute_parametric_sweep("precision_vintage", param="tone", values=[0.2, 0.8])
    records = res.metrics_records()
    assert len(records) == 2
    for r in records:
        assert r.f_res_hz is not None or r.peak_db is not None
        assert isinstance(r.insertion_loss_db, float)

    table_str = res.summary_table()
    assert "Setting / Label" in table_str
    assert "f_res (Hz)" in table_str

    res.print_metrics()
    captured = capsys.readouterr()
    assert "Setting / Label" in captured.out


def test_active_preamp_treble_boost_sweep():
    """Active preamp treble boost sweep must increase high-frequency gain."""
    res = compute_parametric_sweep(
        "jazz_pair_active", param="treble_boost", values=[0.0, 6.0, 12.0]
    )
    assert len(res.curves) == 3
    f_arr = np.asarray(res.freqs)
    idx_4k = int(np.argmin(np.abs(f_arr - 4000.0)))
    boost = res.curves[-1][idx_4k] - res.curves[0][idx_4k]
    assert boost > 4.0, f"Expected active treble boost at 4 kHz, got {boost:.2f} dB"


def test_tone_cap_sweep():
    """Tone cap sweep (10 nF to 100 nF) with rolled-off tone pot must downshift resonant peak frequency."""
    inst, voicing = resolve_target_voicing("precision_vintage")
    v_rolled = voicing.model_copy(deep=True)
    v_rolled.controls["tone"] = 0.0
    res = compute_parametric_sweep((inst, v_rolled), param="tone_cap", values=[10e-9, 47e-9, 100e-9])
    assert len(res.curves) == 3
    records = res.metrics_records()
    f_res_list = [r.f_res_hz for r in records if r.f_res_hz is not None]
    if len(f_res_list) >= 2:
        for i in range(len(f_res_list) - 1):
            assert f_res_list[i] >= f_res_list[i + 1]


def test_sweeps_polymorphic_inputs_and_edge_cases():
    """Verify compute_parametric_sweep polymorphic inputs, metrics, and error handling."""
    from allomorph.config import VOICES
    from allomorph.config.instruments import load_instrument

    inst = load_instrument("34in_standard_jazz")

    # 1. Blend sweep with separate neck_vol and bridge_vol (Jazz Bass)
    res_blend = compute_parametric_sweep(
        "34in_standard_jazz:pair_open", param="blend", values=[0.0, 0.5, 1.0]
    )
    assert len(res_blend.curves) == 3

    # 2. Polymorphic inputs: VoiceConfig, InstrumentConfig
    vcfg = VOICES["precision_vintage"]
    res_vcfg = compute_parametric_sweep(vcfg, param="tone", pot_taper="audio_15")
    assert len(res_vcfg.curves) > 0

    res_inst = compute_parametric_sweep(inst, param="vol", pot_taper="linear")
    assert len(res_inst.curves) > 0

    # 3. Custom labels and print_metrics
    res_custom = compute_parametric_sweep(
        "precision_vintage",
        param="cable",
        values=[200.0, 750.0],
        labels=["200pF", "750pF"],
        pot_taper="reverse_audio",
    )
    res_custom.print_metrics()
    assert len(res_custom.metrics_records()) == 2

    # 4. Error cases: invalid type, mismatched labels
    with pytest.raises(TypeError):
        compute_parametric_sweep(12345, param="tone")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Length of labels"):
        compute_parametric_sweep(
            "precision_vintage", param="tone", values=[0.0, 1.0], labels=["OnlyOne"]
        )


def test_sweeps_bass_treble_boost_and_unsupported_params():
    """Verify sweeps with active bass and treble boost, linear curves, and errors."""
    from allomorph.circuit.sweeps import ParametricSweepResult, compute_parametric_sweep
    from allomorph.config.instruments import load_instrument

    inst = load_instrument("34in_standard_p")

    # Cable sweep with default values
    res_cable = compute_parametric_sweep(inst, param="cable")
    assert len(res_cable.curves) == 5
    assert len(res_cable.curves_db) == 5
    assert len(res_cable.curves_linear) == 5

    # Bass boost sweep (adds low_shelf if not present)
    res_bass = compute_parametric_sweep(inst, param="bass_boost")
    assert len(res_bass.curves) == 5

    # Treble boost sweep (adds high_shelf if not present)
    res_treble = compute_parametric_sweep(inst, param="treble_boost")
    assert len(res_treble.curves) == 5

    # Tone cap sweep
    res_cap = compute_parametric_sweep(inst, param="tone_cap")
    assert len(res_cap.curves) == 5

    # Unsupported parameter raises ValueError
    with pytest.raises(ValueError, match="Unsupported sweep parameter"):
        compute_parametric_sweep(inst, param="nonexistent_flux_capacitor")

    # ParametricSweepResult mismatched curve length raises ValueError
    with pytest.raises(ValueError, match="does not match frequencies length"):
        ParametricSweepResult(
            param="test",
            values=[1.0],
            freqs=np.array([100.0, 200.0]),
            curves=[np.array([0.0])],  # length 1 vs freqs length 2
            labels=["1.0"],
        )

