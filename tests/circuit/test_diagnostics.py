"""Deterministic unit tests for circuit diagnostics, onset detection, and telemetry aggregation."""

from pathlib import Path

import numpy as np

from allomorph.circuit.diagnostics import (
    AudioAuditRecord,
    aggregate_audit_report,
    compute_stem_level_metrics,
    evaluate_causal_onset_and_peak,
    extract_frequency_anchors,
    is_manifest_entry_fresh,
)


def test_evaluate_causal_onset_and_peak_causal():
    """Verify causal impulse response with immediate onset is flagged as causal zero-latency."""
    # Peak at sample 0, decaying exponentially
    ir = np.zeros(256, dtype=np.float64)
    ir[0] = 1.0
    ir[1] = 0.5
    ir[2] = 0.25

    onset_idx, peak_idx, is_causal = evaluate_causal_onset_and_peak(ir)
    assert onset_idx == 0
    assert peak_idx == 0
    assert is_causal is True


def test_evaluate_causal_onset_and_peak_gentle_group_delay():
    """Verify gentle filter group delay (onset at sample 2, peak at sample 15) passes causal check."""
    ir = np.zeros(256, dtype=np.float64)
    ir[2] = 0.05
    ir[15] = 1.0

    onset_idx, peak_idx, is_causal = evaluate_causal_onset_and_peak(ir)
    assert onset_idx == 2
    assert peak_idx == 15
    assert is_causal is True


def test_evaluate_causal_onset_and_peak_dead_latency():
    """Verify artificial leading dead latency (onset > 4) fails causal check."""
    ir = np.zeros(256, dtype=np.float64)
    ir[10] = 0.1
    ir[20] = 1.0

    onset_idx, peak_idx, is_causal = evaluate_causal_onset_and_peak(ir)
    assert onset_idx == 10
    assert peak_idx == 20
    assert is_causal is False


def test_evaluate_causal_onset_and_peak_empty():
    """Verify empty IR returns safe defaults."""
    empty = np.array([], dtype=np.float64)
    onset_idx, peak_idx, is_causal = evaluate_causal_onset_and_peak(empty)
    assert onset_idx == 0
    assert peak_idx == 0
    assert is_causal is False


def test_compute_stem_level_metrics_sine():
    """Verify level metrics for a unity full-scale sine wave."""
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sine = np.sin(2.0 * np.pi * 1000.0 * t)

    metrics = compute_stem_level_metrics(sine)
    assert -0.05 <= metrics["peak_dbfs"] <= 0.05
    assert -3.1 <= metrics["rms_dbfs"] <= -2.9
    assert 2.9 <= metrics["crest_factor_db"] <= 3.1


def test_compute_stem_level_metrics_silence():
    """Verify level metrics for digital silence return low floors."""
    silence = np.zeros(1024, dtype=np.float64)
    metrics = compute_stem_level_metrics(silence)
    assert metrics["peak_dbfs"] <= -100.0
    assert metrics["rms_dbfs"] <= -100.0


def test_extract_frequency_anchors():
    """Verify frequency anchor lookup correctly matches nearest bins."""
    f_bins = np.linspace(0.0, 24000.0, 24001)  # 1 Hz resolution
    # Create simple frequency response where mag_db = -f / 1000
    mags_db = -f_bins / 1000.0

    target_freqs = {
        "dc": 0.0,
        "1khz": 1000.0,
        "5khz": 5000.0,
    }
    anchors = extract_frequency_anchors(f_bins, mags_db, target_freqs)
    assert anchors["dc"] == 0.0
    assert anchors["1khz"] == -1.0
    assert anchors["5khz"] == -5.0


def test_is_manifest_entry_fresh(tmp_path: Path):
    """Verify manifest entry freshness checks against physical file state."""
    test_file = tmp_path / "stem.wav"
    test_file.write_bytes(b"12345678")

    assert is_manifest_entry_fresh(None, test_file) is False

    incomplete_entry = {"size_bytes": 8}
    assert is_manifest_entry_fresh(incomplete_entry, test_file) is False

    fresh_entry = {
        "size_bytes": 8,
        "true_peak_dbfs": -0.5,
        "lufs": -20.5,
    }
    assert is_manifest_entry_fresh(fresh_entry, test_file) is True

    stale_entry = {
        "size_bytes": 100,  # File changed on disk
        "true_peak_dbfs": -0.5,
        "lufs": -20.5,
    }
    assert is_manifest_entry_fresh(stale_entry, test_file) is False


def test_aggregate_audit_report():
    """Verify aggregate_audit_report compiles statistics across multiple records."""
    r1 = AudioAuditRecord(
        file_path="stem1.wav",
        file_name="stem1.wav",
        duration_s=5.0,
        sample_rate=48000,
        peak_dbfs=-1.0,
        true_peak_linear=0.89,
        true_peak_dbfs=-1.0,
        rms_dbfs=-18.0,
        crest_factor_db=17.0,
        lufs=-20.0,
        dc_offset=0.0,
        has_clipping=False,
        is_valid=True,
    )
    r2 = AudioAuditRecord(
        file_path="stem2.wav",
        file_name="stem2.wav",
        duration_s=5.0,
        sample_rate=48000,
        peak_dbfs=0.5,
        true_peak_linear=1.05,
        true_peak_dbfs=0.4,
        rms_dbfs=-16.0,
        crest_factor_db=16.4,
        lufs=-18.0,
        dc_offset=0.0,
        has_clipping=True,
        is_valid=False,
    )

    report = aggregate_audit_report([r1, r2])
    assert report.total_files == 2
    assert report.valid_files == 1
    assert report.clipped_files == 1
    assert report.min_lufs == -20.0
    assert report.max_lufs == -18.0
    assert report.mean_lufs == -19.0
    assert report.max_true_peak_dbfs == 0.4
    assert len(report.records) == 2
