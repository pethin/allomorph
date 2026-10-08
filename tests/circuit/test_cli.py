"""
Tests for circuit simulation CLI in allomorph.circuit.cli.
"""

from pathlib import Path
from typing import Any

import pytest

from allomorph.circuit.cli import main


def test_circuit_cli_argument_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Bug 4 regression: verify normalize and target_dbfs are forwarded to simulate_instrument_voicing."""
    calls = []

    def _mock_sim(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return tmp_path / "out.wav"

    monkeypatch.setattr("allomorph.circuit.cli.simulate_instrument_voicing", _mock_sim)

    dummy_input = tmp_path / "input.wav"
    dummy_input.write_bytes(b"dummy")

    main(
        argv=[
            "--instrument",
            "34in_standard_p",
            "--voice",
            "vintage_open",
            "--input",
            str(dummy_input),
            "--normalize",
            "rms",
            "--target-dbfs",
            "-22.5",
            "--vol",
            "0.8",
            "--tone",
            "0.7",
            "--cable-pf",
            "500",
            "--no-dc-block",
            "--no-dither",
        ]
    )

    assert len(calls) == 1
    call = calls[0]
    assert call.get("instrument") == "34in_standard_p"
    assert call.get("voicing") == "vintage_open"
    assert call.get("normalize") == "rms"
    assert call.get("target_dbfs") == -22.5
    assert call.get("vol_pos") == 0.8
    assert call.get("tone_pos") == 0.7
    assert call.get("cable_pf") == 500.0
    assert call.get("dc_block") is False
    assert call.get("apply_dither") is False


def test_circuit_cli_sweep(capsys: pytest.CaptureFixture[str]):
    """Verify --sweep outputs parametric sweep grid."""
    main(argv=["--sweep", "tone", "--voice", "vintage_open"])
    captured = capsys.readouterr()
    assert "PARAMETRIC SWEEP: vintage_open" in captured.out
    assert "Frequency Response Grid" in captured.out
    assert "Analytical Circuit Metrics" in captured.out
