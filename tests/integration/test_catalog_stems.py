"""
Integration tests for catalog instruments, voicings, and forward stems.
Verifies that all catalog instruments and voicings pass forward digital twin simulation,
causal zero-latency alignment, and true-peak safety.
"""

from allomorph.circuit.stem_debug import debug_voicing_stem
from allomorph.config.instruments import INSTRUMENTS
from allomorph.config.preamps import PREAMPS


def test_all_catalog_instrument_voicings_bug_free():
    """Validates that all native voicings defined across all catalog instruments
    pass causal zero-latency alignment, remain below true-peak ceiling, and maintain stable THD.
    """
    for inst_id, inst in sorted(INSTRUMENTS.items()):
        for v_id, voicing in sorted(inst.voicings.items()):
            report = debug_voicing_stem(
                instrument=inst,
                voicing=voicing,
                preamps=PREAMPS,
            )
            assert report.is_causal_zero_latency, (
                f"{inst_id}:{v_id} failed causal zero-latency (onset={report.onset_sample_index}, peak={report.peak_sample_index})"
            )
            assert report.peak_dbfs <= -0.09, (
                f"{inst_id}:{v_id} clipped above -0.09 dBFS ceiling ({report.peak_dbfs:+.2f} dBFS)"
            )
            assert report.thd_percent <= 50.0, (
                f"{inst_id}:{v_id} showed unstable harmonic distortion ({report.thd_percent:.1f}%)"
            )
