"""
Allomorph CLI - Audit Subcommand
Audio QA telemetry, BS.1770-4 gated LUFS, crest factor, and true-peak clipping audit.
"""

import sys
from pathlib import Path

from allomorph.circuit.audit import audit_stems_audio_catalog
from allomorph.cli.console import log_error, log_info, log_step, log_success, render_table


def audit(
    audio_dir: Path | None = None,
    *,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Audits audio stems for BS.1770-4 loudness, true-peak ceiling, and clipping compliance."""
    log_step("AUDIT", f"Auditing audio files in {audio_dir or 'audio/stems/'} (force={force})...")

    report = audit_stems_audio_catalog(audio_dir=audio_dir, force=force)

    if report.total_files == 0:
        log_info("No WAV files found to audit.")
        return

    # Render summary table
    summary_rows = [
        ["Total Files", str(report.total_files)],
        ["Compliant Files", str(report.valid_files)],
        ["Clipped Files", str(report.clipped_files)],
        ["Mean LUFS", f"{report.mean_lufs:.2f} LUFS" if report.mean_lufs is not None else "N/A"],
        [
            "Min / Max LUFS",
            f"{report.min_lufs:.2f} / {report.max_lufs:.2f} LUFS"
            if report.min_lufs is not None and report.max_lufs is not None
            else "N/A",
        ],
        ["Mean True Peak", f"{report.mean_true_peak_dbfs:.2f} dBFS"],
        ["Max True Peak", f"{report.max_true_peak_dbfs:.2f} dBFS"],
    ]
    render_table("Audio QA Catalog Audit", headers=["Metric", "Value"], rows=summary_rows)

    if verbose or report.clipped_files > 0:
        detail_rows = []
        for r in report.records:
            if verbose or not r.is_valid:
                detail_rows.append(
                    [
                        r.file_name,
                        f"{r.lufs:.2f}" if r.lufs is not None else "N/A",
                        f"{r.true_peak_dbfs:.2f} dBFS",
                        f"{r.crest_factor_db:.2f} dB",
                        "PASS" if r.is_valid else "CLIPPED",
                    ]
                )
        if detail_rows:
            render_table(
                "Audio File Details",
                headers=["File", "LUFS", "True Peak", "Crest Factor", "Status"],
                rows=detail_rows,
                border_style="red" if report.clipped_files > 0 else "dim",
            )

    if report.clipped_files > 0:
        log_error(
            f"Catalog audit failed: {report.clipped_files} file(s) exceed true-peak ceiling (0.9900 / -0.09 dBFS)."
        )
        sys.exit(1)

    log_success("All audited audio files comply with loudness and true-peak standards.")
