"""
Allomorph Visualizer Subpackage
Interactive Altair visualizations and Polars frequency response modeling.
"""

from allomorph.visualizer.charts import (
    DOCS_DIR,
    build_voicings_html,
    generate_all_charts,
    generate_interactive_chart,
    generate_voicings_page,
    render_chart_to_file,
)
from allomorph.visualizer.dataframe import (
    F_MAX,
    F_MIN,
    NUM_POINTS,
    build_voice_dataframe,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_curve_rms_db,
    log_freqs,
)

__all__ = [
    "DOCS_DIR",
    "F_MAX",
    "F_MIN",
    "NUM_POINTS",
    "build_voice_dataframe",
    "build_voicings_comparison_data",
    "build_voicings_comparison_dataframe",
    "build_voicings_html",
    "compute_curve_rms_db",
    "generate_all_charts",
    "generate_interactive_chart",
    "generate_voicings_page",
    "log_freqs",
    "render_chart_to_file",
]
