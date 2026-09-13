"""
Allomorph Visualizer - Interactive Altair Charts Generation
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from allomorph.circuit import compute_differential_circuit_transfer_functions, load_circuit
from allomorph.config.instruments import load_all_instruments, load_instrument
from allomorph.config.schema import InstrumentConfig
from allomorph.config.voices import VOICES
from allomorph.dsp import FREQS
from allomorph.physics import is_voice_matching_source
from allomorph.visualizer.dataframe import (
    build_baked_responses_data,
    build_baked_waterfall_3d_data,
    build_composite_instrument_dataframe,
    build_instrument_frontend_dataframe,
    build_universal_targets_dataframe,
    build_voice_dataframe,
)
from allomorph.visualizer.portal import RESPONSES_DIR, append_spec_panel, generate_portal_pages

alt.data_transformers.disable_max_rows()


def render_chart_to_file(
    master_df: pl.DataFrame,
    target_path: Path,
    chart_title: str,
    chart_subtitle: str,
    y_title: str,
    y_domain: Sequence[float],
    mode: str = "unified",
) -> Path:
    """Renders a Polars master dataframe into an interactive Altair chart HTML file."""
    voice_selection = alt.selection_point(fields=["voice_name"], bind="legend")
    if mode == "output" or mode == "difference":
        params = [voice_selection]
        filters: list[Any] = []
    else:  # unified
        mode_selection = alt.selection_point(
            fields=["mode"],
            bind=alt.binding_radio(
                options=["Input/Output Difference", "Output Voice"], name="Display Mode: "
            ),
            value="Input/Output Difference",
        )
        params = [mode_selection, voice_selection]
        filters = [mode_selection]

    chart = (
        alt.Chart(master_df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(
                "frequency:Q",
                scale=alt.Scale(type="log", domain=[20, 20000]),
                title="Frequency (Hz)",
                axis=alt.Axis(
                    values=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                    grid=True,
                    gridDash=[3, 3],
                    gridColor="#333333",
                ),
            ),
            y=alt.Y(
                "magnitude_db:Q",
                scale=alt.Scale(domain=y_domain),
                title=y_title,
                axis=alt.Axis(grid=True, gridDash=[3, 3], gridColor="#333333"),
            ),
            color=alt.Color(
                "voice_name:N",
                title="Allomorph Pickup Profile (Click to isolate)",
                scale=alt.Scale(scheme="tableau20"),
            ),
            opacity=alt.condition(voice_selection, alt.value(1.0), alt.value(0.12)),
            strokeWidth=alt.condition(voice_selection, alt.value(2.8), alt.value(1.0)),
            tooltip=[
                alt.Tooltip("voice_name:N", title="Pickup Configuration"),
                alt.Tooltip("topology:N", title="Topology"),
                alt.Tooltip("description:N", title="Circuit / Acoustic Description"),
                alt.Tooltip("frequency:Q", title="Frequency (Hz)", format=".1f"),
                alt.Tooltip("magnitude_db:Q", title="Magnitude (dB)", format="+.1f"),
            ],
        )
    )

    for f in filters:
        chart = chart.transform_filter(f)
    for p in params:
        chart = chart.add_params(p)

    chart = (
        chart.properties(
            title=alt.TitleParams(
                text=chart_title,
                subtitle=chart_subtitle,
                fontSize=16,
                subtitleFontSize=12,
                anchor="start",
            ),
            width=740,
            height=480,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(orient="right", labelLimit=320)
        .interactive()
    )

    chart.save(str(target_path))
    print(f"Saved interactive Altair visualization: {target_path}")
    return target_path


def generate_universal_targets_chart(target_path: Path | None = None) -> Path:
    """
    Renders the Mode 1 Universal Target Voicings chart (Block 2):
    Evaluates all 22 target voices relative to Canonical Intermediate (34" @ 93.5mm datum).
    Includes the 3 Dynamic Feel Tiers specification panel (Clean, Dynamic, Hot Rod).
    """
    if target_path is None:
        target_path = RESPONSES_DIR / "universal_targets.html"
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    master_df = build_universal_targets_dataframe()
    voice_selection = alt.selection_point(fields=["voice_name"], bind="legend")

    chart = (
        alt.Chart(master_df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(
                "frequency:Q",
                scale=alt.Scale(type="log", domain=[20, 20000]),
                title="Frequency (Hz)",
                axis=alt.Axis(
                    values=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                    grid=True,
                    gridDash=[3, 3],
                    gridColor="#333333",
                ),
            ),
            y=alt.Y(
                "magnitude_db:Q",
                scale=alt.Scale(domain=[-24, 24]),
                title="Voicing Magnitude relative to Intermediate (dB)",
                axis=alt.Axis(
                    values=[-24, -18, -12, -6, 0, 6, 12, 18, 24],
                    grid=True,
                    gridDash=[3, 3],
                    gridColor="#333333",
                ),
            ),
            color=alt.Color(
                "voice_name:N",
                title="Universal Target Voice (Click to isolate)",
                scale=alt.Scale(scheme="tableau20"),
            ),
            opacity=alt.condition(voice_selection, alt.value(1.0), alt.value(0.12)),
            strokeWidth=alt.condition(voice_selection, alt.value(2.8), alt.value(1.0)),
            tooltip=[
                alt.Tooltip("voice_name:N", title="Pickup Configuration"),
                alt.Tooltip("topology:N", title="Topology"),
                alt.Tooltip("description:N", title="Circuit / Acoustic Description"),
                alt.Tooltip("frequency:Q", title="Frequency (Hz)", format=".1f"),
                alt.Tooltip("magnitude_db:Q", title="Magnitude (dB)", format="+.1f"),
            ],
        )
        .add_params(voice_selection)
        .properties(
            title=alt.TitleParams(
                text="Allomorph Master Voices: Universal Target Voicings (Block 2)",
                subtitle='Target Passive Acoustic Apertures & SPICE Loaded RLC Resonances relative to Canonical Intermediate Baseline (34" @ 93.5mm)',
                fontSize=16,
                subtitleFontSize=12,
                anchor="start",
            ),
            width=740,
            height=480,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(orient="right", labelLimit=320)
        .interactive()
    )

    chart.save(str(target_path))

    spec_panel = """
<style>
  body {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 12px 16px;
    margin: 0;
    overflow-x: hidden;
    box-sizing: border-box;
  }
  #vis {
    display: flex;
    justify-content: center;
    width: 100%;
    overflow-x: hidden;
  }
  .tiers-container { max-width: 1060px; margin: 14px auto 0 auto; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .tier-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 6px; }
  .tier-header { display: flex; align-items: center; justify-content: space-between; }
  .tier-title { font-size: 13px; font-weight: 700; color: #f0f6fc; }
  .tier-badge { font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; text-transform: uppercase; }
  .badge-clean { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #38bdf8; }
  .badge-dynamic, .badge-standard { background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid #4ade80; }
  .badge-hotrod { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid #f87171; }
  .tier-desc { font-size: 11px; color: #8b949e; line-height: 1.4; }
  .tier-formula { font-family: monospace; font-size: 11px; color: #58a6ff; background: #0d1117; padding: 4px 6px; border-radius: 4px; margin-top: 4px; }
</style>
<div class="tiers-container">
  <div class="tier-card">
    <div class="tier-header">
      <span class="tier-title">01 Studio Clean</span>
      <span class="tier-badge badge-clean">cln_</span>
    </div>
    <div class="tier-desc">0% Saturation / Maximum Headroom. Pure linear RLC resonance response with zero magnetic compression. Ideal for pristine DI and clean funk.</div>
    <div class="tier-formula">&alpha; = 0.00 | V_sat = 10.0V</div>
  </div>
  <div class="tier-card">
    <div class="tier-header">
      <span class="tier-title">02 Standard Dynamic</span>
      <span class="tier-badge badge-standard">std_</span>
    </div>
    <div class="tier-desc">Standard Give & Bloom. 100% nominal target dynamic modeling: magnetic string drag damping, 2f0 orbit bloom, dynamic inductance sag (&lambda;_L), and soft-knee compression.</div>
    <div class="tier-formula">&alpha; = &alpha;_tgt | V_sat = V_tgt</div>
  </div>
  <div class="tier-card">
    <div class="tier-header">
      <span class="tier-title">03 Hot Rod</span>
      <span class="tier-badge badge-hotrod">hot_</span>
    </div>
    <div class="tier-desc">175% Overwound Pre-Conditioner. Saturated attack give, compressed low-mids, and elevated harmonic punch to drive downstream Darkglass engines.</div>
    <div class="tier-formula">&alpha; = 1.75 &times; &alpha;_tgt | V_sat / 1.35</div>
  </div>
</div>
"""
    append_spec_panel(target_path, spec_panel)
    print(f"Saved Universal Target Voicings visualization: {target_path}")
    return target_path


def generate_instrument_frontend_chart(
    inst: InstrumentConfig,
    target_path: Path | None = None,
) -> Path:
    """
    Renders the Frontend Deconvolutions chart (Block 1) for a single source instrument.
    Allows users to click any pickup switch position in the legend to isolate that specific pickup key.
    """
    inst_id = inst.id
    inst_name = inst.name
    if target_path is None:
        target_path = RESPONSES_DIR / f"{inst_id}_frontend.html"
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_instrument_frontend_dataframe(inst)
    pickup_selection = alt.selection_point(fields=["pickup_name"], bind="legend")

    rule_df = pl.DataFrame({"frequency": [20.0, 20000.0], "magnitude_db": [0.0, 0.0]})
    baseline = (
        alt.Chart(rule_df)
        .mark_line(color="#8b949e", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(x="frequency:Q", y="magnitude_db:Q")
    )

    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(
                "frequency:Q",
                scale=alt.Scale(type="log", domain=[20, 20000]),
                title="Frequency (Hz)",
                axis=alt.Axis(
                    values=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                    grid=True,
                    gridDash=[3, 3],
                    gridColor="#333333",
                ),
            ),
            y=alt.Y(
                "magnitude_db:Q",
                scale=alt.Scale(domain=[-24, 24]),
                title="Frontend Deconvolution Gain (dB)",
                axis=alt.Axis(
                    values=[-24, -18, -12, -6, 0, 6, 12, 18, 24],
                    grid=True,
                    gridDash=[3, 3],
                    gridColor="#333333",
                ),
            ),
            color=alt.Color(
                "pickup_name:N",
                title="Pickup Switch Position (Click to isolate)",
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=[
                alt.Tooltip("pickup_name:N", title="Pickup Switch Position"),
                alt.Tooltip("position_mm:Q", title="Bridge Distance (mm)", format=".1f"),
                alt.Tooltip("frequency:Q", title="Frequency (Hz)", format=".1f"),
                alt.Tooltip("magnitude_db:Q", title="Gain / Cut (dB)", format="+.1f"),
            ],
            opacity=alt.condition(pickup_selection, alt.value(0.96), alt.value(0.12)),
            strokeWidth=alt.condition(pickup_selection, alt.value(3.0), alt.value(1.2)),
        )
    )

    chart = (
        alt.layer(lines, baseline)
        .add_params(pickup_selection)
        .properties(
            title=alt.TitleParams(
                text=f"Allomorph Frontend Deconvolutions: {inst_name} (Block 1)",
                subtitle="Inverting Physical Pickup Aperture Sinc & RLC Impedance to Canonical Intermediate Baseline (0.00 dB Target)",
                fontSize=16,
                subtitleFontSize=12,
                anchor="start",
            ),
            width=740,
            height=480,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(orient="right", labelLimit=320)
        .interactive()
    )

    chart.save(str(target_path))

    panel = """
<style>
  body {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 12px 16px;
    margin: 0;
    overflow-x: hidden;
    box-sizing: border-box;
  }
  #vis {
    display: flex;
    justify-content: center;
    width: 100%;
    overflow-x: hidden;
  }
  .info-container {
    max-width: 1060px;
    margin: 14px auto 0 auto;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px;
    font-size: 12px;
    color: #8b949e;
    line-height: 1.5;
  }
  .info-title { color: #f0f6fc; font-weight: 700; margin-bottom: 4px; }
  .highlight { color: #58a6ff; font-weight: 600; }
</style>
<div class="info-container">
  <div class="info-title">Block 1 Operational Directives:</div>
  <div>1. <span class="highlight">Knobs Wide Open (100%):</span> Source bass volume and tone knobs must be wide open (100%) so that passive pot loading matches the deconvolution netlist exactly.</div>
  <div>2. <span class="highlight">Strictly Positive Initial Polarity:</span> All FIRs enforce strictly positive initial polarity (&sum; h[:16] &gt; 0) to ensure zero phase cancellation when blended in parallel with DI.</div>
  <div>3. <span class="highlight">Zero-Latency Causal Synthesis:</span> 2048-tap minimum-phase causal FIRs run on Darkglass Anagram Block 1 with 0% DSP overhead and zero perceptible latency.</div>
</div>
"""
    content = target_path.read_text(encoding="utf-8")
    content = content.replace("</body>", f"{panel}</body>")
    target_path.write_text(content, encoding="utf-8")
    print(f"Saved Frontend Deconvolution chart for {inst_name}: {target_path}")
    return target_path


def generate_frontend_deconvolutions_chart(target_path: Path | None = None) -> Path:
    """
    Renders the Mode 2 Frontend Deconvolutions master page (Block 1).
    Provides an instrument selector linking to each source instrument's dedicated frontend chart,
    allowing users to click and isolate individual pickup keys/positions.
    """
    if target_path is None:
        target_path = RESPONSES_DIR / "frontend_deconvolutions.html"
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_insts = load_all_instruments()
    inst_items = []
    for iid, icfg in sorted(all_insts.items()):
        if iid == "canonical_intermediate":
            continue
        iname = icfg.name
        inst_chart_file = target_path.parent / f"{iid}_frontend.html"
        generate_instrument_frontend_chart(icfg, target_path=inst_chart_file)
        inst_items.append({"id": iid, "name": iname, "url": f"{iid}_frontend.html"})

    default_item = (
        inst_items[0]
        if inst_items
        else {"id": "30in_emg_mmtw", "url": "30in_emg_mmtw_frontend.html"}
    )
    options_html = "\n".join(
        [f'        <option value="{item["url"]}">{item["name"]}</option>' for item in inst_items]
    )
    tabs_html = "\n".join(
        [
            f'      <button class="inst-tab-btn{" active" if item["id"] == default_item["id"] else ""}" data-url="{item["url"]}" onclick="switchInstrument(\'{item["url"]}\', this)">{item["name"]}</button>'
            for item in inst_items
        ]
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph Master Voices: Frontend Deconvolutions (Block 1)</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --accent: #38bdf8;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --tag-bg: #21262d;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg) !important;
      color: #c9d1d9 !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 12px 16px;
      margin: 0;
      overflow-x: hidden;
    }}
    .header-card {{
      max-width: 1060px;
      margin: 0 auto 12px auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .header-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .header-title {{
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .inst-picker-select {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 12px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 6px;
      outline: none;
      cursor: pointer;
    }}
    .inst-picker-select:focus {{
      border-color: var(--accent);
    }}
    .inst-tabs-wrap {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      border-top: 1px solid var(--border);
      padding-top: 10px;
    }}
    .inst-tab-btn {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      padding: 5px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .inst-tab-btn:hover {{
      background: #30363d;
      color: var(--text);
    }}
    .inst-tab-btn.active {{
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
      border-color: #38bdf8;
    }}
    .chart-frame-wrap {{
      max-width: 1060px;
      margin: 0 auto;
      display: flex;
      justify-content: center;
    }}
    .chart-frame {{
      width: 100%;
      height: 720px;
      border: none;
      border-radius: 8px;
      background: var(--bg);
    }}
    #pickup-sublevel {{ display: none; }}
  </style>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega@6"></script>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega-lite@6.4.1"></script>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>
</head>
<body>
  <div class="header-card">
    <div class="header-row">
      <div class="header-title">
        <span>Frontend Deconvolutions (Block 1) — Per-Instrument Pickup Key Selector</span>
      </div>
      <div>
        <select id="inst-select" class="inst-picker-select" onchange="switchInstrument(this.value)">
{options_html}
        </select>
      </div>
    </div>
    <div class="inst-tabs-wrap">
{tabs_html}
    </div>
  </div>

  <div id="pickup-sublevel"></div>

  <div class="chart-frame-wrap">
    <iframe id="frontend-frame" class="chart-frame" src="{default_item["url"]}" title="Per-Instrument Frontend Deconvolutions Chart"></iframe>
  </div>

  <script>
    // vegaEmbed reference for embedded and sub-frame charts
    if (typeof vegaEmbed !== 'undefined') {{
      window.vegaEmbed = vegaEmbed;
    }}
    function switchInstrument(url, btnEl) {{
      const frame = document.getElementById('frontend-frame');
      if (frame) frame.src = url;
      const sel = document.getElementById('inst-select');
      if (sel) sel.value = url;
      document.querySelectorAll('.inst-tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.getAttribute('data-url') === url);
      }});
    }}
  </script>
</body>
</html>
"""
    target_path.write_text(html_content, encoding="utf-8")
    print(f"Saved Frontend Deconvolutions master page: {target_path}")
    return target_path


def generate_baked_responses_page(target_html: Path | None = None) -> Path:
    """
    Renders the Mode 3 Baked Transformations master page (1-Block Monolithic Model).
    Embeds the compact responses matrix directly in a script tag for instant,
    zero-CORS local viewing. Allows users to dynamically select multiple source instruments
    and multiple target voicings with sub-millisecond Vega re-rendering.
    """
    if target_html is None:
        target_html = RESPONSES_DIR / "baked_responses.html"
    target_html = Path(target_html)
    target_html.parent.mkdir(parents=True, exist_ok=True)

    data = build_baked_responses_data(step=3)
    data_json = json.dumps(data)

    default_active_insts = {"34in_standard_p", "34in_standard_jazz"}
    inst_buttons: list[str] = []
    for iid, info in sorted(data["instruments"].items()):
        is_act = " active" if iid in default_active_insts else ""
        name = info["name"]
        scale = info["scale_in"]
        inst_buttons.append(
            f'        <button class="chip-btn{is_act}" data-id="{iid}" onclick="toggleInstrument(\'{iid}\')">'
            f'<span class="chip-check">✓</span><span>{name} ({scale:.0f}")</span></button>'
        )
    inst_chips_html = "\n".join(inst_buttons)

    default_active_voices = {
        "05_vintage_62_p_alnico",
        "02_jazz_bass_pair",
        "09_stingray_mm_parallel",
    }
    voice_buttons: list[str] = []
    for vid, vinfo in sorted(data["voices"].items()):
        is_act = " active" if vid in default_active_voices else ""
        vname = vinfo["name"]
        fam = vinfo["family"]
        voice_buttons.append(
            f'        <button class="chip-btn{is_act}" data-id="{vid}" data-family="{fam}" onclick="toggleVoicing(\'{vid}\')">'
            f'<span class="chip-check">✓</span><span>{vname}</span></button>'
        )
    voice_chips_html = "\n".join(voice_buttons)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph Master Voices: Baked Transformations (1-Block Single Model)</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --card-hover: #1c2128;
      --border: #30363d;
      --accent: #38bdf8;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --tag-bg: #21262d;
      --btn-active: #1f6feb;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg) !important;
      color: #c9d1d9 !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 12px 16px;
      margin: 0;
      overflow-x: hidden;
    }}
    .header-card {{
      max-width: 1060px;
      margin: 0 auto 12px auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .header-title-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .header-title {{
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge-counter {{
      font-size: 11px;
      font-weight: 700;
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
      border: 1px solid #38bdf8;
      padding: 3px 10px;
      border-radius: 12px;
    }}
    .section-title-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
      border-top: 1px solid var(--border);
      padding-top: 10px;
    }}
    .section-label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--accent);
    }}
    .quick-actions {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .quick-btn {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .quick-btn:hover {{
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      border-color: #8b949e;
    }}
    .chips-container {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip-btn {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      padding: 5px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      user-select: none;
    }}
    .chip-btn:hover {{
      background: #30363d;
      color: var(--text);
      border-color: #8b949e;
    }}
    .chip-btn.active {{
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
      border-color: #38bdf8;
      box-shadow: 0 0 6px rgba(56, 189, 248, 0.25);
    }}
    .chip-check {{
      display: inline-block;
      font-size: 11px;
      opacity: 0.3;
      transition: opacity 0.15s ease;
    }}
    .chip-btn.active .chip-check {{
      opacity: 1;
      color: #38bdf8;
    }}
    .chart-card {{
      max-width: 1060px;
      margin: 0 auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      justify-content: center;
    }}
    #vis {{
      width: 100%;
      display: flex;
      justify-content: center;
    }}
    .directives-card {{
      max-width: 1060px;
      margin: 14px auto 0 auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.5;
    }}
    .directive-item {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .directive-title {{
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
    }}
    .highlight {{
      color: #58a6ff;
      font-weight: 600;
    }}
  </style>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega@6"></script>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega-lite@6.4.1"></script>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>
</head>
<body>
  <div class="header-card">
    <div class="header-title-row">
      <div class="header-title">
        <span>Baked Transformations (1-Block Single Model) — Multi-Instrument & Multi-Voicing Matrix</span>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <a href="baked_waterfall_3d.html" class="quick-btn" style="text-decoration: none; display: inline-flex; align-items: center; gap: 6px; color: #38bdf8; border-color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline></svg>
          <span>View 3D IR Waterfall &rarr;</span>
        </a>
        <div id="curve-counter" class="badge-counter">6 active curves</div>
      </div>
    </div>

    <div class="section-title-row">
      <span class="section-label">Source Instruments (Multi-Select):</span>
      <div class="quick-actions">
        <button class="quick-btn" onclick="selectAllInstruments()">Select All</button>
        <button class="quick-btn" onclick="clearInstruments()">Clear</button>
        <button class="quick-btn" onclick="selectStandardInstruments()">34" Standard</button>
        <button class="quick-btn" onclick="selectShortMedInstruments()">Short/Medium Scale</button>
      </div>
    </div>
    <div id="inst-chips" class="chips-container">
{inst_chips_html}
    </div>

    <div class="section-title-row">
      <span class="section-label">Target Voicings (Multi-Select):</span>
      <div class="quick-actions">
        <button class="quick-btn" onclick="selectAllVoicings()">All Voicings</button>
        <button class="quick-btn" onclick="clearVoicings()">Clear</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('Precision')">Precision</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('Jazz')">Jazz</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('StingRay')">StingRay</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('PJ')">PJ & P/MM</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('Character')">Character</button>
      </div>
    </div>
    <div id="voice-chips" class="chips-container">
{voice_chips_html}
    </div>
  </div>

  <div class="chart-card">
    <div id="vis"></div>
  </div>

  <div class="directives-card">
    <div class="directive-item">
      <div class="directive-title">1. Monolithic 1-Block Deployment</div>
      <div>Curves plot the regularized differential transfer function <span class="highlight">H<sub>diff</sub>(f) = H<sub>tgt</sub> / H<sub>src</sub></span> generated when executing <span class="highlight">--stage bake</span>. These models run on Darkglass Anagram Block 1 or Neural Amp Modeler without requiring a separate frontend IR block.</div>
    </div>
    <div class="directive-item">
      <div class="directive-title">2. Linear Transfer vs. Dynamic Feel</div>
      <div>Plotted curves depict the continuous frequency response synthesized by the differential acoustic aperture and loaded SPICE circuit. Non-linear dynamic give (&alpha;), attack sag (k<sub>sag</sub>), and rail saturation (V<sub>sat</sub>) operate dynamically on high-amplitude transients.</div>
    </div>
    <div class="directive-item">
      <div class="directive-title">3. Bit-for-Bit Identity Bypass</div>
      <div>When a source instrument already embodies the target voice (e.g. 34" Standard P &rarr; Vintage '62 P), the differential transfer evaluates to an exact flat <span class="highlight">0.00 dB</span>, confirming that identity models are safely omitted via <span class="highlight">skip_identity=True</span>.</div>
    </div>
  </div>

  <script id="baked-data" type="application/json">
{data_json}
  </script>

  <script>
    const bakedData = JSON.parse(document.getElementById('baked-data').textContent);
    const selectedInstruments = new Set(["34in_standard_p", "34in_standard_jazz"]);
    const selectedVoicings = new Set(["05_vintage_62_p_alnico", "02_jazz_bass_pair", "09_stingray_mm_parallel"]);
    let vegaView = null;

    function toggleInstrument(id) {{
      if (selectedInstruments.has(id)) {{
        if (selectedInstruments.size > 1) {{
          selectedInstruments.delete(id);
        }}
      }} else {{
        selectedInstruments.add(id);
      }}
      updateUI();
    }}

    function selectAllInstruments() {{
      Object.keys(bakedData.instruments).forEach(id => selectedInstruments.add(id));
      updateUI();
    }}

    function clearInstruments() {{
      selectedInstruments.clear();
      const first = Object.keys(bakedData.instruments)[0];
      if (first) selectedInstruments.add(first);
      updateUI();
    }}

    function selectStandardInstruments() {{
      selectedInstruments.clear();
      Object.keys(bakedData.instruments).forEach(id => {{
        if (id.startsWith("34in")) selectedInstruments.add(id);
      }});
      if (selectedInstruments.size === 0) {{
        const first = Object.keys(bakedData.instruments)[0];
        if (first) selectedInstruments.add(first);
      }}
      updateUI();
    }}

    function selectShortMedInstruments() {{
      selectedInstruments.clear();
      Object.keys(bakedData.instruments).forEach(id => {{
        if (id.startsWith("30in") || id.startsWith("32in")) selectedInstruments.add(id);
      }});
      if (selectedInstruments.size === 0) {{
        const first = Object.keys(bakedData.instruments)[0];
        if (first) selectedInstruments.add(first);
      }}
      updateUI();
    }}

    function toggleVoicing(id) {{
      if (selectedVoicings.has(id)) {{
        if (selectedVoicings.size > 1) {{
          selectedVoicings.delete(id);
        }}
      }} else {{
        selectedVoicings.add(id);
      }}
      updateUI();
    }}

    function selectAllVoicings() {{
      Object.keys(bakedData.voices).forEach(id => selectedVoicings.add(id));
      updateUI();
    }}

    function clearVoicings() {{
      selectedVoicings.clear();
      const first = Object.keys(bakedData.voices)[0];
      if (first) selectedVoicings.add(first);
      updateUI();
    }}

    function filterVoicingsByFamily(family) {{
      selectedVoicings.clear();
      Object.entries(bakedData.voices).forEach(([vid, vinfo]) => {{
        const fam = vinfo.family || '';
        if (family === 'ALL' || fam.toLowerCase().includes(family.toLowerCase()) || (family === 'PJ' && (fam.includes('PJ') || fam.includes('P∕MM')))) {{
          selectedVoicings.add(vid);
        }}
      }});
      if (selectedVoicings.size === 0) {{
        const first = Object.keys(bakedData.voices)[0];
        if (first) selectedVoicings.add(first);
      }}
      updateUI();
    }}

    function buildCurrentRecords() {{
      const records = [];
      const freqs = bakedData.frequencies;
      const n = freqs.length;

      selectedInstruments.forEach(iid => {{
        const instInfo = bakedData.instruments[iid];
        if (!instInfo) return;
        const iname = instInfo.name;

        selectedVoicings.forEach(vid => {{
          const vinfo = bakedData.voices[vid];
          if (!vinfo) return;
          const vname = vinfo.name;

          const resp = bakedData.responses[iid] && bakedData.responses[iid][vid];
          if (!resp) return;
          const mags = resp.magnitude_db;
          const pname = resp.pickup_name;
          const label = `${{iname}} ➔ ${{vname}}`;

          for (let i = 0; i < n; i++) {{
            records.push({{
              frequency: freqs[i],
              magnitude_db: mags[i],
              label: label,
              instrument_name: iname,
              pickup_name: pname,
              voice_name: vname,
              topology: vinfo.topology,
              description: vinfo.description
            }});
          }}
        }});
      }});
      return records;
    }}

    function updateUI() {{
      document.querySelectorAll('#inst-chips .chip-btn').forEach(btn => {{
        btn.classList.toggle('active', selectedInstruments.has(btn.dataset.id));
      }});
      document.querySelectorAll('#voice-chips .chip-btn').forEach(btn => {{
        btn.classList.toggle('active', selectedVoicings.has(btn.dataset.id));
      }});

      const totalCurves = selectedInstruments.size * selectedVoicings.size;
      const counterEl = document.getElementById('curve-counter');
      if (counterEl) {{
        counterEl.textContent = `${{totalCurves}} active curves (${{selectedInstruments.size}} instruments × ${{selectedVoicings.size}} voicings)`;
      }}

      if (vegaView) {{
        const records = buildCurrentRecords();
        vegaView.change('table', vega.changeset().remove(() => true).insert(records)).runAsync();
      }}
    }}

    const baseSpec = {{
      "config": {{
        "view": {{"continuousWidth": 300, "continuousHeight": 300, "strokeWidth": 0}},
        "legend": {{"labelLimit": 340, "orient": "right"}}
      }},
      "data": {{"name": "table"}},
      "layer": [
        {{
          "mark": {{"type": "line", "color": "#8b949e", "strokeDash": [6, 4], "strokeWidth": 1.5}},
          "data": {{"values": [{{"frequency": 20.0, "magnitude_db": 0.0}}, {{"frequency": 20000.0, "magnitude_db": 0.0}}]}},
          "encoding": {{
            "x": {{"field": "frequency", "type": "quantitative"}},
            "y": {{"field": "magnitude_db", "type": "quantitative"}}
          }}
        }},
        {{
          "mark": {{"type": "line", "strokeWidth": 2.4}},
          "params": [{{"name": "curve_select", "select": {{"type": "point", "fields": ["label"]}}, "bind": "legend"}}],
          "encoding": {{
            "x": {{
              "field": "frequency",
              "type": "quantitative",
              "scale": {{"type": "log", "domain": [20, 20000]}},
              "title": "Frequency (Hz)",
              "axis": {{"grid": true, "gridColor": "#333333", "gridDash": [3, 3], "values": [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]}}
            }},
            "y": {{
              "field": "magnitude_db",
              "type": "quantitative",
              "scale": {{"domain": [-24, 24]}},
              "title": "Differential Gain / Cut (dB)",
              "axis": {{"grid": true, "gridColor": "#333333", "gridDash": [3, 3], "values": [-24, -18, -12, -6, 0, 6, 12, 18, 24]}}
            }},
            "color": {{
              "field": "label",
              "type": "nominal",
              "scale": {{"scheme": "tableau20"}},
              "title": "Transformation (Click to isolate)"
            }},
            "opacity": {{
              "condition": {{"param": "curve_select", "value": 0.96}},
              "value": 0.12
            }},
            "strokeWidth": {{
              "condition": {{"param": "curve_select", "value": 3.0}},
              "value": 1.4
            }},
            "tooltip": [
              {{"field": "label", "type": "nominal", "title": "Transformation"}},
              {{"field": "instrument_name", "type": "nominal", "title": "Source Bass"}},
              {{"field": "pickup_name", "type": "nominal", "title": "Source Pickup"}},
              {{"field": "voice_name", "type": "nominal", "title": "Target Voice"}},
              {{"field": "topology", "type": "nominal", "title": "Topology"}},
              {{"field": "frequency", "type": "quantitative", "format": ".1f", "title": "Frequency (Hz)"}},
              {{"field": "magnitude_db", "type": "quantitative", "format": "+.2f", "title": "Differential (dB)"}}
            ]
          }}
        }}
      ],
      "width": 740,
      "height": 480,
      "title": {{
        "text": "Allomorph Master Voices: Baked Transformations (1-Block Single Model)",
        "subtitle": "Monolithic Transfer Functions (H_diff = H_tgt / H_src) from Source Instrument Pickups to Target Voicings",
        "fontSize": 16,
        "subtitleFontSize": 12,
        "anchor": "start"
      }}
    }};

    window.addEventListener('DOMContentLoaded', () => {{
      vegaEmbed('#vis', baseSpec, {{renderer: 'canvas', actions: false}}).then(res => {{
        vegaView = res.view;
        updateUI();
      }}).catch(err => {{
        console.error("VegaEmbed error:", err);
      }});
    }});
  </script>
</body>
</html>
"""
    target_html.write_text(html_content, encoding="utf-8")
    print(f"Saved Baked Transformations master page: {target_html}")
    return target_html


def generate_baked_waterfall_3d_page(target_html: Path | None = None) -> Path:
    """
    Renders the Mode 3 Baked Voicing IR 3D Waterfall & Topography master page.
    Visualizes the linear impulse response (IR) of monolithic baked transformations via
    Cumulative Spectral Decay (CSD) 3D surfaces and multi-voicing catalog 3D landscapes.
    Embeds the compact CSD matrix directly for zero-latency local viewing using WebGL (Plotly.js).
    """
    if target_html is None:
        target_html = RESPONSES_DIR / "baked_waterfall_3d.html"
    target_html = Path(target_html)
    target_html.parent.mkdir(parents=True, exist_ok=True)

    data = build_baked_waterfall_3d_data(num_freqs=50, num_slices=24, max_time_ms=10.0)
    data_json = json.dumps(data)

    default_inst = (
        "30in_emg_mmtw"
        if "30in_emg_mmtw" in data["instruments"]
        else next(iter(data["instruments"].keys()))
    )
    inst_buttons: list[str] = []
    for iid, info in sorted(data["instruments"].items()):
        is_act = " active" if iid == default_inst else ""
        name = info["name"]
        scale = info["scale_in"]
        inst_buttons.append(
            f'        <button class="chip-btn{is_act}" data-id="{iid}" onclick="selectInstrument(\'{iid}\')">'
            f'<span class="chip-check">✓</span><span>{name} ({scale:.0f}")</span></button>'
        )
    inst_chips_html = "\n".join(inst_buttons)

    default_voice = (
        "05_vintage_62_p_alnico"
        if "05_vintage_62_p_alnico" in data["voices"]
        else next(iter(data["voices"].keys()))
    )
    voice_buttons: list[str] = []
    for vid, vinfo in sorted(data["voices"].items()):
        is_act = " active" if vid == default_voice else ""
        vname = vinfo["name"]
        fam = vinfo["family"]
        voice_buttons.append(
            f'        <button class="chip-btn{is_act}" data-id="{vid}" data-family="{fam}" onclick="selectVoicing(\'{vid}\')">'
            f'<span class="chip-check">✓</span><span>{vname}</span></button>'
        )
    voice_chips_html = "\n".join(voice_buttons)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph Master Voices: Baked Voicing IR 3D Waterfall & Topography</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --card-hover: #1c2128;
      --border: #30363d;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --tag-bg: #21262d;
      --btn-active: #1f6feb;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg) !important;
      color: #c9d1d9 !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 12px 16px;
      margin: 0;
      overflow-x: hidden;
    }}
    .header-card {{
      max-width: 1100px;
      margin: 0 auto 12px auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .header-title-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .header-title {{
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge-counter {{
      font-size: 11px;
      font-weight: 700;
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
      border: 1px solid #38bdf8;
      padding: 3px 10px;
      border-radius: 12px;
    }}
    .section-title-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
      border-top: 1px solid var(--border);
      padding-top: 10px;
    }}
    .section-label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
    }}
    .quick-actions {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .quick-btn {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.12s ease;
    }}
    .quick-btn:hover {{
      color: var(--text);
      border-color: #8b949e;
    }}
    .mode-pill-container {{
      display: inline-flex;
      background: #21262d;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 2px;
      gap: 2px;
    }}
    .mode-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 4px 12px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.12s ease;
    }}
    .mode-btn.active {{
      background: var(--btn-active);
      color: #ffffff;
      box-shadow: 0 0 8px rgba(31, 111, 235, 0.4);
    }}
    .chips-container {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip-btn {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.12s ease;
      user-select: none;
    }}
    .chip-btn:hover {{
      background: var(--card-hover);
      color: var(--text);
      border-color: #8b949e;
    }}
    .chip-btn.active {{
      background: rgba(56, 189, 248, 0.12);
      border-color: #38bdf8;
      color: #38bdf8;
      font-weight: 700;
    }}
    .chip-check {{
      font-size: 10px;
      display: inline-block;
      opacity: 0.25;
    }}
    .chip-btn.active .chip-check {{
      opacity: 1;
      color: #38bdf8;
    }}
    .controls-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      background: #11151c;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
    }}
    .control-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
    }}
    .control-select {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      outline: none;
      cursor: pointer;
    }}
    .chart-card {{
      max-width: 1100px;
      margin: 0 auto 12px auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      overflow: hidden;
    }}
    #plotly-vis {{
      width: 100%;
      height: 620px;
    }}
    .waveform-card {{
      max-width: 1100px;
      margin: 0 auto 12px auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .waveform-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
    }}
    .waveform-metrics {{
      display: flex;
      gap: 16px;
      font-size: 11px;
      color: var(--text-muted);
    }}
    .waveform-metrics span {{
      color: #38bdf8;
      font-weight: 600;
    }}
    #fir-canvas {{
      width: 100%;
      height: 100px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    .directives-card {{
      max-width: 1100px;
      margin: 0 auto 16px auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }}
    .directive-item {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
      line-height: 1.45;
      color: var(--text-muted);
    }}
    .directive-title {{
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
    }}
    .highlight {{
      color: #58a6ff;
      font-weight: 600;
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="header-card">
    <div class="header-title-row">
      <div class="header-title">
        <span>Baked Voicing IR 3D Waterfall & Topography (Mode 3 Single-Block)</span>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <a href="baked_responses.html" class="quick-btn" style="text-decoration: none; display: inline-flex; align-items: center; gap: 6px; color: #38bdf8; border-color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">
          <span>&larr; View 2D Curves</span>
        </a>
        <div id="status-badge" class="badge-counter">CSD Waterfall Active</div>
      </div>
    </div>

    <div class="controls-bar">
      <div class="control-group">
        <span>3D DISPLAY MODE:</span>
        <div class="mode-pill-container">
          <button id="btn-mode-csd" class="mode-btn active" onclick="setViewMode('csd')">3D CSD Waterfall (Decay)</button>
          <button id="btn-mode-catalog" class="mode-btn" onclick="setViewMode('catalog')">3D Voicing Landscape (All Voices)</button>
        </div>
      </div>

      <div class="quick-actions">
        <div class="control-group">
          <span>COLORSCALE:</span>
          <select id="select-colorscale" class="control-select" onchange="changeColorScale(this.value)">
            <option value="Viridis" selected>Viridis (Acoustic)</option>
            <option value="Electric">Electric (Cyan / Neon)</option>
            <option value="Turbo">Turbo (High-Contrast)</option>
            <option value="Sunset">Sunset (Warmth)</option>
            <option value="Plasma">Plasma (Dark Neon)</option>
          </select>
        </div>

        <button id="btn-wireframe" class="quick-btn" onclick="toggleWireframe()">Wireframe: OFF</button>
        <button id="btn-contours" class="quick-btn" onclick="toggleContours()">Contours: ON</button>
        <button class="quick-btn" onclick="resetCamera()">Reset Camera</button>
      </div>
    </div>

    <div class="section-title-row">
      <span class="section-label">Source Instrument:</span>
      <div class="quick-actions">
        <button class="quick-btn" onclick="selectInstrument('30in_emg_mmtw')">30" EMG MMTW</button>
        <button class="quick-btn" onclick="selectInstrument('32in_reverse_pmm')">32" Reverse P/MM</button>
        <button class="quick-btn" onclick="selectInstrument('34in_standard_p')">34" Standard P</button>
        <button class="quick-btn" onclick="selectInstrument('34in_standard_jazz')">34" Standard Jazz</button>
        <button class="quick-btn" onclick="selectInstrument('dingwall_ng3_37in')">Dingwall 34-37"</button>
      </div>
    </div>
    <div id="inst-chips" class="chips-container">
{inst_chips_html}
    </div>

    <div class="section-title-row">
      <span id="voice-section-label" class="section-label">Target Voicing (Select to inspect):</span>
      <div class="quick-actions">
        <button class="quick-btn" onclick="filterVoicingsByFamily('Precision')">Precision</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('Jazz')">Jazz</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('StingRay')">StingRay</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('PJ')">PJ & P/MM</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('Character')">Character</button>
        <button class="quick-btn" onclick="filterVoicingsByFamily('All')">Show All</button>
      </div>
    </div>
    <div id="voice-chips" class="chips-container">
{voice_chips_html}
    </div>
  </div>

  <div class="chart-card">
    <div id="plotly-vis"></div>
  </div>

  <div class="waveform-card" id="waveform-panel">
    <div class="waveform-header">
      <span id="waveform-title">2048-Tap Minimum-Phase Impulse Response (h[n] Transient Window: 0 - 2.67 ms)</span>
      <div class="waveform-metrics">
        <div>Initial Polarity: <span id="metric-polarity">+ Positive</span></div>
        <div>Peak Magnitude: <span id="metric-peak">0.00 dBFS</span></div>
        <div>Sample Rate: <span>48 kHz</span></div>
        <div>Taps: <span>2048 (42.7 ms)</span></div>
      </div>
    </div>
    <canvas id="fir-canvas" width="1060" height="100"></canvas>
  </div>

  <div class="directives-card">
    <div class="directive-item">
      <div class="directive-title">1. Cumulative Spectral Decay (CSD) Mechanics</div>
      <div>Slicing the minimum-phase impulse response <span class="highlight">h[n]</span> with a moving onset taper reveals how energy dissipates across time. High-Q passive resonances (e.g., Alnico P-bass or StingRay MM) ring out for <span class="highlight">4&ndash;8 ms</span>, whereas broad active voicings settle in <span class="highlight">&lt; 1 ms</span>.</div>
    </div>
    <div class="directive-item">
      <div class="directive-title">2. Causal Minimum-Phase Settling & Positive Polarity</div>
      <div>Synthesized via Allomorph's vectorized homomorphic Hilbert real-cepstrum engine. Impulse responses feature zero pre-echo, causal onset, and strictly positive initial polarity (+) to prevent comb cancellations when blended in parallel.</div>
    </div>
    <div class="directive-item">
      <div class="directive-title">3. 3D Topographic Catalog Relief</div>
      <div>Switching to <span class="highlight">3D Voicing Landscape</span> stacks target voicings across the depth axis, exposing harmonic notch geometry and spatial comb-filter shifts directly as a continuous physical acoustic terrain.</div>
    </div>
  </div>

  <script id="waterfall-data" type="application/json">
{data_json}
  </script>

  <script>
    const waterfallData = JSON.parse(document.getElementById('waterfall-data').textContent);
    let currentInst = '{default_inst}';
    let currentVoice = '{default_voice}';
    let viewMode = 'csd'; // 'csd' or 'catalog'
    let currentColorScale = 'Viridis';
    let showWireframe = false;
    let showContours = true;
    let lastCamera = null;

    const logTickVals = [1.301, 1.699, 2.0, 2.301, 2.699, 3.0, 3.301, 3.699, 4.0, 4.301];
    const logTickText = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'];
    const logFreqs = waterfallData.frequencies.map(f => Math.log10(f));

    function setViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btn-mode-csd').classList.toggle('active', mode === 'csd');
      document.getElementById('btn-mode-catalog').classList.toggle('active', mode === 'catalog');
      document.getElementById('status-badge').textContent = mode === 'csd' ? 'CSD Waterfall Active' : 'Catalog Landscape Active';
      document.getElementById('waveform-panel').style.display = mode === 'csd' ? 'flex' : 'none';
      render3DPlot();
    }}

    function selectInstrument(id) {{
      currentInst = id;
      document.querySelectorAll('#inst-chips .chip-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.id === id);
      }});
      render3DPlot();
    }}

    function selectVoicing(id) {{
      currentVoice = id;
      document.querySelectorAll('#voice-chips .chip-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.id === id);
      }});
      if (viewMode === 'catalog') {{
        // Switch to CSD mode when clicking a specific voice to inspect its decay
        setViewMode('csd');
      }} else {{
        render3DPlot();
      }}
    }}

    function filterVoicingsByFamily(family) {{
      const chips = document.querySelectorAll('#voice-chips .chip-btn');
      chips.forEach(btn => {{
        if (family === 'All' || btn.dataset.family === family) {{
          btn.style.display = 'inline-flex';
        }} else {{
          btn.style.display = 'none';
        }}
      }});
      if (viewMode === 'catalog') {{
        render3DPlot();
      }}
    }}

    function changeColorScale(scale) {{
      currentColorScale = scale;
      render3DPlot();
    }}

    function toggleWireframe() {{
      showWireframe = !showWireframe;
      document.getElementById('btn-wireframe').textContent = 'Wireframe: ' + (showWireframe ? 'ON' : 'OFF');
      render3DPlot();
    }}

    function toggleContours() {{
      showContours = !showContours;
      document.getElementById('btn-contours').textContent = 'Contours: ' + (showContours ? 'ON' : 'OFF');
      render3DPlot();
    }}

    function resetCamera() {{
      lastCamera = null;
      render3DPlot();
    }}

    function render3DPlot() {{
      const plotDiv = document.getElementById('plotly-vis');
      const instInfo = waterfallData.instruments[currentInst] || {{ name: currentInst }};

      if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.scene) {{
        lastCamera = plotDiv._fullLayout.scene._scene.getCamera();
      }}

      if (viewMode === 'csd') {{
        renderCsdView(plotDiv, instInfo);
      }} else {{
        renderCatalogView(plotDiv, instInfo);
      }}
    }}

    function renderCsdView(plotDiv, instInfo) {{
      const resp = waterfallData.responses[currentInst][currentVoice];
      const vInfo = waterfallData.voices[currentVoice] || {{ name: currentVoice }};
      const csd = resp.csd_matrix;

      const hoverText = [];
      for (let i = 0; i < waterfallData.time_ms.length; i++) {{
        const row = [];
        const t = waterfallData.time_ms[i];
        for (let j = 0; j < waterfallData.frequencies.length; j++) {{
          const f = waterfallData.frequencies[j];
          const db = csd[i][j];
          row.push(`Freq: ${{f.toFixed(1)}} Hz<br>Decay: ${{t.toFixed(2)}} ms<br>Mag: ${{db.toFixed(1)}} dB`);
        }}
        hoverText.push(row);
      }}

      const trace = {{
        type: 'surface',
        x: logFreqs,
        y: waterfallData.time_ms,
        z: csd,
        colorscale: currentColorScale,
        showscale: true,
        colorbar: {{
          title: {{ text: 'dB', font: {{ color: '#8b949e', size: 11 }} }},
          tickfont: {{ color: '#8b949e', size: 10 }},
          len: 0.8,
          thickness: 14
        }},
        contours: {{
          z: {{
            show: showContours,
            usecolormap: true,
            highlightcolor: '#38bdf8',
            project: {{ z: true }}
          }}
        }},
        hidesurface: showWireframe,
        text: hoverText,
        hoverinfo: 'text'
      }};

      const layout = {{
        title: {{
          text: `<b>Cumulative Spectral Decay (CSD):</b> ${{instInfo.name}} ➔ ${{vInfo.name}}`,
          font: {{ color: '#f0f6fc', size: 14 }},
          x: 0.05
        }},
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#161b22',
        margin: {{ l: 20, r: 20, b: 20, t: 40 }},
        scene: {{
          xaxis: {{
            title: {{ text: 'Frequency (Hz)', font: {{ color: '#8b949e', size: 11 }} }},
            tickvals: logTickVals,
            ticktext: logTickText,
            tickfont: {{ color: '#8b949e', size: 9 }},
            gridcolor: '#30363d',
            backgroundcolor: '#0d1117'
          }},
          yaxis: {{
            title: {{ text: 'Decay Time (ms)', font: {{ color: '#8b949e', size: 11 }} }},
            tickfont: {{ color: '#8b949e', size: 9 }},
            gridcolor: '#30363d',
            backgroundcolor: '#0d1117'
          }},
          zaxis: {{
            title: {{ text: 'Magnitude (dB)', font: {{ color: '#8b949e', size: 11 }} }},
            range: [-60, 15],
            tickfont: {{ color: '#8b949e', size: 9 }},
            gridcolor: '#30363d',
            backgroundcolor: '#0d1117'
          }},
          camera: lastCamera || {{ eye: {{ x: -1.75, y: -1.65, z: 1.15 }} }}
        }}
      }};

      Plotly.react(plotDiv, [trace], layout, {{ responsive: true, displayModeBar: false }});
      renderWaveform(resp.fir_waveform, vInfo.name);
    }}

    function renderCatalogView(plotDiv, instInfo) {{
      const instResp = waterfallData.responses[currentInst];
      const visibleChips = Array.from(document.querySelectorAll('#voice-chips .chip-btn'))
        .filter(btn => btn.style.display !== 'none');
      const voiceIds = visibleChips.map(btn => btn.dataset.id);

      const zMatrix = [];
      const yLabels = [];
      const yIndices = [];
      const hoverText = [];

      voiceIds.forEach((vid, idx) => {{
        const r = instResp[vid];
        const vinfo = waterfallData.voices[vid] || {{ name: vid }};
        zMatrix.push(r.magnitude_db);
        yLabels.push(vinfo.name.replace(/ Bass| Passive| Active/g, ''));
        yIndices.push(idx);

        const row = [];
        for (let j = 0; j < waterfallData.frequencies.length; j++) {{
          const f = waterfallData.frequencies[j];
          const db = r.magnitude_db[j];
          row.push(`Voice: ${{vinfo.name}}<br>Freq: ${{f.toFixed(1)}} Hz<br>Mag: ${{db.toFixed(1)}} dB`);
        }}
        hoverText.push(row);
      }});

      const trace = {{
        type: 'surface',
        x: logFreqs,
        y: yIndices,
        z: zMatrix,
        colorscale: currentColorScale,
        showscale: true,
        colorbar: {{
          title: {{ text: 'dB', font: {{ color: '#8b949e', size: 11 }} }},
          tickfont: {{ color: '#8b949e', size: 10 }},
          len: 0.8,
          thickness: 14
        }},
        contours: {{
          z: {{
            show: showContours,
            usecolormap: true,
            highlightcolor: '#38bdf8',
            project: {{ z: true }}
          }}
        }},
        hidesurface: showWireframe,
        text: hoverText,
        hoverinfo: 'text'
      }};

      const layout = {{
        title: {{
          text: `<b>Baked Voicing 3D Topographic Catalog:</b> ${{instInfo.name}} (${{voiceIds.length}} Voicings)`,
          font: {{ color: '#f0f6fc', size: 14 }},
          x: 0.05
        }},
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#161b22',
        margin: {{ l: 20, r: 20, b: 20, t: 40 }},
        scene: {{
          xaxis: {{
            title: {{ text: 'Frequency (Hz)', font: {{ color: '#8b949e', size: 11 }} }},
            tickvals: logTickVals,
            ticktext: logTickText,
            tickfont: {{ color: '#8b949e', size: 9 }},
            gridcolor: '#30363d',
            backgroundcolor: '#0d1117'
          }},
          yaxis: {{
            title: {{ text: 'Target Voicing', font: {{ color: '#8b949e', size: 11 }} }},
            tickvals: yIndices,
            ticktext: yLabels,
            tickfont: {{ color: '#8b949e', size: 8 }},
            gridcolor: '#30363d',
            backgroundcolor: '#0d1117'
          }},
          zaxis: {{
            title: {{ text: 'Magnitude (dB)', font: {{ color: '#8b949e', size: 11 }} }},
            range: [-30, 18],
            tickfont: {{ color: '#8b949e', size: 9 }},
            gridcolor: '#30363d',
            backgroundcolor: '#0d1117'
          }},
          camera: lastCamera || {{ eye: {{ x: -1.75, y: -1.65, z: 1.15 }} }}
        }}
      }};

      Plotly.react(plotDiv, [trace], layout, {{ responsive: true, displayModeBar: false }});
    }}

    function renderWaveform(samples, voiceName) {{
      const canvas = document.getElementById('fir-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // Grid line at y = 0
      const midY = h / 2;
      ctx.strokeStyle = '#21262d';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(w, midY);
      ctx.stroke();

      if (!samples || samples.length === 0) return;

      const maxVal = Math.max(...samples.map(Math.abs), 0.001);
      const scaleY = (midY - 10) / maxVal;

      document.getElementById('metric-peak').textContent = (20 * Math.log10(maxVal)).toFixed(2) + ' dBFS';
      document.getElementById('metric-polarity').textContent = samples[0] >= 0 ? '+ Positive' : '- Inverted';

      // Draw glowing waveform
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 6;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;

      ctx.beginPath();
      const stepX = w / (samples.length - 1);
      for (let i = 0; i < samples.length; i++) {{
        const x = i * stepX;
        const y = midY - (samples[i] * scaleY);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.shadowBlur = 0;
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      render3DPlot();
    }});
  </script>
</body>
</html>
"""
    target_html.write_text(html_content, encoding="utf-8")
    print(f"Saved Baked Voicing IR 3D Waterfall page: {target_html}")
    return target_html


def generate_composite_instrument_chart(
    instrument: InstrumentConfig | str = "30in",
    out_html: str | Path | None = None,
) -> Path:
    """
    Renders the Signal Flow Inspector chart:
      1. Source Bass Input (Entering Block 1, relative to Canonical Intermediate datum)
      2. Block 1 Deconvolution (Deconvolution FIR filter with Wiener regularization)
      3. Canonical Intermediate (0 dB Neutral Baseline Datum)
      4. Block 2 Target Voicing (Universal target transfer function from Canonical datum)
      5. Target Voice Output (Authentic target voice response)
    Illustrates: Source Bass Input + Block 1 Deconvolution = Canonical Intermediate (0 dB) -> Block 2 Target Voicing -> Target Voice Output.
    """
    inst = instrument if isinstance(instrument, InstrumentConfig) else load_instrument(instrument)
    inst_id = inst.id
    inst_name = inst.name

    if out_html is None:
        target_path = RESPONSES_DIR / f"{inst_id}.html"
    else:
        p = Path(out_html)
        if p.is_dir() or (not p.suffix and not p.exists()):
            target_path = p / f"{inst_id}.html"
        else:
            target_path = p
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    master_df = build_composite_instrument_dataframe(inst)
    voice_names = [v for v in master_df["voice_name"].unique().sort().to_list() if v]
    pickup_names = [p for p in master_df["pickup_name"].unique().sort().to_list() if p]

    default_pickup_key = inst.default_pickup or (
        next(iter(inst.pickups.keys())) if inst.pickups else ""
    )
    default_pickup_cfg = inst.pickups.get(default_pickup_key)
    default_pickup = (
        default_pickup_cfg.name
        if default_pickup_cfg and default_pickup_cfg.name in pickup_names
        else (pickup_names[0] if pickup_names else "")
    )

    matching_voice_name: str | None = None
    for vid, vcfg in sorted(VOICES.items()):
        if vid == "00_canonical_intermediate":
            continue
        if inst.pickup_mapping.get(vid, inst.default_pickup) != default_pickup_key:
            continue
        if not is_voice_matching_source(inst, vid, vcfg):
            continue
        if default_pickup_cfg and default_pickup_cfg.circuit and vcfg.circuit:
            src_m = load_circuit(default_pickup_cfg.circuit)
            tgt_m = load_circuit(vcfg.circuit)
            diff_c = compute_differential_circuit_transfer_functions(tgt_m, src_m, freqs=FREQS)
            if np.allclose(diff_c[0], 1.0, rtol=1e-3):
                matching_voice_name = vcfg.name
                break
        elif not (default_pickup_cfg and default_pickup_cfg.circuit) and not vcfg.circuit:
            matching_voice_name = vcfg.name
            break

    default_voice = (
        matching_voice_name
        if matching_voice_name and matching_voice_name in voice_names
        else (voice_names[0] if voice_names else "")
    )

    voice_select = alt.selection_point(
        fields=["voice_name"],
        bind=alt.binding_select(options=voice_names, name="Target Voicing (Block 2): "),
        value=default_voice,
    )

    stage_selection = alt.selection_point(fields=["stage"], bind="legend")

    stage_order = [
        "1. Source Bass Input",
        "2. Block 1 Deconvolution",
        "3. Canonical Intermediate (0 dB)",
        "4. Block 2 Target Voicing",
        "5. Target Voice Output",
    ]
    color_scale = alt.Scale(
        domain=stage_order, range=["#38bdf8", "#26a69a", "#8b949e", "#ff7043", "#ffd54f"]
    )
    dash_scale = alt.Scale(domain=stage_order, range=[[0], [3, 3], [6, 4], [8, 4], [0]])

    base_chart = (
        alt.Chart(master_df)
        .mark_line()
        .encode(
            x=alt.X(
                "frequency:Q",
                scale=alt.Scale(type="log", domain=[20, 20000]),
                title="Frequency (Hz)",
                axis=alt.Axis(
                    values=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                    grid=True,
                    gridDash=[3, 3],
                    gridColor="#333333",
                ),
            ),
            y=alt.Y(
                "magnitude_db:Q",
                scale=alt.Scale(domain=[-24, 24]),
                title="Magnitude / Gain relative to Canonical Intermediate (dB)",
                axis=alt.Axis(
                    values=[-24, -18, -12, -6, 0, 6, 12, 18, 24],
                    grid=True,
                    gridDash=[3, 3],
                    gridColor="#333333",
                ),
            ),
            color=alt.Color(
                "stage:O",
                scale=color_scale,
                sort=stage_order,
                title="Signal Flow Stage (Click to isolate)",
            ),
            strokeDash=alt.StrokeDash(
                "stage:O",
                scale=dash_scale,
                sort=stage_order,
                title="Signal Flow Stage (Click to isolate)",
            ),
            opacity=alt.condition(stage_selection, alt.value(0.96), alt.value(0.12)),
            strokeWidth=alt.StrokeWidth(
                "stage:O",
                scale=alt.Scale(domain=stage_order, range=[2.2, 1.8, 1.5, 1.8, 3.2]),
                sort=stage_order,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("stage:O", title="Signal Stage"),
                alt.Tooltip("pickup_name:N", title="Source Pickup"),
                alt.Tooltip("voice_name:N", title="Target Voicing"),
                alt.Tooltip("frequency:Q", title="Frequency (Hz)", format=".1f"),
                alt.Tooltip("magnitude_db:Q", title="Magnitude (dB)", format="+.1f"),
            ],
        )
    )

    pred_stages_123 = alt.FieldOneOfPredicate(
        field="stage",
        oneOf=[
            "1. Source Bass Input",
            "2. Block 1 Deconvolution",
            "3. Canonical Intermediate (0 dB)",
        ],
    )
    pred_stage_4 = alt.FieldEqualPredicate(field="stage", equal="4. Block 2 Target Voicing")
    pred_stage_5 = alt.FieldEqualPredicate(field="stage", equal="5. Target Voice Output")

    if len(pickup_names) > 1:
        pickup_select = alt.selection_point(
            fields=["pickup_name"],
            bind=alt.binding_select(options=pickup_names, name="Source Pickup (Block 1): "),
            value=default_pickup,
        )
        filter_comp = (
            (pickup_select & pred_stages_123)
            | (voice_select & pred_stage_4)
            | (pickup_select & voice_select & pred_stage_5)
        )
        chart = base_chart.add_params(
            voice_select, pickup_select, stage_selection
        ).transform_filter(filter_comp)
    else:
        filter_comp = (
            pred_stages_123 | (voice_select & pred_stage_4) | (voice_select & pred_stage_5)
        )
        chart = base_chart.add_params(voice_select, stage_selection).transform_filter(filter_comp)

    chart = (
        chart.properties(
            title=alt.TitleParams(
                text=f"Allomorph Master Voices: Signal Flow Inspector ({inst_name})",
                subtitle="Signal Flow (All Relative to Canonical Intermediate): Source Bass Input ➔ [Block 1 Deconvolution] ➔ Canonical Intermediate (0 dB) ➔ [Block 2 Voicing] ➔ Target Voice Output",
                fontSize=16,
                subtitleFontSize=12,
                anchor="start",
            ),
            width=740,
            height=480,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(orient="right", labelLimit=320)
        .interactive()
    )

    chart.save(str(target_path))

    spec_panel = """
<style>
  body {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 12px 16px;
    margin: 0;
    overflow-x: hidden;
    box-sizing: border-box;
  }
  #vis {
    display: flex;
    justify-content: center;
    width: 100%;
    overflow-x: hidden;
  }
  .composite-banner {
    max-width: 1060px;
    margin: 14px auto 0 auto;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    font-size: 11px;
  }
  .comp-item { display: flex; flex-direction: column; gap: 4px; }
  .comp-title { font-weight: 700; display: flex; align-items: center; gap: 6px; }
  .dot-cyan { width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; display: inline-block; }
  .dot-teal { width: 8px; height: 8px; border-radius: 50%; background: #26a69a; display: inline-block; }
  .dot-gray { width: 8px; height: 8px; border-radius: 50%; background: #8b949e; display: inline-block; }
  .dot-orange { width: 8px; height: 8px; border-radius: 50%; background: #ff7043; display: inline-block; }
  .dot-gold { width: 8px; height: 8px; border-radius: 50%; background: #ffd54f; display: inline-block; }
  .comp-desc { color: #8b949e; line-height: 1.4; }
</style>
<div class="composite-banner">
  <div class="comp-item">
    <div class="comp-title"><span class="dot-cyan"></span> 1. Source Bass Input</div>
    <div class="comp-desc">Cyan curve: Physical acoustic aperture and RLC response of the selected source pickup entering Block 1 (relative to Canonical Intermediate Datum).</div>
  </div>
  <div class="comp-item">
    <div class="comp-title"><span class="dot-teal"></span> 2. Block 1 Deconvolution</div>
    <div class="comp-desc">Dotted Teal curve: 2048-tap FIR deconvolution filter (H<sub>front</sub> = H<sub>can</sub> / H<sub>src</sub>) with Wiener regularization & HF clamping neutralizing source pickup to Canonical Intermediate.</div>
  </div>
  <div class="comp-item">
    <div class="comp-title"><span class="dot-gray"></span> 3. Canonical Intermediate (0 dB)</div>
    <div class="comp-desc">Dashed Gray line: Standardized neutral 0.00 dB baseline datum achieved when Source Input passes through Block 1 (Stage 1 + Stage 2 = 0 dB).</div>
  </div>
  <div class="comp-item">
    <div class="comp-title"><span class="dot-orange"></span> 4. Block 2 Target Voicing</div>
    <div class="comp-desc">Dashed Orange curve: Universal target transfer function (H<sub>back</sub> = H<sub>tgt</sub> / H<sub>can</sub>) applied by Block 2 NAM relative to Canonical Intermediate.</div>
  </div>
  <div class="comp-item">
    <div class="comp-title"><span class="dot-gold"></span> 5. Target Voice Output</div>
    <div class="comp-desc">Solid Gold curve: Authentic target acoustic voice produced after Block 2 processing (Canonical Baseline + Target Voicing).</div>
  </div>
</div>
"""
    append_spec_panel(target_path, spec_panel)
    print(f"Saved Signal Flow Inspector chart: {target_path}")
    return target_path


def generate_interactive_chart(
    instrument: InstrumentConfig | str = "30in",
    out_html: str | Path | None = None,
    mode: str = "composite",
) -> Path:
    """
    Calculates voice responses and renders an interactive Altair chart.
    mode:
      - 'composite': 3-curve overlay (Frontend IR + Backend Voicing = Composite Result).
      - 'unified': embeds both Output Voice and Input/Output Difference curves with interactive radio buttons.
      - 'output': standalone chart strictly plotting the 12 target Output Voice curves.
      - 'difference': standalone chart strictly plotting the Input/Output Difference curves for this instrument.
      - 'targets': standalone chart strictly plotting the Universal Target curves.
      - 'frontends': standalone chart strictly plotting the Frontend Deconvolution curves.
    """
    if mode == "targets":
        return generate_universal_targets_chart(
            target_path=Path(out_html) if out_html is not None else None
        )
    elif mode == "frontends":
        return generate_frontend_deconvolutions_chart(
            target_path=Path(out_html) if out_html is not None else None
        )
    elif mode == "baked":
        return generate_baked_responses_page(
            target_html=Path(out_html) if out_html is not None else None
        )

    inst = instrument if isinstance(instrument, InstrumentConfig) else load_instrument(instrument)
    inst_id = inst.id
    inst_name = inst.name

    if out_html is None:
        target_path = RESPONSES_DIR / f"{inst_id}.html"
    else:
        p = Path(out_html)
        if p.is_dir() or (not p.suffix and not p.exists()):
            target_path = p / f"{inst_id}.html"
        else:
            target_path = p

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "composite":
        return generate_composite_instrument_chart(inst, out_html=target_path)

    print(f"Computing voice frequency responses (mode={mode}, instrument={inst_name})...")
    if mode == "output":
        dfs = [
            build_voice_dataframe(vid, cfg, instrument=inst, mode="output")
            for vid, cfg in VOICES.items()
        ]
        master_df = pl.concat(dfs)
        chart_title = "Allomorph Master Voices: Output Voice Frequency Responses"
        chart_subtitle = f"Target Passive Acoustic Apertures & SPICE Loaded RLC Resonances (Reference: {inst_name})"
        y_title = "Normalized Output Magnitude (dB)"
        y_domain = [-30, 10]
    elif mode == "difference":
        dfs = [
            build_voice_dataframe(vid, cfg, instrument=inst, mode="difference")
            for vid, cfg in VOICES.items()
        ]
        master_df = pl.concat(dfs)
        chart_title = "Allomorph Master Voices: Input/Output Differential Transfer Functions"
        chart_subtitle = f'Source: {inst_name} -> Target: 34" Standard & 37" Multi-Scale Datums (Δ Transfer Filter)'
        y_title = "Differential Transfer Magnitude (dB)"
        y_domain = [-28, 15]
    else:  # mode == "unified"
        dfs_out = [
            build_voice_dataframe(vid, cfg, instrument=inst, mode="output", include_mode_col=True)
            for vid, cfg in VOICES.items()
        ]
        dfs_diff = [
            build_voice_dataframe(
                vid, cfg, instrument=inst, mode="difference", include_mode_col=True
            )
            for vid, cfg in VOICES.items()
        ]
        master_df = pl.concat(dfs_diff + dfs_out)
        chart_title = "Allomorph Master Voices: Acoustic & Electrical Response Curves"
        chart_subtitle = f"Interactive View ({inst_name}) — Switch between Input/Output Difference and Output Voice"
        y_title = "Normalized Magnitude / Differential Gain (dB)"
        y_domain = [-30, 15]

    print("Rendering interactive chart using Altair...")
    return render_chart_to_file(
        master_df, target_path, chart_title, chart_subtitle, y_title, y_domain, mode=mode
    )


def generate_all_charts(output_dir: str | Path | None = None) -> dict[str, Path]:
    """Generates standalone Altair interactive charts for all configured instruments and Architecture C master views."""
    out_dir = Path(output_dir) if output_dir else RESPONSES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    all_insts = load_all_instruments()

    # 1. Mode 1: Universal Target Voicings (Block 2)
    print("Generating Universal Target Voicings (Block 2)...")
    generate_universal_targets_chart(out_dir / "universal_targets.html")

    # 2. Mode 2: Frontend Deconvolutions (Block 1)
    print("Generating Frontend Deconvolutions (Block 1)...")
    generate_frontend_deconvolutions_chart(out_dir / "frontend_deconvolutions.html")

    # 3. Mode 3: Baked Transformations (1-Block Single Model)
    print("Generating Baked Transformations (1-Block Single Model)...")
    baked_html = generate_baked_responses_page(out_dir / "baked_responses.html")

    # 4. Mode 3 3D: Baked Voicing IR 3D Waterfall & Topography
    print("Generating Baked Voicing IR 3D Waterfall & Topography...")
    waterfall_3d_html = generate_baked_waterfall_3d_page(out_dir / "baked_waterfall_3d.html")

    generated: dict[str, Path] = {
        "baked": baked_html,
        "baked_waterfall_3d": waterfall_3d_html,
    }
    for inst_id, inst_cfg in all_insts.items():
        if inst_id == "canonical_intermediate":
            continue
        inst_name = inst_cfg.name

        # Signal Flow Inspector (End-to-End: Bass Input -> Deconv -> Canonical (0 dB) -> Voicing -> Target Output)
        print(f"Generating Signal Flow Inspector for {inst_name}...")
        out_composite_file = out_dir / f"{inst_id}.html"
        generate_composite_instrument_chart(inst_cfg, out_html=out_composite_file)
        generated[inst_id] = out_composite_file

    generate_portal_pages(output_dir=out_dir)
    return generated
