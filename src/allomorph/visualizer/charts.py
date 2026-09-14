"""
Allomorph Visualizer - Interactive Altair Charts Generation
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import altair as alt
import polars as pl

from allomorph.config.instruments import load_instrument
from allomorph.config.schema import InstrumentConfig
from allomorph.config.voices import VOICES
from allomorph.visualizer.dataframe import (
    build_voice_dataframe,
    build_voicing_ir_diff_3d_data,
    build_voicings_comparison_data,
)
from allomorph.visualizer.portal import RESPONSES_DIR, generate_portal_pages

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


def generate_voicings_page(target_html: Path | str | None = None) -> Path:
    """
    Renders the interactive 2D Voicing Comparison page (voicings.html):
    Allows selecting any Source Voicing and Target Voicing from the 28 standard catalog voicings in VOICES.
    Displays an interactive 3-line graph:
      1. Source Voicing (H_src)
      2. Target Voicing (H_tgt)
      3. Normalized Difference (Norm. Diff = H_tgt,norm - H_src,norm)
    When Source == Target, the differential line evaluates to exact 0.00 dB.
    Includes live metrics, preset comparisons, and zero-scroll responsiveness.
    """
    if target_html is None:
        target_html = RESPONSES_DIR / "voicings.html"
    target_html = Path(target_html)
    target_html.parent.mkdir(parents=True, exist_ok=True)

    voicings_data = build_voicings_comparison_data(step=1)
    data_json = json.dumps(voicings_data, separators=(",", ":"))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph | Voicing Comparisons (3-Line Response)</title>
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
      --teal: #10b981;
      --amber: #f59e0b;
      --blue: #38bdf8;
      --red: #f43f5e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 16px;
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .header-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header-title-col {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .header-title {{
      font-size: 18px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 2px 8px;
      border-radius: 12px;
    }}
    .header-sub {{
      font-size: 12px;
      color: var(--text-muted);
    }}
    .header-links {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .nav-btn {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .nav-btn:hover {{
      background: var(--card-hover);
      color: var(--text);
      border-color: #8b949e;
    }}
    .nav-btn.primary {{
      color: var(--accent);
      border-color: var(--accent);
    }}
    .control-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .selectors-row {{
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 12px;
    }}
    .selector-box {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .selector-label {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .dot-blue {{ width: 8px; height: 8px; border-radius: 50%; background: var(--blue); display: inline-block; }}
    .dot-amber {{ width: 8px; height: 8px; border-radius: 50%; background: var(--amber); display: inline-block; }}
    .dot-green {{ width: 8px; height: 8px; border-radius: 50%; background: var(--teal); display: inline-block; }}
    select {{
      background: #0d1117;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      outline: none;
      width: 100%;
    }}
    select:focus {{
      border-color: var(--accent);
    }}
    .btn-swap {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 16px;
      font-weight: 700;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      margin-top: 18px;
      transition: all 0.15s ease;
    }}
    .btn-swap:hover {{
      background: var(--card-hover);
      border-color: var(--accent);
      color: var(--accent);
      transform: rotate(180deg);
    }}
    .presets-row {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .preset-label {{
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      margin-right: 4px;
    }}
    .preset-chip {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .preset-chip:hover {{
      background: var(--card-hover);
      color: var(--text);
      border-color: #8b949e;
    }}
    .metrics-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 10px;
    }}
    .metric-card {{
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .metric-name {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      font-weight: 700;
    }}
    .metric-val {{
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
    }}
    .metric-val.boost {{ color: #38bdf8; }}
    .metric-val.cut {{ color: #f43f5e; }}
    .metric-val.identity {{ color: #10b981; }}
    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }}
    #plot-div {{
      width: 100%;
      height: 520px;
    }}
    .specs-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .spec-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
    }}
    .spec-header {{
      font-size: 13px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .spec-desc {{
      color: var(--text-muted);
      line-height: 1.4;
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="container">
    <div class="header-card">
      <div class="header-title-col">
        <div class="header-title">
          <span>Allomorph Voicing Comparisons</span>
          <span class="badge">3-Line Direct Forward Twin</span>
        </div>
        <div class="header-sub">
          Source Voicing (H<sub>src</sub>) &times; Normalized Difference (Norm. Diff = H&#770;<sub>tgt</sub> / H&#770;<sub>src</sub>) = Target Voicing (H<sub>tgt</sub>)
        </div>
      </div>
      <div class="header-links">
        <a href="index.html" class="nav-btn">&larr; Master Portal</a>
        <a href="voicing_ir_3d.html" class="nav-btn primary">View 3D IR Waterfall &rarr;</a>
      </div>
    </div>

    <div class="control-card">
      <div class="selectors-row">
        <div class="selector-box">
          <div class="selector-label" style="color: var(--blue);">
            <span class="dot-blue"></span> Source Voicing (Input)
          </div>
          <select id="source-select" onchange="onSelectionChanged()"></select>
        </div>

        <button id="btn-swap" class="btn-swap" onclick="swapVoicings()" title="Swap Source & Target">&#x21C6;</button>

        <div class="selector-box">
          <div class="selector-label" style="color: var(--amber);">
            <span class="dot-amber"></span> Target Voicing (Output)
          </div>
          <select id="target-select" onchange="onSelectionChanged()"></select>
        </div>
      </div>

      <div class="presets-row">
        <span class="preset-label">Quick Comparisons:</span>
        <button class="preset-chip" onclick="applyPreset('precision_vintage', 'jazz_bridge_growl')">P vs J Bridge</button>
        <button class="preset-chip" onclick="applyPreset('precision_vintage', 'stingray_parallel')">P vs StingRay</button>
        <button class="preset-chip" onclick="applyPreset('stingray_parallel', 'jazz_bridge_growl')">StingRay vs J Bridge</button>
        <button class="preset-chip" onclick="applyPreset('pj_passive', 'pj_active')">Passive vs Active PJ</button>
        <button class="preset-chip" onclick="applyPreset('upright_acoustic', 'studio_direct')">Upright vs Studio DI</button>
        <button class="preset-chip" onclick="applyPreset('precision_vintage', 'precision_vintage')">Identity (Reset)</button>
      </div>

      <div class="metrics-row">
        <div class="metric-card">
          <div class="metric-name">Norm. Diff Max Boost</div>
          <div class="metric-val boost" id="metric-boost">+0.0 dB</div>
        </div>
        <div class="metric-card">
          <div class="metric-name">Norm. Diff Max Cut</div>
          <div class="metric-val cut" id="metric-cut">-0.0 dB</div>
        </div>
        <div class="metric-card">
          <div class="metric-name">Norm. Diff Dynamic Range</div>
          <div class="metric-val" id="metric-range">0.0 dB</div>
        </div>
        <div class="metric-card">
          <div class="metric-name">Digital Twin Status</div>
          <div class="metric-val identity" id="metric-status">Identity Match</div>
        </div>
      </div>
    </div>

    <div class="chart-card">
      <div id="plot-div"></div>
    </div>

    <div class="specs-grid">
      <div class="spec-card">
        <div class="spec-header" style="color: var(--blue);">
          <span id="src-spec-title">Source Voicing</span>
          <span class="badge" id="src-spec-topo">Topology</span>
        </div>
        <div class="spec-desc" id="src-spec-desc">Description</div>
      </div>
      <div class="spec-card">
        <div class="spec-header" style="color: var(--amber);">
          <span id="tgt-spec-title">Target Voicing</span>
          <span class="badge" id="tgt-spec-topo">Topology</span>
        </div>
        <div class="spec-desc" id="tgt-spec-desc">Description</div>
      </div>
    </div>
  </div>

  <script id="voicings-data" type="application/json">
{data_json}
  </script>

  <script>
    const data = JSON.parse(document.getElementById('voicings-data').textContent);
    const freqs = data.frequencies;
    const voices = data.voices;
    const families = data.families;

    const sourceSelect = document.getElementById('source-select');
    const targetSelect = document.getElementById('target-select');

    function populateSelect(selectEl, selectedId) {{
      selectEl.innerHTML = '';
      families.forEach(fam => {{
        const famVoices = Object.values(voices).filter(v => v.family === fam);
        if (famVoices.length === 0) return;
        const optgroup = document.createElement('optgroup');
        optgroup.label = fam;
        famVoices.forEach(v => {{
          const opt = document.createElement('option');
          opt.value = v.id;
          opt.textContent = `${{v.name}} (${{v.topology}})`;
          if (v.id === selectedId) opt.selected = true;
          optgroup.appendChild(opt);
        }});
        selectEl.appendChild(optgroup);
      }});
    }}

    populateSelect(sourceSelect, data.default_source || 'precision_vintage');
    populateSelect(targetSelect, data.default_target || 'jazz_bridge_growl');

    function applyPreset(srcId, tgtId) {{
      sourceSelect.value = srcId;
      targetSelect.value = tgtId;
      renderChart();
    }}

    function swapVoicings() {{
      const temp = sourceSelect.value;
      sourceSelect.value = targetSelect.value;
      targetSelect.value = temp;
      renderChart();
    }}

    function onSelectionChanged() {{
      renderChart();
    }}

    function cinfSmoothstep(t) {{
      if (t <= 0.0) return 0.0;
      if (t >= 1.0) return 1.0;
      let arg = (1.0 - 2.0 * t) / (t * (1.0 - t));
      arg = Math.min(Math.max(arg, -80.0), 80.0);
      return 1.0 / (1.0 + Math.exp(arg));
    }}

    function computeCurveRmsDb(mags) {{
      let sum = 0.0;
      for (let i = 0; i < mags.length; i++) {{
        sum += Math.pow(10.0, mags[i] / 10.0);
      }}
      return 10.0 * Math.log10(Math.max(sum / mags.length, 1e-12));
    }}

    function smoothSoftKneeDb(x, thresh, ceiling, alpha) {{
      const w = Math.max(ceiling - thresh, 1e-6);
      const z = alpha * (x - thresh);
      const excess = (z > 50.0 ? z : (z < -50.0 ? 0.0 : Math.log1p(Math.exp(z)))) / alpha;
      return x - excess + w * Math.tanh(excess / w);
    }}

    function renderChart() {{
      const srcId = sourceSelect.value;
      const tgtId = targetSelect.value;

      const src = voices[srcId];
      const tgt = voices[tgtId];

      const srcMags = src.magnitude_db;
      const tgtMags = tgt.magnitude_db;
      const srcRms = (typeof src.rms_db === 'number') ? src.rms_db : computeCurveRmsDb(srcMags);
      const tgtRms = (typeof tgt.rms_db === 'number') ? tgt.rms_db : computeCurveRmsDb(tgtMags);

      const diffMags = [];
      let maxBoost = -999;
      let maxBoostFreq = 0;
      let maxCut = 999;
      let maxCutFreq = 0;

      const isIdentity = (srcId === tgtId);

      if (isIdentity) {{
        for (let i = 0; i < freqs.length; i++) {{
          diffMags.push(0.0);
        }}
        maxBoost = 0.0;
        maxBoostFreq = freqs[0];
        maxCut = 0.0;
        maxCutFreq = freqs[0];
      }} else {{
        const hSrc = [];
        const hTgt = [];
        const pIn = [];
        let maxPIn = 1e-12;
        const fC = 1200.0;
        const snrDb = 35.0;
        const maxBoostDb = 7.0;

        for (let i = 0; i < freqs.length; i++) {{
          const sn = srcMags[i] - srcRms;
          const tn = tgtMags[i] - tgtRms;
          const hs = Math.pow(10.0, sn / 20.0);
          const ht = Math.pow(10.0, tn / 20.0);
          hSrc.push(hs);
          hTgt.push(ht);
          const sDry = 1.0 / (1.0 + Math.pow(freqs[i] / fC, 2));
          const pin = hs * hs * sDry;
          pIn.push(pin);
          if (pin > maxPIn) maxPIn = pin;
        }}

        const eps = Math.pow(10.0, -snrDb / 10.0) * maxPIn;
        const kneeWidth = Math.min(2.5, maxBoostDb / 2.0);
        const thresh = maxBoostDb - kneeWidth;

        for (let i = 0; i < freqs.length; i++) {{
          const sDry = 1.0 / (1.0 + Math.pow(freqs[i] / fC, 2));
          const hNam = (hTgt[i] * hSrc[i] * sDry) / (pIn[i] + eps);
          const dbNam = 20.0 * Math.log10(Math.max(hNam, 1e-3));
          let d = smoothSoftKneeDb(dbNam, thresh, maxBoostDb, 2.0);
          d = Math.round(d * 100) / 100;
          diffMags.push(d);

          if (freqs[i] <= 8000.0) {{
            if (d > maxBoost) {{ maxBoost = d; maxBoostFreq = freqs[i]; }}
            if (d < maxCut) {{ maxCut = d; maxCutFreq = freqs[i]; }}
          }}
        }}
      }}

      const dynRange = Math.round((maxBoost - maxCut) * 10) / 10;

      document.getElementById('metric-boost').textContent = (maxBoost >= 0 ? '+' : '') + maxBoost.toFixed(1) + ' dB @ ' + Math.round(maxBoostFreq) + ' Hz';
      document.getElementById('metric-cut').textContent = (maxCut >= 0 ? '+' : '') + maxCut.toFixed(1) + ' dB @ ' + Math.round(maxCutFreq) + ' Hz';
      document.getElementById('metric-range').textContent = isIdentity ? '0.0 dB (Flat)' : dynRange.toFixed(1) + ' dB';
      document.getElementById('metric-status').textContent = isIdentity ? 'Identity Match (0.00 dB)' : 'Normalized Transform';
      document.getElementById('metric-status').className = 'metric-val ' + (isIdentity ? 'identity' : 'boost');

      document.getElementById('src-spec-title').textContent = src.name;
      document.getElementById('src-spec-topo').textContent = src.topology;
      document.getElementById('src-spec-desc').textContent = `${{src.description}} (fr: ${{src.fr.toFixed(0)}} Hz, Q: ${{src.q.toFixed(2)}})`;

      document.getElementById('tgt-spec-title').textContent = tgt.name;
      document.getElementById('tgt-spec-topo').textContent = tgt.topology;
      document.getElementById('tgt-spec-desc').textContent = `${{tgt.description}} (fr: ${{tgt.fr.toFixed(0)}} Hz, Q: ${{tgt.q.toFixed(2)}})`;

      const traceSource = {{
        x: freqs,
        y: srcMags,
        name: `Source: ${{src.name}}`,
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#38bdf8', width: 2.2 }},
        hovertemplate: `Source: %{{y:+.1f}} dB<extra></extra>`
      }};

      const traceTarget = {{
        x: freqs,
        y: tgtMags,
        name: `Target: ${{tgt.name}}`,
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#f59e0b', width: 2.2, dash: 'solid' }},
        hovertemplate: `Target: %{{y:+.1f}} dB<extra></extra>`
      }};

      const traceDiff = {{
        x: freqs,
        y: diffMags,
        name: `Norm. Diff`,
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#10b981', width: 3.0 }},
        hovertemplate: `Norm. Diff: %{{y:+.1f}} dB<extra></extra>`
      }};

      const traceZero = {{
        x: [20, 20000],
        y: [0, 0],
        name: '0 dB Reference',
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#4b5563', width: 1, dash: 'dash' }},
        hoverinfo: 'none',
        showlegend: false
      }};

      const layout = {{
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#161b22',
        margin: {{ l: 55, r: 25, t: 30, b: 45 }},
        hovermode: 'x unified',
        xaxis: {{
          type: 'log',
          range: [Math.log10(20), Math.log10(20000)],
          title: {{ text: 'Frequency (Hz)', font: {{ color: '#8b949e', size: 11 }} }},
          tickvals: [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
          ticktext: ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'],
          tickfont: {{ color: '#8b949e', size: 10 }},
          gridcolor: '#30363d',
          linecolor: '#30363d'
        }},
        yaxis: {{
          title: {{ text: 'Magnitude / Normalized Difference (dB)', font: {{ color: '#8b949e', size: 11 }} }},
          range: [-18, 18],
          tickvals: [-18, -12, -6, 0, 6, 12, 18],
          tickfont: {{ color: '#8b949e', size: 10 }},
          gridcolor: '#30363d',
          linecolor: '#30363d'
        }},
        legend: {{
          orientation: 'h',
          x: 0,
          y: 1.12,
          font: {{ color: '#f0f6fc', size: 11 }},
          bgcolor: 'rgba(0,0,0,0)'
        }}
      }};

      Plotly.react('plot-div', [traceZero, traceSource, traceTarget, traceDiff], layout, {{ responsive: true, displayModeBar: false }});
    }}

    window.addEventListener('DOMContentLoaded', renderChart);
  </script>
</body>
</html>
"""
    target_html.write_text(html_content, encoding="utf-8")
    print(f"Saved Voicings Comparison page: {target_html}")
    return target_html


def generate_voicing_ir_3d_page(target_html: Path | str | None = None) -> Path:
    """
    Renders the interactive WebGL 3D Voicing IR Difference Waterfall page (voicing_ir_3d.html):
    Allows selecting any Source Voicing and Target Voicing from VOICES.
    Displays a 3D Cumulative Spectral Decay (CSD) waterfall surface of the difference impulse response (h_diff(t))
    and the 128-tap time-domain FIR waveform.
    When Source == Target, the IR collapses to an ideal unit delta impulse and flat CSD.
    """
    if target_html is None:
        target_html = RESPONSES_DIR / "voicing_ir_3d.html"
    target_html = Path(target_html)
    target_html.parent.mkdir(parents=True, exist_ok=True)

    waterfall_data = build_voicing_ir_diff_3d_data(num_freqs=50, num_slices=24, max_time_ms=3.0)
    data_json = json.dumps(waterfall_data, separators=(",", ":"))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph | Voicing IR Difference 3D Waterfall & Waveform</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --accent: #38bdf8;
      --text: #f0f6fc;
      --text-muted: #8b949e;
      --tag-bg: #21262d;
      --teal: #10b981;
      --amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 16px;
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .header-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header-title-col {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .header-title {{
      font-size: 18px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 2px 8px;
      border-radius: 12px;
    }}
    .header-sub {{
      font-size: 12px;
      color: var(--text-muted);
    }}
    .nav-btn {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .nav-btn:hover {{
      background: #1c2128;
      color: var(--text);
      border-color: #8b949e;
    }}
    .control-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .selectors-row {{
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 12px;
    }}
    .selector-box {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .selector-label {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    select {{
      background: #0d1117;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      outline: none;
      width: 100%;
    }}
    .btn-swap {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 16px;
      font-weight: 700;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      margin-top: 18px;
      transition: all 0.15s ease;
    }}
    .btn-swap:hover {{
      border-color: var(--accent);
      color: var(--accent);
      transform: rotate(180deg);
    }}
    .options-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      padding-top: 4px;
      border-top: 1px solid #21262d;
    }}
    .toggles-group {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .toggle-btn {{
      background: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }}
    .toggle-btn.active {{
      background: #1f6feb;
      border-color: #388bfd;
      color: #ffffff;
    }}
    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }}
    #plot-div {{
      width: 100%;
      height: 600px;
    }}
    .waveform-card {{
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
      justify-content: space-between;
      align-items: center;
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
      color: var(--accent);
      font-weight: 600;
    }}
    #fir-canvas {{
      width: 100%;
      height: 90px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="container">
    <div class="header-card">
      <div class="header-title-col">
        <div class="header-title">
          <span>Voicing IR Difference 3D Waterfall & Waveform</span>
          <span class="badge">WebGL 3D Surface</span>
        </div>
        <div class="header-sub">
          Cumulative Spectral Decay (CSD) of the Difference Impulse Response h<sub>diff</sub>(t) across Time (0–3 ms) & Frequency (20–20,000 Hz)
        </div>
      </div>
      <div>
        <a href="voicings.html" class="nav-btn">&larr; View 2D Curves</a>
        <a href="index.html" class="nav-btn">Master Portal</a>
      </div>
    </div>

    <div class="control-card">
      <div class="selectors-row">
        <div class="selector-box">
          <div class="selector-label" style="color: var(--accent);">Source Voicing</div>
          <select id="source-select" onchange="onSelectionChanged()"></select>
        </div>

        <button id="btn-swap" class="btn-swap" onclick="swapVoicings()" title="Swap Source & Target">&#x21C6;</button>

        <div class="selector-box">
          <div class="selector-label" style="color: var(--amber);">Target Voicing</div>
          <select id="target-select" onchange="onSelectionChanged()"></select>
        </div>
      </div>

      <div class="options-row">
        <div class="toggles-group">
          <button id="btn-surface" class="toggle-btn active" onclick="toggleSurface()">Surface</button>
          <button id="btn-wireframe" class="toggle-btn" onclick="toggleWireframe()">Wireframe</button>
          <button id="btn-contours" class="toggle-btn active" onclick="toggleContours()">Contours</button>
        </div>

        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 11px; color: var(--text-muted); font-weight: 700;">COLORSCALE:</span>
          <select id="colorscale-select" onchange="onColorScaleChanged()" style="width: auto; padding: 4px 8px; font-size: 11px;">
            <option value="Viridis" selected>Viridis</option>
            <option value="Inferno">Inferno</option>
            <option value="Turbo">Turbo</option>
            <option value="Magma">Magma</option>
            <option value="Plasma">Plasma</option>
            <option value="Electric">Electric</option>
          </select>
        </div>
      </div>
    </div>

    <div class="chart-card">
      <div id="plot-div"></div>
    </div>

    <div class="waveform-card">
      <div class="waveform-header">
        <span>Time-Domain Difference Impulse Response h<sub>diff</sub>(t) [First 128 taps]</span>
        <div class="waveform-metrics">
          <div>Peak Level: <span id="metric-peak">0.00 dBFS</span></div>
          <div>Polarity: <span id="metric-polarity">+ Positive</span></div>
          <div>Filter Length: <span>2048 taps (Minimum-Phase)</span></div>
        </div>
      </div>
      <canvas id="fir-canvas" width="1200" height="90"></canvas>
    </div>
  </div>

  <script id="waterfall-data" type="application/json">
{data_json}
  </script>

  <script>
    const data = JSON.parse(document.getElementById('waterfall-data').textContent);
    const frequencies = data.frequencies;
    const timeMs = data.time_ms;
    const voices = data.voices;
    const responses = data.responses;

    const sourceSelect = document.getElementById('source-select');
    const targetSelect = document.getElementById('target-select');

    let showSurface = true;
    let showWireframe = false;
    let showContours = true;
    let currentColorScale = 'Viridis';
    let lastCamera = null;

    const logFreqs = frequencies.map(f => Math.log10(f));
    const logTickVals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].map(f => Math.log10(f));
    const logTickText = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'];

    function populateSelects() {{
      const vList = Object.values(voices);
      [sourceSelect, targetSelect].forEach(sel => {{
        sel.innerHTML = '';
        vList.forEach(v => {{
          const opt = document.createElement('option');
          opt.value = v.id;
          opt.textContent = `${{v.name}} (${{v.family}})`;
          sel.appendChild(opt);
        }});
      }});
      sourceSelect.value = data.default_source || 'precision_vintage';
      targetSelect.value = data.default_target || 'jazz_bridge_growl';
    }}

    populateSelects();

    function swapVoicings() {{
      const temp = sourceSelect.value;
      sourceSelect.value = targetSelect.value;
      targetSelect.value = temp;
      renderPlot();
    }}

    function onSelectionChanged() {{
      renderPlot();
    }}

    function onColorScaleChanged() {{
      currentColorScale = document.getElementById('colorscale-select').value;
      renderPlot();
    }}

    function toggleSurface() {{
      showSurface = !showSurface;
      document.getElementById('btn-surface').classList.toggle('active', showSurface);
      renderPlot();
    }}

    function toggleWireframe() {{
      showWireframe = !showWireframe;
      document.getElementById('btn-wireframe').classList.toggle('active', showWireframe);
      renderPlot();
    }}

    function toggleContours() {{
      showContours = !showContours;
      document.getElementById('btn-contours').classList.toggle('active', showContours);
      renderPlot();
    }}

    function renderPlot() {{
      const plotDiv = document.getElementById('plot-div');
      if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.scene) {{
        lastCamera = plotDiv._fullLayout.scene._scene.getCamera();
      }}

      const srcId = sourceSelect.value;
      const tgtId = targetSelect.value;

      const src = voices[srcId] || {{ name: srcId }};
      const tgt = voices[tgtId] || {{ name: tgtId }};

      const resp = responses[srcId][tgtId];
      const csd = resp.csd_matrix;

      const hoverText = [];
      for (let i = 0; i < timeMs.length; i++) {{
        const t = timeMs[i];
        const row = [];
        for (let j = 0; j < frequencies.length; j++) {{
          const f = frequencies[j];
          const db = csd[i][j];
          row.push(`Freq: ${{f.toFixed(1)}} Hz<br>Decay: ${{t.toFixed(2)}} ms<br>Mag: ${{db.toFixed(1)}} dB`);
        }}
        hoverText.push(row);
      }}

      const trace = {{
        type: 'surface',
        x: logFreqs,
        y: timeMs,
        z: csd,
        colorscale: currentColorScale,
        showscale: true,
        colorbar: {{
          title: {{ text: 'dB', font: {{ color: '#8b949e', size: 11 }} }},
          tickfont: {{ color: '#8b949e', size: 10 }},
          len: 0.8,
          thickness: 14
        }},
        hidesurface: !showSurface,
        contours: {{
          z: {{
            show: showContours,
            usecolormap: true,
            highlightcolor: '#38bdf8',
            project: {{ z: true }}
          }}
        }},
        wireframe: {{
          show: showWireframe
        }},
        text: hoverText,
        hoverinfo: 'text'
      }};

      const layout = {{
        title: {{
          text: `<b>Difference IR CSD Waterfall:</b> ${{src.name}} → ${{tgt.name}}`,
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
      renderWaveform(resp.fir_waveform);
    }}

    function renderWaveform(samples) {{
      const canvas = document.getElementById('fir-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // Center reference line
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

    window.addEventListener('DOMContentLoaded', renderPlot);
  </script>
</body>
</html>
"""
    target_html.write_text(html_content, encoding="utf-8")
    print(f"Saved Voicing IR Difference 3D Waterfall page: {target_html}")
    return target_html


def generate_interactive_chart(
    instrument: InstrumentConfig | str = "30in",
    out_html: str | Path | None = None,
    mode: str = "voicings",
) -> Path:
    """
    Calculates voice responses and renders an interactive chart.
    mode:
      - 'voicings': interactive 3-line voicing comparison (Source, Target, Diff).
      - 'voicing_ir_3d' or 'waterfall3d': interactive WebGL 3D difference IR waterfall.
      - 'unified': embeds both Output Voice and Input/Output Difference curves with interactive radio buttons.
      - 'output': standalone chart strictly plotting the target Output Voice curves.
      - 'difference': standalone chart strictly plotting the Input/Output Difference curves for this instrument.
    """
    if mode == "voicings":
        return generate_voicings_page(target_html=Path(out_html) if out_html is not None else None)
    elif mode in ("voicing_ir_3d", "waterfall3d"):
        return generate_voicing_ir_3d_page(
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
        y_domain = [-18, 18]
    elif mode == "difference":
        dfs = [
            build_voice_dataframe(vid, cfg, instrument=inst, mode="difference")
            for vid, cfg in VOICES.items()
        ]
        master_df = pl.concat(dfs)
        chart_title = "Allomorph Master Voices: Input/Output Differential Transfer Functions"
        chart_subtitle = f'Source: {inst_name} -> Target: 34" Standard & 37" Multi-Scale Datums (Δ Transfer Filter)'
        y_title = "Differential Transfer Magnitude (dB)"
        y_domain = [-18, 18]
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
        y_domain = [-18, 18]

    print("Rendering interactive chart using Altair...")
    return render_chart_to_file(
        master_df, target_path, chart_title, chart_subtitle, y_title, y_domain, mode=mode
    )


def generate_all_charts(output_dir: str | Path | None = None) -> dict[str, Path]:
    """Generates standalone interactive charts for voicings comparison, 3D IR difference, and the portal."""
    out_dir = Path(output_dir) if output_dir else RESPONSES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Interactive Voicings Comparison (3-line graph: Source, Target, Diff)
    print("Generating Voicings Comparison (voicings.html)...")
    voicings_html = generate_voicings_page(out_dir / "voicings.html")

    # 2. 3D Difference IR Waterfall & Waveform (voicing_ir_3d.html)
    print("Generating Voicing IR 3D Difference Waterfall (voicing_ir_3d.html)...")
    waterfall_3d_html = generate_voicing_ir_3d_page(out_dir / "voicing_ir_3d.html")

    generated: dict[str, Path] = {
        "voicings": voicings_html,
        "voicing_ir_3d": waterfall_3d_html,
    }

    generate_portal_pages(output_dir=out_dir)
    return generated
