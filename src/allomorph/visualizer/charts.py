"""
Allomorph Visualizer - Interactive Altair Charts Generation
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import altair as alt
import polars as pl

from allomorph.config.instruments import load_instrument
from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig
from allomorph.visualizer.dataframe import (
    build_voice_dataframe,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
)

DOCS_DIR = REPO_ROOT / "docs"

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
                alt.Tooltip("sensor_type:N", title="Transducer"),
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


def build_voicings_html(
    instruments: Mapping[str, InstrumentConfig] | None = None,
) -> str:
    """Constructs the flat, self-contained HTML string for the Allomorph Frequency Response Suite."""
    if instruments is None:
        from allomorph.config.instruments import INSTRUMENTS

        instruments = INSTRUMENTS

    voicings_data = build_voicings_comparison_data(instruments=instruments, step=1)
    data_json = json.dumps(voicings_data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph | Frequency Response Suite & Voicing Comparisons</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --card-hover: #1c2128;
      --border: #30363d;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --accent-subtle: rgba(56, 189, 248, 0.12);
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
      padding: 24px;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .header h1 {{
      font-size: 24px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .badge {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      background: var(--accent-subtle);
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 2px 8px;
      border-radius: 12px;
    }}
    .header .subtitle {{
      font-size: 14px;
      color: var(--text-muted);
      margin-top: 6px;
    }}
    .flow-pipeline-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      overflow-x: auto;
    }}
    .flow-step {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 160px;
    }}
    .flow-step-num {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      font-weight: 700;
    }}
    .flow-step-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
    }}
    .flow-step-desc {{
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.3;
    }}
    .flow-step.flow-active .flow-step-num {{
      color: var(--accent);
    }}
    .flow-arrow {{
      color: var(--border);
      font-size: 18px;
      font-weight: 700;
      padding: 0 4px;
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
    <div class="header">
      <h1>Allomorph Frequency Response Suite <span class="badge">Universal Voicings</span> <span class="badge">Direct Forward Simulation</span></h1>
      <div class="subtitle">Interactive Source vs Target Voicing Comparisons &amp; Analytical Differential Transformations.</div>
    </div>

    <div class="flow-pipeline-card">
      <div class="flow-step">
        <div class="flow-step-num">Step 1</div>
        <div class="flow-step-title">Source Digital Twin</div>
        <div class="flow-step-desc">Forward source instrument simulation (H<sub>src</sub>)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-num">Step 2</div>
        <div class="flow-step-title">Target Digital Twin</div>
        <div class="flow-step-desc">Forward target instrument simulation (H<sub>tgt</sub>)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step flow-active">
        <div class="flow-step-num">Training</div>
        <div class="flow-step-title">NAM Block 1 Training</div>
        <div class="flow-step-desc">WaveNet A2 learns Source &rarr; Target transform</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-num">Output</div>
        <div class="flow-step-title">Stage Physical Twin</div>
        <div class="flow-step-desc">Physical Bass A &rarr; NAM Block 1 &approx; Physical Bass B</div>
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

      <div id="presets-container" class="presets-row">
        <span class="preset-label">Quick Comparisons:</span>
      </div>

      <div class="metrics-row">
        <div class="metric-card">
          <div class="metric-name">Difference Max Boost</div>
          <div class="metric-val boost" id="metric-boost">+0.0 dB</div>
        </div>
        <div class="metric-card">
          <div class="metric-name">Difference Max Cut</div>
          <div class="metric-val cut" id="metric-cut">-0.0 dB</div>
        </div>
        <div class="metric-card">
          <div class="metric-name">Difference Dynamic Range</div>
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
          <span class="badge" id="src-spec-topo">Transducer</span>
        </div>
        <div class="spec-desc" id="src-spec-desc">Description</div>
      </div>
      <div class="spec-card">
        <div class="spec-header" style="color: var(--amber);">
          <span id="tgt-spec-title">Target Voicing</span>
          <span class="badge" id="tgt-spec-topo">Transducer</span>
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
    const instruments = data.instruments;
    const instrumentOrder = data.instrument_order || Object.keys(instruments || {{}});
    const chips = data.chips || [];

    const sourceSelect = document.getElementById('source-select');
    const targetSelect = document.getElementById('target-select');
    const presetsContainer = document.getElementById('presets-container');

    if (presetsContainer && chips.length > 0) {{
      presetsContainer.innerHTML = '<span class="preset-label">Quick Comparisons:</span>';
      chips.forEach(chip => {{
        const btn = document.createElement('button');
        btn.className = 'preset-chip';
        btn.textContent = chip.label;
        btn.onclick = () => applyPreset(chip.source, chip.target);
        presetsContainer.appendChild(btn);
      }});
    }}

    function populateSelect(selectEl, selectedId) {{
      selectEl.innerHTML = '';
      instrumentOrder.forEach(instId => {{
        const instMeta = instruments[instId];
        if (!instMeta || !instMeta.voice_ids || instMeta.voice_ids.length === 0) return;
        const optgroup = document.createElement('optgroup');
        optgroup.label = instMeta.label || instMeta.name;
        instMeta.voice_ids.forEach(vid => {{
          const v = voices[vid];
          if (!v) return;
          const opt = document.createElement('option');
          opt.value = v.id;
          opt.textContent = v.name;
          if (v.id === selectedId) opt.selected = true;
          optgroup.appendChild(opt);
        }});
        selectEl.appendChild(optgroup);
      }});
    }}

    populateSelect(sourceSelect, data.default_source || '34in_standard_p:vintage_open');
    populateSelect(targetSelect, data.default_target || '34in_standard_jazz:bridge_growl');

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

    function renderChart() {{
      const srcId = sourceSelect.value;
      const tgtId = targetSelect.value;

      const src = voices[srcId];
      const tgt = voices[tgtId];

      const srcMags = src.magnitude_db;
      const tgtMags = tgt.magnitude_db;

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
        for (let i = 0; i < freqs.length; i++) {{
          let d = Math.round((tgtMags[i] - srcMags[i]) * 100) / 100;
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
      document.getElementById('src-spec-topo').textContent = (src.sensor_type || 'magnetic').toUpperCase();
      const srcDesc = src.fr != null ? `${{src.description}} (fr: ${{src.fr.toFixed(0)}} Hz, Q: ${{src.q.toFixed(2)}})` : `${{src.description}} (Flat / Wideband)`;
      document.getElementById('src-spec-desc').textContent = srcDesc;

      document.getElementById('tgt-spec-title').textContent = tgt.name;
      document.getElementById('tgt-spec-topo').textContent = (tgt.sensor_type || 'magnetic').toUpperCase();
      const tgtDesc = tgt.fr != null ? `${{tgt.description}} (fr: ${{tgt.fr.toFixed(0)}} Hz, Q: ${{tgt.q.toFixed(2)}})` : `${{tgt.description}} (Flat / Wideband)`;
      document.getElementById('tgt-spec-desc').textContent = tgtDesc;

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
        name: `Difference`,
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#10b981', width: 3.0 }},
        hovertemplate: `Difference: %{{y:+.1f}} dB<extra></extra>`
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


def generate_voicings_page(
    target_html: Path | str | None = None,
    instruments: Mapping[str, InstrumentConfig] | None = None,
) -> Path:
    """Renders the flat, self-contained interactive Voicing Comparison page (default: docs/frequency_responses.html)."""
    if target_html is None:
        target_html = DOCS_DIR / "frequency_responses.html"
    target_html = Path(target_html)
    target_html.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_voicings_html(instruments=instruments)
    target_html.write_text(html_content, encoding="utf-8")
    print(f"Saved Voicings Comparison page: {target_html}")
    return target_html


def generate_interactive_chart(
    instrument: InstrumentConfig | str = "30in_mustang_pj",
    out_html: str | Path | None = None,
    mode: str = "voicings",
    instruments: Mapping[str, InstrumentConfig] | None = None,
) -> Path:
    """
    Calculates voicing responses and renders an interactive chart.
    mode:
      - 'voicings': interactive 3-line voicing comparison (Source, Target, Diff).
      - 'unified': embeds both Output Voice and Input/Output Difference curves with interactive radio buttons.
      - 'output': standalone chart strictly plotting the target Output Voice curves.
      - 'difference': standalone chart strictly plotting the Input/Output Difference curves for this instrument.
    """
    if mode == "voicings":
        return generate_voicings_page(
            target_html=Path(out_html) if out_html is not None else None,
            instruments=instruments,
        )

    if instruments is None:
        from allomorph.config.instruments import INSTRUMENTS

        instruments = INSTRUMENTS

    inst = (
        instrument
        if isinstance(instrument, InstrumentConfig)
        else (instruments.get(instrument) or load_instrument(instrument))
    )
    inst_id = inst.id
    inst_name = inst.name

    if out_html is None:
        target_path = DOCS_DIR / f"{inst_id}.html"
    else:
        p = Path(out_html)
        if p.is_dir() or (not p.suffix and not p.exists()):
            target_path = p / f"{inst_id}.html"
        else:
            target_path = p

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Computing voice frequency responses (mode={mode}, instrument={inst_name})...")
    src_voicing = next(iter(inst.voicings.values()))

    target_dfs: list[pl.DataFrame] = []
    diff_dfs: list[pl.DataFrame] = []

    for _, t_inst in sorted(instruments.items()):
        for v_id, v_obj in t_inst.voicings.items():
            v_label = v_obj.tone_name or v_obj.name or v_id
            v_desc = f"{v_obj.name} ({t_inst.name})"

            if mode in ("output", "unified"):
                df_out = build_voice_dataframe(t_inst, v_obj, mode="output")
                df_out = df_out.with_columns(
                    pl.lit(v_label).alias("voice_name"),
                    pl.lit(v_obj.sensor_type).alias("sensor_type"),
                    pl.lit(v_desc).alias("description"),
                )
                if mode == "unified":
                    df_out = df_out.with_columns(pl.lit("Output Voice").alias("mode"))
                target_dfs.append(df_out)

            if mode in ("difference", "unified"):
                df_diff_pair = build_voicings_comparison_dataframe(
                    source_instrument=inst,
                    source_voicing=src_voicing,
                    target_instrument=t_inst,
                    target_voicing=v_obj,
                )
                df_diff = df_diff_pair.filter(pl.col("line_type") == "3. Difference")
                df_diff = df_diff.with_columns(
                    pl.lit(v_label).alias("voice_name"),
                    pl.lit(v_obj.sensor_type).alias("sensor_type"),
                    pl.lit(v_desc).alias("description"),
                )
                if mode == "unified":
                    df_diff = df_diff.with_columns(pl.lit("Input/Output Difference").alias("mode"))
                diff_dfs.append(df_diff)

    if mode == "output":
        master_df = pl.concat(target_dfs)
        chart_title = "Allomorph Master Voices: Output Voice Frequency Responses"
        chart_subtitle = f"Target Passive Acoustic Apertures & SPICE Loaded RLC Resonances (Reference: {inst_name})"
        y_title = "Normalized Output Magnitude (dB)"
        y_domain = [-18, 18]
    elif mode == "difference":
        master_df = pl.concat(diff_dfs)
        chart_title = "Allomorph Master Voices: Input/Output Differential Transfer Functions"
        chart_subtitle = f'Source: {inst_name} -> Target: 34" Standard & 37" Multi-Scale Datums (Δ Transfer Filter)'
        y_title = "Differential Transfer Magnitude (dB)"
        y_domain = [-18, 18]
    else:  # mode == "unified"
        master_df = pl.concat(diff_dfs + target_dfs)
        chart_title = "Allomorph Master Voices: Acoustic & Electrical Response Curves"
        chart_subtitle = f"Interactive View ({inst_name}) — Switch between Input/Output Difference and Output Voice"
        y_title = "Normalized Magnitude / Differential Gain (dB)"
        y_domain = [-18, 18]

    print("Rendering interactive chart using Altair...")
    return render_chart_to_file(
        master_df, target_path, chart_title, chart_subtitle, y_title, y_domain, mode=mode
    )


def generate_all_charts(output_dir: str | Path | None = None) -> dict[str, Path]:
    """Generates the flat, self-contained interactive Frequency Response Suite page (docs/frequency_responses.html)."""
    out_dir = Path(output_dir) if output_dir else DOCS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir if out_dir.suffix == ".html" else (out_dir / "frequency_responses.html")
    print(f"Generating Frequency Response Suite ({target_path.name})...")
    voicings_html = generate_voicings_page(target_path)
    return {
        "voicings": voicings_html,
    }
