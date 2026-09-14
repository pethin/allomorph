"""
Allomorph Visualizer - Interactive HTML Portal Generation
"""

from collections.abc import Mapping
from pathlib import Path

from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig
from allomorph.visualizer.schema import PortalInstrumentMeta

DOCS_DIR = REPO_ROOT / "docs"
RESPONSES_DIR = DOCS_DIR / "frequency_responses"


def append_spec_panel(html_path: Path, panel_html: str) -> None:
    """Appends an informational HTML spec/directive panel before </body>."""
    content = html_path.read_text(encoding="utf-8")
    if "</body>" in content:
        content = content.replace("</body>", f"{panel_html}\n</body>")
        html_path.write_text(content, encoding="utf-8")


def format_instrument_meta(inst: InstrumentConfig) -> PortalInstrumentMeta:
    """Formats an InstrumentConfig into metadata suitable for the portal."""
    inst_id = inst.id
    inst_name = inst.name
    scale_in = inst.scale_length_in or 34.0
    scale_m = inst.scale_length_m or (scale_in * 0.0254)
    speeds = inst.string_wave_speeds or []
    speeds_str = ", ".join(f"{float(s):.1f} m/s" for s in speeds) if speeds else "N/A"

    parts = []
    for pid, pcfg in inst.pickups.items():
        if pcfg.type == "composite":
            continue
        pname = pcfg.name or pid
        pos_m = pcfg.position_from_bridge_m
        if pos_m:
            pos_mm = float(pos_m) * 1000.0
            parts.append(f"{pname} (@ {pos_mm:.1f}mm)")
        else:
            parts.append(pname)
    pickups_summary = " | ".join(parts) if parts else "Standard Pickups"

    return PortalInstrumentMeta(
        id=inst_id,
        name=inst_name,
        scale_in=scale_in,
        scale_m=round(scale_m, 4),
        speeds_str=speeds_str,
        pickups_summary=pickups_summary,
        default_pickup=inst.default_pickup or "default",
    )


def build_portal_html(
    instruments_meta: Mapping[str, PortalInstrumentMeta] | None = None,
    default_id: str = "voicings",
    base_url_prefix: str = "./",
) -> str:
    """Constructs a responsive, dark-mode portal HTML string with interactive Voicing Comparisons and 3D IR Difference Waterfall navigation."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph | Voicing Comparisons & Frequency Response Suite</title>
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
      --orange: #ff7043;
      --purple: #a855f7;
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
    }}
    .header {{
      margin-bottom: 16px;
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
    .header .badge {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
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
      margin-bottom: 20px;
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
    .primary-nav-bar {{
      display: flex;
      gap: 10px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .primary-nav-btn {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
    }}
    .primary-nav-btn:hover {{
      background: var(--tag-bg);
      color: var(--text);
      border-color: #8b949e;
    }}
    .primary-nav-btn.active {{
      background: var(--btn-active);
      border-color: #388bfd;
      color: #ffffff;
      box-shadow: 0 0 12px rgba(31, 111, 235, 0.4);
    }}
    .meta-panel {{
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 20px;
      margin-bottom: 20px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) auto;
      gap: 16px;
      align-items: center;
    }}
    .meta-item .label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      font-weight: 600;
    }}
    .meta-item .value {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      margin-top: 2px;
    }}
    .open-standalone-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background-color: var(--tag-bg);
      border: 1px solid var(--border);
      color: var(--accent);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.15s ease;
      white-space: nowrap;
    }}
    .open-standalone-btn:hover {{
      background-color: #30363d;
      color: #ffffff;
      border-color: var(--accent);
    }}
    .chart-card {{
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }}
    iframe {{
      width: 100%;
      min-height: 720px;
      height: 820px;
      border: none;
      display: block;
      background-color: #0d1117;
      overflow: hidden;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Allomorph Frequency Response Suite <span class="badge">Universal Voicings</span> <span class="badge">Direct Forward Simulation</span></h1>
      <div class="subtitle">Interactive Source vs Target Voicing Comparisons & 3D Differential IR Waterfalls.</div>
    </div>

    <div class="flow-pipeline-card">
      <div class="flow-step">
        <div class="flow-step-num">Step 1</div>
        <div class="flow-step-title">Source Voicing</div>
        <div class="flow-step-desc">Acoustic aperture + RLC pickup circuit (H<sub>src</sub>)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step flow-active">
        <div class="flow-step-num">Transformation</div>
        <div class="flow-step-title">Differential Filter</div>
        <div class="flow-step-desc">Regularized minimum-phase FIR (H<sub>diff</sub> = H<sub>tgt</sub> / H<sub>src</sub>)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-num">Step 2</div>
        <div class="flow-step-title">Target Voicing</div>
        <div class="flow-step-desc">Acoustic aperture + SPICE loaded RLC resonance (H<sub>tgt</sub>)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-num">Output</div>
        <div class="flow-step-title">Acoustic Digital Twin</div>
        <div class="flow-step-desc">Exact response: H<sub>src</sub> &times; H<sub>diff</sub> = H<sub>tgt</sub></div>
      </div>
    </div>

    <div class="primary-nav-bar" role="tablist">
      <button class="primary-nav-btn active" id="pnav-voicings" onclick="selectPrimaryView('voicings')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
        <span>Voicing Comparisons (3-Line Response)</span>
      </button>
      <button class="primary-nav-btn" id="pnav-waterfall3d" onclick="selectPrimaryView('waterfall3d')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline></svg>
        <span>Voicing IR 3D Difference Waterfall</span>
      </button>
    </div>

    <div class="meta-panel">
      <div class="meta-item">
        <div class="label" id="meta-label-primary">Active View</div>
        <div class="value" id="meta-name">Voicing Comparisons (3-Line Response)</div>
      </div>
      <div class="meta-item">
        <div class="label" id="meta-label-secondary">Voicing Matrix</div>
        <div class="value" id="meta-scale">28 Catalog Voicings: Select Any Source and Target</div>
      </div>
      <div class="meta-item">
        <div class="label" id="meta-label-tertiary">Response Overlay</div>
        <div class="value" id="meta-pickups">H_src (Cyan) | H_tgt (Orange) | H_diff (Purple) with Identity (0.00 dB)</div>
      </div>
      <div class="meta-item">
        <div class="label">Response View Mode</div>
        <div class="value" id="meta-view-mode">Interactive 3-Line Voicings</div>
      </div>
      <div>
        <a id="standalone-link" class="open-standalone-btn" href="{base_url_prefix}voicings.html" target="_blank">
          Open Standalone Chart ↗
        </a>
      </div>
    </div>

    <div class="chart-card">
      <iframe id="chart-frame" src="{base_url_prefix}voicings.html" title="Interactive Voicing Comparisons"></iframe>
    </div>
  </div>

  <script>
    const baseUrlPrefix = "{base_url_prefix}";
    let currentPrimaryView = "{default_id}" === "waterfall3d" ? "waterfall3d" : "voicings";

    function updateView() {{
      const pnavVoicings = document.getElementById('pnav-voicings');
      const pnavWaterfall3d = document.getElementById('pnav-waterfall3d');
      const frame = document.getElementById('chart-frame');
      const standaloneLink = document.getElementById('standalone-link');
      const metaName = document.getElementById('meta-name');
      const metaScale = document.getElementById('meta-scale');
      const metaPickups = document.getElementById('meta-pickups');
      const metaViewMode = document.getElementById('meta-view-mode');

      pnavVoicings.classList.toggle('active', currentPrimaryView === 'voicings');
      if (pnavWaterfall3d) pnavWaterfall3d.classList.toggle('active', currentPrimaryView === 'waterfall3d');

      if (currentPrimaryView === 'voicings') {{
        const url = `${{baseUrlPrefix}}voicings.html`;
        frame.src = url;
        standaloneLink.href = url;
        metaName.textContent = 'Voicing Comparisons (3-Line Response)';
        metaScale.textContent = '28 Catalog Voicings: Select Any Source and Target Voicing';
        metaPickups.textContent = 'H_src (Cyan) | H_tgt (Orange) | H_diff (Purple) with Identity (0.00 dB)';
        metaViewMode.textContent = 'Interactive 3-Line Voicings';
        if (window.history.replaceState) window.history.replaceState(null, null, '#voicings');
        return;
      }}

      if (currentPrimaryView === 'waterfall3d') {{
        const url = `${{baseUrlPrefix}}voicing_ir_3d.html`;
        frame.src = url;
        standaloneLink.href = url;
        metaName.textContent = 'Voicing IR Difference 3D Waterfall & Waveform';
        metaScale.textContent = 'Cumulative Spectral Decay (CSD) & Difference Impulse Response (h_diff)';
        metaPickups.textContent = 'Interactive 3D WebGL Surface: Time Decay (0-10ms) x Frequency x dB with FIR Waveform';
        metaViewMode.textContent = '3D Impulse Response Analysis';
        if (window.history.replaceState) window.history.replaceState(null, null, '#waterfall3d');
        return;
      }}
    }}

    function resizeIframe() {{
      const frame = document.getElementById('chart-frame');
      if (!frame) return;
      try {{
        const doc = frame.contentDocument || frame.contentWindow.document;
        if (doc && doc.body) {{
          const h = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
          if (h > 250) {{
            frame.style.height = (h + 16) + 'px';
          }}
        }}
      }} catch (err) {{}}
    }}

    function selectPrimaryView(view) {{
      currentPrimaryView = view;
      updateView();
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      const frame = document.getElementById('chart-frame');
      if (frame) {{
        frame.addEventListener('load', () => {{
          resizeIframe();
          setTimeout(resizeIframe, 150);
          setTimeout(resizeIframe, 450);
        }});
      }}
      window.addEventListener('resize', resizeIframe);

      const hash = window.location.hash.replace('#', '');
      if (hash === 'waterfall3d' || hash === '3d') {{
        currentPrimaryView = 'waterfall3d';
      }} else {{
        currentPrimaryView = 'voicings';
      }}
      updateView();
    }});
  </script>
</body>
</html>
"""
    return html


def generate_portal_pages(
    output_dir: str | Path | None = None,
    default_id: str | None = None,
) -> None:
    """
    Builds the interactive index portal:
    1. docs/frequency_responses/index.html (relative links './{id}.html')
    2. docs/frequency_responses.html (relative links './frequency_responses/{id}.html')
    """
    out_dir = Path(output_dir) if output_dir else RESPONSES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    def_id = default_id or "voicings"

    # 1. Write docs/frequency_responses/index.html
    index_html = build_portal_html(default_id=def_id, base_url_prefix="./")
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"Saved interactive portal: {index_path}")

    # 2. Write docs/frequency_responses.html at root of docs/ (only if target directory is default RESPONSES_DIR)
    if out_dir.resolve() == RESPONSES_DIR.resolve():
        root_portal_html = build_portal_html(
            default_id=def_id, base_url_prefix="./frequency_responses/"
        )
        root_portal_path = DOCS_DIR / "frequency_responses.html"
        root_portal_path.write_text(root_portal_html, encoding="utf-8")
        print(f"Saved master portal: {root_portal_path}")
