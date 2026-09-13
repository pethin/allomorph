"""
Allomorph Visualizer - Interactive HTML Portal Generation
"""

from collections.abc import Mapping
from pathlib import Path

from pydantic import TypeAdapter

from allomorph.config.instruments import load_all_instruments
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
    instruments_meta: Mapping[str, PortalInstrumentMeta],
    default_id: str,
    base_url_prefix: str = "./",
) -> str:
    """Constructs a responsive, dark-mode portal HTML string with 3-way Architecture C signal flow navigation."""
    meta_json = (
        TypeAdapter(dict[str, PortalInstrumentMeta])
        .dump_json(dict(instruments_meta), indent=2)
        .decode("utf-8")
    )

    buttons_html = []
    for inst_id, meta in instruments_meta.items():
        is_active = " active" if inst_id == default_id else ""
        buttons_html.append(
            f'<button class="tab-btn{is_active}" data-id="{inst_id}" onclick="selectInstrument(\'{inst_id}\')">'
            f"<span>{meta.name}</span>"
            f"</button>"
        )
    tabs_markup = "\n    ".join(buttons_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Allomorph | Two-Stage Frequency Response Suite</title>
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
      --teal: #26a69a;
      --orange: #ff7043;
      --gold: #ffd54f;
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
      min-width: 150px;
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
    .inspector-subcontrols {{
      display: block;
    }}
    .tabs-container {{
      margin-bottom: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .tab-btn {{
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .tab-btn:hover {{
      background-color: var(--tag-bg);
      color: var(--text);
      border-color: #8b949e;
    }}
    .tab-btn.active {{
      background-color: #238636;
      border-color: #2ea043;
      color: #ffffff;
      box-shadow: 0 0 8px rgba(46, 160, 67, 0.4);
    }}
    .view-mode-bar {{
      margin-bottom: 16px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 16px;
    }}
    .mode-toggle-group {{
      display: inline-flex;
      background-color: var(--tag-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 3px;
      gap: 4px;
      flex-wrap: wrap;
    }}
    .mode-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .mode-btn:hover {{
      color: var(--text);
      background-color: rgba(255, 255, 255, 0.05);
    }}
    .mode-btn.active {{
      background-color: var(--btn-active);
      color: #ffffff;
      box-shadow: 0 0 8px rgba(31, 111, 235, 0.4);
    }}
    .mode-hint {{
      font-size: 12px;
      color: var(--text-muted);
      font-style: italic;
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
      height: 760px;
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
      <h1>Allomorph Frequency Response Suite <span class="badge">Architecture C</span> <span class="badge">Two-Stage Signal Flow</span></h1>
      <div class="subtitle">Universal 2-Stage Acoustic Aperture Deconvolution & SPICE Digital Twins for Darkglass Anagram & Neural Amp Modeler.</div>
    </div>

    <div class="flow-pipeline-card">
      <div class="flow-step">
        <div class="flow-step-num">Step 1</div>
        <div class="flow-step-title">Source Bass</div>
        <div class="flow-step-desc">Physical pickups & scale (Knobs wide open 100%)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step flow-active">
        <div class="flow-step-num">Block 1</div>
        <div class="flow-step-title">Frontend IR</div>
        <div class="flow-step-desc">2048-tap causal minimum-phase FIR</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-num">Datum</div>
        <div class="flow-step-title">Canonical Intermediate</div>
        <div class="flow-step-desc">34" @ 93.5mm median (0.00 dB baseline)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step flow-active">
        <div class="flow-step-num">Block 2</div>
        <div class="flow-step-title">Universal Target NAM</div>
        <div class="flow-step-desc">A2-Lite Preamp (Clean / Dynamic / Hot Rod)</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-num">Output</div>
        <div class="flow-step-title">Target Acoustic Voice</div>
        <div class="flow-step-desc">Exact tone: Block 1 &times; Block 2 = Acoustic Twin</div>
      </div>
    </div>

    <div class="primary-nav-bar" role="tablist">
      <button class="primary-nav-btn active" id="pnav-targets" onclick="selectPrimaryView('targets')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
        <span>Universal Targets (Block 2)</span>
      </button>
      <button class="primary-nav-btn" id="pnav-frontends" onclick="selectPrimaryView('frontends')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        <span>Frontend Deconvolutions (Block 1)</span>
      </button>
      <button class="primary-nav-btn" id="pnav-baked" onclick="selectPrimaryView('baked')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
        <span>Baked Responses (1-Block)</span>
      </button>
      <button class="primary-nav-btn" id="pnav-inspector" onclick="selectPrimaryView('inspector')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
        <span>Signal Flow Inspector (End-to-End)</span>
      </button>
    </div>

    <div class="inspector-subcontrols" id="inspector-subcontrols" style="display: none;">
      <div class="tabs-container" id="tabs">
      {tabs_markup}
      </div>
    </div>

    <div class="meta-panel">
      <div class="meta-item">
        <div class="label" id="meta-label-primary">Active Stage</div>
        <div class="value" id="meta-name">Universal Targets (Block 2)</div>
      </div>
      <div class="meta-item">
        <div class="label" id="meta-label-secondary">Reference / Datum</div>
        <div class="value" id="meta-scale">Canonical Intermediate: 34" @ 93.5mm median</div>
      </div>
      <div class="meta-item">
        <div class="label" id="meta-label-tertiary">Pickup & Circuit</div>
        <div class="value" id="meta-pickups">All 22 Universal Targets across 3 Dynamic Feel Tiers</div>
      </div>
      <div class="meta-item">
        <div class="label">Response View Mode</div>
        <div class="value" id="meta-view-mode">Universal Targets (Block 2)</div>
      </div>
      <div>
        <a id="standalone-link" class="open-standalone-btn" href="{base_url_prefix}universal_targets.html" target="_blank">
          Open Standalone Chart ↗
        </a>
      </div>
    </div>

    <div class="chart-card">
      <iframe id="chart-frame" src="{base_url_prefix}universal_targets.html" title="Interactive Vega Frequency Response Chart"></iframe>
    </div>
  </div>

  <script>
    const instruments = {meta_json};
    const baseUrlPrefix = "{base_url_prefix}";
    let currentPrimaryView = "targets"; // 'targets', 'frontends', 'baked', 'inspector'
    let currentId = "{default_id}";

    function updateView() {{
      const pnavTargets = document.getElementById('pnav-targets');
      const pnavFrontends = document.getElementById('pnav-frontends');
      const pnavBaked = document.getElementById('pnav-baked');
      const pnavInspector = document.getElementById('pnav-inspector');
      const subcontrols = document.getElementById('inspector-subcontrols');
      const frame = document.getElementById('chart-frame');
      const standaloneLink = document.getElementById('standalone-link');
      const metaName = document.getElementById('meta-name');
      const metaScale = document.getElementById('meta-scale');
      const metaPickups = document.getElementById('meta-pickups');
      const metaViewMode = document.getElementById('meta-view-mode');

      pnavTargets.classList.toggle('active', currentPrimaryView === 'targets');
      pnavFrontends.classList.toggle('active', currentPrimaryView === 'frontends');
      if (pnavBaked) pnavBaked.classList.toggle('active', currentPrimaryView === 'baked');
      pnavInspector.classList.toggle('active', currentPrimaryView === 'inspector');

      if (currentPrimaryView === 'targets') {{
        subcontrols.style.display = 'none';
        const url = `${{baseUrlPrefix}}universal_targets.html`;
        frame.src = url;
        standaloneLink.href = url;
        metaName.textContent = 'Universal Target Voicings (Block 2)';
        metaScale.textContent = 'Canonical Intermediate: 34" @ 93.5mm median';
        metaPickups.textContent = '22 Target Profiles across 3 Dynamic Feel Tiers (Clean, Dynamic, Hot Rod)';
        metaViewMode.textContent = 'Universal Target Profiles';
        if (window.history.replaceState) window.history.replaceState(null, null, '#targets');
        return;
      }}

      if (currentPrimaryView === 'frontends') {{
        subcontrols.style.display = 'none';
        const url = `${{baseUrlPrefix}}frontend_deconvolutions.html`;
        frame.src = url;
        standaloneLink.href = url;
        metaName.textContent = 'Frontend Deconvolutions (Block 1)';
        metaScale.textContent = '32 Physical Switch Positions across 11 Source Basses';
        metaPickups.textContent = '2048-tap causal minimum-phase FIRs with strictly positive initial polarity';
        metaViewMode.textContent = 'Frontend Deconvolution IRs';
        if (window.history.replaceState) window.history.replaceState(null, null, '#frontends');
        return;
      }}

      if (currentPrimaryView === 'baked') {{
        subcontrols.style.display = 'none';
        const url = `${{baseUrlPrefix}}baked_responses.html`;
        frame.src = url;
        standaloneLink.href = url;
        metaName.textContent = 'Baked Transformations (1-Block Single Model)';
        metaScale.textContent = 'Multi-Instrument & Multi-Voicing Differential Matrix';
        metaPickups.textContent = '276 Monolithic Transfer Curves (H_diff = H_tgt / H_src) with dynamic multi-selection';
        metaViewMode.textContent = 'Baked End-to-End Single-Block';
        if (window.history.replaceState) window.history.replaceState(null, null, '#baked');
        return;
      }}

      // Inspector Mode
      subcontrols.style.display = 'block';
      const inst = instruments[currentId] || Object.values(instruments)[0];

      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.id === currentId);
      }});

      const chartUrl = `${{baseUrlPrefix}}${{currentId}}.html`;
      frame.src = chartUrl;
      standaloneLink.href = chartUrl;
      metaName.textContent = inst.name;
      metaScale.textContent = `${{inst.scale_in}}" scale (${{inst.scale_m}} m) | ${{inst.speeds_str}}`;
      metaPickups.textContent = inst.pickups_summary;
      metaViewMode.textContent = 'End-to-End Signal Flow';

      if (window.history.replaceState) {{
        window.history.replaceState(null, null, `#${{currentId}}`);
      }}
      setTimeout(resizeIframe, 150);
      setTimeout(resizeIframe, 450);
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

    function selectInstrument(id) {{
      if (!instruments[id]) return;
      currentId = id;
      currentPrimaryView = 'inspector';
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
      if (hash === 'targets') {{
        currentPrimaryView = 'targets';
      }} else if (hash === 'frontends') {{
        currentPrimaryView = 'frontends';
      }} else if (hash === 'baked') {{
        currentPrimaryView = 'baked';
      }} else if (hash) {{
        const hashId = hash.split(':')[0];
        if (instruments[hashId]) {{
          currentId = hashId;
          currentPrimaryView = 'inspector';
        }}
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
    all_insts = load_all_instruments()

    active_meta: dict[str, PortalInstrumentMeta] = {}
    for inst_id, inst_cfg in all_insts.items():
        if inst_id == "canonical_intermediate":
            continue
        chart_file = out_dir / f"{inst_id}.html"
        if chart_file.exists() or not any(out_dir.glob("*.html")):
            active_meta[inst_id] = format_instrument_meta(inst_cfg)

    if not active_meta:
        for inst_id, inst_cfg in all_insts.items():
            if inst_id == "canonical_intermediate":
                continue
            active_meta[inst_id] = format_instrument_meta(inst_cfg)

    def_id = default_id if (default_id and default_id in active_meta) else next(iter(active_meta))

    # 1. Write docs/frequency_responses/index.html
    index_html = build_portal_html(active_meta, default_id=def_id, base_url_prefix="./")
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"Saved interactive portal: {index_path}")

    # 2. Write docs/frequency_responses.html at root of docs/ (only if target directory is default RESPONSES_DIR)
    if out_dir.resolve() == RESPONSES_DIR.resolve():
        root_portal_html = build_portal_html(
            active_meta, default_id=def_id, base_url_prefix="./frequency_responses/"
        )
        root_portal_path = DOCS_DIR / "frequency_responses.html"
        root_portal_path.write_text(root_portal_html, encoding="utf-8")
        print(f"Saved master portal: {root_portal_path}")
