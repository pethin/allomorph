"""Generate Tone3000-compliant SPICE schematic SVG and 1024x1024 JPG artwork for Allomorph Tone Packs.

Adheres strictly to Tone3000's Tone Sharing Guidelines for Circuit Models:
- Zero promotional logos, marketing banners, or text badges that trigger automated graphic removal
- Dominant SPICE circuit schematic diagram (inductors, resistors, capacitors, op-amps, diodes, pots, sources, grounds)
- Circuit topology sculpted directly in the silhouette/shape of each instrument's pickups
- Overlaid on a dark-mode ghosted pickup and string hardware silhouette on an EDA schematic grid canvas
- Distinct signature color scheme per Tone Pack edition for instant visual differentiation
"""

import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from allomorph.pipeline.schema import ArtworkPackConfig


def _svg_content(fn: Callable[[str], str]) -> Callable[[str], str]:
    return fn


# ==============================================================================
# 1. DARK HARDWARE SILHOUETTE UNDERLAY PRIMITIVES
# ==============================================================================


def make_ghost_pole(x: float, y: float, r: float = 15.0, accent: str = "#64748b") -> str:
    """Generate a muted dark-mode Alnico pole piece underlay behind the SPICE circuit."""
    return f"""
    <g transform="translate({x:.1f}, {y:.1f})" opacity="0.60">
      <circle cx="0" cy="0" r="{r + 3.5:.1f}" fill="#06080c"/>
      <circle cx="0" cy="0" r="{r:.1f}" fill="#171d28" stroke="#334155" stroke-width="1.8"/>
      <circle cx="0" cy="0" r="{r * 0.48:.1f}" fill="none" stroke="{accent}" stroke-width="1.2" stroke-opacity="0.50"/>
    </g>
    """


def make_pbass_ghost_shell(
    cx: float,
    cy: float,
    x_strings: tuple[float, float],
    accent: str,
    w: float = 460.0,
    h: float = 216.0,
) -> str:
    """Muted P-Bass split-coil bobbin casing underlay with glowing schematic outline."""
    rx = 24.0
    x = cx - w / 2
    y = cy - h / 2
    ear_dx = w / 2 + 20.0
    s1, s2 = x_strings
    pole_off = 25.0

    return f"""
    <g filter="url(#dropShadow)">
      <!-- Left & Right Mounting Ears -->
      <path d="M {x:.1f} {cy - 38:.1f} A 40 40 0 0 0 {x:.1f} {cy + 38:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.55"/>
      <circle cx="{cx - ear_dx:.1f}" cy="{cy:.1f}" r="7.0" fill="#090b10" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.65"/>
      <path d="M {x + w:.1f} {cy - 38:.1f} A 40 40 0 0 1 {x + w:.1f} {cy + 38:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.55"/>
      <circle cx="{cx + ear_dx:.1f}" cy="{cy:.1f}" r="7.0" fill="#090b10" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.65"/>

      <!-- Main Bobbin Silhouette -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="#10141d" fill-opacity="0.92" stroke="#334155" stroke-width="3.0"/>
      <rect x="{x + 8:.1f}" y="{y + 8:.1f}" width="{w - 16:.1f}" height="{h - 16:.1f}" rx="{rx - 5:.1f}" fill="#0a0d14" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.45" stroke-dasharray="10 7"/>

      <!-- Ghost Pole Pieces -->
      {make_ghost_pole(s1 - pole_off, cy, r=16.5, accent=accent)}
      {make_ghost_pole(s1 + pole_off, cy, r=16.5, accent=accent)}
      {make_ghost_pole(s2 - pole_off, cy, r=16.5, accent=accent)}
      {make_ghost_pole(s2 + pole_off, cy, r=16.5, accent=accent)}
    </g>
    """


def make_jbass_ghost_shell(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    w: float = 800.0,
    h: float = 128.0,
) -> str:
    """Slim Jazz Bass single-coil housing underlay with 4 prominent mounting ears and 8 poles."""
    rx = 18.0
    x = cx - w / 2
    y = cy - h / 2
    ear_dx = 195.0
    pole_off = 24.0

    poles = "".join(
        make_ghost_pole(s - pole_off, cy, r=13.5, accent=accent)
        + make_ghost_pole(s + pole_off, cy, r=13.5, accent=accent)
        for s in x_strings
    )

    return f"""
    <g filter="url(#dropShadow)">
      <!-- Top & Bottom J-Bass Semicircular Mounting Ears (Prominent Single-Coil Signature) -->
      <path d="M {cx - ear_dx - 36:.1f} {y:.1f} A 38 38 0 0 1 {cx - ear_dx + 36:.1f} {y:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.4" stroke-opacity="0.70"/>
      <circle cx="{cx - ear_dx:.1f}" cy="{y - 12:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.80"/>
      <path d="M {cx - ear_dx - 36:.1f} {y + h:.1f} A 38 38 0 0 0 {cx - ear_dx + 36:.1f} {y + h:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.4" stroke-opacity="0.70"/>
      <circle cx="{cx - ear_dx:.1f}" cy="{y + h + 12:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.80"/>

      <path d="M {cx + ear_dx - 36:.1f} {y:.1f} A 38 38 0 0 1 {cx + ear_dx + 36:.1f} {y:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.4" stroke-opacity="0.70"/>
      <circle cx="{cx + ear_dx:.1f}" cy="{y - 12:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.80"/>
      <path d="M {cx + ear_dx - 36:.1f} {y + h:.1f} A 38 38 0 0 0 {cx + ear_dx + 36:.1f} {y + h:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.4" stroke-opacity="0.70"/>
      <circle cx="{cx + ear_dx:.1f}" cy="{y + h + 12:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.80"/>

      <!-- Slim Single-Coil J Bobbin Silhouette -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="#10141d" fill-opacity="0.92" stroke="#334155" stroke-width="3.0"/>
      <rect x="{x + 8:.1f}" y="{y + 8:.1f}" width="{w - 16:.1f}" height="{h - 16:.1f}" rx="{rx - 5:.1f}" fill="#0a0d14" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.48" stroke-dasharray="10 7"/>

      {poles}
    </g>
    """


def make_stingray_ghost_shell(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    w: float = 820.0,
    h: float = 480.0,
) -> str:
    """Muted Music Man StingRay humbucker casing underlay with 3 ears and 8 large poles."""
    rx = 28.0
    x = cx - w / 2
    y = cy - h / 2
    ear_y1 = cy - 125.0
    ear_y2 = cy + 125.0
    row_off = 120.0

    upper_poles = "".join(
        make_ghost_pole(s, cy - row_off, r=30.0, accent=accent) for s in x_strings
    )
    lower_poles = "".join(
        make_ghost_pole(s, cy + row_off, r=30.0, accent=accent) for s in x_strings
    )

    return f"""
    <g filter="url(#dropShadow)">
      <!-- 2 Left Mounting Tabs & 1 Right Mounting Tab -->
      <path d="M {x:.1f} {ear_y1 - 34:.1f} A 36 36 0 0 0 {x:.1f} {ear_y1 + 34:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.55"/>
      <circle cx="{x - 18:.1f}" cy="{ear_y1:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.65"/>
      <path d="M {x:.1f} {ear_y2 - 34:.1f} A 36 36 0 0 0 {x:.1f} {ear_y2 + 34:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.55"/>
      <circle cx="{x - 18:.1f}" cy="{ear_y2:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.65"/>
      <path d="M {x + w:.1f} {cy - 40:.1f} A 42 42 0 0 1 {x + w:.1f} {cy + 40:.1f} Z" fill="#12161f" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.55"/>
      <circle cx="{x + w + 18:.1f}" cy="{cy:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.65"/>

      <!-- Main Humbucker Housing -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="#10141d" fill-opacity="0.92" stroke="#334155" stroke-width="3.0"/>
      <rect x="{x + 10:.1f}" y="{y + 10:.1f}" width="{w - 20:.1f}" height="{h / 2 - 14:.1f}" rx="16" fill="#0a0d14" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.42" stroke-dasharray="10 7"/>
      <rect x="{x + 10:.1f}" y="{cy + 4:.1f}" width="{w - 20:.1f}" height="{h / 2 - 14:.1f}" rx="16" fill="#0a0d14" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.42" stroke-dasharray="10 7"/>

      {upper_poles}
      {lower_poles}
    </g>
    """


def make_soapbar_ghost_shell(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    w: float = 840.0,
    h: float = 276.0,
    blades: bool = False,
) -> str:
    """Muted 5-string soapbar casing underlay with inset left/right screw recesses and dual-coil chambers."""
    rx = 26.0
    x = cx - w / 2
    y = cy - h / 2
    row_off = 68.0

    if blades:
        bw = w - 96.0
        bx = cx - bw / 2
        internals = f"""
        <rect x="{bx:.1f}" y="{cy - row_off - 10:.1f}" width="{bw:.1f}" height="20" rx="10" fill="#0d131a" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.42"/>
        <rect x="{bx:.1f}" y="{cy + row_off - 10:.1f}" width="{bw:.1f}" height="20" rx="10" fill="#0d131a" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.42"/>
        """
    else:
        upper_poles = "".join(
            make_ghost_pole(s, cy - row_off, r=18.5, accent=accent) for s in x_strings
        )
        lower_poles = "".join(
            make_ghost_pole(s, cy + row_off, r=18.5, accent=accent) for s in x_strings
        )
        internals = upper_poles + lower_poles

    return f"""
    <g filter="url(#dropShadow)">
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="#10141d" fill-opacity="0.92" stroke="#334155" stroke-width="3.2"/>
      <rect x="{x + 12:.1f}" y="{y + 12:.1f}" width="{w - 24:.1f}" height="{h / 2 - 16:.1f}" rx="14" fill="#0a0d14" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.45" stroke-dasharray="10 7"/>
      <rect x="{x + 12:.1f}" y="{cy + 4:.1f}" width="{w - 24:.1f}" height="{h / 2 - 16:.1f}" rx="14" fill="#0a0d14" fill-opacity="0.94" stroke="{accent}" stroke-width="1.6" stroke-opacity="0.45" stroke-dasharray="10 7"/>
      <!-- Authentic Soapbar Left & Right Inset Semicircular Screw Cutouts -->
      <path d="M {x + 4:.1f} {cy - 28:.1f} A 28 28 0 0 1 {x + 4:.1f} {cy + 28:.1f} Z" fill="#080a0f" stroke="{accent}" stroke-width="1.8" stroke-opacity="0.60"/>
      <circle cx="{x + 19:.1f}" cy="{cy:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.75"/>
      <path d="M {x + w - 4:.1f} {cy - 28:.1f} A 28 28 0 0 0 {x + w - 4:.1f} {cy + 28:.1f} Z" fill="#080a0f" stroke="{accent}" stroke-width="1.8" stroke-opacity="0.60"/>
      <circle cx="{x + w - 19:.1f}" cy="{cy:.1f}" r="6.5" fill="#090b10" stroke="{accent}" stroke-width="1.5" stroke-opacity="0.75"/>
      {internals}
    </g>
    """


# ==============================================================================
# 2. SPICE SCHEMATIC COMPONENT PRIMITIVES (CLEAN SINGLE-SIDE LABELS)
# ==============================================================================


def _badge_label(x: float, y: float, text: str, accent: str, anchor: str = "middle") -> str:
    """Render a high-contrast monospace component label with a dark backing pill so wires never clash."""
    if not text:
        return ""
    est_w = len(text) * 8.6 + 12.0
    rx = (
        x - est_w / 2 if anchor == "middle" else (x - 4.0 if anchor == "start" else x - est_w + 4.0)
    )
    return f"""
    <g>
      <rect x="{rx:.1f}" y="{y - 13:.1f}" width="{est_w:.1f}" height="18" rx="4" fill="#0a0d14" fill-opacity="0.88"/>
      <text x="{x:.1f}" y="{y:.1f}" fill="{accent}" font-size="13.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-weight="700" text-anchor="{anchor}">{text}</text>
    </g>
    """


def spice_node(x: float, y: float, accent: str) -> str:
    """Solder junction node on a SPICE net wire."""
    return f"""<circle cx="{x:.1f}" cy="{y:.1f}" r="6.0" fill="{accent}" stroke="#ffffff" stroke-width="1.4"/>"""


def spice_port(x: float, y: float, label: str, accent: str, align: str = "right") -> str:
    """SPICE terminal port flag/node."""
    tx = x + 15 if align == "right" else x - 15
    anchor = "start" if align == "right" else "end"
    return f"""
    <g>
      <circle cx="{x:.1f}" cy="{y:.1f}" r="7.5" fill="#0b0e14" stroke="{accent}" stroke-width="3.0"/>
      <circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="#ffffff"/>
      {_badge_label(tx, y + 5, label, "#f8fafc", anchor=anchor)}
    </g>
    """


def spice_gnd(x: float, y: float, accent: str, stem: float = 18.0) -> str:
    """Standard 3-line SPICE ground symbol."""
    y1 = y + stem
    return f"""
    <g>
      <line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{accent}" stroke-width="3.0"/>
      <line x1="{x - 20:.1f}" y1="{y1:.1f}" x2="{x + 20:.1f}" y2="{y1:.1f}" stroke="#f1f5f9" stroke-width="3.2"/>
      <line x1="{x - 13:.1f}" y1="{y1 + 7:.1f}" x2="{x + 13:.1f}" y2="{y1 + 7:.1f}" stroke="#f1f5f9" stroke-width="2.8"/>
      <line x1="{x - 5:.1f}" y1="{y1 + 14:.1f}" x2="{x + 5:.1f}" y2="{y1 + 14:.1f}" stroke="#f1f5f9" stroke-width="2.4"/>
    </g>
    """


def spice_resistor_h(
    cx: float,
    cy: float,
    label: str = "",
    w: float = 68.0,
    h: float = 22.0,
    accent: str = "#38bdf8",
    label_pos: str = "top",
    pot: bool = False,
) -> str:
    """Horizontal SPICE resistor or potentiometer symbol with single-side pill label."""
    x = cx - w / 2
    y = cy - h / 2
    ty = y - 8 if label_pos == "top" else y + h + 18
    dx = w / 6.0
    zig = (
        f"M {x:.1f} {cy:.1f} "
        f"L {x + dx * 0.5:.1f} {cy - h * 0.42:.1f} "
        f"L {x + dx * 1.5:.1f} {cy + h * 0.42:.1f} "
        f"L {x + dx * 2.5:.1f} {cy - h * 0.42:.1f} "
        f"L {x + dx * 3.5:.1f} {cy + h * 0.42:.1f} "
        f"L {x + dx * 4.5:.1f} {cy - h * 0.42:.1f} "
        f"L {x + dx * 5.5:.1f} {cy + h * 0.42:.1f} "
        f"L {x + w:.1f} {cy:.1f}"
    )
    wiper = ""
    if pot:
        wiper = f"""
        <line x1="{cx - 16:.1f}" y1="{cy + 18:.1f}" x2="{cx + 16:.1f}" y2="{cy - 18:.1f}" stroke="{accent}" stroke-width="2.6"/>
        <polygon points="{cx + 19:.1f},{cy - 21:.1f} {cx + 9:.1f},{cy - 18:.1f} {cx + 16:.1f},{cy - 10:.1f}" fill="{accent}"/>
        """
    return f"""
    <g>
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="#0d1118" stroke="#f8fafc" stroke-width="2.6"/>
      <path d="{zig}" fill="none" stroke="{accent}" stroke-width="1.8" stroke-opacity="0.85" stroke-linejoin="round"/>
      {wiper}
      {_badge_label(cx, ty, label, accent)}
    </g>
    """


def spice_resistor_v(
    cx: float,
    cy: float,
    label: str = "",
    w: float = 22.0,
    h: float = 64.0,
    accent: str = "#38bdf8",
    pot: bool = False,
    align: str = "right",
) -> str:
    """Vertical SPICE resistor or potentiometer symbol."""
    x = cx - w / 2
    y = cy - h / 2
    tx = cx + w / 2 + 12 if align == "right" else cx - w / 2 - 12
    anchor = "start" if align == "right" else "end"
    dy = h / 6.0
    zig = (
        f"M {cx:.1f} {y:.1f} "
        f"L {cx - w * 0.42:.1f} {y + dy * 0.5:.1f} "
        f"L {cx + w * 0.42:.1f} {y + dy * 1.5:.1f} "
        f"L {cx - w * 0.42:.1f} {y + dy * 2.5:.1f} "
        f"L {cx + w * 0.42:.1f} {y + dy * 3.5:.1f} "
        f"L {cx - w * 0.42:.1f} {y + dy * 4.5:.1f} "
        f"L {cx + w * 0.42:.1f} {y + dy * 5.5:.1f} "
        f"L {cx:.1f} {y + h:.1f}"
    )
    wiper = ""
    if pot:
        wiper = f"""
        <line x1="{cx - 20:.1f}" y1="{cy + 15:.1f}" x2="{cx + 20:.1f}" y2="{cy - 15:.1f}" stroke="{accent}" stroke-width="2.6"/>
        <polygon points="{cx + 23:.1f},{cy - 18:.1f} {cx + 13:.1f},{cy - 16:.1f} {cx + 19:.1f},{cy - 8:.1f}" fill="{accent}"/>
        """
    return f"""
    <g>
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="#0d1118" stroke="#f8fafc" stroke-width="2.6"/>
      <path d="{zig}" fill="none" stroke="{accent}" stroke-width="1.8" stroke-opacity="0.85" stroke-linejoin="round"/>
      {wiper}
      {_badge_label(tx, cy + 5, label, accent, anchor=anchor)}
    </g>
    """


def spice_inductor_h(
    cx: float,
    cy: float,
    loops: int = 4,
    r: float = 15.0,
    label: str = "",
    accent: str = "#38bdf8",
    label_pos: str = "top",
) -> str:
    """Horizontal multi-loop SPICE inductor with ferromagnetic core bars."""
    w = loops * 2.0 * r
    x0 = cx - w / 2
    arcs = f"M {x0:.1f} {cy:.1f} " + " ".join(
        f"A {r:.1f} {r:.1f} 0 0 1 {x0 + (i + 1) * 2.0 * r:.1f} {cy:.1f}" for i in range(loops)
    )
    ty = cy - r - 15 if label_pos == "top" else cy + 20
    return f"""
    <g>
      <rect x="{x0 - 3:.1f}" y="{cy - r - 9:.1f}" width="{w + 6:.1f}" height="{r + 14:.1f}" fill="#0a0d14" fill-opacity="0.82" rx="4"/>
      <line x1="{x0 + 4:.1f}" y1="{cy - r - 5:.1f}" x2="{x0 + w - 4:.1f}" y2="{cy - r - 5:.1f}" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.90"/>
      <line x1="{x0 + 4:.1f}" y1="{cy - r - 10:.1f}" x2="{x0 + w - 4:.1f}" y2="{cy - r - 10:.1f}" stroke="{accent}" stroke-width="1.6" stroke-dasharray="7 5" stroke-opacity="0.70"/>
      <path d="{arcs}" fill="none" stroke="#f8fafc" stroke-width="3.2" stroke-linecap="round"/>
      <circle cx="{x0 + 6:.1f}" cy="{cy - r + 5:.1f}" r="3.8" fill="{accent}"/>
      {_badge_label(cx, ty, label, "#f8fafc")}
    </g>
    """


def spice_capacitor_h(
    cx: float,
    cy: float,
    plate_h: float = 32.0,
    gap: float = 12.0,
    label: str = "",
    accent: str = "#38bdf8",
    label_pos: str = "bottom",
) -> str:
    """Horizontal SPICE capacitor (vertical parallel plates on a horizontal wire)."""
    x1 = cx - gap / 2
    x2 = cx + gap / 2
    ty = cy + plate_h / 2 + 18 if label_pos == "bottom" else cy - plate_h / 2 - 8
    return f"""
    <g>
      <rect x="{x1 - 4:.1f}" y="{cy - plate_h / 2:.1f}" width="{gap + 8:.1f}" height="{plate_h:.1f}" fill="#0a0d14"/>
      <line x1="{x1:.1f}" y1="{cy - plate_h / 2:.1f}" x2="{x1:.1f}" y2="{cy + plate_h / 2:.1f}" stroke="#f8fafc" stroke-width="3.5"/>
      <line x1="{x2:.1f}" y1="{cy - plate_h / 2:.1f}" x2="{x2:.1f}" y2="{cy + plate_h / 2:.1f}" stroke="#f8fafc" stroke-width="3.5"/>
      {_badge_label(cx, ty, label, accent)}
    </g>
    """


def spice_capacitor_v(
    cx: float,
    cy: float,
    plate_w: float = 32.0,
    gap: float = 12.0,
    label: str = "",
    accent: str = "#38bdf8",
    align: str = "right",
) -> str:
    """Vertical SPICE capacitor (horizontal parallel plates on a vertical wire)."""
    y1 = cy - gap / 2
    y2 = cy + gap / 2
    tx = cx + plate_w / 2 + 10 if align == "right" else cx - plate_w / 2 - 10
    anchor = "start" if align == "right" else "end"
    return f"""
    <g>
      <rect x="{cx - plate_w / 2:.1f}" y="{y1 - 4:.1f}" width="{plate_w:.1f}" height="{gap + 8:.1f}" fill="#0a0d14"/>
      <line x1="{cx - plate_w / 2:.1f}" y1="{y1:.1f}" x2="{cx + plate_w / 2:.1f}" y2="{y1:.1f}" stroke="#f8fafc" stroke-width="3.5"/>
      <line x1="{cx - plate_w / 2:.1f}" y1="{y2:.1f}" x2="{cx + plate_w / 2:.1f}" y2="{y2:.1f}" stroke="#f8fafc" stroke-width="3.5"/>
      {_badge_label(tx, cy + 5, label, accent, anchor=anchor)}
    </g>
    """


def spice_ac_source(
    cx: float,
    cy: float,
    r: float = 23.0,
    label: str = "",
    accent: str = "#38bdf8",
    label_pos: str = "top",
) -> str:
    """SPICE AC/EMF induced string flux voltage source symbol."""
    sine = f"M {cx - 12:.1f} {cy:.1f} Q {cx - 6:.1f} {cy - 12:.1f} {cx:.1f} {cy:.1f} T {cx + 12:.1f} {cy:.1f}"
    ty = cy - r - 8 if label_pos == "top" else cy + r + 18
    return f"""
    <g>
      <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#0d1118" stroke="#f8fafc" stroke-width="2.8"/>
      <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r - 4.0:.1f}" fill="none" stroke="{accent}" stroke-width="1.4" stroke-opacity="0.60"/>
      <path d="{sine}" fill="none" stroke="{accent}" stroke-width="2.8" stroke-linecap="round"/>
      {_badge_label(cx, ty, label, "#f8fafc")}
    </g>
    """


def spice_zener_pair_h(
    cx: float,
    cy: float,
    label: str = "D_sat",
    accent: str = "#38bdf8",
    label_pos: str = "bottom",
) -> str:
    """Back-to-back Zener diode pair representing nonlinear magnetic flux saturation."""
    ty = cy + 28 if label_pos == "bottom" else cy - 16
    return f"""
    <g>
      <rect x="{cx - 30:.1f}" y="{cy - 14:.1f}" width="60" height="28" fill="#0a0d14"/>
      <line x1="{cx - 30:.1f}" y1="{cy:.1f}" x2="{cx + 30:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="2.6"/>
      <polygon points="{cx - 19:.1f},{cy - 11:.1f} {cx - 19:.1f},{cy + 11:.1f} {cx - 4:.1f},{cy:.1f}" fill="#0d1118" stroke="#f8fafc" stroke-width="2.2"/>
      <polyline points="{cx - 8:.1f},{cy - 12:.1f} {cx - 4:.1f},{cy - 12:.1f} {cx - 4:.1f},{cy + 12:.1f} {cx:.1f},{cy + 12:.1f}" fill="none" stroke="{accent}" stroke-width="2.2"/>
      <polygon points="{cx + 19:.1f},{cy - 11:.1f} {cx + 19:.1f},{cy + 11:.1f} {cx + 4:.1f},{cy:.1f}" fill="#0d1118" stroke="#f8fafc" stroke-width="2.2"/>
      <polyline points="{cx + 8:.1f},{cy - 12:.1f} {cx + 4:.1f},{cy - 12:.1f} {cx + 4:.1f},{cy + 12:.1f} {cx:.1f},{cy + 12:.1f}" fill="none" stroke="{accent}" stroke-width="2.2"/>
      {_badge_label(cx, ty, label, accent)}
    </g>
    """


def spice_opamp(
    cx: float,
    cy: float,
    w: float = 86.0,
    h: float = 76.0,
    label: str = "U1",
    accent: str = "#38bdf8",
) -> str:
    """SPICE operational amplifier triangle pointing right (+ on top-left, - on bottom-left)."""
    xL = cx - w / 2
    xR = cx + w / 2
    yT = cy - h / 2
    yB = cy + h / 2
    return f"""
    <g>
      <line x1="{cx - 6:.1f}" y1="{cy - h * 0.28:.1f}" x2="{cx - 6:.1f}" y2="{cy - h * 0.52:.1f}" stroke="{accent}" stroke-width="2.2"/>
      <circle cx="{cx - 6:.1f}" cy="{cy - h * 0.52:.1f}" r="3.5" fill="{accent}"/>
      <line x1="{cx - 6:.1f}" y1="{cy + h * 0.28:.1f}" x2="{cx - 6:.1f}" y2="{cy + h * 0.52:.1f}" stroke="{accent}" stroke-width="2.2"/>
      <circle cx="{cx - 6:.1f}" cy="{cy + h * 0.52:.1f}" r="3.5" fill="{accent}"/>

      <polygon points="{xL:.1f},{yT:.1f} {xL:.1f},{yB:.1f} {xR:.1f},{cy:.1f}" fill="#0d1118" stroke="#f8fafc" stroke-width="3.0" stroke-linejoin="round"/>
      <polygon points="{xL + 5:.1f},{yT + 7:.1f} {xL + 5:.1f},{yB - 7:.1f} {xR - 9:.1f},{cy:.1f}" fill="none" stroke="{accent}" stroke-width="1.4" stroke-opacity="0.50"/>

      <text x="{xL + 15:.1f}" y="{cy - 13:.1f}" fill="{accent}" font-size="18" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-weight="800" text-anchor="middle">+</text>
      <text x="{xL + 15:.1f}" y="{cy + 23:.1f}" fill="#f8fafc" font-size="18" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-weight="800" text-anchor="middle">−</text>
      <text x="{cx - 4:.1f}" y="{cy + 5:.1f}" fill="#f8fafc" font-size="14" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-weight="700" text-anchor="middle">{label}</text>
    </g>
    """


# ==============================================================================
# 3. COMPOSITE SPICE PICKUP MACRO BLOCKS (SCULPTED IN EXACT PICKUP SHAPE)
# ==============================================================================


def make_pbass_spice_bobbin(
    cx: float,
    cy: float,
    x_strings: tuple[float, float],
    accent: str,
    idx: int = 1,
    l_lbl: str = "L1 1.9H",
    rdc_lbl: str = "R_dc 5.25k",
    reddy_lbl: str = "R_eddy 90k",
    cp_lbl: str = "C_p 120pF",
    w: float = 396.0,
    h: float = 216.0,
    extra_right_pot: str = "",
) -> str:
    """Render a Split-P bobbin silhouette centered on its string pair with a 100% symmetric SPICE tank."""
    shell = make_pbass_ghost_shell(cx, cy, x_strings, accent, w=w, h=h)
    x_ear_L = cx - w / 2 - 20.0
    x_ear_R = cx + w / 2 + 20.0
    x_n1 = cx - 154.0
    x_n2 = cx + 154.0
    yT = cy - h / 2 + 34.0
    yB = cy + h / 2 - 34.0

    return f"""
    {shell}
    <!-- Symmetric SPICE Subcircuit Sculpted Inside Split-P Bobbin {idx} -->
    <g>
      <rect x="{x_n1:.1f}" y="{yT:.1f}" width="{x_n2 - x_n1:.1f}" height="{yB - yT:.1f}" fill="none" stroke="{accent}" stroke-width="2.8"/>
      <line x1="{x_ear_L:.1f}" y1="{cy:.1f}" x2="{x_ear_R:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="3.2"/>

      <!-- Top Rail: Eddy Loss / Pot Resistor (Dead-Center at cx) -->
      {spice_resistor_h(cx, yT, label=reddy_lbl, w=72.0, h=22.0, accent=accent, label_pos="top", pot=bool(extra_right_pot))}

      <!-- Center Rail: AC String Source (Left String) + Inductor (Center) + DC Resistance (Right String) -->
      {spice_ac_source(cx - 100.0, cy, r=21.0, label=f"V_str{idx}", accent=accent, label_pos="top")}
      {spice_inductor_h(cx, cy, loops=4, r=14.0, label=l_lbl, accent=accent, label_pos="bottom")}
      {spice_resistor_h(cx + 100.0, cy, label=rdc_lbl, w=64.0, h=21.0, accent=accent, label_pos="bottom")}

      <!-- Bottom Rail: Winding Capacitance + Saturation Clipper (Symmetric around cx) -->
      {spice_capacitor_h(cx - 66.0, yB, plate_h=28.0, gap=11.0, label=cp_lbl, accent=accent, label_pos="bottom")}
      {spice_zener_pair_h(cx + 66.0, yB, label=f"D_sat{idx}", accent=accent, label_pos="bottom")}

      {spice_node(x_ear_L, cy, accent)}
      {spice_node(x_n1, cy, accent)}
      {spice_node(x_n2, cy, accent)}
      {spice_node(x_ear_R, cy, accent)}
    </g>
    """


def make_jbass_spice_bobbin(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    tag: str = "N",
    l_lbl: str = "L_N 3.2H",
    rdc_lbl: str = "R_dc 7.2k",
    reddy_lbl: str = "R_eddy 135k",
    cp_lbl: str = "C_p 70pF",
    ctone_lbl: str = "C_t 47nF",
    w: float = 800.0,
    h: float = 128.0,
) -> str:
    """Render a slim Jazz Bass single-coil silhouette with a single-line inline SPICE coil network."""
    _ = ctone_lbl
    shell = make_jbass_ghost_shell(cx, cy, x_strings, accent, w=w, h=h)
    xL = cx - w / 2 + 34.0
    xR = cx + w / 2 - 34.0
    yT = cy - 38.0

    x_src = cx - 265.0
    x_n1 = cx - 195.0
    x_n_mid = cx
    x_n2 = cx + 195.0
    x_zener = cx + 265.0

    return f"""
    {shell}
    <!-- Slim Single-Coil Inline SPICE Subcircuit Inside Jazz Bobbin ({tag}) -->
    <g>
      <!-- Compact Upper Parasitic Shunt Arch (R_eddy & C_p) -->
      <polyline points="{x_n1:.1f},{cy:.1f} {x_n1:.1f},{yT:.1f} {x_n2:.1f},{yT:.1f} {x_n2:.1f},{cy:.1f}" fill="none" stroke="{accent}" stroke-width="2.4"/>
      <line x1="{x_n_mid:.1f}" y1="{yT:.1f}" x2="{x_n_mid:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="2.2"/>

      <!-- Main Single-Coil Series Signal Axis -->
      <line x1="{xL:.1f}" y1="{cy:.1f}" x2="{xR + 34:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="3.2"/>

      <!-- Left Ground at Neck/Bridge Bobbin End -->
      <line x1="{xL:.1f}" y1="{cy:.1f}" x2="{xL - 40:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="3.0"/>
      {spice_gnd(xL - 40, cy, accent, stem=16.0)}

      <!-- Single-Coil Inline Components Along Center Axis -->
      {spice_ac_source(x_src, cy, r=20.0, label=f"E_{tag}", accent=accent, label_pos="bottom")}
      {spice_inductor_h(cx - 98.0, cy, loops=4, r=14.0, label=l_lbl, accent=accent, label_pos="bottom")}
      {spice_resistor_h(cx + 98.0, cy, label=rdc_lbl, w=70.0, h=20.0, accent=accent, label_pos="bottom")}
      {spice_zener_pair_h(x_zener, cy, label=f"D_{tag}", accent=accent, label_pos="bottom")}

      <!-- Compact Shunt Components Along Top Arch (Labels Above Pickup) -->
      {spice_resistor_h(cx - 98.0, yT, label=reddy_lbl, w=68.0, h=18.0, accent=accent, label_pos="top")}
      {spice_capacitor_h(cx + 98.0, yT, plate_h=24.0, gap=10.0, label=cp_lbl, accent=accent, label_pos="top")}

      {spice_node(xL, cy, accent)}
      {spice_node(x_n1, cy, accent)}
      {spice_node(x_n_mid, yT, accent)}
      {spice_node(x_n_mid, cy, accent)}
      {spice_node(x_n2, cy, accent)}
      {spice_node(xR + 34, cy, accent)}
    </g>
    """


def make_soapbar_spice_bobbin(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    tag: str = "N",
    l_lbl: str = "3.8H",
    rdc_lbl: str = "8.6k",
    cp_lbl: str = "245pF",
    w: float = 820.0,
    h: float = 264.0,
    blades: bool = False,
    with_internal_opamp: bool = False,
) -> str:
    """Render a large 5-string dual-coil Soapbar silhouette filled with a dual-row humbucker SPICE circuit."""
    shell = make_soapbar_ghost_shell(cx, cy, x_strings, accent, w=w, h=h, blades=blades)
    xL = cx - w / 2 + 38.0
    xR = cx + w / 2 - 38.0
    y_up = cy - 64.0
    y_lo = cy + 64.0
    y_top = cy - h / 2 + 28.0
    y_bot = cy + h / 2 - 28.0

    x_src = xL + 50.0
    x_n1 = xL + 112.0
    x_sum = xR - 150.0 if with_internal_opamp else xR - 50.0

    if with_internal_opamp:
        x_op = xR - 70.0
        right_stage = f"""
        <line x1="{x_sum:.1f}" y1="{y_top:.1f}" x2="{x_sum:.1f}" y2="{y_bot:.1f}" stroke="{accent}" stroke-width="2.6"/>
        <line x1="{x_sum:.1f}" y1="{y_up:.1f}" x2="{x_sum + 24:.1f}" y2="{y_up:.1f}" stroke="{accent}" stroke-width="2.8"/>
        <line x1="{x_sum + 24:.1f}" y1="{y_up:.1f}" x2="{x_sum + 24:.1f}" y2="{cy - 20:.1f}" stroke="{accent}" stroke-width="2.8"/>
        <line x1="{x_sum + 24:.1f}" y1="{cy - 20:.1f}" x2="{x_op - 43:.1f}" y2="{cy - 20:.1f}" stroke="{accent}" stroke-width="2.8"/>

        <line x1="{x_sum:.1f}" y1="{y_lo:.1f}" x2="{x_sum + 24:.1f}" y2="{y_lo:.1f}" stroke="{accent}" stroke-width="2.8"/>
        <line x1="{x_sum + 24:.1f}" y1="{y_lo:.1f}" x2="{x_sum + 24:.1f}" y2="{cy + 20:.1f}" stroke="{accent}" stroke-width="2.8"/>
        <line x1="{x_sum + 24:.1f}" y1="{cy + 20:.1f}" x2="{x_op - 43:.1f}" y2="{cy + 20:.1f}" stroke="{accent}" stroke-width="2.8"/>

        {spice_opamp(x_op, cy, w=82.0, h=72.0, label=f"U_{tag}", accent=accent)}
        <line x1="{x_op + 41:.1f}" y1="{cy:.1f}" x2="{xR + 38:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="3.2"/>
        {spice_node(x_sum, y_up, accent)}
        {spice_node(x_sum, y_lo, accent)}
        {spice_node(xR + 38, cy, accent)}
        """
    else:
        right_stage = f"""
        <line x1="{x_sum:.1f}" y1="{y_top:.1f}" x2="{x_sum:.1f}" y2="{y_bot:.1f}" stroke="{accent}" stroke-width="2.8"/>
        <line x1="{x_sum:.1f}" y1="{cy:.1f}" x2="{xR + 38:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="3.2"/>
        {spice_capacitor_v(x_sum - 78, cy, plate_w=28.0, gap=11.0, label=f"C_p {cp_lbl}", accent=accent, align="left")}
        <line x1="{x_sum - 78:.1f}" y1="{y_up:.1f}" x2="{x_sum - 78:.1f}" y2="{cy - 6:.1f}" stroke="{accent}" stroke-width="2.6"/>
        <line x1="{x_sum - 78:.1f}" y1="{cy + 6:.1f}" x2="{x_sum - 78:.1f}" y2="{y_lo:.1f}" stroke="{accent}" stroke-width="2.6"/>
        {spice_node(x_sum - 78, y_up, accent)}
        {spice_node(x_sum - 78, y_lo, accent)}
        {spice_node(x_sum, y_up, accent)}
        {spice_node(x_sum, cy, accent)}
        {spice_node(x_sum, y_lo, accent)}
        {spice_node(xR + 38, cy, accent)}
        """

    return f"""
    {shell}
    <!-- Dual-Coil Humbucker SPICE Network Sculpted Inside Soapbar ({tag}) -->
    <g>
      <line x1="{x_n1:.1f}" y1="{y_top:.1f}" x2="{x_sum:.1f}" y2="{y_top:.1f}" stroke="{accent}" stroke-width="2.4"/>
      <line x1="{x_n1:.1f}" y1="{y_bot:.1f}" x2="{x_sum:.1f}" y2="{y_bot:.1f}" stroke="{accent}" stroke-width="2.4"/>
      <line x1="{x_n1:.1f}" y1="{y_top:.1f}" x2="{x_n1:.1f}" y2="{y_bot:.1f}" stroke="{accent}" stroke-width="2.6"/>

      <!-- Left Ground Reference -->
      <line x1="{xL:.1f}" y1="{y_up:.1f}" x2="{xL:.1f}" y2="{y_lo:.1f}" stroke="{accent}" stroke-width="2.8"/>
      <line x1="{xL:.1f}" y1="{cy:.1f}" x2="{xL - 40:.1f}" y2="{cy:.1f}" stroke="{accent}" stroke-width="3.0"/>
      {spice_gnd(xL - 40, cy, accent, stem=16.0)}
      {spice_node(xL, cy, accent)}

      <!-- Upper Coil Row -->
      <line x1="{xL:.1f}" y1="{y_up:.1f}" x2="{x_sum:.1f}" y2="{y_up:.1f}" stroke="{accent}" stroke-width="3.0"/>
      {spice_ac_source(x_src, y_up, r=20.0, label=f"V_{tag}1", accent=accent, label_pos="top")}
      {spice_inductor_h(cx - 135, y_up, loops=4, r=14.0, label=f"L_{tag}1 {l_lbl}", accent=accent, label_pos="bottom")}
      {spice_resistor_h(cx + 35, y_up, label=f"R_{tag}1 {rdc_lbl}", w=68.0, h=20.0, accent=accent, label_pos="bottom")}
      {spice_resistor_h(cx - 45, y_top, label="R_eddy 90k", w=64.0, h=18.0, accent=accent, label_pos="top")}

      <!-- Lower Coil Row -->
      <line x1="{xL:.1f}" y1="{y_lo:.1f}" x2="{x_sum:.1f}" y2="{y_lo:.1f}" stroke="{accent}" stroke-width="3.0"/>
      {spice_ac_source(x_src, y_lo, r=20.0, label=f"V_{tag}2", accent=accent, label_pos="bottom")}
      {spice_inductor_h(cx - 135, y_lo, loops=4, r=14.0, label=f"L_{tag}2 {l_lbl}", accent=accent, label_pos="top")}
      {spice_resistor_h(cx + 35, y_lo, label=f"R_{tag}2 {rdc_lbl}", w=68.0, h=20.0, accent=accent, label_pos="top")}
      {spice_capacitor_h(cx - 45, y_bot, plate_h=26.0, gap=10.0, label=f"C_{tag} {cp_lbl}", accent=accent, label_pos="bottom")}

      {spice_node(x_n1, y_up, accent)}
      {spice_node(x_n1, y_lo, accent)}
      {right_stage}
    </g>
    """


# ==============================================================================
# 4. FULL TONE PACK SCHEMATIC LAYOUTS (TIGHT PICKUP-SILHOUETTE GEOMETRY)
# ==============================================================================


def generate_pack_svg(model_key: str) -> str:
    """Generate a Tone3000-compliant SPICE schematic SVG sculpted in the shape of the pickups."""
    strings_4 = [300.0, 500.0, 700.0, 900.0]
    strings_5 = [280.0, 440.0, 600.0, 760.0, 920.0]

    configs = {
        "precision": {
            "accent": "#f59e0b",
            "title": "STANDARD PRECISION BASS",
            "scale": '34" SCALE',
            "badge2": "ALNICO SPLIT-P",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 35,
            "num_strings": 4,
            "hero_scale": 1.0,
            "content": _svg_content(
                lambda accent: (
                    f"""
                    <!-- Upper-Left Split-P Half (Centered on Bass Strings E=300, A=500 -> cx=400, w=396 -> x=202..598) -->
                    {make_pbass_spice_bobbin(400, 430, (strings_4[0], strings_4[1]), accent, idx=1, l_lbl="L1 1.9H", rdc_lbl="R1 5.25k", reddy_lbl="R_e1 90k", cp_lbl="C1 120pF", w=396, h=224)}

                    <!-- Lower-Right Split-P Half (Centered on Treble Strings D=700, G=900 -> cx=800, w=396 -> x=602..998) -->
                    {make_pbass_spice_bobbin(800, 690, (strings_4[2], strings_4[3]), accent, idx=2, l_lbl="L2 1.9H", rdc_lbl="R2 5.25k", reddy_lbl="R_vol 250k", cp_lbl="C_t 47nF", w=396, h=224, extra_right_pot="pot")}

                    <!-- Left Ground on E/A Bobbin Mounting Ear -->
                    <line x1="182" y1="430" x2="125" y2="430" stroke="{accent}" stroke-width="3.2"/>
                    {spice_gnd(125, 430, accent)}

                    <!-- Tight Diagonal Series Hum-Canceling Link Between the Two Split-P Halves -->
                    <polyline points="618,430 656,430 544,690 582,690" fill="none" stroke="{accent}" stroke-width="3.6" stroke-linejoin="round"/>

                    <!-- Output Terminal on Right Mounting Ear of D/G Bobbin -->
                    <line x1="1018" y1="690" x2="1065" y2="690" stroke="{accent}" stroke-width="3.2"/>
                    {spice_port(1065, 690, "V_OUT", accent, align="right")}
                    """
                )
            ),
        },
        "jazz": {
            "accent": "#0ea5e9",
            "title": "STANDARD JAZZ BASS",
            "scale": '34" SCALE',
            "badge2": "60s DUAL-J",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 25,
            "num_strings": 4,
            "hero_scale": 1.0,
            "content": _svg_content(
                lambda accent: (
                    f"""
                    <!-- Slim Neck Single-Coil J Pickup SPICE Bobbin (Shorter Width w=780, Slim h=128) -->
                    {make_jbass_spice_bobbin(600, 320, strings_4, accent, tag="N", l_lbl="L_N 3.2H", rdc_lbl="R_N 7.2k", reddy_lbl="R_eN 135k", cp_lbl="C_N 70pF", w=780, h=128)}

                    <!-- Slim Bridge Single-Coil J Pickup SPICE Bobbin (Wider Bridge Width w=820, Slim h=128) -->
                    {make_jbass_spice_bobbin(600, 790, strings_4, accent, tag="B", l_lbl="L_B 3.6H", rdc_lbl="R_B 7.8k", reddy_lbl="R_eB 125k", cp_lbl="C_B 70pF", w=820, h=128)}

                    <!-- Passive J-Bass Control Plate Ladder (Zero Wire Crossings, Grounded Bottom Rail) -->
                    <polyline points="990,320 1045,320 1045,475 300,475" fill="none" stroke="{accent}" stroke-width="3.0"/>
                    <polyline points="1010,790 1045,790 1045,625 300,625" fill="none" stroke="{accent}" stroke-width="3.0"/>

                    <!-- Neck Volume Pot (Aligned on E String x=300) -->
                    <line x1="300" y1="475" x2="300" y2="625" stroke="{accent}" stroke-width="2.8"/>
                    {spice_resistor_v(300, 550, label="R_vol_N 250k", accent=accent, pot=True, align="left")}
                    {spice_gnd(300, 625, accent, stem=16.0)}
                    {spice_node(300, 625, accent)}

                    <!-- Bridge Volume Pot (Aligned on A String x=500) -->
                    <line x1="500" y1="475" x2="500" y2="625" stroke="{accent}" stroke-width="2.8"/>
                    {spice_resistor_v(500, 550, label="R_vol_B 250k", accent=accent, pot=True, align="right")}
                    {spice_node(500, 475, accent)}
                    {spice_node(500, 625, accent)}
                    {spice_gnd(500, 625, accent, stem=16.0)}

                    <!-- Master Tone Pot (Aligned on D String x=700) -->
                    <line x1="700" y1="475" x2="700" y2="625" stroke="{accent}" stroke-width="2.8"/>
                    {spice_resistor_v(700, 550, label="R_tone 250k", accent=accent, pot=True, align="right")}
                    {spice_node(700, 475, accent)}
                    {spice_node(700, 625, accent)}
                    {spice_gnd(700, 625, accent, stem=16.0)}

                    <!-- Tone Shunt Capacitor (Aligned on G String x=900) -->
                    <line x1="900" y1="475" x2="900" y2="625" stroke="{accent}" stroke-width="2.8"/>
                    {spice_capacitor_v(900, 550, plate_w=28.0, gap=10.0, label="C_t 47nF", accent=accent, align="right")}
                    {spice_node(900, 475, accent)}
                    {spice_node(900, 625, accent)}
                    {spice_gnd(900, 625, accent, stem=16.0)}

                    <!-- Passive Output Jack Bus -->
                    <line x1="1045" y1="475" x2="1045" y2="625" stroke="{accent}" stroke-width="3.0"/>
                    <line x1="1045" y1="550" x2="1085" y2="550" stroke="{accent}" stroke-width="3.2"/>
                    {spice_node(1045, 475, accent)}
                    {spice_node(1045, 550, accent)}
                    {spice_node(1045, 625, accent)}
                    {spice_port(1085, 550, "V_OUT", accent, align="right")}
                    """
                )
            ),
        },
        "pj": {
            "accent": "#a855f7",
            "title": "STANDARD P/J BASS",
            "scale": '34" SCALE',
            "badge2": "P/J HYBRID",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 25,
            "num_strings": 4,
            "hero_scale": 1.0,
            "content": _svg_content(
                lambda accent: (
                    f"""
                    <!-- Upper-Left Split-P E/A Half (Centered on E=300, A=500 -> cx=400, w=396) -->
                    {make_pbass_spice_bobbin(400, 280, (strings_4[0], strings_4[1]), accent, idx=1, l_lbl="L_P1 1.9H", rdc_lbl="R_P1 5.25k", reddy_lbl="R_e1 90k", cp_lbl="C1 120pF", w=396, h=204)}

                    <!-- Middle-Right Split-P D/G Half (Centered on D=700, G=900 -> cx=800, w=396) -->
                    {make_pbass_spice_bobbin(800, 520, (strings_4[2], strings_4[3]), accent, idx=2, l_lbl="L_P2 1.9H", rdc_lbl="R_P2 5.25k", reddy_lbl="R_vol 250k", cp_lbl="C2 120pF", w=396, h=204, extra_right_pot="pot")}

                    <!-- Slim Lower J-Bridge Single-Coil Bobbin (Centered on all 4 strings -> cx=600, Slim h=128) -->
                    {make_jbass_spice_bobbin(600, 825, strings_4, accent, tag="J", l_lbl="L_J 3.6H", rdc_lbl="R_J 7.8k", reddy_lbl="R_eJ 125k", cp_lbl="C_J 70pF", w=820, h=128)}

                    <!-- Split-P Ground & Tight Diagonal Series Bus -->
                    <line x1="182" y1="280" x2="125" y2="280" stroke="{accent}" stroke-width="3.0"/>
                    {spice_gnd(125, 280, accent)}
                    <polyline points="618,280 656,280 544,520 582,520" fill="none" stroke="{accent}" stroke-width="3.4" stroke-linejoin="round"/>

                    <!-- Right P/J Parallel Summing Bus Between Split-P and J-Bridge -->
                    <polyline points="1018,520 1045,520 1045,825 1010,825" fill="none" stroke="{accent}" stroke-width="3.2"/>
                    <line x1="1045" y1="672" x2="1085" y2="672" stroke="{accent}" stroke-width="3.2"/>
                    {spice_node(1045, 672, accent)}
                    {spice_port(1085, 672, "V_OUT", accent, align="right")}
                    """
                )
            ),
        },
        "mustang": {
            "accent": "#f43f5e",
            "title": "MUSTANG P/J BASS",
            "scale": '30" SHORT SCALE',
            "badge2": "MUSTANG P/J",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 26,
            "num_strings": 4,
            "hero_scale": 1.0,
            "extra_bg": """
                <g opacity="0.14">
                  <line x1="940" y1="0" x2="940" y2="1200" stroke="#f43f5e" stroke-width="32"/>
                  <line x1="982" y1="0" x2="982" y2="1200" stroke="#f43f5e" stroke-width="14"/>
                </g>
            """,
            "content": _svg_content(
                lambda accent: (
                    f"""
                    <!-- 30" Short-Scale Mustang Split-P Bobbins (Centered on E/A=400 and D/G=800, w=396) -->
                    {make_pbass_spice_bobbin(400, 280, (strings_4[0], strings_4[1]), accent, idx=1, l_lbl="L_M1 1.9H", rdc_lbl="R_M1 5.25k", reddy_lbl="R_e1 90k", cp_lbl="C1 120pF", w=396, h=204)}
                    {make_pbass_spice_bobbin(800, 520, (strings_4[2], strings_4[3]), accent, idx=2, l_lbl="L_M2 1.9H", rdc_lbl="R_M2 5.25k", reddy_lbl="R_vol 250k", cp_lbl="C2 120pF", w=396, h=204, extra_right_pot="pot")}

                    <!-- Slim Mustang J-Bridge Bobbin (Centered at cx=600, Slim h=128) -->
                    {make_jbass_spice_bobbin(600, 825, strings_4, accent, tag="MJ", l_lbl="L_MJ 3.6H", rdc_lbl="R_MJ 7.8k", reddy_lbl="R_eJ 125k", cp_lbl="C_J 70pF", w=820, h=128)}

                    <!-- Split-P Ground & Tight Diagonal Series Bus -->
                    <line x1="182" y1="280" x2="125" y2="280" stroke="{accent}" stroke-width="3.0"/>
                    {spice_gnd(125, 280, accent)}
                    <polyline points="618,280 656,280 544,520 582,520" fill="none" stroke="{accent}" stroke-width="3.4" stroke-linejoin="round"/>

                    <!-- Mustang 3-Way Toggle Switch & Tone Shunt on Left String Axis (E=300) -->
                    <line x1="300" y1="520" x2="544" y2="520" stroke="{accent}" stroke-width="2.8"/>
                    <line x1="300" y1="520" x2="300" y2="640" stroke="{accent}" stroke-width="2.8"/>
                    {spice_capacitor_v(300, 580, label="C_c 750pF", accent=accent, align="left")}
                    {spice_gnd(300, 640, accent)}
                    {spice_node(544, 520, accent)}

                    <!-- Right 3-Way Toggle Summing Bus Between Split-P and J-Bridge -->
                    <polyline points="1018,520 1045,520 1045,825 1010,825" fill="none" stroke="{accent}" stroke-width="3.2"/>
                    <line x1="1045" y1="672" x2="1085" y2="672" stroke="{accent}" stroke-width="3.2"/>
                    {spice_node(1045, 672, accent)}
                    {spice_port(1085, 672, "V_OUT", accent, align="right")}
                    """
                )
            ),
        },
        "preamp_soapbar": {
            "accent": "#a3e635",
            "title": "PREAMP SOAPBAR BASS",
            "scale": '34" SCALE',
            "badge2": "DUAL SOAPBAR",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 26,
            "num_strings": 5,
            "hero_scale": 1.0,
            "content": _svg_content(
                lambda accent: (
                    f"""
                    <!-- Massive Neck 5-String Dual-Coil Soapbar SPICE Bobbin (Tall h=276, No Ears, Inset Side Cutouts) -->
                    {make_soapbar_spice_bobbin(600, 250, strings_5, accent, tag="N", l_lbl="3.8H", rdc_lbl="8.6k", cp_lbl="245pF", w=840, h=276, blades=False, with_internal_opamp=False)}

                    <!-- Massive Bridge 5-String Dual-Coil Soapbar SPICE Bobbin (Tall h=276, No Ears, Inset Side Cutouts) -->
                    {make_soapbar_spice_bobbin(600, 910, strings_5, accent, tag="B", l_lbl="4.2H", rdc_lbl="9.4k", cp_lbl="116pF", w=840, h=276, blades=False, with_internal_opamp=False)}

                    <!-- Active 3-Band EQ Preamp Subcircuit Box Between the Dual Soapbars -->
                    <rect x="220" y="430" width="760" height="300" rx="18" fill="#090c12" fill-opacity="0.88" stroke="{accent}" stroke-width="2.0" stroke-opacity="0.55" stroke-dasharray="12 8"/>

                    <!-- Dual Soapbar Outputs Routed into Preamp Blend Potentiometer -->
                    <polyline points="1020,250 1050,250 1050,410 280,410 280,520" fill="none" stroke="{accent}" stroke-width="3.0"/>
                    <polyline points="1020,910 1050,910 1050,750 280,750 280,640" fill="none" stroke="{accent}" stroke-width="3.0"/>
                    <line x1="280" y1="520" x2="280" y2="640" stroke="{accent}" stroke-width="2.8"/>
                    {spice_resistor_v(280, 580, label="R_blend 500k", accent=accent, pot=True, align="left")}

                    <!-- Stage 1: Input JFET Buffer Op-Amp -->
                    <line x1="280" y1="560" x2="367" y2="560" stroke="{accent}" stroke-width="2.8"/>
                    <line x1="340" y1="600" x2="367" y2="600" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="340" y1="600" x2="340" y2="640" stroke="{accent}" stroke-width="2.6"/>
                    {spice_gnd(340, 640, accent, stem=12.0)}
                    {spice_opamp(410, 580, w=86.0, h=76.0, label="U1_BUF", accent=accent)}

                    <!-- Stage 2: 3-Band Active Baxandall EQ Parallel Filter Bank (BASS / MID / TREBLE) -->
                    <line x1="453" y1="580" x2="500" y2="580" stroke="{accent}" stroke-width="3.0"/>
                    <line x1="500" y1="485" x2="500" y2="675" stroke="{accent}" stroke-width="2.8"/>
                    <line x1="720" y1="485" x2="720" y2="675" stroke="{accent}" stroke-width="2.8"/>

                    <!-- Top Branch: Bass Shelf (40 Hz) -->
                    <line x1="500" y1="485" x2="720" y2="485" stroke="{accent}" stroke-width="2.6"/>
                    {spice_capacitor_h(610, 485, plate_h=26.0, gap=10.0, label="C_bass 47nF", accent=accent, label_pos="top")}

                    <!-- Center Branch: Mid Bandpass (500 Hz) -->
                    <line x1="500" y1="580" x2="720" y2="580" stroke="{accent}" stroke-width="2.8"/>
                    {spice_inductor_h(610, 580, loops=4, r=13.0, label="L_mid 120mH", accent=accent, label_pos="bottom")}

                    <!-- Bottom Branch: Treble Shelf (4 kHz) -->
                    <line x1="500" y1="675" x2="720" y2="675" stroke="{accent}" stroke-width="2.6"/>
                    {spice_capacitor_h(610, 675, plate_h=26.0, gap=10.0, label="C_treb 2.2nF", accent=accent, label_pos="bottom")}

                    {spice_node(500, 485, accent)}
                    {spice_node(500, 580, accent)}
                    {spice_node(500, 675, accent)}
                    {spice_node(720, 485, accent)}
                    {spice_node(720, 580, accent)}
                    {spice_node(720, 675, accent)}

                    <!-- Stage 3: Output EQ Summing Amplifier & V_OUT -->
                    <line x1="720" y1="560" x2="797" y2="560" stroke="{accent}" stroke-width="2.8"/>
                    <line x1="770" y1="600" x2="797" y2="600" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="770" y1="600" x2="770" y2="640" stroke="{accent}" stroke-width="2.6"/>
                    {spice_gnd(770, 640, accent, stem=12.0)}
                    {spice_opamp(840, 580, w=86.0, h=76.0, label="U2_EQ", accent=accent)}

                    <line x1="883" y1="580" x2="1085" y2="580" stroke="{accent}" stroke-width="3.2"/>
                    {spice_resistor_h(955, 580, label="R_out 100Ω", w=68.0, h=20.0, accent=accent, label_pos="top")}
                    {spice_port(1085, 580, "V_OUT", accent, align="right")}
                    """
                )
            ),
        },
        "active_stingray": {
            "accent": "#f97316",
            "title": "ACTIVE STINGRAY BASS",
            "scale": '34" SCALE',
            "badge2": "MM HUMBUCKER",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 25,
            "num_strings": 4,
            "hero_scale": 1.0,
            "content": _svg_content(
                lambda accent: (
                    f"""
                    <!-- Massive Central Music Man StingRay Humbucker Ghost Underlay (Centered at cx=600) -->
                    {make_stingray_ghost_shell(600, 550, strings_4, accent, w=820, h=480)}

                    <!-- Upper Coil Row SPICE Network (Across Upper 4 Giant Poles) -->
                    <line x1="230" y1="430" x2="790" y2="430" stroke="{accent}" stroke-width="3.2"/>
                    <line x1="360" y1="350" x2="760" y2="350" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="360" y1="350" x2="360" y2="430" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="760" y1="350" x2="760" y2="430" stroke="{accent}" stroke-width="2.6"/>
                    {spice_ac_source(295, 430, r=22.0, label="V_MM1", accent=accent, label_pos="top")}
                    {spice_inductor_h(475, 430, loops=4, r=15.0, label="L1 2.4H", accent=accent, label_pos="bottom")}
                    {spice_resistor_h(660, 430, label="R_dc1 4.4k", w=72.0, h=22.0, accent=accent, label_pos="bottom")}
                    {spice_resistor_h(475, 350, label="R_eddy1 75k", w=70.0, h=20.0, accent=accent, label_pos="top")}
                    {spice_capacitor_h(660, 350, plate_h=28.0, gap=11.0, label="C_p1 180pF", accent=accent, label_pos="top")}

                    <!-- Lower Coil Row SPICE Network (Across Lower 4 Giant Poles) -->
                    <line x1="230" y1="670" x2="790" y2="670" stroke="{accent}" stroke-width="3.2"/>
                    <line x1="360" y1="750" x2="760" y2="750" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="360" y1="670" x2="360" y2="750" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="760" y1="670" x2="760" y2="750" stroke="{accent}" stroke-width="2.6"/>
                    {spice_ac_source(295, 670, r=22.0, label="V_MM2", accent=accent, label_pos="bottom")}
                    {spice_inductor_h(475, 670, loops=4, r=15.0, label="L2 2.4H", accent=accent, label_pos="top")}
                    {spice_resistor_h(660, 670, label="R_dc2 4.4k", w=72.0, h=22.0, accent=accent, label_pos="top")}
                    {spice_resistor_h(475, 750, label="R_eddy2 75k", w=70.0, h=20.0, accent=accent, label_pos="bottom")}
                    {spice_zener_pair_h(660, 750, label="D_sat", accent=accent, label_pos="bottom")}

                    <!-- Left Mounting Ear Ground & Right Internal Music Man 2-Band Active Op-Amp U1 -->
                    <line x1="230" y1="430" x2="230" y2="670" stroke="{accent}" stroke-width="3.2"/>
                    <line x1="230" y1="550" x2="145" y2="550" stroke="{accent}" stroke-width="3.2"/>
                    {spice_gnd(145, 550, accent)}

                    <!-- Active Op-Amp Stage Inside Right Side of Humbucker Housing -->
                    <line x1="790" y1="430" x2="790" y2="530" stroke="{accent}" stroke-width="3.0"/>
                    <line x1="790" y1="530" x2="862" y2="530" stroke="{accent}" stroke-width="3.0"/>
                    <line x1="790" y1="670" x2="790" y2="570" stroke="{accent}" stroke-width="3.0"/>
                    <line x1="790" y1="570" x2="862" y2="570" stroke="{accent}" stroke-width="3.0"/>
                    {spice_opamp(905, 550, w=86.0, h=76.0, label="U1_MM", accent=accent)}

                    <line x1="948" y1="550" x2="1075" y2="550" stroke="{accent}" stroke-width="3.4"/>
                    {spice_node(230, 550, accent)}
                    {spice_node(360, 430, accent)}
                    {spice_node(760, 430, accent)}
                    {spice_node(360, 670, accent)}
                    {spice_node(760, 670, accent)}
                    {spice_port(1075, 550, "V_OUT", accent, align="right")}
                    """
                )
            ),
        },
        "active_emg": {
            "accent": "#10b981",
            "title": "ACTIVE EMG BASS",
            "scale": '34" SCALE',
            "badge2": "ACTIVE SOAPBAR",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 26,
            "num_strings": 5,
            "hero_scale": 1.0,
            "content": _svg_content(
                lambda accent: (
                    f"""
                    <!-- Neck Active EMG 40 Soapbar with Internal Differential Preamp U_N -->
                    {make_soapbar_spice_bobbin(600, 300, strings_5, accent, tag="N", l_lbl="2.0H", rdc_lbl="4.5k", cp_lbl="584pF", w=820, h=256, blades=True, with_internal_opamp=True)}

                    <!-- Bridge Active EMG 40 Soapbar with Internal Differential Preamp U_B -->
                    {make_soapbar_spice_bobbin(600, 800, strings_5, accent, tag="B", l_lbl="2.0H", rdc_lbl="4.5k", cp_lbl="584pF", w=820, h=256, blades=True, with_internal_opamp=True)}

                    <!-- Central Active ABCX Summing & Tone Bus Directly Between EMG Soapbars -->
                    <polyline points="1010,300 1045,300 1045,490 360,490" fill="none" stroke="{accent}" stroke-width="3.0"/>
                    <polyline points="1010,800 1045,800 1045,610 360,610" fill="none" stroke="{accent}" stroke-width="3.0"/>

                    <line x1="360" y1="490" x2="360" y2="610" stroke="{accent}" stroke-width="2.8"/>
                    {spice_resistor_v(360, 550, label="R_bal 25k", accent=accent, pot=True, align="left")}

                    <line x1="360" y1="530" x2="557" y2="530" stroke="{accent}" stroke-width="2.8"/>
                    <line x1="505" y1="570" x2="557" y2="570" stroke="{accent}" stroke-width="2.6"/>
                    <line x1="505" y1="570" x2="505" y2="610" stroke="{accent}" stroke-width="2.6"/>
                    {spice_node(505, 610, accent)}
                    {spice_opamp(600, 550, w=86.0, h=76.0, label="U_SUM", accent=accent)}

                    <line x1="643" y1="550" x2="1085" y2="550" stroke="{accent}" stroke-width="3.2"/>
                    {spice_resistor_h(820, 550, label="R_out 100Ω", w=72.0, h=22.0, accent=accent, label_pos="top")}
                    {spice_port(1085, 550, "V_OUT", accent, align="right")}
                    """
                )
            ),
        },
    }

    cfg = ArtworkPackConfig.model_validate(configs[model_key])
    acc = cfg.accent
    strings = strings_5 if cfg.num_strings == 5 else strings_4

    # Full-bleed vertical physical strings underlay extending from top (y=0) to bottom (y=1200)
    strings_svg = "\n".join(
        f"""
        <line x1="{sx:.1f}" y1="0" x2="{sx:.1f}" y2="1200" stroke="#1e293b" stroke-width="{16.0 - i * 1.8:.1f}" opacity="0.65"/>
        <line x1="{sx:.1f}" y1="0" x2="{sx:.1f}" y2="1200" stroke="{acc}" stroke-width="1.4" stroke-dasharray="10 8" opacity="0.35"/>
        """
        for i, sx in enumerate(strings)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1200" width="1200" height="1200">
  <defs>
    <!-- Dark SPICE Schematic Canvas Background -->
    <radialGradient id="bgGrad" cx="50%" cy="50%" r="70%">
      <stop offset="0%" stop-color="#191d26"/>
      <stop offset="55%" stop-color="#12151c"/>
      <stop offset="100%" stop-color="#0b0d12"/>
    </radialGradient>

    <!-- Subtle Edition Color Aura Behind Schematic -->
    <radialGradient id="heroGlow" cx="50%" cy="50%" r="54%">
      <stop offset="0%" stop-color="{acc}" stop-opacity="0.26"/>
      <stop offset="48%" stop-color="{acc}" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="{acc}" stop-opacity="0.0"/>
    </radialGradient>

    <!-- Authentic EDA / SPICE Schematic Coordinate Grid Pattern -->
    <pattern id="spiceGridMinor" width="30" height="30" patternUnits="userSpaceOnUse">
      <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#ffffff" stroke-width="0.6" stroke-opacity="0.04"/>
    </pattern>
    <pattern id="spiceGridMajor" width="150" height="150" patternUnits="userSpaceOnUse">
      <rect width="150" height="150" fill="url(#spiceGridMinor)"/>
      <path d="M 150 0 L 0 0 0 150" fill="none" stroke="{acc}" stroke-width="1.0" stroke-opacity="0.12"/>
    </pattern>

    <!-- Clean Drop Shadow for Ghost Pickup Silhouette Underlay -->
    <filter id="dropShadow" x="-25%" y="-25%" width="150%" height="150%">
      <feDropShadow dx="0" dy="16" stdDeviation="16" flood-color="#000000" flood-opacity="0.88"/>
    </filter>
  </defs>

  <!-- 1. SPICE Canvas & Grid -->
  <rect width="1200" height="1200" fill="url(#bgGrad)"/>
  <rect width="1200" height="1200" fill="url(#heroGlow)"/>
  <rect width="1200" height="1200" fill="url(#spiceGridMajor)"/>
  {cfg.extra_bg}

  <!-- 2. Faint Physical Bass Strings Underlay -->
  <g>
    {strings_svg}
  </g>

  <!-- 3. Dominant Pickup-Shaped SPICE Schematic & Hardware Underlay -->
  <g transform="translate(600, 600) scale({cfg.hero_scale:.2f}) translate(-600, -550)">
    {cfg.content(acc)}
  </g>
</svg>"""
    # Strict validation of XML syntax
    ET.fromstring(svg)
    return svg


def main() -> None:
    assets_dir = Path("tone3000/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    packs_dir = Path("tone3000/packs")

    editions = {
        "precision": ("allomorph_standard_precision_bass", "34in_standard_p"),
        "jazz": ("allomorph_standard_jazz_bass", "34in_standard_jazz"),
        "pj": ("allomorph_standard_pj_bass", "34in_standard_pj"),
        "mustang": ("allomorph_mustang_pj_bass", "30in_mustang_pj"),
        "preamp_soapbar": ("allomorph_preamp_soapbar_bass", "34in_preamp_soapbar"),
        "active_stingray": ("allomorph_active_stingray_bass", "34in_active_stingray"),
        "active_emg": ("allomorph_active_emg_bass", "34in_active_emg"),
    }

    for key, (name, pack_id) in editions.items():
        svg_code = generate_pack_svg(key)
        svg_path = assets_dir / f"{name}.svg"
        jpg_path = assets_dir / f"{name}.jpg"

        svg_path.write_text(svg_code, encoding="utf-8")
        print(f"Generated valid SVG: {svg_path}")

        # Render 1024x1024 PNG via qlmanage, then convert to high-quality JPG via sips
        res = subprocess.run(
            ["qlmanage", "-t", "-s", "1024", "-o", str(assets_dir), str(svg_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(f"qlmanage error for {svg_path}: {res.stderr}")
            continue

        png_rendered = assets_dir / f"{name}.svg.png"
        if png_rendered.exists():
            subprocess.run(
                [
                    "sips",
                    "-s",
                    "format",
                    "jpeg",
                    "-s",
                    "formatOptions",
                    "95",
                    str(png_rendered),
                    "--out",
                    str(jpg_path),
                ],
                check=True,
                capture_output=True,
            )
            png_rendered.unlink()
            print(f"Rendered production JPG: {jpg_path}")

            # Sync to exported tone pack directory if present
            pack_artwork = packs_dir / pack_id / "artwork.jpg"
            if pack_artwork.parent.exists():
                pack_artwork.write_bytes(jpg_path.read_bytes())
                print(f"Synced pack artwork: {pack_artwork}")


if __name__ == "__main__":
    main()
