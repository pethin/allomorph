"""Generate production-ready vector SVG and 1024x1024 JPG artwork for Allomorph Tone3000 Tone Packs.

Features:
- Pure dark-mode studio hardware aesthetic with soft accent vignette glow
- Dominant, enlarged pickup silhouettes commanding the center of the artwork for 1"x1" thumbnail discernability
- Authentic pickup geometries (split-P, dual-J, sweet-spot humbucker, dual soapbars, active blades)
- Bold, high-contrast typography (prominent ALLOMORPH header, massive instrument titles, clear voicing counts)
- Zero micro-jargon or microscopic CAD datums that clutter small mobile and storefront thumbnails
- 3 high-contrast specification badges (scale, pickup architecture, and platform compatibility: "FOR NAM & ANAGRAM")
- High-resolution SVG and converted 1024x1024 JPG assets for Tone3000 storefront
"""

import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from allomorph.pipeline.schema import ArtworkPackConfig


def _svg_content(fn: Callable[[str], str]) -> Callable[[str], str]:
    return fn


def make_pole(x: float, y: float, r: float = 11.0) -> str:
    """Generate an anatomically accurate cylindrical Alnico V pole piece with metallic shading."""
    return f"""
    <!-- Pole piece at {x:.1f}, {y:.1f} -->
    <g transform="translate({x:.1f}, {y:.1f})">
      <circle cx="0" cy="0" r="{r + 2.0:.1f}" fill="#080a0f" opacity="0.85"/>
      <circle cx="0" cy="0" r="{r:.1f}" fill="url(#poleGrad)"/>
      <circle cx="0" cy="0" r="{r - 1.5:.1f}" fill="none" stroke="#f8fafc" stroke-width="0.85" opacity="0.75"/>
      <ellipse cx="-2.5" cy="-2.5" rx="{r * 0.4:.1f}" ry="{r * 0.25:.1f}" fill="#ffffff" opacity="0.65" transform="rotate(-30, -2.5, -2.5)"/>
      <circle cx="0" cy="0" r="{r * 0.5:.1f}" fill="none" stroke="#64748b" stroke-width="0.5" opacity="0.5"/>
    </g>
    """


def make_pbass_half(
    cx: float,
    cy: float,
    x_strings: tuple[float, float],
    accent: str,
    label: str = "",
    w: float = 240.0,
    h: float = 114.0,
) -> str:
    """Generate an authentic Precision Bass split bobbin with high-contrast chamfer styling."""
    rx = 13.5
    x = cx - w / 2
    y = cy - h / 2

    ear_screw_dx = 131.0
    left_screw_x = cx - ear_screw_dx
    right_screw_x = cx + ear_screw_dx

    s1, s2 = x_strings
    pole_offset = 15.0

    return f"""
    <!-- P-Bass Bobbin Half at {cx:.1f}, {cy:.1f} ({label}) -->
    <g filter="url(#dropShadow)">
      <!-- Left Mounting Tab -->
      <path d="M {x:.1f} {cy - 24.2:.1f} A 25 25 0 0 0 {x:.1f} {cy + 24.2:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{cy:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{cy:.1f}" r="2.0" fill="#334155"/>

      <!-- Right Mounting Tab -->
      <path d="M {x + w:.1f} {cy - 24.2:.1f} A 25 25 0 0 1 {x + w:.1f} {cy + 24.2:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.2"/>
      <circle cx="{right_screw_x:.1f}" cy="{cy:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{right_screw_x:.1f}" cy="{cy:.1f}" r="2.0" fill="#334155"/>

      <!-- Main Bobbin Casing with Chamfer Bevel -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="url(#coverGrad)" stroke="#64748b" stroke-width="3.0"/>
      <rect x="{x + 2:.1f}" y="{y + 2:.1f}" width="{w - 4:.1f}" height="{h - 4:.1f}" rx="{rx - 1:.1f}" fill="none" stroke="#94a3b8" stroke-width="1.2" opacity="0.6"/>
      <rect x="{x + 7:.1f}" y="{y + 7:.1f}" width="{w - 14:.1f}" height="{h - 14:.1f}" rx="{rx - 4:.1f}" fill="#161c26" stroke="#2a374a" stroke-width="1.5"/>

      <!-- Internal Coil Indicator -->
      <rect x="{x + 16:.1f}" y="{y + 16:.1f}" width="{w - 32:.1f}" height="{h - 32:.1f}" rx="8" fill="none" stroke="{accent}" stroke-width="1.0" stroke-opacity="0.45" stroke-dasharray="5 5"/>
      
      <!-- 4 Poles (2 pairs straddling each string) -->
      {make_pole(s1 - pole_offset, cy)}
      {make_pole(s1 + pole_offset, cy)}
      {make_pole(s2 - pole_offset, cy)}
      {make_pole(s2 + pole_offset, cy)}
    </g>
    """


def make_jbass_pickup(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    w: float = 396.0,
    h: float = 78.0,
    label: str = "",
) -> str:
    """Generate a Jazz Bass single-coil pickup with high-contrast chamfer styling."""
    rx = 10.0
    x = cx - w / 2
    y = cy - h / 2
    pole_offset = 15.0

    ear_dx = 83.4
    left_ear_x = cx - ear_dx
    right_ear_x = cx + ear_dx

    return f"""
    <!-- Jazz Bass Bobbin at {cx:.1f}, {cy:.1f} ({label}) -->
    <g filter="url(#dropShadow)">
      <!-- Left Mounting Tab Pair -->
      <path d="M {left_ear_x - 21:.1f} {y:.1f} A 26.7 26.7 0 0 1 {left_ear_x + 21:.1f} {y:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.0"/>
      <circle cx="{left_ear_x:.1f}" cy="{y - 4.5:.1f}" r="3.5" fill="#0d1117" stroke="#64748b" stroke-width="1.0"/>
      <circle cx="{left_ear_x:.1f}" cy="{y - 4.5:.1f}" r="1.6" fill="#334155"/>
      
      <path d="M {left_ear_x - 21:.1f} {y + h:.1f} A 26.7 26.7 0 0 0 {left_ear_x + 21:.1f} {y + h:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.0"/>
      <circle cx="{left_ear_x:.1f}" cy="{y + h + 4.5:.1f}" r="3.5" fill="#0d1117" stroke="#64748b" stroke-width="1.0"/>
      <circle cx="{left_ear_x:.1f}" cy="{y + h + 4.5:.1f}" r="1.6" fill="#334155"/>

      <!-- Right Mounting Tab Pair -->
      <path d="M {right_ear_x - 21:.1f} {y:.1f} A 26.7 26.7 0 0 1 {right_ear_x + 21:.1f} {y:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.0"/>
      <circle cx="{right_ear_x:.1f}" cy="{y - 4.5:.1f}" r="3.5" fill="#0d1117" stroke="#64748b" stroke-width="1.0"/>
      <circle cx="{right_ear_x:.1f}" cy="{y - 4.5:.1f}" r="1.6" fill="#334155"/>
      
      <path d="M {right_ear_x - 21:.1f} {y + h:.1f} A 26.7 26.7 0 0 0 {right_ear_x + 21:.1f} {y + h:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.0"/>
      <circle cx="{right_ear_x:.1f}" cy="{y + h + 4.5:.1f}" r="3.5" fill="#0d1117" stroke="#64748b" stroke-width="1.0"/>
      <circle cx="{right_ear_x:.1f}" cy="{y + h + 4.5:.1f}" r="1.6" fill="#334155"/>

      <!-- Main Bobbin Casing with Chamfer Bevel -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="url(#coverGrad)" stroke="#64748b" stroke-width="3.0"/>
      <rect x="{x + 2:.1f}" y="{y + 2:.1f}" width="{w - 4:.1f}" height="{h - 4:.1f}" rx="{rx - 1:.1f}" fill="none" stroke="#94a3b8" stroke-width="1.2" opacity="0.6"/>
      <rect x="{x + 6:.1f}" y="{y + 6:.1f}" width="{w - 12:.1f}" height="{h - 12:.1f}" rx="{rx - 3:.1f}" fill="#161c26" stroke="#2a374a" stroke-width="1.5"/>

      <!-- Internal Coil Indicator -->
      <rect x="{x + 14:.1f}" y="{y + 14:.1f}" width="{w - 28:.1f}" height="{h - 28:.1f}" rx="6" fill="none" stroke="{accent}" stroke-width="1.0" stroke-opacity="0.45" stroke-dasharray="6 6"/>

      <!-- 8 Poles (4 pairs of 2 straddling E, A, D, G) -->
      {make_pole(x_strings[0] - pole_offset, cy)}
      {make_pole(x_strings[0] + pole_offset, cy)}
      {make_pole(x_strings[1] - pole_offset, cy)}
      {make_pole(x_strings[1] + pole_offset, cy)}
      {make_pole(x_strings[2] - pole_offset, cy)}
      {make_pole(x_strings[2] + pole_offset, cy)}
      {make_pole(x_strings[3] - pole_offset, cy)}
      {make_pole(x_strings[3] + pole_offset, cy)}
    </g>
    """


def make_stingray_pickup(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    w: float = 380.0,
    h: float = 204.0,
    label: str = "MUSIC MAN HUMBUCKER",
) -> str:
    """Generate an authentic Music Man StingRay humbucker with high-contrast chamfer styling."""
    rx = 15.0
    x = cx - w / 2
    y = cy - h / 2

    left_screw_x = cx - 198.0
    ear_y1 = cy - 54.0
    ear_y2 = cy + 54.0

    right_screw_x = cx + 198.0
    ear_yr = cy

    pole_r = 18.0
    row_offset = 38.0

    upper_poles = "".join(make_pole(s, cy - row_offset, r=pole_r) for s in x_strings)
    lower_poles = "".join(make_pole(s, cy + row_offset, r=pole_r) for s in x_strings)

    return f"""
    <!-- Music Man StingRay Humbucker at {cx:.1f}, {cy:.1f} ({label}) -->
    <g filter="url(#dropShadow)">
      <!-- Left Upper Mounting Tab (Bass Side) -->
      <path d="M {x:.1f} {ear_y1 - 18:.1f} A 20 20 0 0 0 {x:.1f} {ear_y1 + 18:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{ear_y1:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{ear_y1:.1f}" r="2.0" fill="#334155"/>

      <!-- Left Lower Mounting Tab (Bass Side) -->
      <path d="M {x:.1f} {ear_y2 - 18:.1f} A 20 20 0 0 0 {x:.1f} {ear_y2 + 18:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{ear_y2:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{ear_y2:.1f}" r="2.0" fill="#334155"/>

      <!-- Right Center Mounting Tab (Treble Side - Single Ear) -->
      <path d="M {x + w:.1f} {ear_yr - 22:.1f} A 24 24 0 0 1 {x + w:.1f} {ear_yr + 22:.1f} Z" fill="url(#coverGrad)" stroke="#475569" stroke-width="2.2"/>
      <circle cx="{right_screw_x:.1f}" cy="{ear_yr:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{right_screw_x:.1f}" cy="{ear_yr:.1f}" r="2.0" fill="#334155"/>

      <!-- Main Pickup Housing with Chamfer Bevel -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="url(#coverGrad)" stroke="#64748b" stroke-width="3.0"/>
      <rect x="{x + 2:.1f}" y="{y + 2:.1f}" width="{w - 4:.1f}" height="{h - 4:.1f}" rx="{rx - 1:.1f}" fill="none" stroke="#94a3b8" stroke-width="1.2" opacity="0.6"/>
      <rect x="{x + 7:.1f}" y="{y + 7:.1f}" width="{w - 14:.1f}" height="{h - 14:.1f}" rx="{rx - 4:.1f}" fill="#161c26" stroke="#2a374a" stroke-width="1.5"/>

      <!-- Internal Dual Coil Bobbin Outlines -->
      <rect x="{x + 14:.1f}" y="{y + 12:.1f}" width="{w - 28:.1f}" height="{h / 2 - 16:.1f}" rx="6" fill="none" stroke="{accent}" stroke-width="1.0" stroke-opacity="0.45" stroke-dasharray="6 6"/>
      <rect x="{x + 14:.1f}" y="{cy + 4:.1f}" width="{w - 28:.1f}" height="{h / 2 - 16:.1f}" rx="6" fill="none" stroke="{accent}" stroke-width="1.0" stroke-opacity="0.45" stroke-dasharray="6 6"/>

      <!-- 8 Massive Alnico V 3/8" Pole Pieces -->
      {upper_poles}
      {lower_poles}
    </g>
    """


def make_soapbar_pickup(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    w: float | None = None,
    h: float = 160.0,
    label: str = "PASSIVE SOAPBAR",
) -> str:
    """Generate an authentic passive soapbar pickup with high-contrast chamfer styling."""
    is_5string = len(x_strings) == 5
    if w is None:
        w = 428.0 if is_5string else 375.0
    rx = 14.0
    x = cx - w / 2
    y = cy - h / 2

    screw_dx = 200.5 if is_5string else 174.0
    left_screw_x = cx - screw_dx
    right_screw_x = cx + screw_dx

    pole_r = 10.5
    row_offset = 28.0

    upper_poles = "".join(make_pole(s, cy - row_offset, r=pole_r) for s in x_strings)
    lower_poles = "".join(make_pole(s, cy + row_offset, r=pole_r) for s in x_strings)

    return f"""
    <!-- Passive Soapbar Pickup at {cx:.1f}, {cy:.1f} ({label}) -->
    <g filter="url(#dropShadow)">
      <!-- Main Soapbar Casing with Chamfer Bevel -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="url(#coverGrad)" stroke="#64748b" stroke-width="3.0"/>
      <rect x="{x + 2:.1f}" y="{y + 2:.1f}" width="{w - 4:.1f}" height="{h - 4:.1f}" rx="{rx - 1:.1f}" fill="none" stroke="#94a3b8" stroke-width="1.2" opacity="0.6"/>
      <rect x="{x + 7:.1f}" y="{y + 7:.1f}" width="{w - 14:.1f}" height="{h - 14:.1f}" rx="{rx - 4:.1f}" fill="#161c26" stroke="#2a374a" stroke-width="1.5"/>

      <!-- Inset Mounting Tabs & Screw Recesses -->
      <!-- Left Inset Recess -->
      <path d="M {x + 6:.1f} {cy - 20:.1f} A 20 20 0 0 1 {x + 6:.1f} {cy + 20:.1f} Z" fill="#0d1117" stroke="#334155" stroke-width="1.5"/>
      <circle cx="{left_screw_x:.1f}" cy="{cy:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{cy:.1f}" r="2.0" fill="#334155"/>

      <!-- Right Inset Recess -->
      <path d="M {x + w - 6:.1f} {cy - 20:.1f} A 20 20 0 0 0 {x + w - 6:.1f} {cy + 20:.1f} Z" fill="#0d1117" stroke="#334155" stroke-width="1.5"/>
      <circle cx="{right_screw_x:.1f}" cy="{cy:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{right_screw_x:.1f}" cy="{cy:.1f}" r="2.0" fill="#334155"/>

      <!-- Internal Dual Coil Bobbin Outlines -->
      <rect x="{x + 16:.1f}" y="{y + 12:.1f}" width="{w - 32:.1f}" height="{h / 2 - 16:.1f}" rx="6" fill="none" stroke="{accent}" stroke-width="1.0" stroke-opacity="0.4" stroke-dasharray="6 6"/>
      <rect x="{x + 16:.1f}" y="{cy + 4:.1f}" width="{w - 32:.1f}" height="{h / 2 - 16:.1f}" rx="6" fill="none" stroke="{accent}" stroke-width="1.0" stroke-opacity="0.4" stroke-dasharray="6 6"/>

      <!-- Exposed Cylindrical Alnico V Pole Pieces -->
      {upper_poles}
      {lower_poles}
    </g>
    """


def make_emg40_pickup(
    cx: float,
    cy: float,
    x_strings: list[float],
    accent: str,
    w: float = 428.0,
    h: float = 160.0,
    label: str = "ACTIVE EMG 40",
) -> str:
    """Generate an authentic active 5-string EMG 40 soapbar pickup with high-contrast styling."""
    rx = 14.0
    _ = x_strings
    x = cx - w / 2
    y = cy - h / 2

    screw_dx = 200.5
    left_screw_x = cx - screw_dx
    right_screw_x = cx + screw_dx

    blade_offset = 32.0
    blade_w = 360.0
    blade_x = cx - blade_w / 2

    return f"""
    <!-- Active EMG 40 Soapbar Pickup at {cx:.1f}, {cy:.1f} ({label}) -->
    <g filter="url(#dropShadow)">
      <!-- Main Soapbar Casing with Crisp Light-Catching Chamfer -->
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="url(#coverGrad)" stroke="#64748b" stroke-width="3.0"/>
      <!-- Top Specular Highlight Bevel -->
      <rect x="{x + 2:.1f}" y="{y + 2:.1f}" width="{w - 4:.1f}" height="{h - 4:.1f}" rx="{rx - 1:.1f}" fill="none" stroke="#94a3b8" stroke-width="1.2" opacity="0.6"/>
      <!-- Inner Bobbin Recess -->
      <rect x="{x + 7:.1f}" y="{y + 7:.1f}" width="{w - 14:.1f}" height="{h - 14:.1f}" rx="{rx - 4:.1f}" fill="#161c26" stroke="#2a374a" stroke-width="1.5"/>

      <!-- Inset Mounting Tabs & Screw Recesses -->
      <!-- Left Inset Recess -->
      <path d="M {x + 6:.1f} {cy - 20:.1f} A 20 20 0 0 1 {x + 6:.1f} {cy + 20:.1f} Z" fill="#0d1117" stroke="#334155" stroke-width="1.5"/>
      <circle cx="{left_screw_x:.1f}" cy="{cy:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{left_screw_x:.1f}" cy="{cy:.1f}" r="2.0" fill="#334155"/>

      <!-- Right Inset Recess -->
      <path d="M {x + w - 6:.1f} {cy - 20:.1f} A 20 20 0 0 0 {x + w - 6:.1f} {cy + 20:.1f} Z" fill="#0d1117" stroke="#334155" stroke-width="1.5"/>
      <circle cx="{right_screw_x:.1f}" cy="{cy:.1f}" r="4.5" fill="#0d1117" stroke="#64748b" stroke-width="1.2"/>
      <circle cx="{right_screw_x:.1f}" cy="{cy:.1f}" r="2.0" fill="#334155"/>

      <!-- Internal Dual Sensing Blade Rails (Luminous, High Vibrancy) -->
      <!-- Upper Blade Rail -->
      <rect x="{blade_x:.1f}" y="{cy - blade_offset - 5:.1f}" width="{blade_w:.1f}" height="10" rx="5" fill="#09131a" stroke="{accent}" stroke-width="1.5"/>
      <rect x="{blade_x + 2:.1f}" y="{cy - blade_offset - 3:.1f}" width="{blade_w - 4:.1f}" height="6" rx="3" fill="{accent}" fill-opacity="0.95"/>
      <line x1="{blade_x + 6:.1f}" y1="{cy - blade_offset:.1f}" x2="{blade_x + blade_w - 6:.1f}" y2="{cy - blade_offset:.1f}" stroke="#ffffff" stroke-width="1.5" opacity="0.9"/>

      <!-- Lower Blade Rail -->
      <rect x="{blade_x:.1f}" y="{cy + blade_offset - 5:.1f}" width="{blade_w:.1f}" height="10" rx="5" fill="#09131a" stroke="{accent}" stroke-width="1.5"/>
      <rect x="{blade_x + 2:.1f}" y="{cy + blade_offset - 3:.1f}" width="{blade_w - 4:.1f}" height="6" rx="3" fill="{accent}" fill-opacity="0.95"/>
      <line x1="{blade_x + 6:.1f}" y1="{cy + blade_offset:.1f}" x2="{blade_x + blade_w - 6:.1f}" y2="{cy + blade_offset:.1f}" stroke="#ffffff" stroke-width="1.5" opacity="0.9"/>

      <!-- Internal Coil Boundary Indication -->
      <rect x="{x + 24:.1f}" y="{y + 16:.1f}" width="{w - 48:.1f}" height="{h - 32:.1f}" rx="8" fill="none" stroke="{accent}" stroke-width="1.0" stroke-opacity="0.4" stroke-dasharray="6 6"/>
    </g>
    """


def generate_pack_svg(model_key: str) -> str:
    """Generate pristine standalone SVG for a Tone3000 Tone Pack edition."""
    strings_4 = [480.0, 560.0, 640.0, 720.0]  # E, A, D, G string axes (80px / 19mm scale)
    strings_5 = [440.0, 520.0, 600.0, 680.0, 760.0]  # B, E, A, D, G string axes (5-string)
    strings = strings_4

    configs = {
        "precision": {
            "accent": "#f59e0b",
            "title": "STANDARD PRECISION BASS",
            "scale": '34" SCALE',
            "badge2": "ALNICO SPLIT-P",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 21,
            "num_strings": 4,
            "hero_scale": 1.35,
            "content": _svg_content(
                lambda accent: (
                    f"""
                {make_pbass_half(520, 458, (strings[0], strings[1]), accent, "BASS E/A")}
                {make_pbass_half(680, 572, (strings[2], strings[3]), accent, "TREBLE D/G")}
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
            "voicing_count": 20,
            "num_strings": 4,
            "hero_scale": 1.30,
            "content": _svg_content(
                lambda accent: (
                    f"""
                {make_jbass_pickup(600, 415, strings, accent, w=385, h=78, label="NECK PICKUP")}
                {make_jbass_pickup(600, 615, strings, accent, w=396, h=78, label="BRIDGE PICKUP")}
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
            "voicing_count": 19,
            "num_strings": 4,
            "hero_scale": 1.25,
            "content": _svg_content(
                lambda accent: (
                    f"""
                {make_pbass_half(520, 388, (strings[0], strings[1]), accent, "P-BASS EA")}
                {make_pbass_half(680, 502, (strings[2], strings[3]), accent, "P-BASS DG")}
                {make_jbass_pickup(600, 650, strings, accent, w=396, h=78, label="J-BRIDGE")}
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
            "voicing_count": 22,
            "num_strings": 4,
            "hero_scale": 1.25,
            "extra_bg": """
                <g opacity="0.10">
                  <line x1="880" y1="60" x2="880" y2="1140" stroke="#f43f5e" stroke-width="28"/>
                  <line x1="915" y1="60" x2="915" y2="1140" stroke="#f43f5e" stroke-width="12"/>
                </g>
            """,
            "content": _svg_content(
                lambda accent: (
                    f"""
                {make_pbass_half(520, 388, (strings[0], strings[1]), accent, "MUSTANG P-EA")}
                {make_pbass_half(680, 502, (strings[2], strings[3]), accent, "MUSTANG P-DG")}
                {make_jbass_pickup(600, 650, strings, accent, w=396, h=78, label="MUSTANG J-BRIDGE")}
                """
                )
            ),
        },
        "preamp_soapbar": {
            "accent": "#06b6d4",
            "title": "PREAMP SOAPBAR BASS",
            "scale": '34" SCALE',
            "badge2": "DUAL SOAPBAR",
            "badge3": "FOR NAM &amp; ANAGRAM",
            "voicing_count": 22,
            "num_strings": 5,
            "hero_scale": 1.24,
            "content": _svg_content(
                lambda accent: (
                    f"""
                {make_soapbar_pickup(600, 415, strings_5, accent, label="NECK SOAPBAR")}
                {make_soapbar_pickup(600, 615, strings_5, accent, label="BRIDGE SOAPBAR")}
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
            "voicing_count": 21,
            "num_strings": 4,
            "hero_scale": 1.45,
            "content": _svg_content(
                lambda accent: (
                    f"""
                {make_stingray_pickup(600, 515, strings, accent, label="SWEET-SPOT HUMBUCKER")}
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
            "voicing_count": 22,
            "num_strings": 5,
            "hero_scale": 1.24,
            "content": _svg_content(
                lambda accent: (
                    f"""
                {make_emg40_pickup(600, 415, strings_5, accent, label="NECK EMG SOAPBAR")}
                {make_emg40_pickup(600, 615, strings_5, accent, label="BRIDGE EMG SOAPBAR")}
                """
                )
            ),
        },
    }

    cfg = ArtworkPackConfig.model_validate(configs[model_key])
    acc = cfg.accent
    strings = strings_5 if cfg.num_strings == 5 else strings_4
    hero_scale = cfg.hero_scale
    cy_hero = 550.0

    sy1 = 515.0 + (270.0 - cy_hero) / hero_scale
    sy2 = 515.0 + (830.0 - cy_hero) / hero_scale

    if cfg.num_strings == 5:
        strings_svg = f"""  <!-- Physical Bass Strings (5 Strings) -->
    <g filter="url(#stringShadow)">
      <!-- B String -->
      <line x1="{strings[0]:.1f}" y1="{sy1:.1f}" x2="{strings[0]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="16.0"/>
      <line x1="{strings[0]:.1f}" y1="{sy1:.1f}" x2="{strings[0]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="3.5" opacity="0.95"/>

      <!-- E String -->
      <line x1="{strings[1]:.1f}" y1="{sy1:.1f}" x2="{strings[1]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="13.5"/>
      <line x1="{strings[1]:.1f}" y1="{sy1:.1f}" x2="{strings[1]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="3.0" opacity="0.95"/>

      <!-- A String -->
      <line x1="{strings[2]:.1f}" y1="{sy1:.1f}" x2="{strings[2]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="11.5"/>
      <line x1="{strings[2]:.1f}" y1="{sy1:.1f}" x2="{strings[2]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="2.5" opacity="0.95"/>

      <!-- D String -->
      <line x1="{strings[3]:.1f}" y1="{sy1:.1f}" x2="{strings[3]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="9.5"/>
      <line x1="{strings[3]:.1f}" y1="{sy1:.1f}" x2="{strings[3]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="2.2" opacity="0.95"/>

      <!-- G String -->
      <line x1="{strings[4]:.1f}" y1="{sy1:.1f}" x2="{strings[4]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="8.0"/>
      <line x1="{strings[4]:.1f}" y1="{sy1:.1f}" x2="{strings[4]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="2.0" opacity="0.95"/>
    </g>"""
    else:
        strings_svg = f"""  <!-- Physical Bass Strings (4 Strings) -->
    <g filter="url(#stringShadow)">
      <!-- E String -->
      <line x1="{strings[0]:.1f}" y1="{sy1:.1f}" x2="{strings[0]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="16.0"/>
      <line x1="{strings[0]:.1f}" y1="{sy1:.1f}" x2="{strings[0]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="3.5" opacity="0.95"/>

      <!-- A String -->
      <line x1="{strings[1]:.1f}" y1="{sy1:.1f}" x2="{strings[1]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="13.5"/>
      <line x1="{strings[1]:.1f}" y1="{sy1:.1f}" x2="{strings[1]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="3.0" opacity="0.95"/>

      <!-- D String -->
      <line x1="{strings[2]:.1f}" y1="{sy1:.1f}" x2="{strings[2]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="11.5"/>
      <line x1="{strings[2]:.1f}" y1="{sy1:.1f}" x2="{strings[2]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="2.5" opacity="0.95"/>

      <!-- G String -->
      <line x1="{strings[3]:.1f}" y1="{sy1:.1f}" x2="{strings[3]:.1f}" y2="{sy2:.1f}" stroke="url(#stringGrad)" stroke-width="9.0"/>
      <line x1="{strings[3]:.1f}" y1="{sy1:.1f}" x2="{strings[3]:.1f}" y2="{sy2:.1f}" stroke="#ffffff" stroke-width="2.2" opacity="0.95"/>
    </g>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1200" width="1200" height="1200">
  <defs>
    <!-- Background Radial Gradient: Rich graphite slate center fading smoothly to deep black -->
    <radialGradient id="bgGrad" cx="50%" cy="46%" r="65%">
      <stop offset="0%" stop-color="#222b3b"/>
      <stop offset="35%" stop-color="#151b25"/>
      <stop offset="70%" stop-color="#0d1117"/>
      <stop offset="100%" stop-color="#080a0f"/>
    </radialGradient>

    <!-- Outer Bezel Linear -->
    <linearGradient id="bezelGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#475569"/>
      <stop offset="30%" stop-color="#2a374a"/>
      <stop offset="70%" stop-color="#141c28"/>
      <stop offset="100%" stop-color="#334155"/>
    </linearGradient>

    <!-- Silky-Smooth Studio Backlight Bloom: Pure accent color diffusion with zero hard boundaries -->
    <radialGradient id="heroGlow" cx="50%" cy="46%" r="48%">
      <stop offset="0%" stop-color="{acc}" stop-opacity="0.45"/>
      <stop offset="25%" stop-color="{acc}" stop-opacity="0.28"/>
      <stop offset="50%" stop-color="{acc}" stop-opacity="0.12"/>
      <stop offset="75%" stop-color="{acc}" stop-opacity="0.03"/>
      <stop offset="100%" stop-color="{acc}" stop-opacity="0.0"/>
    </radialGradient>

    <!-- Pickup Cover Gradient -->
    <linearGradient id="coverGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#3b485d"/>
      <stop offset="35%" stop-color="#263142"/>
      <stop offset="100%" stop-color="#151b24"/>
    </linearGradient>

    <!-- Metallic Pole Gradient -->
    <radialGradient id="poleGrad" cx="35%" cy="35%" r="70%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="25%" stop-color="#f1f5f9"/>
      <stop offset="60%" stop-color="#94a3b8"/>
      <stop offset="100%" stop-color="#475569"/>
    </radialGradient>

    <!-- Steel String Linear (High-Gloss Steel) -->
    <linearGradient id="stringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#64748b"/>
      <stop offset="25%" stop-color="#ffffff"/>
      <stop offset="60%" stop-color="#cbd5e1"/>
      <stop offset="100%" stop-color="#475569"/>
    </linearGradient>

    <!-- Clean Drop Shadow: Generous filter region so no blur clipping occurs -->
    <filter id="dropShadow" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="0" dy="20" stdDeviation="18" flood-color="#000000" flood-opacity="0.92"/>
      <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000000" flood-opacity="0.65"/>
    </filter>

    <filter id="stringShadow" x="-50%" y="-20%" width="200%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#000000" flood-opacity="0.80"/>
    </filter>
  </defs>

  <!-- Canvas Background -->
  <rect width="1200" height="1200" fill="url(#bgGrad)"/>
  <!-- Seamless Studio Backlight Bloom across canvas (no shape clipping) -->
  <rect width="1200" height="1200" fill="url(#heroGlow)"/>
  {cfg.extra_bg}

  <!-- Outer Precision Enclosure Bezel -->
  <rect x="24" y="24" width="1152" height="1152" rx="32" fill="none" stroke="url(#bezelGrad)" stroke-width="4"/>
  <rect x="36" y="36" width="1128" height="1128" rx="24" fill="none" stroke="#ffffff" stroke-opacity="0.08" stroke-width="1.5"/>

  <!-- Precision Corner Registration Crosshairs -->
  <g stroke="{acc}" stroke-width="2.2" stroke-opacity="0.85">
    <line x1="60" y1="75" x2="90" y2="75"/><line x1="75" y1="60" x2="75" y2="90"/><circle cx="75" cy="75" r="6" fill="none"/>
    <line x1="1110" y1="75" x2="1140" y2="75"/><line x1="1125" y1="60" x2="1125" y2="90"/><circle cx="1125" cy="75" r="6" fill="none"/>
    <line x1="60" y1="1125" x2="90" y2="1125"/><line x1="75" y1="1110" x2="75" y2="1140"/><circle cx="75" cy="1125" r="6" fill="none"/>
    <line x1="1110" y1="1125" x2="1140" y2="1125"/><line x1="1125" y1="1110" x2="1125" y2="1140"/><circle cx="1125" cy="1125" r="6" fill="none"/>
  </g>

  <!-- Primary Branding Header: Perfectly balanced from top -->
  <g transform="translate(600, 184)">
    <text x="0" y="0" fill="#ffffff" font-size="108" font-family="system-ui, -apple-system, sans-serif" font-weight="900" letter-spacing="16" text-anchor="middle">ALLOMORPH</text>
    <text x="0" y="46" fill="{acc}" font-size="25" font-family="system-ui, -apple-system, sans-serif" font-weight="800" letter-spacing="6" text-anchor="middle">ANALOG PICKUP DIGITAL TWINS</text>
  </g>

  <!-- Header Technical Divider -->
  <line x1="80" y1="270" x2="1120" y2="270" stroke="#334155" stroke-width="2"/>
  <line x1="440" y1="270" x2="760" y2="270" stroke="{acc}" stroke-width="4.5"/>

  <!-- Center Hero Pickup & Strings Container -->
  <g transform="translate(600, {cy_hero:.1f}) scale({hero_scale:.2f}) translate(-600, -515)">
    {cfg.content(acc)}

    {strings_svg}
  </g>

  <!-- Bottom Hero Section -->
  <line x1="80" y1="830" x2="1120" y2="830" stroke="#334155" stroke-width="2"/>
  <line x1="80" y1="830" x2="420" y2="830" stroke="{acc}" stroke-width="4.5"/>

  <!-- Tone Pack Title (BIG & BOLD) -->
  <text x="80" y="902" fill="#ffffff" font-size="66" font-family="system-ui, -apple-system, sans-serif" font-weight="900" letter-spacing="1.5">{cfg.title}</text>
  
  <!-- Edition Subtitle / Voicing Count (BIG & BOLD) -->
  <text x="80" y="960" fill="{acc}" font-size="28" font-family="system-ui, -apple-system, sans-serif" font-weight="800" letter-spacing="3.0">{cfg.voicing_count} DIGITAL TWIN VOICINGS</text>
  
  <!-- Specification Badges (3 Large High-Contrast Badges) -->
  <g transform="translate(0, 1000)">
    <!-- Badge 1 (Scale) -->
    <rect x="80" y="0" width="230" height="68" rx="16" fill="#141a24" stroke="#3b485d" stroke-width="2.2"/>
    <text x="195" y="43" fill="#f8fafc" font-size="22" font-family="system-ui, -apple-system, sans-serif" font-weight="800" letter-spacing="1.5" text-anchor="middle">{cfg.scale}</text>

    <!-- Badge 2 (Pickup) -->
    <rect x="330" y="0" width="370" height="68" rx="16" fill="#141a24" stroke="#3b485d" stroke-width="2.2"/>
    <text x="515" y="43" fill="#f8fafc" font-size="22" font-family="system-ui, -apple-system, sans-serif" font-weight="800" letter-spacing="1.5" text-anchor="middle">{cfg.badge2}</text>

    <!-- Badge 3 (Target / Platform) -->
    <rect x="720" y="0" width="400" height="68" rx="16" fill="#141a24" stroke="{acc}" stroke-width="3.0"/>
    <text x="920" y="43" fill="{acc}" font-size="22" font-family="system-ui, -apple-system, sans-serif" font-weight="900" letter-spacing="1.5" text-anchor="middle">{cfg.badge3}</text>
  </g>
</svg>"""
    # Strict validation of XML syntax
    ET.fromstring(svg)
    return svg


def main():
    assets_dir = Path("tone3000/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    editions = {
        "precision": "allomorph_standard_precision_bass",
        "jazz": "allomorph_standard_jazz_bass",
        "pj": "allomorph_standard_pj_bass",
        "mustang": "allomorph_mustang_pj_bass",
        "preamp_soapbar": "allomorph_preamp_soapbar_bass",
        "active_stingray": "allomorph_active_stingray_bass",
        "active_emg": "allomorph_active_emg_bass",
    }

    # Remove obsolete active_soapbar and placeholder assets if present
    for old_file in (
        list(assets_dir.glob("allomorph_active_soapbar_bass.*"))
        + list(assets_dir.glob("allomorph_emg_soapbar_bass.*"))
        + list(assets_dir.glob("allomorph_34in_emg_soapbar_bass.*"))
        + list(assets_dir.glob("coilshift_*.jpg"))
    ):
        print(f"Removing obsolete asset: {old_file}")
        old_file.unlink()

    for key, name in editions.items():
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


if __name__ == "__main__":
    main()
