"""Deterministic resolvers and decomposition primitives for instrument voicings and pickups.

Provides pure, testable units for pickup resonance inference, coil resolution
and compounding, voicing token parsing, slug matching, and bundle assignment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from allomorph.config.geometry import resolve_pickup_coils
from allomorph.config.schema import (
    BundleTargetConfig,
    InstrumentConfig,
    PackBundleConfig,
    PickupConfig,
    VoiceCoilConfig,
    VoicePickupConfig,
    VoicingConfig,
)

if TYPE_CHECKING:
    from allomorph.config.instruments import VoicingBundle


import numpy as np

from allomorph.dsp.constants import FREQS


def determine_pickup_resonance_and_q(
    mag_curve: Sequence[float] | np.ndarray | None,
    freqs: Sequence[float] | np.ndarray = FREQS,
) -> tuple[float | None, float | None]:
    """Dynamically determines the electrical resonant frequency (fr in Hz) and Q factor from an MNA circuit curve.

    Uses sub-bin quadratic interpolation in log-frequency space and analytical 2nd-order RLC / half-power bandwidth.
    Enforces a strict +0.20 dB threshold above passband baseline to distinguish true electrical resonance
    from flat buffers, tone roll-offs, or wideband sensors.

    Returns:
        (fr, Q): (float, float) if a genuine resonance peak exists, or (None, None) for flat/wideband responses.
    """
    if mag_curve is None:
        return None, None

    mag = np.asarray(mag_curve, dtype=np.float64)
    f_arr = np.asarray(freqs, dtype=np.float64)
    if len(mag) < 3 or len(f_arr) < 3 or len(mag) != len(f_arr):
        return None, None

    # Passband baseline (around 100-200 Hz)
    idx_100 = int(np.argmin(np.abs(f_arr - 100.0)))
    idx_200 = int(np.argmin(np.abs(f_arr - 200.0)))
    base_idx_lo = min(idx_100, idx_200)
    base_idx_hi = max(idx_100, idx_200) + 1
    base_mag = float(np.mean(mag[base_idx_lo:base_idx_hi]))
    if base_mag <= 1e-9:
        base_mag = float(mag[0])

    peak_idx = int(np.argmax(mag))
    peak_mag = float(mag[peak_idx])

    peak_db_above_base = 20.0 * np.log10(max(peak_mag / max(base_mag, 1e-9), 1e-9))

    # Reject if below +0.20 dB threshold or peak is at boundary (monotonic lowpass/highpass)
    if peak_db_above_base < 0.20 or peak_idx == 0 or peak_idx >= len(mag) - 1:
        return None, None

    # Sub-bin quadratic peak interpolation in log-frequency space
    x = np.log(f_arr[peak_idx - 1 : peak_idx + 2])
    y = mag[peak_idx - 1 : peak_idx + 2]

    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    if abs(denom) < 1e-12:
        return float(f_arr[peak_idx]), None

    a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
    if a >= 0:  # Not a local maximum (concave upwards)
        return float(f_arr[peak_idx]), None

    b = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0]) + x[0] ** 2 * (y[1] - y[2])) / denom
    x_peak = -b / (2.0 * a)
    fr = float(np.exp(x_peak))
    interp_peak_mag = float(a * x_peak**2 + b * x_peak + (y[0] - a * x[0] ** 2 - b * x[0]))

    # Q factor extraction
    m_p = max(interp_peak_mag / max(base_mag, 1e-9), 1.0)
    target_mag = interp_peak_mag / np.sqrt(2.0)

    left_side = mag[:peak_idx]
    right_side = mag[peak_idx:]

    has_left_3db = bool(np.any(left_side <= target_mag))
    has_right_3db = bool(np.any(right_side <= target_mag))

    if has_left_3db and has_right_3db:
        i1 = int(np.where(left_side <= target_mag)[0][-1])
        f1 = float(np.interp(target_mag, [mag[i1], mag[i1 + 1]], [f_arr[i1], f_arr[i1 + 1]]))

        i2 = int(np.where(right_side <= target_mag)[0][0] + peak_idx)
        f2 = float(np.interp(target_mag, [mag[i2 - 1], mag[i2]], [f_arr[i2 - 1], f_arr[i2]]))

        bw = f2 - f1
        q = float(fr / bw) if bw > 0 else None
    else:
        # Analytical 2nd-order RLC peak gain formula: M_p = 1 / (2*zeta*sqrt(1-zeta^2))
        if m_p > 1.001:
            term = max(1.0 - 1.0 / (m_p**2), 0.0)
            zeta2 = 0.5 * (1.0 - np.sqrt(term))
            q = float(1.0 / (2.0 * np.sqrt(zeta2))) if zeta2 > 0 else None
        else:
            q = None

    return round(fr, 1), round(q, 2) if q is not None else None


def build_voice_coils_for_pickup(
    pickup: PickupConfig,
    active_coil_ids: Sequence[str] | None = None,
    instrument: InstrumentConfig | None = None,
) -> list[VoiceCoilConfig]:
    """Resolves individual physical coils for a pickup filtered by active coil IDs."""
    raw_coils = resolve_pickup_coils(pickup, instrument=instrument)
    if active_coil_ids:
        raw_coils = [c for c in raw_coils if c.id in active_coil_ids or not c.id]

    return [
        VoiceCoilConfig(
            position_from_bridge_m=c.position_from_bridge_m,
            aperture_width_in=c.aperture_width_in,
            weight=c.weight,
            polarity=c.polarity,
            strings=c.strings,
            pole_type=c.pole_type,
        )
        for c in raw_coils
    ]


def compound_multi_pickup_coils(
    voice_pickups: Sequence[VoicePickupConfig],
) -> list[VoiceCoilConfig]:
    """Compounds coils from multiple voice pickups by scaling weight and polarity by parent pickup."""
    return [
        VoiceCoilConfig(
            position_from_bridge_m=c.position_from_bridge_m,
            aperture_width_in=c.aperture_width_in,
            weight=c.weight * p.weight,
            polarity=c.polarity * p.polarity,
            strings=c.strings,
            pole_type=c.pole_type,
        )
        for p in voice_pickups
        for c in p.coils
    ]


def parse_target_voicing_token(
    target_token: str,
    default_inst_id: str,
) -> tuple[str, str]:
    """Parses a target voicing token into (instrument_id, voicing_id).

    Supports canonical 'instrument_id:voicing_id', universal catalog voice slugs,
    or local voicing IDs defaulting to default_inst_id.
    """
    if ":" in target_token:
        tgt_inst_id, tgt_vid = target_token.split(":", 1)
        return tgt_inst_id, tgt_vid

    from allomorph.config.voices import VOICES

    if target_token in VOICES:
        v_obj = VOICES[target_token]
        tgt_inst_id = v_obj.instrument_id or default_inst_id
        tgt_vid = v_obj.id
        return tgt_inst_id, tgt_vid

    return default_inst_id, target_token


def find_voicing_by_slug_or_id(
    instrument: InstrumentConfig,
    voicing_id_or_slug: str,
) -> VoicingConfig | None:
    """Finds a voicing on an instrument by direct ID or normalized tone name slug."""
    if voicing_id_or_slug in instrument.voicings:
        return instrument.voicings[voicing_id_or_slug]

    clean_target = voicing_id_or_slug.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
    for v in instrument.voicings.values():
        if v.id == voicing_id_or_slug or v.id == clean_target:
            return v
        if v.tone_name:
            slug = v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            if slug == clean_target or slug == voicing_id_or_slug:
                return v

    return None


def assign_target_to_bundle(
    target_vid: str,
    pack_bundles: Mapping[str, VoicingBundle],
    bundle_configs: Sequence[PackBundleConfig],
) -> str:
    """Assigns a target voicing ID to an appropriate pack bundle name."""
    # 1. Check explicit bundle targets in config
    for b_cfg in bundle_configs:
        for t in b_cfg.targets:
            t_str = t.target if isinstance(t, BundleTargetConfig) else t
            t_vid = t_str.split(":", 1)[1] if ":" in t_str else t_str
            if target_vid == t_str or target_vid == t_vid:
                return b_cfg.name

    for b_cfg in bundle_configs:
        for t in b_cfg.targets:
            t_str = t.target if isinstance(t, BundleTargetConfig) else t
            if target_vid in t_str:
                return b_cfg.name

    # 2. Check position keyword match if multiple bundles
    if len(pack_bundles) > 1:
        for bn in pack_bundles:
            if bn in target_vid.lower():
                return bn

    # 3. Fallback to first available bundle
    return next(iter(pack_bundles.keys()))
