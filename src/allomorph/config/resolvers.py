"""Deterministic resolvers and decomposition primitives for instrument voicings and pickups.

Provides pure, testable units for pickup resonance inference, active pickup/coil
resolution, electroacoustic voicing metadata extraction, and voicing token parsing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from allomorph.config.geometry import resolve_pickup_coils
from allomorph.config.schema import (
    CoilConfig,
    InstrumentConfig,
    PickupConfig,
    VoicingConfig,
)
from allomorph.dsp.constants import FREQS


@dataclass(frozen=True)
class VoicingElectroacoustics:
    """Resolved physical coil array, RLC resonance, and magnetic core parameters for a voicing."""

    coils: list[CoilConfig]
    fr: float | None
    q: float | None
    magnet_type: str
    alpha: float
    vsat: float


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

    peak_idx = int(base_idx_lo + np.argmax(mag[base_idx_lo:]))
    peak_mag = float(mag[peak_idx])

    peak_db_above_base = 20.0 * np.log10(max(peak_mag / max(base_mag, 1e-9), 1e-9))

    # Reject if below +0.20 dB threshold or peak is at boundary (monotonic lowpass/highpass)
    if (
        peak_db_above_base < 0.20
        or peak_idx <= base_idx_lo
        or peak_idx >= len(mag) - 1
        or mag[peak_idx] < mag[peak_idx - 1]
        or mag[peak_idx] < mag[peak_idx + 1]
    ):
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


def resolve_voicing_active_pickups(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
) -> list[str]:
    """Resolves the active physical pickup IDs for an instrument voicing by evaluating
    electrical transmission through the Modified Nodal Analysis (MNA) harness.
    """
    all_pickups = list(instrument.pickups.keys())
    if len(all_pickups) <= 1:
        return all_pickups

    h_id = voicing.harness
    harnesses = getattr(instrument, "harnesses", {})
    if not harnesses or h_id not in harnesses:
        return all_pickups

    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.preamps import PREAMPS

    curves = solve_mna_harness(
        instrument,
        harnesses[h_id],
        voicing,
        freqs=np.array([200.0, 1000.0], dtype=np.float64),
        preamps=PREAMPS,
    )
    max_trans = (
        max(float(np.max(np.abs(curves[p]))) for p in all_pickups if p in curves)
        if any(p in curves for p in all_pickups)
        else 1.0
    )
    thresh = max(0.10 * max_trans, 1e-4)
    active = [p for p in all_pickups if p in curves and np.max(np.abs(curves[p])) >= thresh]
    return active if active else all_pickups


def resolve_voicing_active_coils(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    pickup_key: str,
) -> list[str]:
    """Resolves active coil IDs for a specific pickup by evaluating electrical transmission
    through the Modified Nodal Analysis (MNA) harness.
    """
    p_cfg = instrument.pickups.get(pickup_key)
    if not p_cfg or not p_cfg.coils:
        return []
    if len(p_cfg.coils) <= 1:
        return [p_cfg.coils[0].id or pickup_key]

    h_id = voicing.harness
    harnesses = getattr(instrument, "harnesses", {})
    if not harnesses or h_id not in harnesses:
        return [c.id for c in p_cfg.coils if c.id]

    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.preamps import PREAMPS

    curves = solve_mna_harness(
        instrument,
        harnesses[h_id],
        voicing,
        freqs=np.array([200.0, 1000.0], dtype=np.float64),
        preamps=PREAMPS,
    )
    all_c_keys = [
        k for c in p_cfg.coils if c.id for k in [f"{pickup_key}.{c.id}", f"{p_cfg.id}.{c.id}", c.id]
    ]
    max_c_trans = (
        max(float(np.max(np.abs(curves[k]))) for k in all_c_keys if k in curves)
        if any(k in curves for k in all_c_keys)
        else 1.0
    )
    thresh = max(0.10 * max_c_trans, 1e-4)
    active_coils: list[str] = []
    for c in p_cfg.coils:
        if not c.id:
            continue
        c_keys = [f"{pickup_key}.{c.id}", f"{p_cfg.id}.{c.id}", c.id]
        if any(k in curves and np.max(np.abs(curves[k])) >= thresh for k in c_keys):
            active_coils.append(c.id)

    return active_coils if active_coils else [c.id for c in p_cfg.coils if c.id]


def build_voice_coils_for_pickup(
    pickup: PickupConfig,
    active_coil_ids: Sequence[str] | None = None,
    instrument: InstrumentConfig | None = None,
) -> list[CoilConfig]:
    """Resolves individual physical coils for a pickup filtered by active coil IDs."""
    all_coils = resolve_pickup_coils(pickup, instrument=instrument)
    raw_coils = all_coils
    if active_coil_ids:
        filtered = [c for c in all_coils if c.id in active_coil_ids or not c.id]
        if filtered:
            raw_coils = filtered

    # Re-normalize active coil weights when a full-string-bed multi-coil pickup is coil-split
    scale_w = 1.0
    if len(raw_coils) < len(all_coils) and all(
        not c.strings or c.strings == ["all"] for c in raw_coils
    ):
        w_sum = sum(float(c.weight) for c in raw_coils)
        if w_sum > 0.0:
            scale_w = 1.0 / w_sum

    return [
        CoilConfig(
            id=c.id,
            position_from_bridge_m=c.position_from_bridge_m,
            aperture_width_in=c.aperture_width_in,
            weight=float(c.weight) * scale_w,
            polarity=c.polarity,
            strings=list(c.strings),
            pole_type=c.pole_type,
            L=c.L,
            Rdc=c.Rdc,
            Reddy=c.Reddy,
            Ccoil=c.Ccoil,
        )
        for c in raw_coils
    ]


def resolve_voicing_electroacoustics(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
) -> VoicingElectroacoustics:
    """Resolves active physical coils, electrical resonance (fr, Q), and magnetic saturation
    parameters directly from an InstrumentConfig and VoicingConfig.
    """
    active_pickup_keys = resolve_voicing_active_pickups(instrument, voicing)

    for k in active_pickup_keys:
        if k not in instrument.pickups:
            raise KeyError(
                f"Pickup '{k}' not found on instrument '{instrument.id}'. "
                f"Available pickups: {list(instrument.pickups.keys())}"
            )

    h_id = voicing.harness
    if not h_id and instrument.harnesses:
        h_id = next(iter(instrument.harnesses.keys()))
    curves_dict: dict[str, np.ndarray] = {}
    if h_id and h_id in instrument.harnesses:
        from allomorph.circuit.solver import solve_mna_harness

        curves_dict = solve_mna_harness(
            instrument, instrument.harnesses[h_id], voicing, freqs=FREQS
        )

    coils: list[CoilConfig] = []
    active_mags: list[str] = []
    if len(active_pickup_keys) >= 2:
        idx_100 = int(np.argmin(np.abs(np.asarray(FREQS, dtype=float) - 100.0)))
        raw_weights = [
            float(np.abs(curves_dict[p_key][idx_100]))
            if (p_key in curves_dict and len(curves_dict[p_key]) > idx_100)
            else 1.0
            for p_key in active_pickup_keys
        ]
        w_total = sum(raw_weights)
        norm_weights = (
            [w / w_total for w in raw_weights]
            if w_total > 1e-9
            else [1.0 / len(active_pickup_keys)] * len(active_pickup_keys)
        )

        for idx, p_key in enumerate(active_pickup_keys):
            p = instrument.pickups[p_key]
            m_type = p.magnet_type or "alnico_v"
            if m_type not in active_mags:
                active_mags.append(m_type)
            active_c_ids = resolve_voicing_active_coils(instrument, voicing, p_key)
            sub_coils = build_voice_coils_for_pickup(p, active_c_ids, instrument=instrument)
            p_w = norm_weights[idx]
            for c in sub_coils:
                coils.append(
                    CoilConfig(
                        id=c.id,
                        position_from_bridge_m=c.position_from_bridge_m,
                        aperture_width_in=c.aperture_width_in,
                        weight=c.weight * p_w,
                        polarity=c.polarity,
                        strings=list(c.strings),
                        pole_type=c.pole_type,
                        L=c.L,
                        Rdc=c.Rdc,
                        Reddy=c.Reddy,
                        Ccoil=c.Ccoil,
                    )
                )
    else:
        primary_key = (
            active_pickup_keys[0] if active_pickup_keys else next(iter(instrument.pickups.keys()))
        )
        pickup = instrument.pickups[primary_key]
        active_c_ids = resolve_voicing_active_coils(instrument, voicing, primary_key)
        coils = build_voice_coils_for_pickup(pickup, active_c_ids, instrument=instrument)

    primary_key_val = ""
    target_text = f"{voicing.id or ''} {voicing.tone_name or ''} {voicing.name or ''}".lower()
    for cand in ("bridge", "neck", "middle"):
        if cand in target_text:
            matched = next((k for k in active_pickup_keys if cand in k.lower()), None)
            if matched:
                primary_key_val = matched
                break
    if not primary_key_val:
        primary_key_val = (
            active_pickup_keys[0] if active_pickup_keys else next(iter(instrument.pickups.keys()))
        )

    primary_pickup = (
        instrument.pickups[primary_key_val]
        if primary_key_val in instrument.pickups
        else next(iter(instrument.pickups.values()))
    )

    if voicing.sensor_type == "bridge_force":
        fr: float | None = 800.0
        q: float | None = 1.8
    elif len(active_pickup_keys) >= 2 and any(k in curves_dict for k in active_pickup_keys):
        active_curves = [curves_dict[k] for k in active_pickup_keys if k in curves_dict]
        compound_curve: np.ndarray = np.sum(active_curves, axis=0)
        fr, q = determine_pickup_resonance_and_q(compound_curve, FREQS)
        if fr is None and primary_key_val in curves_dict:
            fr, q = determine_pickup_resonance_and_q(curves_dict[primary_key_val], FREQS)
    elif primary_key_val in curves_dict:
        fr, q = determine_pickup_resonance_and_q(curves_dict[primary_key_val], FREQS)
    else:
        fr, q = None, None

    if len(active_mags) == 1:
        magnet_type = active_mags[0]
    elif len(active_mags) > 1:
        magnet_type = " / ".join(active_mags)
    else:
        magnet_type = primary_pickup.magnet_type or "alnico_v"

    alpha = float(primary_pickup.alpha or 0.25)
    vsat = float(primary_pickup.vsat or 1.0)

    return VoicingElectroacoustics(
        coils=coils,
        fr=fr,
        q=q,
        magnet_type=magnet_type,
        alpha=alpha,
        vsat=vsat,
    )


def parse_target_voicing_token(
    target_token: str,
    default_inst_id: str,
) -> tuple[str, str]:
    """Parses a target voicing token into (instrument_id, voicing_id).

    Supports canonical 'instrument_id:voicing_id' or a local voicing ID defaulting to default_inst_id.
    """
    if ":" in target_token:
        tgt_inst_id, tgt_vid = target_token.split(":", 1)
        return tgt_inst_id, tgt_vid
    return default_inst_id, target_token


def find_voicing_by_slug_or_id(
    instrument: InstrumentConfig,
    voicing_id: str,
) -> VoicingConfig | None:
    """Finds a voicing on an instrument strictly by its canonical voicing ID."""
    if voicing_id in instrument.voicings:
        return instrument.voicings[voicing_id]
    for v in instrument.voicings.values():
        if v.id == voicing_id:
            return v
    return None
