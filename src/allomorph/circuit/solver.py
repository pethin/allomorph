"""
Allomorph - Nodal RLC Matrix Solver & Differential Transfer Functions
Evaluates closed-form nodal AC transfer functions across frequencies using
vectorized NumPy SIMD operations, Cole-Davidson dielectric relaxation,
Jordan after-effect permeability dispersion, and Wiener-regularized deconvolution.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from allomorph.circuit.mna import (
    DisjointSet,
    _CoilBranch,
    _get_cached_s_ratio_power,
    compute_active_preamp_biquads,
    compute_active_preamp_eq,
    compute_active_preamp_transfer,
    compute_core_impedance,
    compute_core_impedance_jacobians,
    evaluate_analog_band,
    smooth_soft_knee_db,
    solve_mna_linear_system,
)
from allomorph.circuit.parser import MAGNET_PROPERTIES, eval_pot_taper
from allomorph.config.schema import (
    CoilConfig,
    HarnessConfig,
    InstrumentConfig,
    PreampConfig,
    VoicingConfig,
)
from allomorph.dsp import FREQS

__all__ = [
    "DisjointSet",
    "_CoilBranch",
    "_get_cached_s_ratio_power",
    "compute_active_preamp_biquads",
    "compute_active_preamp_eq",
    "compute_active_preamp_transfer",
    "compute_core_impedance",
    "compute_core_impedance_jacobians",
    "evaluate_analog_band",
    "smooth_soft_knee_db",
    "solve_mna_harness",
    "solve_mna_linear_system",
]


def solve_mna_harness(
    instrument: InstrumentConfig,
    harness: HarnessConfig,
    voicing: VoicingConfig,
    freqs: Sequence[float] | np.ndarray = FREQS,
    return_complex: bool = False,
    preamps: Mapping[str, PreampConfig] | None = None,
) -> dict[str, np.ndarray]:
    """
    Vectorized Modified Nodal Analysis (MNA) solver for multi-harness, multi-coil bass circuits.
    Solves Y(s) * V(s) = I(s) across all frequency bins simultaneously in <0.3 ms (>3000x real-time).
    Seamlessly handles 1V/1T, 2V/1T, 2V/2T, series/parallel switches, mutual coupling, and active preamps.
    """
    if preamps is None:
        from allomorph.config.preamps import PREAMPS

        preamps = PREAMPS

    f = np.asarray(freqs, dtype=np.float64)
    w = 2.0 * np.pi * f
    s = 1j * w
    n_freqs = len(f)

    if harness.type == "direct":
        res_direct: dict[str, np.ndarray] = {}
        for p_id, p_cfg in instrument.pickups.items():
            arr = np.ones(n_freqs, dtype=np.complex128 if return_complex else np.float64)
            res_direct[p_id] = arr
            for c in p_cfg.coils:
                if c.id:
                    res_direct[f"{p_id}.{c.id}"] = arr
        return res_direct

    # 1. Gather active switch position connections and component overrides
    active_switch_connects: list[list[str]] = []
    active_components: dict[str, float] = {}
    switch_k_mutual: float | None = None

    for sw_id, sw_cfg in harness.switches.items():
        pos_name = voicing.switches.get(sw_id)
        if pos_name is None and voicing.switch is not None and len(harness.switches) == 1:
            pos_name = voicing.switch
        if pos_name is None:
            pos_name = sw_cfg.default
        if pos_name not in sw_cfg.positions:
            raise ValueError(
                f"Invalid position '{pos_name}' for switch '{sw_id}'. "
                f"Available positions: {list(sw_cfg.positions.keys())}"
            )
        pos_cfg = sw_cfg.positions[pos_name]
        active_switch_connects.extend(pos_cfg.connect)
        active_components.update(pos_cfg.components)
        if pos_cfg.k_mutual is not None:
            switch_k_mutual = pos_cfg.k_mutual

    # Apply voicing component overrides
    active_components.update(voicing.components)

    # 2. Build Disjoint Set (Nets) for wiring and active switch connections
    ds = DisjointSet()
    ds.union("GND", "GND")
    ds.union("out", "out")
    if harness.type == "active_preamp":
        ds.union("preamp.in", "preamp.in")
        ds.union("preamp.out", "preamp.out")
        ds.union("preamp.gnd", "GND")

    for conn in harness.wiring:
        if len(conn) >= 2:
            ds.union(conn[0], conn[1])

    for conn in active_switch_connects:
        if len(conn) >= 2:
            ds.union(conn[0], conn[1])

    gnd_names = {"GND", "gnd", "0", "preamp.gnd"}

    # 3. Identify all distinct non-ground nets in the circuit
    all_terms = set(ds.parent.keys())
    all_terms.add("out")
    if harness.type == "active_preamp":
        all_terms.add("preamp.in")
        all_terms.add("preamp.out")

    net_roots: set[str] = set()
    for t in all_terms:
        root = ds.find(t)
        if root not in gnd_names:
            net_roots.add(root)

    sorted_nets = sorted(net_roots)
    node_map: dict[str, int] = {net: i for i, net in enumerate(sorted_nets)}
    n_nodes = len(sorted_nets)

    if n_nodes == 0:
        return {}

    # Initialize vectorized admittance matrix Y(M, N, N)
    Y = np.zeros((n_freqs, n_nodes, n_nodes), dtype=np.complex128)
    GMIN = 1e-12
    for i in range(n_nodes):
        Y[:, i, i] += GMIN

    def stamp_adm(na: str, nb: str, y_val: complex | np.ndarray) -> None:
        ra = ds.find(na)
        rb = ds.find(nb)
        is_gnd_a = ra in gnd_names
        is_gnd_b = rb in gnd_names
        ia = node_map.get(ra, -1) if not is_gnd_a else -1
        ib = node_map.get(rb, -1) if not is_gnd_b else -1

        # If a terminal is neither in node_map nor ground, it is floating (open-circuit)
        if not is_gnd_a and ia < 0:
            return
        if not is_gnd_b and ib < 0:
            return

        if ia >= 0:
            Y[:, ia, ia] += y_val
        if ib >= 0:
            Y[:, ib, ib] += y_val
        if ia >= 0 and ib >= 0:
            Y[:, ia, ib] -= y_val
            Y[:, ib, ia] -= y_val

    # 4. Stamp Potentiometers and Controls
    for ctrl_id, ctrl in harness.controls.items():
        pos = voicing.controls.get(ctrl_id)
        if pos is None:
            pos = ctrl.default
        pos = min(max(float(pos), 0.0), 1.0)

        # Check component overrides
        r_total = float(active_components.get(f"controls.{ctrl_id}.resistance", ctrl.resistance))
        c_tone = active_components.get(f"controls.{ctrl_id}.cap", ctrl.cap)

        if ctrl.type == "pot":
            if c_tone is not None and float(c_tone) > 0.0:
                # Integrated Tone Pot (Rheostat + Cap in series)
                r_frac = eval_pot_taper(pos, ctrl.taper)
                r_rheo = r_frac * r_total
                if f"controls.{ctrl_id}.Rtone" in active_components:
                    r_rheo = float(active_components[f"controls.{ctrl_id}.Rtone"])
                elif "Rtone" in active_components:
                    r_rheo = float(active_components["Rtone"])
                alpha_tone = 0.988
                w0 = 2.0 * np.pi * 1000.0
                s_ratio = np.where(w > 0.0, w / w0, 0.0)
                kappa_tone = (
                    1j * (w0 * float(c_tone)) * np.exp(1j * (alpha_tone - 1.0) * (np.pi / 2.0))
                )
                Y_c = kappa_tone * (s_ratio**alpha_tone)
                Y_tone = Y_c / (1.0 + Y_c * r_rheo) if r_rheo > 0.0 else Y_c

                term_in = f"controls.{ctrl_id}.in"
                term_gnd = f"controls.{ctrl_id}.gnd"
                term_wiper = f"controls.{ctrl_id}.wiper"
                used_terms = [t for t in (term_in, term_wiper, term_gnd) if t in ds.parent]
                if len(used_terms) >= 2:
                    stamp_adm(used_terms[0], used_terms[1], Y_tone)
                else:
                    stamp_adm(term_in, "GND", Y_tone)

            elif ctrl.taper == "mn_blend":
                # MN Blend Potentiometer
                term_wiper = f"controls.{ctrl_id}.wiper"
                term_neck = f"controls.{ctrl_id}.neck_in"
                term_bridge = f"controls.{ctrl_id}.bridge_in"
                if term_neck not in ds.parent:
                    term_neck = f"controls.{ctrl_id}.in_a"
                if term_bridge not in ds.parent:
                    term_bridge = f"controls.{ctrl_id}.in_b"

                if pos <= 0.5:
                    r_neck = 10.0
                    atten_b = (0.5 - pos) / 0.5
                    r_bridge = max(eval_pot_taper(atten_b, "audio15") * r_total, 10.0)
                else:
                    r_bridge = 10.0
                    atten_n = (pos - 0.5) / 0.5
                    r_neck = max(eval_pot_taper(atten_n, "audio15") * r_total, 10.0)

                stamp_adm(term_neck, term_wiper, 1.0 / r_neck)
                stamp_adm(term_bridge, term_wiper, 1.0 / r_bridge)

            else:
                # Standard Volume Potentiometer (Voltage Divider)
                r_frac = eval_pot_taper(pos, ctrl.taper)
                r_top = max((1.0 - r_frac) * r_total, 10.0)
                r_bot = max(r_frac * r_total, 10.0)
                term_in = f"controls.{ctrl_id}.in"
                term_wiper = f"controls.{ctrl_id}.wiper"
                term_gnd = f"controls.{ctrl_id}.gnd"
                stamp_adm(term_in, term_wiper, 1.0 / r_top)
                stamp_adm(term_wiper, term_gnd, 1.0 / r_bot)

    # 5. Stamp Standalone Components (e.g. vintage_cap, treble bleed)
    for comp_name, comp_val in active_components.items():
        if comp_name.startswith("controls.") or comp_name in (
            "cable_pf",
            "load_resistance",
            "k_mutual",
        ):
            continue
        c_val = float(comp_val)
        term_in = f"{comp_name}.in"
        term_out = f"{comp_name}.out"
        if term_in in ds.parent or term_out in ds.parent:
            if c_val < 1e-3:
                stamp_adm(term_in, term_out, s * c_val)
            else:
                stamp_adm(term_in, term_out, 1.0 / c_val)

    # 6. Stamp Load & Cable Admittance
    cable_pf = float(active_components.get("cable_pf", harness.cable_pf))
    c_cable = cable_pf * 1e-12
    r_load = float(active_components.get("load_resistance", harness.load_resistance))
    c_anagram = 100e-12
    alpha_cable = 0.994
    w0 = 2.0 * np.pi * 1000.0
    s_ratio = np.where(w > 0.0, w / w0, 0.0)
    kappa_cable = 1j * (w0 * c_cable) * np.exp(1j * (alpha_cable - 1.0) * (np.pi / 2.0))
    y_cable = kappa_cable * (s_ratio**alpha_cable)
    y_load_out = (1.0 / r_load) + y_cable + s * c_anagram

    if harness.type == "passive":
        stamp_adm("out", "GND", y_load_out)
    elif harness.type == "active_preamp":
        y_preamp_in = (1.0 / 1e6) + s * 20e-12
        stamp_adm("preamp.in", "GND", y_preamp_in)

    # 7. Collect Coils and Compute Intrinsic Branch Impedances
    coils_dict: dict[str, _CoilBranch] = {}
    for p_id, p_cfg in instrument.pickups.items():
        p_coils = (
            p_cfg.coils
            if p_cfg.coils
            else [
                CoilConfig(
                    id=p_id,
                    position_from_bridge_m=p_cfg.position_from_bridge_m or 0.10,
                    aperture_width_in=p_cfg.aperture_width_in,
                )
            ]
        )
        for c in p_coils:
            c_id = c.id or p_id
            coil_key = f"{p_id}.{c_id}"
            alias_key = p_id if len(p_coils) == 1 else None

            t_hot = f"pickups.{p_id}.{c.id}.hot" if c.id else f"pickups.{p_id}.hot"
            if t_hot not in ds.parent and f"pickups.{p_id}.hot" in ds.parent:
                t_hot = f"pickups.{p_id}.hot"
            t_cold = f"pickups.{p_id}.{c.id}.cold" if c.id else f"pickups.{p_id}.cold"
            if t_cold not in ds.parent and f"pickups.{p_id}.cold" in ds.parent:
                t_cold = f"pickups.{p_id}.cold"

            # Electrical properties
            L_val = active_components.get(
                f"pickups.{p_id}.{c.id}.L",
                active_components.get(f"pickups.{p_id}.L", c.L if c.L is not None else None),
            )
            L = float(L_val) if L_val is not None else (float(c.L) if c.L is not None else 4.0)

            Rdc_val = active_components.get(
                f"pickups.{p_id}.{c.id}.Rdc",
                active_components.get(f"pickups.{p_id}.Rdc", c.Rdc if c.Rdc is not None else None),
            )
            Rdc = (
                float(Rdc_val)
                if Rdc_val is not None
                else (float(c.Rdc) if c.Rdc is not None else 8000.0)
            )

            Reddy_val = active_components.get(
                f"pickups.{p_id}.{c.id}.Reddy",
                active_components.get(
                    f"pickups.{p_id}.Reddy", c.Reddy if c.Reddy is not None else None
                ),
            )
            Reddy = (
                float(Reddy_val)
                if Reddy_val is not None
                else (float(c.Reddy) if c.Reddy is not None else 100000.0)
            )

            Ccoil_val = active_components.get(
                f"pickups.{p_id}.{c.id}.Ccoil",
                active_components.get(
                    f"pickups.{p_id}.Ccoil", c.Ccoil if c.Ccoil is not None else None
                ),
            )
            Ccoil = (
                float(Ccoil_val)
                if Ccoil_val is not None
                else (float(c.Ccoil) if c.Ccoil is not None else 80e-12)
            )

            mag_type = c.pole_type or p_cfg.magnet_type or "alnico_v"
            props = MAGNET_PROPERTIES.get(mag_type, MAGNET_PROPERTIES["alnico_v"])
            L_core = props.k_core * L if props.k_core > 0.0 else 0.0
            f_core = props.f_core
            R_core = 2.0 * math.pi * f_core * L_core if f_core > 0.0 else 0.0
            chi_mu = props.chi_mu
            k_skin = props.k_skin
            f_skin = props.f_skin

            Z_L = compute_core_impedance(
                s,
                L,
                L_core,
                R_core,
                chi_mu=chi_mu,
                omega_mu=2.0 * math.pi * 1200.0,
                k_skin=k_skin,
                omega_skin=2.0 * math.pi * f_skin if f_skin > 0.0 else 1.0,
                Rdc=Rdc,
            )
            Y_br = 1.0 / (Rdc + Z_L) + (1.0 / Reddy if Reddy > 0.0 else 0.0)
            Z_br = 1.0 / Y_br

            h_int: np.ndarray | None = None
            if p_cfg.has_internal_buffer:
                # Active internal buffer: op-amp input senses internal coil RLC tank voltage
                Y_shunt_int = s * Ccoil + (1.0 / 1e6)
                h_int = Y_br / (Y_br + Y_shunt_int)
                r_buf = float(p_cfg.buffer_output_impedance)
                Z_br = np.full(n_freqs, r_buf, dtype=np.complex128)
                Y_br = np.full(n_freqs, 1.0 / r_buf, dtype=np.complex128)
                Ccoil = 0.0

            cb = _CoilBranch(
                pickup_id=p_id,
                coil_id=c_id,
                alias_key=alias_key,
                t_hot=t_hot,
                t_cold=t_cold,
                L=L,
                Rdc=Rdc,
                Ccoil=Ccoil,
                Z_br=np.asarray(Z_br, dtype=np.complex128),
                Y_br=np.asarray(Y_br, dtype=np.complex128),
            )
            cb.h_internal = h_int
            coils_dict[coil_key] = cb

    # 8. Stamp Coils with Mutual Inductance
    k_m = (
        switch_k_mutual
        if switch_k_mutual is not None
        else (harness.k_mutual if harness.k_mutual is not None else 0.0)
    )
    coil_keys = list(coils_dict.keys())
    stamped_pairs: set[tuple[int, int]] = set()

    for idx_a in range(len(coil_keys)):
        k_a = coil_keys[idx_a]
        c_a = coils_dict[k_a]
        stamp_adm(c_a.t_hot, c_a.t_cold, s * c_a.Ccoil)

        coupled_b: _CoilBranch | None = None
        idx_b = -1
        if k_m > 0.0 and len(coil_keys) >= 2:
            for j in range(idx_a + 1, len(coil_keys)):
                k_candidate = coil_keys[j]
                c_cand = coils_dict[k_candidate]
                if c_a.pickup_id == c_cand.pickup_id or len(coil_keys) == 2:
                    coupled_b = c_cand
                    idx_b = j
                    break

        if coupled_b is not None and (idx_a, idx_b) not in stamped_pairs:
            stamped_pairs.add((idx_a, idx_b))
            stamped_pairs.add((idx_b, idx_a))
            c_b = coupled_b
            M = k_m * np.sqrt(c_a.L * c_b.L)
            Z_m = s * M
            Z_a = c_a.Z_br
            Z_b = c_b.Z_br
            delta = Z_a * Z_b - Z_m**2
            y_aa = Z_b / delta
            y_bb = Z_a / delta
            y_m = Z_m / delta

            c_a.y_self = y_aa
            c_a.y_mutual = y_m
            c_a.coupled_key = coil_keys[idx_b]
            c_b.y_self = y_bb
            c_b.y_mutual = y_m
            c_b.coupled_key = k_a

            stamp_adm(c_a.t_hot, c_a.t_cold, y_aa)
            stamp_adm(c_b.t_hot, c_b.t_cold, y_bb)

            ra_p = ds.find(c_a.t_hot)
            ra_m = ds.find(c_a.t_cold)
            rb_p = ds.find(c_b.t_hot)
            rb_m = ds.find(c_b.t_cold)
            ia_p = node_map.get(ra_p, -1) if ra_p not in gnd_names else -1
            ia_m = node_map.get(ra_m, -1) if ra_m not in gnd_names else -1
            ib_p = node_map.get(rb_p, -1) if rb_p not in gnd_names else -1
            ib_m = node_map.get(rb_m, -1) if rb_m not in gnd_names else -1

            if ia_p >= 0 and ib_p >= 0:
                Y[:, ia_p, ib_p] -= y_m
                Y[:, ib_p, ia_p] -= y_m
            if ia_p >= 0 and ib_m >= 0:
                Y[:, ia_p, ib_m] += y_m
                Y[:, ib_m, ia_p] += y_m
            if ia_m >= 0 and ib_p >= 0:
                Y[:, ia_m, ib_p] += y_m
                Y[:, ib_p, ia_m] += y_m
            if ia_m >= 0 and ib_m >= 0:
                Y[:, ia_m, ib_m] -= y_m
                Y[:, ib_m, ia_m] -= y_m

        elif (idx_a, idx_a) not in stamped_pairs and not any(
            idx_a in pair for pair in stamped_pairs
        ):
            c_a.y_self = c_a.Y_br
            c_a.y_mutual = 0.0
            c_a.coupled_key = None
            stamp_adm(c_a.t_hot, c_a.t_cold, c_a.Y_br)

    # 9. Solve MNA system for each coil's open-circuit EMF (E = 1 V)
    target_net = "preamp.in" if harness.type == "active_preamp" else "out"
    r_target = ds.find(target_net)
    target_idx = node_map.get(r_target, -1) if r_target not in gnd_names else -1

    H_coils: dict[str, np.ndarray] = {}

    for k_coil, c_info in coils_dict.items():
        I_vec = np.zeros((n_freqs, n_nodes), dtype=np.complex128)
        rh = ds.find(c_info.t_hot)
        rc = ds.find(c_info.t_cold)
        is_gnd_h = rh in gnd_names
        is_gnd_c = rc in gnd_names
        ih = node_map.get(rh, -1) if not is_gnd_h else -1
        ic = node_map.get(rc, -1) if not is_gnd_c else -1

        # If either coil terminal is floating (not in node_map and not ground), the coil loop is open
        if (not is_gnd_h and ih < 0) or (not is_gnd_c and ic < 0):
            H_coils[k_coil] = np.zeros(n_freqs, dtype=np.complex128)
            continue

        y_self = c_info.y_self if c_info.y_self is not None else c_info.Y_br
        scale_emf = c_info.h_internal if c_info.h_internal is not None else 1.0
        inj_current = y_self * scale_emf
        if ih >= 0:
            I_vec[:, ih] += inj_current
        if ic >= 0:
            I_vec[:, ic] -= inj_current

        coup_key = c_info.coupled_key
        if coup_key and coup_key in coils_dict:
            c_coup = coils_dict[coup_key]
            y_m = c_info.y_mutual
            rh_c = ds.find(c_coup.t_hot)
            rc_c = ds.find(c_coup.t_cold)
            is_gnd_hc = rh_c in gnd_names
            is_gnd_cc = rc_c in gnd_names
            ih_c = node_map.get(rh_c, -1) if not is_gnd_hc else -1
            ic_c = node_map.get(rc_c, -1) if not is_gnd_cc else -1
            if not ((not is_gnd_hc and ih_c < 0) or (not is_gnd_cc and ic_c < 0)):
                if ih_c >= 0:
                    I_vec[:, ih_c] -= y_m
                if ic_c >= 0:
                    I_vec[:, ic_c] += y_m

        # Solve system across all frequencies simultaneously
        V_sol = solve_mna_linear_system(Y, I_vec)

        if target_idx >= 0:
            H_coil = V_sol[:, target_idx]
        else:
            H_coil = np.zeros(n_freqs, dtype=np.complex128)

        H_coils[k_coil] = H_coil

    # 10. Active Preamp Cascading (Stage 2: Active EQ, Stage 3: Low-Z Driver)
    if harness.type == "active_preamp":
        if voicing.preamp_bands:
            H_eq = compute_active_preamp_transfer(voicing.preamp_bands, s, gain_db=voicing.gain_db)
        elif harness.preamp:
            if preamps is None or harness.preamp not in preamps:
                raise KeyError(
                    f"Preamp '{harness.preamp}' required by harness '{harness.name}' "
                    "but not provided in preamps mapping."
                )
            preamp_spec = preamps[harness.preamp]
            adjusted_bands = []
            for b in preamp_spec.bands:
                b_gain = b.gain_db
                for ctrl_id, ctrl in harness.controls.items():
                    if ctrl.type == "preamp_band" and ctrl.band == b.type:
                        ctrl_val = voicing.controls.get(ctrl_id, ctrl.default)
                        if ctrl.default == 0.5:
                            b_gain = (ctrl_val - 0.5) * 24.0
                        elif ctrl.default > 0.0:
                            b_gain = b.gain_db * (ctrl_val / ctrl.default)
                        else:
                            b_gain = ctrl_val * 12.0
                adjusted_bands.append(b.model_copy(update={"gain_db": b_gain}))
            H_eq = compute_active_preamp_transfer(
                adjusted_bands, s, gain_db=float(preamp_spec.gain_db) + voicing.gain_db
            )
        else:
            H_eq = np.ones(n_freqs, dtype=np.complex128)

        r_out_stage = 100.0
        r_pre_out = ds.find("preamp.out")
        r_out = ds.find("out")

        vol_r_top = 0.0
        vol_r_bot: float | None = None
        for ctrl_id, ctrl in harness.controls.items():
            if ctrl.type == "pot" and ctrl.cap is None and ctrl.taper != "mn_blend":
                t_in = ds.find(f"controls.{ctrl_id}.in")
                t_w = ds.find(f"controls.{ctrl_id}.wiper")
                if t_in == r_pre_out and t_w == r_out:
                    pos = min(max(float(voicing.controls.get(ctrl_id, ctrl.default)), 0.0), 1.0)
                    r_frac = eval_pot_taper(pos, ctrl.taper)
                    vol_r_top = max((1.0 - r_frac) * ctrl.resistance, 10.0)
                    vol_r_bot = max(r_frac * ctrl.resistance, 10.0)
                    break

        if vol_r_bot is not None:
            z_bot = 1.0 / (1.0 / vol_r_bot + y_load_out)
            H_post = z_bot / (r_out_stage + vol_r_top + z_bot)
        else:
            z_load = 1.0 / y_load_out
            H_post = z_load / (r_out_stage + z_load)

        for k_coil in list(H_coils.keys()):
            H_coils[k_coil] = H_coils[k_coil] * H_eq * H_post

    for k_coil, c_info in list(coils_dict.items()):
        alias = c_info.alias_key
        if alias and alias not in H_coils and k_coil in H_coils:
            H_coils[alias] = H_coils[k_coil]

    for p_id, p_cfg in instrument.pickups.items():
        if p_id not in H_coils:
            sub_coils = [v for k, v in H_coils.items() if k.startswith(f"{p_id}.")]
            if sub_coils:
                # For split coils (disjoint string coverage), each string only excites one coil half
                if p_cfg.type == "split_coil" or any(
                    c.strings and c.strings != ["all"] for c in p_cfg.coils
                ):
                    H_coils[p_id] = np.mean(sub_coils, axis=0)
                else:
                    H_coils[p_id] = np.sum(sub_coils, axis=0)

    if return_complex:
        return H_coils
    return {k: np.abs(v) for k, v in H_coils.items()}
