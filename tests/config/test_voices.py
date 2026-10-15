"""
Tests for target voices catalog, electrical parameters, SPICE netlist existence, and transparency.
"""

import numpy as np

from allomorph.circuit.solver import solve_mna_harness
from allomorph.config import VOICES
from allomorph.config.instruments import load_instrument
from allomorph.config.schema import VoiceConfig
from allomorph.dsp import FREQS


def test_voice_parameter_validity():
    """Verify that all target voices are validated VoiceConfig models satisfying physical bounds."""
    for vid, cfg in VOICES.items():
        assert isinstance(cfg, VoiceConfig), f"{vid} is not a VoiceConfig instance"
        assert cfg.name and cfg.description
        if cfg.fr is not None:
            assert cfg.fr > 0
            is_active = (
                bool(
                    cfg.instrument_id
                    and (
                        "active" in cfg.instrument_id
                        or "dingwall" in cfg.instrument_id
                        or "preamp" in cfg.instrument_id
                        or "emg" in cfg.instrument_id
                    )
                )
                or "active" in vid
                or "emg" in vid
            )
            max_fr = 20000.0 if is_active else 6000.0
            assert 200.0 <= cfg.fr <= max_fr, f"{vid} fr outside audible musical range: {cfg.fr}"
            assert cfg.Q is not None and cfg.Q > 0
        assert len(cfg.coils) >= 1
        for c in cfg.coils:
            assert c.position_from_bridge_m > 0
            assert c.aperture_width_in > 0
            assert c.weight > 0
            assert len(c.strings) >= 1

        if cfg.pickups:
            assert len(cfg.pickups) >= 2, f"{vid} pickups list has fewer than 2 pickups"
            for p in cfg.pickups:
                assert p.name
                if p.fr is not None:
                    assert p.fr > 0
                    assert p.Q is not None and p.Q > 0
                assert p.weight > 0
                assert len(p.coils) >= 1


def test_voice_netlist_existence():
    """Verify that every target voice defines a valid instrument harness and resolves in MNA."""
    for vid, cfg in VOICES.items():
        assert cfg.instrument_id is not None, f"{vid} missing instrument_id"
        inst = load_instrument(cfg.instrument_id)
        assert inst is not None, f"Instrument '{cfg.instrument_id}' could not be loaded for {vid}"
        assert inst.voicings, f"Instrument '{cfg.instrument_id}' has no voicings"
        v_match = None
        for v in inst.voicings.values():
            if (v.tone_name and v.tone_name == cfg.tone_name) or v.id == vid or v.name == cfg.name:
                v_match = v
                break
        if v_match is None:
            v_match = next(iter(inst.voicings.values()))
        assert v_match.harness in inst.harnesses, (
            f"Harness '{v_match.harness}' missing from {inst.id}"
        )
        harness = inst.harnesses[v_match.harness]
        curves = solve_mna_harness(inst, harness, v_match, freqs=FREQS)
        assert len(curves) >= 1
        for c in curves.values():
            assert len(c) == len(FREQS)
            assert np.all(np.isfinite(c))


def test_voices_have_no_hardcoded_source_datums():
    # Target voices should be purely decoupled from source instrument geometries
    for vid, cfg in VOICES.items():
        assert not hasattr(cfg, "src_32"), f"{vid} contains deprecated hardcoded 'src_32' datum"
        assert not hasattr(cfg, "src_30"), f"{vid} contains deprecated hardcoded 'src_30' datum"


def test_multi_magnet_target_voicing_composite_formatting():
    """Verify that multi-pickup voicings with distinct magnets format composite magnet_type cleanly."""
    from allomorph.config.voices import voicing_to_voice_config

    inst = load_instrument("32in_custom_pmm")
    v_blend = voicing_to_voice_config(inst, inst.voicings["blend_parallel"])
    assert v_blend.magnet_type == "ceramic / ceramic_steel"
    assert v_blend.pickups is not None
    assert len(v_blend.pickups) == 2
    assert v_blend.pickups[0].magnet_type == "ceramic"
    assert v_blend.pickups[1].magnet_type == "ceramic_steel"

    # Single pickup voicing preserves that pickup's magnet
    v_solo = voicing_to_voice_config(inst, inst.voicings["px_solo"])
    assert v_solo.magnet_type == "ceramic"


def test_voice_aliases_resolution():
    """Verify that voice aliases resolve cleanly to valid target VoiceConfigs."""
    assert "precision_active" in VOICES
    assert VOICES["precision_active"].id == "precision_active"
    assert "p_active" in VOICES
    assert VOICES["p_active"].id == "precision_active"
    assert "j_active" in VOICES
    assert VOICES["j_active"].id == "jazz_pair_active"
    assert "pj_pair_active" in VOICES
    assert VOICES["pj_pair_active"].id == "pj_active"


def test_phase2_determine_pickup_resonance_ignores_dc_shelf():
    """Finding 2.3: determine_pickup_resonance_and_q must ignore sub-100 Hz bass shelf peaks."""
    from allomorph.config.resolvers import determine_pickup_resonance_and_q

    # Precision Active and PJ Active must resolve valid 3.5-4.5 kHz resonant peaks
    p_act = VOICES["precision_active"]
    assert p_act.fr is not None and 3500.0 <= p_act.fr <= 4500.0
    assert p_act.Q is not None and 0.7 <= p_act.Q <= 1.5

    pj_act = VOICES["pj_active"]
    assert pj_act.fr is not None and 3200.0 <= pj_act.fr <= 4200.0
    assert pj_act.Q is not None and 0.7 <= pj_act.Q <= 1.5

    # Synthetic curve with +6 dB sub-100 Hz DC shelf and a resonant peak at 3000 Hz
    f_arr = np.asarray(FREQS, dtype=float)
    dc_shelf = 1.0 + 1.0 / (1.0 + (f_arr / 25.0) ** 4)
    rlc_peak = 1.0 / np.sqrt((1.0 - (f_arr / 3000.0) ** 2) ** 2 + (f_arr / (1.5 * 3000.0)) ** 2)
    synthetic = dc_shelf * rlc_peak
    fr_det, q_det = determine_pickup_resonance_and_q(synthetic, f_arr)
    assert fr_det is not None and 2600.0 <= fr_det <= 3100.0
    assert q_det is not None and 1.0 <= q_det <= 1.8


def test_phase2_voicing_to_voice_config_mna_weights_and_bridge_force():
    """Finding 2.4: voicing_to_voice_config computes MNA 100 Hz weights and sets bridge_force fr/Q."""
    from allomorph.config.geometry import compute_effective_position

    # Jazz Bridge Growl (neck_vol=0.99, bridge_vol=1.0) weights bridge > neck
    j_growl = VOICES["jazz_bridge_growl"]
    assert j_growl.pickups is not None and len(j_growl.pickups) == 2
    w_neck = j_growl.pickups[0].weight
    w_bridge = j_growl.pickups[1].weight
    assert w_bridge > 2.0 * w_neck
    eff_pos = compute_effective_position(j_growl.coils)
    assert 0.088 <= eff_pos <= 0.098  # ~93.2 mm (closer to 63.5mm bridge than 111.3mm center)

    # Upright Piezo (sensor_type == 'bridge_force') sets maple bridge rocking resonance (800 Hz, Q=1.8)
    upright = VOICES["upright_piezo"]
    assert upright.sensor_type == "bridge_force"
    assert upright.fr == 800.0
    assert upright.Q == 1.8


def test_phase2_exact_scales_mapping_for_all_instruments():
    """Finding 2.5: Every instrument maps to its exact physical scale in SCALES."""
    from allomorph.config.instruments import load_all_instruments
    from allomorph.config.scales import SCALES
    from allomorph.config.voices import voicing_to_voice_config

    expected_scales = {
        "30in_gibson_eb0": "30.5in",
        "33in_rickenbacker_4003": "33.25in",
        "34in_active_emg": "34in_5string",
        "34in_preamp_soapbar": "34in_5string",
        "34in_dingwall_sp1": "multiscale_super",
        "37in_multiscale_dingwall": "multiscale",
        "41in_upright_bass": "upright",
    }
    all_insts = load_all_instruments()
    for inst_id, expected_scale in expected_scales.items():
        inst = all_insts[inst_id]
        v0 = next(iter(inst.voicings.values()))
        vcfg = voicing_to_voice_config(inst, v0)
        assert vcfg.scale == expected_scale
        sc = SCALES[vcfg.scale]
        assert sc.scale_length_in is not None and inst.scale_length_in is not None
        assert np.isclose(sc.scale_length_in, inst.scale_length_in)
        assert len(sc.string_wave_speeds) == len(inst.string_wave_speeds)


def test_phase2_coil_split_weight_renormalization():
    """Finding 2.9: Coil-splitting a dual-coil pickup re-normalizes active coil weight to 1.0."""
    from allomorph.config.voices import voicing_to_voice_config

    for inst_id, v_id in [
        ("30in_emg_mmtw", "single_mode"),
        ("32in_custom_pmm", "mm_single"),
        ("32in_fretless_pmm", "jaco_growl"),
        ("34in_active_pmm", "mm_single_solo"),
    ]:
        inst = load_instrument(inst_id)
        vcfg = voicing_to_voice_config(inst, inst.voicings[v_id])
        assert len(vcfg.coils) == 1
        assert np.isclose(vcfg.coils[0].weight, 1.0)


def test_all_instrument_voicings_loaded_in_voices_registry():
    """Finding 3.2: All 60 native voicings across all 16 instruments are auto-loaded into VOICES and mapped in VOICE_FAMILIES."""
    from allomorph.config.instruments import load_all_instruments
    from allomorph.visualizer.dataframe import VOICE_FAMILIES

    instruments = load_all_instruments()
    total_voicings = sum(len(inst.voicings) for inst in instruments.values())
    assert len(VOICES) == total_voicings == 60
    assert set(VOICES.keys()) == set(VOICE_FAMILIES.keys())
    for inst_id, inst in instruments.items():
        for v_id in inst.voicings:
            ref = f"{inst_id}:{v_id}"
            assert ref in VOICES, f"Canonical voicing reference {ref} missing from VOICES"
