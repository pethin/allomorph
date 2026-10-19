"""Tests for the Allomorph Hardware Reamper (`src/reamper/`) and Native NAM A2 Trainer."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest
import torch

from allomorph.dsp.constants import FS
from allomorph.dsp.io import read_wav, write_wav_24bit
from allomorph.trainer.native.models import AllomorphWaveNetA2
from reamper.capture import (
    align_hardware_latency,
    calibrate_output_gain,
    capture_idle_noise,
    compute_max_unclipped_send_gain_db,
    extract_calibration_probe,
    find_anagram_device,
    read_reamp_sidecar,
    reamp,
)
from reamper.generate import (
    generate_optimal_reamp_dry,
    generate_reamp_input,
    synth_reamp_alignment_pulses,
)
from reamper.train import (
    apply_model_io_scale_compensation,
    load_signal_chains,
    resolve_signal_chain_for_stem,
    resolve_stem_send_gain_db,
    select_warm_start_offset,
    train_reamp,
    verify_or_align_stems,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeSoundDeviceModule(types.ModuleType):
    query_devices: Callable[[], list[dict[str, Any]]]
    Stream: type[Any]


def _install_fake_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
    transfer_fn: Callable[[np.ndarray], np.ndarray],
    devices: list[dict[str, Any]] | None = None,
) -> None:
    """Installs a deterministic in-memory fake sounddevice module simulating an Anagram USB audio interface."""
    dev_list = devices or [
        {
            "name": "Darkglass Anagram USB Audio",
            "max_input_channels": 3,
            "max_output_channels": 9,
        }
    ]

    class _FakeStream:
        def __init__(
            self,
            *,
            device: int,
            samplerate: int,
            blocksize: int,
            channels: tuple[int, int],
            dtype: str,
        ) -> None:
            self.device = device
            self.samplerate = samplerate
            self.blocksize = blocksize
            self.in_ch, self.out_ch = channels
            self.dtype = dtype
            self._last_out: np.ndarray = np.zeros((blocksize, self.out_ch), dtype=np.float32)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
            pass

        def write(self, data: np.ndarray) -> None:
            self._last_out = np.asarray(data, dtype=np.float32).copy()

        def read(self, frames: int) -> tuple[np.ndarray, bool]:
            out = transfer_fn(self._last_out[:frames])
            return out.astype(np.float32), False

    fake_sd = _FakeSoundDeviceModule("sounddevice")
    fake_sd.query_devices = lambda: dev_list
    fake_sd.Stream = _FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)


def test_load_and_resolve_all_anagram_signal_chains() -> None:
    """All 10 Darkglass Anagram signal chains in reamp/signal_chains.md must load and resolve cleanly."""
    chains = load_signal_chains(REPO_ROOT / "reamp" / "signal_chains.md")
    assert len(chains) == 10

    expected_stems = [
        ("peggy_classic_8x10.wav", 1, "Peggy Classic", "Vintage Peggy 8×10", "Dynamic 421"),
        ("peggy_blueline_8x10.wav", 2, "Peggy Bass", "Peggy 8×10", "Dynamic 421"),
        ("peggy_fliptop_1x15.wav", 3, "Peggy FlipTop", "Peggy Fliptop 1×15", "Condenser 67"),
        (
            "super_california_4x10.wav",
            4,
            "Super California",
            "California Bass 4×10",
            "Condenser 67",
        ),
        ("kallinen_kruukkeri_4x10.wav", 5, "Kallinen Kruukkeri", "Peggy 4×10", "Dynamic 52"),
        (
            "modern_isobrick750_4x10.wav",
            6,
            "Modern ISOBRICK750 OD",
            "Modern Bass 4×10",
            "Dynamic 52",
        ),
        ("fruity_horror_4x10.wav", 7, "Fruity Horror Bass 500", "Fruity Bass 4×10", "Dynamic 7"),
        ("swirl_900_4x10.wav", 8, "Swirl-900", "SWIRL 4×10", "Condenser 103"),
        (
            "black_and_yellow_4x10.wav",
            9,
            "Black and Yellow",
            "Black & Yellow 4×10",
            "Condenser 414",
        ),
        ("swe_tc650_4x10.wav", 10, "SWE TC650", "Darkglass Neo 4×10", "Dynamic 112"),
    ]

    for filename, exp_idx, exp_amp, exp_cab, exp_mic in expected_stems:
        matched = resolve_signal_chain_for_stem(filename, chains=chains)
        assert matched is not None, f"Failed to resolve signal chain for {filename}"
        assert matched.index == exp_idx
        assert matched.amp_block == exp_amp
        assert matched.cab_block == exp_cab
        assert matched.mic == exp_mic


def test_verify_or_align_stems_long_latency_and_phase_inversion_dual_blip() -> None:
    """verify_or_align_stems and align_hardware_latency must resolve >400ms (24,000-sample) latency
    even when phase-inverted and when Dry Blip 1 (+0.38 at 14,400) and Dry Blip 2 (-0.38 at 38,400)
    are separated by 24,000 samples."""
    sr = FS
    n_total = sr * 6  # 6.0s signal
    rng = np.random.default_rng(1234)

    dry = np.zeros(n_total, dtype=np.float32)
    dry[14400] = 0.38
    dry[38400] = -0.38
    t_music = np.arange(sr * 3, dtype=np.float32) / sr
    bursts = 0.25 * np.sin(2.0 * np.pi * 82.41 * t_music) * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * 3.5 * t_music)
    ) + 0.02 * rng.standard_normal(len(t_music)).astype(np.float32)
    dry[86400 : 86400 + len(bursts)] = bursts

    true_peak_delay = 24000
    wet = np.zeros(n_total, dtype=np.float32)
    wet[true_peak_delay:] = -0.5 * dry[: n_total - true_peak_delay]

    delay, aligned = verify_or_align_stems(
        dry,
        wet,
        sample_rate=sr,
        max_search_ms=1000.0,
        causal_guard_samples=4,
    )
    assert delay == true_peak_delay - 4
    assert len(aligned) == n_total - delay

    hw_delay, hw_aligned = align_hardware_latency(
        dry,
        wet,
        sample_rate=sr,
        max_search_ms=1000.0,
        causal_guard_samples=4,
    )
    assert hw_delay == true_peak_delay - 4
    assert len(hw_aligned) == len(dry)


def test_extract_calibration_probe_and_unclipped_headroom() -> None:
    """extract_calibration_probe must skip leading 1-sample blips and capture peak + high-RMS musical sections."""
    sr = FS
    dry = np.zeros(sr * 10, dtype=np.float32)
    dry[14400] = 0.89
    t_seg = np.arange(int(1.2 * sr), dtype=np.float32) / sr
    dry[3 * sr : 3 * sr + len(t_seg)] = 0.25 * np.sin(2.0 * np.pi * 110.0 * t_seg)
    dry[7 * sr + 5000] = 0.38

    max_clean_db = compute_max_unclipped_send_gain_db(dry, sample_rate=sr)
    assert 8.20 <= max_clean_db <= 8.45

    probe = extract_calibration_probe(dry, sample_rate=sr, window_sec=1.2)
    assert len(probe) > int(1.2 * sr)
    assert float(np.max(np.abs(probe))) >= 0.37
    assert abs(float(probe[0])) < 1e-6
    assert abs(float(probe[-1])) < 1e-6


def test_calibrate_output_gain_closed_loop_boost_and_clip_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calibrate_output_gain must probe reducing gain, boost quiet presets, back off warm/clipping presets, and find the compression knee."""
    sr = FS
    t = np.arange(sr * 6, dtype=np.float32) / sr
    dry = (0.35 * np.sin(2.0 * np.pi * 82.41 * t)).astype(np.float32)

    probed_peaks_mod: list[float] = []

    def _moderate_rig(chunk_out: np.ndarray) -> np.ndarray:
        send_sig = chunk_out[:, 8]
        pk = float(np.max(np.abs(send_sig)))
        if pk > 1e-4:
            probed_peaks_mod.append(pk)
        ret = np.zeros((len(chunk_out), 3), dtype=np.float32)
        ret[:, 0] = np.clip(1.0 * send_sig, -1.0, 1.0)
        return ret

    _install_fake_sounddevice(monkeypatch, _moderate_rig)
    res_mod = calibrate_output_gain(
        dry,
        device_idx=0,
        output_ch=9,
        input_ch=1,
        sample_rate=sr,
        target_peak_dbfs=-3.0,
        tolerance_db=0.5,
        tail_sec=0.05,
    )
    assert res_mod.converged
    assert not res_mod.clipped
    assert abs(res_mod.measured_peak_dbfs - (-3.0)) <= 0.5
    assert 5.6 <= res_mod.output_gain_db <= 6.6
    assert min(probed_peaks_mod) < 0.20

    def _very_quiet_rig(chunk_out: np.ndarray) -> np.ndarray:
        send_sig = chunk_out[:, 8]
        ret = np.zeros((len(chunk_out), 3), dtype=np.float32)
        ret[:, 0] = np.clip(0.5 * send_sig, -1.0, 1.0)
        return ret

    _install_fake_sounddevice(monkeypatch, _very_quiet_rig)
    res_ceiling = calibrate_output_gain(
        dry,
        device_idx=0,
        output_ch=9,
        input_ch=1,
        sample_rate=sr,
        target_peak_dbfs=-3.0,
        tolerance_db=0.5,
        tail_sec=0.05,
    )
    assert not res_ceiling.clipped
    assert res_ceiling.output_gain_db <= res_ceiling.max_unclipped_gain_db + 1e-6
    assert res_ceiling.output_gain_db >= 8.9

    def _warm_rig(chunk_out: np.ndarray) -> np.ndarray:
        send_sig = chunk_out[:, 8]
        ret = np.zeros((len(chunk_out), 3), dtype=np.float32)
        ret[:, 0] = np.clip(2.2 * send_sig, -1.0, 1.0)
        return ret

    _install_fake_sounddevice(monkeypatch, _warm_rig)
    res_warm = calibrate_output_gain(
        dry,
        device_idx=0,
        output_ch=9,
        input_ch=1,
        sample_rate=sr,
        target_peak_dbfs=-3.0,
        tolerance_db=0.5,
        tail_sec=0.05,
    )
    assert res_warm.converged
    assert not res_warm.clipped
    assert abs(res_warm.measured_peak_dbfs - (-3.0)) <= 0.5
    assert -1.3 <= res_warm.output_gain_db <= -0.3

    def _hot_rig(chunk_out: np.ndarray) -> np.ndarray:
        send_sig = chunk_out[:, 8]
        ret = np.zeros((len(chunk_out), 3), dtype=np.float32)
        ret[:, 0] = np.clip(4.0 * send_sig, -1.0, 1.0)
        return ret

    _install_fake_sounddevice(monkeypatch, _hot_rig)
    res_hot = calibrate_output_gain(
        dry,
        device_idx=0,
        output_ch=9,
        input_ch=1,
        sample_rate=sr,
        target_peak_dbfs=-3.0,
        tolerance_db=0.5,
        tail_sec=0.05,
    )
    assert res_hot.converged
    assert not res_hot.clipped
    assert abs(res_hot.measured_peak_dbfs - (-3.0)) <= 0.5
    assert res_hot.output_gain_db < -2.0

    dry_hot = (0.708 * np.sin(2.0 * np.pi * 82.41 * t)).astype(np.float32)

    def _cab_attenuated_saturated_rig(chunk_out: np.ndarray) -> np.ndarray:
        send_sig = chunk_out[:, 8]
        ret = np.zeros((len(chunk_out), 3), dtype=np.float32)
        ret[:, 0] = (0.42 * np.tanh(3.5 * send_sig)).astype(np.float32)
        return ret

    _install_fake_sounddevice(monkeypatch, _cab_attenuated_saturated_rig)
    res_sat = calibrate_output_gain(
        dry_hot,
        device_idx=0,
        output_ch=9,
        input_ch=1,
        sample_rate=sr,
        target_peak_dbfs=-3.0,
        tolerance_db=0.5,
        tail_sec=0.05,
    )
    assert res_sat.converged
    assert not res_sat.clipped
    assert res_sat.output_gain_db <= -5.5


def test_find_anagram_device_prefers_full_duplex(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_anagram_device must select the full-duplex endpoint when half-duplex entries also match."""
    devices = [
        {
            "name": "Darkglass Anagram (Input Only)",
            "max_input_channels": 3,
            "max_output_channels": 0,
        },
        {
            "name": "Darkglass Anagram (Full Duplex)",
            "max_input_channels": 3,
            "max_output_channels": 9,
        },
    ]
    _install_fake_sounddevice(
        monkeypatch,
        lambda x: np.zeros((len(x), 3), dtype=np.float32),
        devices=devices,
    )
    idx, info = find_anagram_device("Anagram", min_in_channels=1, min_out_channels=9)
    assert idx == 1
    assert info["name"] == "Darkglass Anagram (Full Duplex)"


def test_reamp_and_capture_idle_noise_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reamp() and capture_idle_noise() must stream audio, write 24-bit WAVs, and save JSON sidecars."""
    sr = FS
    t = np.arange(sr * 2, dtype=np.float32) / sr
    dry = (0.30 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)
    dry[100] = 0.38
    in_wav = tmp_path / "input.wav"
    out_wav = tmp_path / "stereo_wet.wav"
    write_wav_24bit(in_wav, dry, sample_rate=sr)

    def _stereo_rig(chunk_out: np.ndarray) -> np.ndarray:
        send_sig = chunk_out[:, 8]
        ret = np.zeros((len(chunk_out), 2), dtype=np.float32)
        t_chunk = np.arange(len(chunk_out), dtype=np.float32) / sr
        hum = 0.005 * np.sin(2.0 * np.pi * 60.0 * t_chunk)
        ret[:, 0] = np.clip(0.8 * send_sig + hum, -1.0, 1.0)
        ret[:, 1] = np.clip(-0.6 * send_sig + hum, -1.0, 1.0)
        return ret

    _install_fake_sounddevice(monkeypatch, _stereo_rig)
    reamp(
        input_file=in_wav,
        output_file=out_wav,
        stereo=True,
        tail=0.1,
        save_raw=True,
    )

    assert out_wav.is_file()
    assert (tmp_path / "stereo_wet_raw.wav").is_file()
    sidecar = read_reamp_sidecar(out_wav)
    assert sidecar is not None
    assert "send_gain_db" in sidecar
    assert "send_gain_linear" in sidecar
    audio_out, sr_out = read_wav(out_wav)
    assert sr_out == sr
    assert audio_out.shape == (2, len(dry))

    noise_wav = tmp_path / "captured_noise.wav"
    captured_path = capture_idle_noise(output_file=noise_wav, duration=0.5, sample_rate=sr)
    assert captured_path == noise_wav
    assert noise_wav.is_file()
    assert noise_wav.with_suffix(".json").is_file()


def test_select_warm_start_offset_skips_leading_silence() -> None:
    """select_warm_start_offset must skip leading silence and sparse blips to select dense active audio."""
    x = np.zeros(160_000, dtype=np.float32)
    x[14400] = 0.38
    t = np.arange(32768, dtype=np.float32) / FS
    x[96000 : 96000 + 32768] = 0.25 * np.sin(2.0 * np.pi * 110.0 * t)

    offset = select_warm_start_offset(x, window_len=32768)
    assert 65536 <= offset <= 98304


def test_generate_reamp_input_signal_properties(tmp_path: Path) -> None:
    """generate_reamp_input must synthesize a -3.0 dBFS peak re-amp excitation with asymmetric band-limited pulses."""
    pulses = synth_reamp_alignment_pulses(sample_rate=FS, scale=1.0)
    assert len(pulses) == FS
    assert float(np.max(pulses)) > 0.85
    assert float(np.min(pulses)) < -0.55

    out_wav = tmp_path / "reamp" / "input.wav"
    res_path = generate_reamp_input(output_file=out_wav, duration=8.0, peak_dbfs=-3.0)
    assert res_path == out_wav
    assert out_wav.is_file()

    audio, sr = read_wav(out_wav)
    assert sr == FS
    assert len(audio) == 8 * FS
    peak_db = 20.0 * np.log10(float(np.max(np.abs(audio))))
    assert abs(peak_db - (-3.0)) < 0.05

    dry_arr = generate_optimal_reamp_dry(duration_sec=8.0, sample_rate=FS, peak_dbfs=-3.0)
    assert len(dry_arr) == 8 * FS


def test_apply_model_io_scale_compensation_preserves_exact_hardware_io_levels() -> None:
    """Folding x_scale_factor into input 1x1 convs and 1/y_scale_factor into head_scale must satisfy M_exported(u) == (1/s_y) * M_trained(s_x * u)."""
    torch.manual_seed(42)
    model = AllomorphWaveNetA2()
    model.activation_tau = 0.0
    model.eval()

    s_x = 0.4892
    s_y = 0.7250
    u = torch.randn(2, 6346 + 1024, dtype=torch.float32) * 0.25

    with torch.no_grad():
        y_full_ref, y_lite_ref = model(u * s_x)
        y_full_expected = y_full_ref * (1.0 / s_y)
        y_lite_expected = y_lite_ref * (1.0 / s_y)

    apply_model_io_scale_compensation(model, x_scale_factor=s_x, y_scale_factor=s_y)

    with torch.no_grad():
        y_full_actual, y_lite_actual = model(u)

    assert float(torch.max(torch.abs(y_full_actual - y_full_expected)).item()) < 1e-6
    assert float(torch.max(torch.abs(y_lite_actual - y_lite_expected)).item()) < 1e-6


def test_train_reamp_fast_dev_run_exports_signal_chain_metadata_and_sidecar(tmp_path: Path) -> None:
    """train_reamp fast_dev_run must align stems, consume <stem>.json send_gain_db sidecar, train dual-tier A2 model, and export .nam + .json sidecar."""
    sr = FS
    n_samples = 96_000
    t = np.arange(n_samples, dtype=np.float32) / sr
    dry = (0.30 * np.sin(2.0 * np.pi * 82.41 * t)).astype(np.float32)
    send_gain_db = -4.5
    send_gain_lin = float(10.0 ** (send_gain_db / 20.0))
    wet = np.tanh(2.0 * (dry * send_gain_lin)).astype(np.float32)

    reamp_dir = tmp_path / "reamp"
    out_dir = tmp_path / "models"
    reamp_dir.mkdir(parents=True, exist_ok=True)
    dry_wav = reamp_dir / "input.wav"
    wet_wav = reamp_dir / "peggy_classic_8x10.wav"

    write_wav_24bit(dry_wav, dry, sample_rate=sr)
    write_wav_24bit(wet_wav, wet, sample_rate=sr)
    wet_wav.with_suffix(".json").write_text(
        json.dumps({"send_gain_db": send_gain_db, "send_gain_linear": send_gain_lin}),
        encoding="utf-8",
    )

    resolved_gain, resolved_src = resolve_stem_send_gain_db(wet_wav)
    assert abs(resolved_gain - send_gain_db) < 1e-6
    assert "sidecar" in resolved_src

    train_reamp(
        reamp_target=reamp_dir,
        input_file=dry_wav,
        output_dir=out_dir,
        epochs=1,
        min_epochs=1,
        batch_size=2,
        device="cpu",
        fast_dev_run=True,
    )

    nam_path = out_dir / "peggy_classic_8x10.nam"
    sidecar_path = out_dir / "peggy_classic_8x10.json"
    assert nam_path.is_file()
    assert sidecar_path.is_file()
    assert not (out_dir / "input.nam").exists()

    container = json.loads(nam_path.read_text(encoding="utf-8"))
    assert container["architecture"] == "SlimmableContainer"
    assert len(container["config"]["submodels"]) == 2
    meta = container["metadata"]
    assert meta["signal_chain_index"] == 1
    assert meta["anagram_amp_block"] == "Peggy Classic"
    assert meta["anagram_cab_block"] == "Vintage Peggy 8×10"
    assert meta["anagram_mic"] == "Dynamic 421"
    assert meta["anagram_mic_position"] == "Cone On Axis"
    assert meta["gear_make"] == "Ampeg"
    assert abs(meta["send_gain_db"] - send_gain_db) < 1e-4
    assert "sidecar" in meta["send_gain_source"]
    assert meta["input_scale_compensation"] > 0.0
    assert meta["output_scale_compensation"] > 0.0

    sidecar_meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_meta["nam_file"] == "peggy_classic_8x10.nam"
    assert sidecar_meta["anagram_amp_block"] == "Peggy Classic"
