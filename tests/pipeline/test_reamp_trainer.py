"""Tests for the Darkglass Anagram hardware re-amper (scripts/reamp_anagram.py) and re-amped rig native NAM trainer (scripts/train_reamp.py)."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from allomorph.dsp.constants import FS
from allomorph.dsp.io import read_wav, write_wav_24bit
from scripts.generate_reamp_input import (
    generate_optimal_reamp_dry,
    generate_reamp_input,
    synth_reamp_alignment_pulses,
)
from scripts.reamp_anagram import (
    align_hardware_latency,
    calibrate_output_gain,
    compute_max_unclipped_send_gain_db,
    extract_calibration_probe,
    find_anagram_device,
    reamp,
)
from scripts.train_reamp import (
    load_signal_chains,
    resolve_signal_chain_for_stem,
    select_warm_start_offset,
    train_reamp,
    verify_or_align_stems,
)


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
    # 1-sample calibration blip in first 2 seconds
    dry[14400] = 0.89
    # High-RMS sustained section at t=3..4.2s
    t_seg = np.arange(int(1.2 * sr), dtype=np.float32) / sr
    dry[3 * sr : 3 * sr + len(t_seg)] = 0.25 * np.sin(2.0 * np.pi * 110.0 * t_seg)
    # High-peak transient burst at t=7..8.2s
    dry[7 * sr + 5000] = 0.38

    max_clean_db = compute_max_unclipped_send_gain_db(dry, sample_rate=sr)
    # Musical peak is 0.38 (-8.40 dBFS) -> max clean send gain to 0.9900 (-0.09 dBFS) is ~+8.32 dB
    assert 8.20 <= max_clean_db <= 8.45

    probe = extract_calibration_probe(dry, sample_rate=sr, window_sec=1.2)
    assert len(probe) > int(1.2 * sr)
    assert float(np.max(np.abs(probe))) >= 0.37
    # Boundary samples must be zero due to C-infinity fades
    assert abs(float(probe[0])) < 1e-6
    assert abs(float(probe[-1])) < 1e-6


def test_calibrate_output_gain_closed_loop_boost_and_clip_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calibrate_output_gain must boost quiet presets, respect clean send ceiling, and back off clipping presets."""
    sr = FS
    t = np.arange(sr * 6, dtype=np.float32) / sr
    dry = (0.35 * np.sin(2.0 * np.pi * 82.41 * t)).astype(np.float32)

    # Case 1: Unity/moderate preset (1.0x gain -> -9.12 dBFS at 0 dB send, needs +6.12 dB to reach -3.0 dBFS)
    def _moderate_rig(chunk_out: np.ndarray) -> np.ndarray:
        send_sig = chunk_out[:, 8]
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

    # Case 2: Very quiet preset (0.5x gain) where reaching -3.0 dBFS would clip Output 9 -> clamps to max_unclipped_gain_db
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

    # Case 3: Hot preset (+12 dB internal gain, 4.0x) that clips at 0 dB send gain
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


def test_reamp_end_to_end_with_auto_gain_and_stereo_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reamp() must calibrate send gain, stream audio, align latency, and preserve stereo channels when requested."""
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
        ret[:, 0] = np.clip(0.8 * send_sig, -1.0, 1.0)
        ret[:, 1] = np.clip(-0.6 * send_sig, -1.0, 1.0)
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
    audio_out, sr_out = read_wav(out_wav)
    assert sr_out == sr
    assert audio_out.shape == (2, len(dry))


def test_select_warm_start_offset_skips_leading_silence() -> None:
    """select_warm_start_offset must skip leading silence and sparse blips to select dense active audio."""
    x = np.zeros(160_000, dtype=np.float32)
    x[14400] = 0.38  # single 1-sample blip in first window
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


def test_train_reamp_fast_dev_run_exports_signal_chain_metadata(tmp_path: Path) -> None:
    """train_reamp fast_dev_run must align stems, exclude reamp/input.wav from targets, train dual-tier A2 model, and export signal chain metadata."""
    sr = FS
    n_samples = 96_000  # 2.0 seconds (>= 65,536 minimum)
    t = np.arange(n_samples, dtype=np.float32) / sr
    dry = (0.15 * np.sin(2.0 * np.pi * 82.41 * t)).astype(np.float32)
    wet = np.tanh(2.0 * dry).astype(np.float32)

    reamp_dir = tmp_path / "reamp"
    out_dir = tmp_path / "models"
    reamp_dir.mkdir(parents=True, exist_ok=True)
    dry_wav = reamp_dir / "input.wav"
    wet_wav = reamp_dir / "peggy_classic_8x10.wav"

    write_wav_24bit(dry_wav, dry, sample_rate=sr)
    write_wav_24bit(wet_wav, wet, sample_rate=sr)

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
    assert nam_path.is_file()
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
