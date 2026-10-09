"""
Allomorph - Universal Pickup & Transducer Analog Modeling Engine
"""

__version__ = "0.4.1"

import sys
import types
from typing import Any

# Ensure headless matplotlib raster backend is active
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

# Headless Tkinter fallback: neural-amp-modeler's core trainer imports tkinter at top-level
# for an unused GUI warning modal. On headless Linux environments or systems without python3-tk,
# this raises ModuleNotFoundError during import. Provide a minimal shim if tkinter is unavailable.
if "tkinter" not in sys.modules:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        def _dummy_mainloop(*args: object, **kwargs: object) -> None:
            pass

        class _DummyTkMisc:
            mainloop = _dummy_mainloop

        class _DummyTkWidget:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __getattr__(self, name: str) -> Any:
                return _dummy_mainloop

        class _DummyTkModule(types.ModuleType):
            Tk = _DummyTkWidget
            Toplevel = _DummyTkWidget
            Label = _DummyTkWidget
            Button = _DummyTkWidget
            Misc = _DummyTkMisc
            mainloop = _dummy_mainloop

        sys.modules["tkinter"] = _DummyTkModule("tkinter")

from allomorph.circuit import (
    CALIBRATION_PEAK_CEILING,
    audit_audio_file,
    audit_wet_audio_catalog,
    find_default_input_audio,
    resolve_target_voicing,
    simulate_all_instrument_voicings,
    simulate_instrument_voicing,
    simulate_voice,
)
from allomorph.config.geometry import (
    compute_effective_position,
    resolve_pickup_coils,
    resolve_voice_coils,
    resolve_voice_pickups,
)
from allomorph.config.instruments import (
    INSTRUMENTS,
    get_source_pickup,
    load_all_instruments,
    load_instrument,
)
from allomorph.config.scales import (
    SCALES,
    load_scales,
)
from allomorph.config.strings import (
    STRINGS,
    load_strings_config,
)
from allomorph.config.voices import (
    VOICES,
    load_voices_config,
)
from allomorph.dsp import (
    DEFAULT_INPUT_PATH,
    FREQS,
    FS,
    NUM_TAPS,
    NYQ,
    calibrate_nam_v3_latency,
    cinf_smoothstep,
    compute_lufs,
    compute_true_peak,
    compute_true_peak_dbfs,
    ensure_input_audio_wav,
    fft_convolve,
    generate_input_audio,
    generate_optimal_bass_dry,
    read_wav,
    read_wav_24bit,
    synthesize_minimum_phase_fir,
    write_wav_24bit,
)
from allomorph.naming import (
    VOICE_CONCISE_SLUGS,
    get_default_input_path,
    get_instrument_pickup_basename,
    get_t3k_basename,
    resolve_instruments,
    resolve_voices,
)
from allomorph.physics import (
    compute_coil_aperture,
    generate_wave_speed_continuum,
    numpy_pickup_acoustic_response,
    numpy_pickup_macro_aperture,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    compute_file_sha256,
    get_version_info,
    is_wet_stem_valid,
    resolve_tri_part_version,
    write_manifest,
)

__all__ = [
    "ALLOMORPH_VERSION",
    "CALIBRATION_PEAK_CEILING",
    "DEFAULT_INPUT_PATH",
    "DSP_GENERATION",
    "FREQS",
    "FS",
    "INSTRUMENTS",
    "NUM_TAPS",
    "NYQ",
    "SCALES",
    "STRINGS",
    "VOICES",
    "VOICE_CONCISE_SLUGS",
    "audit_audio_file",
    "audit_wet_audio_catalog",
    "calibrate_nam_v3_latency",
    "cinf_smoothstep",
    "compute_coil_aperture",
    "compute_effective_position",
    "compute_file_sha256",
    "compute_lufs",
    "compute_true_peak",
    "compute_true_peak_dbfs",
    "ensure_input_audio_wav",
    "fft_convolve",
    "find_default_input_audio",
    "generate_input_audio",
    "generate_optimal_bass_dry",
    "generate_wave_speed_continuum",
    "get_default_input_path",
    "get_instrument_pickup_basename",
    "get_source_pickup",
    "get_t3k_basename",
    "get_version_info",
    "is_wet_stem_valid",
    "load_all_instruments",
    "load_instrument",
    "load_scales",
    "load_strings_config",
    "load_voices_config",
    "numpy_pickup_acoustic_response",
    "numpy_pickup_macro_aperture",
    "read_wav",
    "read_wav_24bit",
    "resolve_instruments",
    "resolve_pickup_coils",
    "resolve_target_voicing",
    "resolve_tri_part_version",
    "resolve_voice_coils",
    "resolve_voice_pickups",
    "resolve_voices",
    "simulate_all_instrument_voicings",
    "simulate_instrument_voicing",
    "simulate_voice",
    "synthesize_minimum_phase_fir",
    "write_manifest",
    "write_wav_24bit",
]
