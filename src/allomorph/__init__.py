"""
Allomorph - Universal Pickup & Transducer Analog Modeling Engine
"""

__version__ = "0.4.0"

from allomorph.circuit import (
    CALIBRATION_PEAK_CEILING,
    audit_audio_file,
    audit_wet_audio_catalog,
    export_instrument_pickup_wav,
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
    compute_voice_prefilter_firs,
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
    "compute_voice_prefilter_firs",
    "ensure_input_audio_wav",
    "export_instrument_pickup_wav",
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
