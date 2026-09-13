"""
Allomorph - Universal Pickup & Transducer Analog Modeling Engine
"""

__version__ = "0.2.0"

from allomorph.circuit import (
    CALIBRATION_PEAK_CEILING,
    compute_frontend_deconvolution_fir,
    export_all_frontend_irs,
    export_all_frontend_wet_wavs,
    export_frontend_ir,
    export_frontend_wet_wav,
    find_default_canonical_sweep,
    find_default_input_audio,
    generate_canonical_sweep,
    simulate_backend_targets,
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
    FREQS,
    FS,
    NUM_TAPS,
    NYQ,
    OPTIMAL_DRY_PATH,
    calibrate_nam_v3_latency,
    ensure_optimal_dry_wav,
    fft_convolve,
    generate_optimal_bass_dry,
    read_wav,
    read_wav_24bit,
    synthesize_minimum_phase_fir,
    write_wav_24bit,
)
from allomorph.naming import (
    VOICE_CONCISE_SLUGS,
    get_baked_basename,
    get_canonical_sweep_basename,
    get_canonical_sweep_path,
    get_optimal_dry_basename,
    get_optimal_dry_path,
    get_t3k_basename,
    resolve_instruments,
    resolve_voices,
)
from allomorph.physics import (
    compute_aperture_prefilter_fir,
    compute_coil_aperture,
    compute_voice_prefilter_firs,
    generate_wave_speed_continuum,
    numpy_pickup_acoustic_response,
    numpy_pickup_macro_aperture,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    get_version_info,
    resolve_tri_part_version,
    write_manifest,
)

__all__ = [
    "ALLOMORPH_VERSION",
    "CALIBRATION_PEAK_CEILING",
    "DSP_GENERATION",
    "FREQS",
    "FS",
    "INSTRUMENTS",
    "NUM_TAPS",
    "NYQ",
    "OPTIMAL_DRY_PATH",
    "SCALES",
    "STRINGS",
    "VOICES",
    "VOICE_CONCISE_SLUGS",
    "calibrate_nam_v3_latency",
    "compute_aperture_prefilter_fir",
    "compute_coil_aperture",
    "compute_effective_position",
    "compute_frontend_deconvolution_fir",
    "compute_voice_prefilter_firs",
    "ensure_optimal_dry_wav",
    "export_all_frontend_irs",
    "export_all_frontend_wet_wavs",
    "export_frontend_ir",
    "export_frontend_wet_wav",
    "fft_convolve",
    "find_default_canonical_sweep",
    "find_default_input_audio",
    "generate_canonical_sweep",
    "generate_optimal_bass_dry",
    "generate_wave_speed_continuum",
    "get_baked_basename",
    "get_canonical_sweep_basename",
    "get_canonical_sweep_path",
    "get_optimal_dry_basename",
    "get_optimal_dry_path",
    "get_source_pickup",
    "get_t3k_basename",
    "get_version_info",
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
    "resolve_tri_part_version",
    "resolve_voice_coils",
    "resolve_voice_pickups",
    "resolve_voices",
    "simulate_backend_targets",
    "simulate_voice",
    "synthesize_minimum_phase_fir",
    "write_manifest",
    "write_wav_24bit",
]
