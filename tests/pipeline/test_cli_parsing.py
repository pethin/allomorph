"""
Deterministic unit tests for CLI argument parsing, schema validation, and target resolution.
"""

import pytest

from allomorph.config.instruments import INSTRUMENTS
from allomorph.pipeline.cli import (
    build_pipeline_arg_parser,
    parse_and_validate_pipeline_args,
    resolve_execution_targets,
)
from allomorph.pipeline.schema import PipelineCliConfig
from allomorph.pipeline.sim_cli import (
    build_sim_arg_parser,
    parse_and_validate_sim_args,
)


def test_build_pipeline_arg_parser_creation() -> None:
    parser = build_pipeline_arg_parser()
    assert parser is not None
    assert parser.description == "Allomorph SPICE -> NAM Automation Pipeline"


def test_parse_pipeline_args_defaults() -> None:
    _args, config = parse_and_validate_pipeline_args([])
    assert config.stage == "all"
    assert config.instrument == "all"
    assert config.voice == "all"
    assert config.normalize == "auto"
    assert config.target_dbfs is None
    assert config.train is False
    assert config.force is False
    assert config.overwrite is False
    assert config.clean_audio is False
    assert config.jobs is None
    assert config.max_samples is None


def test_parse_pipeline_args_explicit() -> None:
    argv = [
        "--stage",
        "sim",
        "--instrument",
        "30in",
        "--voice",
        "jazz_bridge_growl",
        "--jobs",
        "2",
        "--max-samples",
        "48000",
        "--normalize",
        "rms",
        "--target-dbfs",
        "-20.5",
        "--force",
    ]
    _args, config = parse_and_validate_pipeline_args(argv)
    assert config.stage == "sim"
    assert config.instrument == "30in"
    assert config.voice == "jazz_bridge_growl"
    assert config.jobs == 2
    assert config.max_samples == 48000
    assert config.normalize == "rms"
    assert config.target_dbfs == -20.5
    assert config.force is True


def test_parse_pipeline_args_invalid_jobs() -> None:
    with pytest.raises(ValueError, match="--jobs must be a positive integer >= 1"):
        parse_and_validate_pipeline_args(["--jobs", "0"])

    with pytest.raises(ValueError, match="--jobs must be a positive integer >= 1"):
        parse_and_validate_pipeline_args(["--jobs", "-1"])


def test_parse_pipeline_args_invalid_max_samples() -> None:
    with pytest.raises(ValueError, match="--max-samples must be a positive integer >= 1"):
        parse_and_validate_pipeline_args(["--max-samples", "0"])

    with pytest.raises(ValueError, match="--max-samples must be a positive integer >= 1"):
        parse_and_validate_pipeline_args(["--max-samples", "-100"])


def test_resolve_execution_targets_alias() -> None:
    config = PipelineCliConfig(instrument="30in", voice="jazz_bridge_growl", jobs=2)
    insts, voices, jobs = resolve_execution_targets(config)
    assert insts == ["30in_mustang_pj"]
    assert "jazz_bridge_growl" in voices
    assert jobs == 2


def test_resolve_execution_targets_all() -> None:
    config = PipelineCliConfig(instrument="all", voice="all", jobs=None)
    insts, voices, jobs = resolve_execution_targets(config)
    assert len(insts) == len(INSTRUMENTS)
    assert set(insts) == set(INSTRUMENTS.keys())
    assert len(voices) > 0
    assert jobs >= 1


def test_build_sim_arg_parser_creation() -> None:
    parser = build_sim_arg_parser()
    assert parser is not None
    assert parser.description == "Allomorph Native Virtual Analog Circuit Simulator."


def test_parse_sim_args_defaults() -> None:
    args = parse_and_validate_sim_args([])
    assert args.instrument == "all"
    assert args.voice == "all"
    assert args.normalize == "auto"
    assert args.target_dbfs is None
    assert args.oversample == 2
    assert args.no_displacement_weighting is False
    assert args.no_magnet_drag is False
    assert args.jobs is None
    assert args.max_samples is None


def test_parse_sim_args_explicit() -> None:
    argv = [
        "--instrument",
        "30in",
        "--voice",
        "jazz_bridge_growl",
        "--vol",
        "0.8",
        "--tone",
        "0.7",
        "--blend",
        "0.5",
        "--oversample",
        "4",
        "--jobs",
        "3",
        "--max-samples",
        "96000",
    ]
    args = parse_and_validate_sim_args(argv)
    assert args.instrument == "30in"
    assert args.voice == "jazz_bridge_growl"
    assert args.vol == 0.8
    assert args.tone == 0.7
    assert args.blend == 0.5
    assert args.oversample == 4
    assert args.jobs == 3
    assert args.max_samples == 96000


def test_parse_sim_args_invalid_jobs() -> None:
    with pytest.raises(ValueError, match="--jobs must be a positive integer >= 1"):
        parse_and_validate_sim_args(["--jobs", "0"])


def test_parse_sim_args_invalid_max_samples() -> None:
    with pytest.raises(ValueError, match="--max-samples must be a positive integer >= 1"):
        parse_and_validate_sim_args(["--max-samples", "0"])
