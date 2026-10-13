"""
Deterministic unit tests for allomorph-sim CLI argument parsing and schema validation.
"""

import pytest

from allomorph.pipeline.sim_cli import (
    build_sim_arg_parser,
    parse_and_validate_sim_args,
)


def test_build_sim_arg_parser_creation() -> None:
    parser = build_sim_arg_parser()
    assert parser is not None
    assert parser.description == "Allomorph Native Virtual Analog Circuit Simulator."


def test_parse_sim_args_defaults() -> None:
    args = parse_and_validate_sim_args([])
    assert args.instrument == "all"
    assert args.voice == "all"
    assert args.vol is None
    assert args.tone is None
    assert args.blend is None
    assert args.oversample == 2
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
