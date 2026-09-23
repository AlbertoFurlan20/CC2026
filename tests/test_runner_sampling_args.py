import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse


def _build_parser():
    """Re-create runner.py's argparse parser for isolated testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--work-dir", type=str, default="./workspace")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--llm-name", type=str, default="claude-v1")
    parser.add_argument("--fast-llm-name", type=str, default="claude-v1")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    return parser


def test_top_p_flag_parsed():
    parser = _build_parser()
    args = parser.parse_args(["--task", "cifar10", "--top-p", "0.7"])
    assert args.top_p == 0.7


def test_best_of_flag_parsed():
    parser = _build_parser()
    args = parser.parse_args(["--task", "cifar10", "--best-of", "3"])
    assert args.best_of == 3


def test_n_samples_flag_parsed():
    parser = _build_parser()
    args = parser.parse_args(["--task", "cifar10", "--n-samples", "2"])
    assert args.n_samples == 2


def test_temperature_flag_parsed():
    parser = _build_parser()
    args = parser.parse_args(["--task", "cifar10", "--temperature", "0.35"])
    assert args.temperature == 0.35


def test_sampling_flags_default_to_none():
    parser = _build_parser()
    args = parser.parse_args(["--task", "cifar10"])
    assert args.top_p is None
    assert args.best_of is None
    assert args.n_samples is None
    assert args.temperature is None
