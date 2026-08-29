# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signal capture flags: parsing, validation, and the resolved worker config."""

import pytest

from vllm.config.observability import ObservabilityConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.signals.config import SignalCaptureConfig
from vllm.signals.tiers import Tier
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse(args: list[str]) -> ObservabilityConfig:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(
        parser.parse_args(["--model", "facebook/opt-125m", *args])
    ).create_observability_config()


def test_capture_is_off_by_default():
    config = parse([])
    assert config.signal_capture_tier == "off"
    assert not config.signal_capture_enabled


def test_flags_reach_the_observability_config(tmp_path):
    config = parse(
        [
            "--signal-capture-tier",
            "residual_raw",
            "--signal-capture-dir",
            str(tmp_path),
            "--signal-capture-layers",
            "all",
            "--signal-capture-token-step",
            "4",
            "--signal-capture-layer-step",
            "2",
            "--signal-capture-dtype",
            "float32",
            "--signal-capture-max-bytes",
            "1024",
            "--signal-capture-session",
            "sweep-a",
        ]
    )
    assert config.signal_capture_enabled
    resolved = SignalCaptureConfig.from_observability(config)
    assert resolved.tier is Tier.RESIDUAL_RAW
    assert resolved.output_dir == str(tmp_path)
    assert resolved.layers == "all"
    assert resolved.token_step == 4
    assert resolved.layer_step == 2
    assert resolved.dtype == "float32"
    assert resolved.max_bytes == 1024
    assert resolved.session == "sweep-a"
    assert resolved.enabled


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("t4", Tier.RESIDUAL_RAW),
        ("residual_raw", Tier.RESIDUAL_RAW),
        ("t2", Tier.LAYER_STATS),
        ("logit", Tier.LOGIT),
        ("t5", Tier.FULL_RAW),
    ],
)
def test_tier_spellings(tmp_path, spelling, expected):
    config = ObservabilityConfig(
        signal_capture_tier=spelling, signal_capture_dir=str(tmp_path)
    )
    assert SignalCaptureConfig.from_observability(config).tier is expected
    # Stored canonically, so deposit metadata is stable across spellings.
    assert config.signal_capture_tier == expected.wire_name


def test_tier_without_a_directory_is_rejected():
    with pytest.raises(ValueError, match="nowhere to write"):
        ObservabilityConfig(signal_capture_tier="residual_raw")


def test_attention_weight_tier_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="fused attention"):
        ObservabilityConfig(
            signal_capture_tier="full_raw_attn", signal_capture_dir=str(tmp_path)
        )


def test_unknown_tier_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown capture tier"):
        ObservabilityConfig(
            signal_capture_tier="everything", signal_capture_dir=str(tmp_path)
        )


@pytest.mark.parametrize("field", ["layer_step", "token_step"])
def test_steps_must_be_positive(tmp_path, field):
    with pytest.raises(ValueError):
        ObservabilityConfig(
            signal_capture_tier="residual_raw",
            signal_capture_dir=str(tmp_path),
            **{f"signal_capture_{field}": 0},
        )


def test_hook_tiers_force_eager(tmp_path):
    """Hooks never re-run on CUDA graph replay, so capture must disable graphs."""
    args = EngineArgs(
        model="facebook/opt-125m",
        signal_capture_tier="residual_raw",
        signal_capture_dir=str(tmp_path),
    )
    config = args.create_engine_config()
    assert config.model_config.enforce_eager


def test_logit_tier_does_not_force_eager(tmp_path):
    """T1 reads the sampler's logits, so it needs no hooks and no eager fallback."""
    args = EngineArgs(
        model="facebook/opt-125m",
        signal_capture_tier="logit",
        signal_capture_dir=str(tmp_path),
    )
    config = args.create_engine_config()
    assert not config.model_config.enforce_eager
