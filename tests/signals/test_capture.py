# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signal capture: tap discovery, deposit shape/size, and reduction agreement."""

import json
import struct

import pytest
import torch
import torch.nn as nn

from vllm.signals.capturer import SignalCapturer, find_decoder_layers
from vllm.signals.config import SignalCaptureConfig
from vllm.signals.tiers import STATS_WIDTH, Tier

HIDDEN = 128
INTERMEDIATE = 256
N_HEAD = 8
N_KV = 2
HEAD_DIM = 16
N_LAYERS = 4


class FakeRotary(nn.Module):
    def forward(self, q, k):
        return q, k


class FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(HIDDEN, (N_HEAD + 2 * N_KV) * HEAD_DIM)
        self.o_proj = nn.Linear(N_HEAD * HEAD_DIM, HIDDEN)
        self.rotary_emb = FakeRotary()

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(
            [N_HEAD * HEAD_DIM, N_KV * HEAD_DIM, N_KV * HEAD_DIM], dim=-1
        )
        q, k = self.rotary_emb(q, k)
        return self.o_proj(q)


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(HIDDEN, 2 * INTERMEDIATE)
        self.act_fn = nn.SiLU()
        self.down_proj = nn.Linear(2 * INTERMEDIATE, HIDDEN)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class FakeNorm(nn.Module):
    """Mirrors vLLM RMSNorm's fused add+norm signature."""

    def forward(self, x, residual=None):
        if residual is None:
            return x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        residual = residual + x
        return residual / (residual.norm(dim=-1, keepdim=True) + 1e-6), residual


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttn()
        self.mlp = FakeMLP()
        self.input_layernorm = FakeNorm()
        self.post_attention_layernorm = FakeNorm()

    def forward(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class FakeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer() for _ in range(N_LAYERS))

    def forward(self, x):
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        return x + residual


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInner()

    def forward(self, x):
        return self.model(x)


def read_deposit(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header.pop("__metadata__", {}), header


def run_steps(capturer, model, req_ids, n_steps, num_tokens=None):
    """Drive n decode steps with one sampled row per request."""
    num_tokens = num_tokens or len(req_ids)
    index = torch.arange(len(req_ids), dtype=torch.long)
    for _ in range(n_steps):
        capturer.begin_step(list(req_ids), index, num_tokens)
        model(torch.randn(num_tokens, HIDDEN))
        capturer.end_step()


def test_finds_decoder_stack():
    layers = find_decoder_layers(FakeModel())
    assert len(layers) == N_LAYERS
    assert all(isinstance(layer, FakeLayer) for layer in layers)


def test_residual_raw_deposit_is_one_vector_per_token(tmp_path):
    config = SignalCaptureConfig(tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path))
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)

    run_steps(capturer, model, ["req-a"], n_steps=5)
    capturer.finish_requests(["req-a"])
    capturer.shutdown()

    path = tmp_path / "req-a.safetensors"
    assert path.exists()
    metadata, header = read_deposit(path)

    assert metadata["format"] == "sigcap-v1"
    assert metadata["tier"] == "residual_raw"
    assert metadata["num_tokens"] == "5"
    assert metadata["truncated"] == "false"

    # Default layer selector is "last": one row per token, from one layer.
    assert header["residual"]["shape"] == [5, HIDDEN]
    assert header["residual.index"]["shape"] == [5, 2]
    assert set(header) == {"residual", "residual.index"}


def test_residual_raw_all_layers(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)
    run_steps(capturer, model, ["r0", "r1"], n_steps=3)
    capturer.finish_requests(["r0", "r1"])
    capturer.shutdown()

    _, header = read_deposit(tmp_path / "r0.safetensors")
    # 3 tokens x 4 layers.
    assert header["residual"]["shape"] == [3 * N_LAYERS, HIDDEN]

    # Index tensor carries the (token, layer) coordinate of every row.
    from safetensors.torch import load_file

    index = load_file(tmp_path / "r0.safetensors")["residual.index"]
    assert sorted(index[:, 0].tolist()) == sorted([t for t in range(3)] * N_LAYERS)
    assert sorted(set(index[:, 1].tolist())) == list(range(N_LAYERS))


def test_native_dtype_halves_the_deposit(tmp_path):
    sizes = {}
    for dtype in ("native", "float32"):
        out = tmp_path / dtype
        config = SignalCaptureConfig(
            tier=Tier.RESIDUAL_RAW, output_dir=str(out), dtype=dtype
        )
        model = FakeModel().to(torch.bfloat16)
        capturer = SignalCapturer(config, model, model_name="fake")
        for _ in range(4):
            capturer.begin_step(["r"], torch.arange(1), 1)
            model(torch.randn(1, HIDDEN, dtype=torch.bfloat16))
            capturer.end_step()
        capturer.finish_requests(["r"])
        capturer.shutdown()
        _, header = read_deposit(out / "r.safetensors")
        start, end = header["residual"]["data_offsets"]
        sizes[dtype] = end - start

    assert sizes["native"] == 4 * HIDDEN * 2  # bf16, as the activations exist
    assert sizes["float32"] == 4 * HIDDEN * 4


def test_full_raw_captures_every_forward_signal(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.FULL_RAW, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)
    run_steps(capturer, model, ["r"], n_steps=2)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(tmp_path / "r.safetensors")
    rows = 2 * N_LAYERS
    assert header["residual"]["shape"] == [rows, HIDDEN]
    assert header["attn_norm"]["shape"] == [rows, HIDDEN]
    assert header["ffn_norm"]["shape"] == [rows, HIDDEN]
    assert header["gate"]["shape"] == [rows, 2 * INTERMEDIATE]
    assert header["qcur"]["shape"] == [rows, N_HEAD * HEAD_DIM]
    assert header["kcur"]["shape"] == [rows, N_KV * HEAD_DIM]


def test_layer_stats_tier_writes_scalar_tuples(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.LAYER_STATS, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)
    run_steps(capturer, model, ["r"], n_steps=3)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(tmp_path / "r.safetensors")
    rows = 3 * N_LAYERS
    assert header["stats.residual"]["shape"] == [rows, STATS_WIDTH]
    assert header["stats.gate"]["shape"] == [rows, STATS_WIDTH]
    # Stats are ~500x smaller than the raw vectors they summarize.
    assert all(not k.startswith("residual") for k in header)


def test_token_step_thins_the_capture(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), token_step=3
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["r"], n_steps=9)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(tmp_path / "r.safetensors")
    assert header["residual"]["shape"] == [3, HIDDEN]  # tokens 0, 3, 6


def test_max_bytes_truncates_rather_than_growing(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), max_bytes=HIDDEN * 4 * 3
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["r"], n_steps=20)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    metadata, header = read_deposit(tmp_path / "r.safetensors")
    assert metadata["truncated"] == "true"
    assert header["residual"]["shape"][0] == 3


def test_two_requests_get_separate_deposits(tmp_path):
    config = SignalCaptureConfig(tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path))
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["alpha", "beta"], n_steps=4)
    capturer.finish_requests(["alpha"])
    run_steps(capturer, model, ["beta"], n_steps=2)
    capturer.finish_requests(["beta"])
    capturer.shutdown()

    _, alpha = read_deposit(tmp_path / "alpha.safetensors")
    _, beta = read_deposit(tmp_path / "beta.safetensors")
    assert alpha["residual"]["shape"] == [4, HIDDEN]
    assert beta["residual"]["shape"] == [6, HIDDEN]


def test_hooks_are_inert_outside_a_step(tmp_path):
    """A dummy/profile run between steps must not land in any deposit."""
    config = SignalCaptureConfig(tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path))
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    model(torch.randn(16, HIDDEN))  # profile run, hooks disarmed
    run_steps(capturer, model, ["r"], n_steps=2)
    model(torch.randn(16, HIDDEN))  # another one
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(tmp_path / "r.safetensors")
    assert header["residual"]["shape"] == [2, HIDDEN]


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("last", (N_LAYERS - 1,)),
        ("all", tuple(range(N_LAYERS))),
        ("0,2", (0, 2)),
        ("-1", (N_LAYERS - 1,)),
    ],
)
def test_layer_selector(spec, expected):
    config = SignalCaptureConfig(tier=Tier.RESIDUAL_RAW, layers=spec)
    assert config.resolve_layers(N_LAYERS) == expected
