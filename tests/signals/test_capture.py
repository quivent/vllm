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


def deposit_path(directory, request_id):
    """Deposits are named `<timestamp>-<request_id>.safetensors`."""
    matches = sorted(directory.glob(f"*-{request_id}.safetensors"))
    assert matches, f"no deposit for {request_id} in {list(directory.iterdir())}"
    return matches[-1]


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
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)

    run_steps(capturer, model, ["req-a"], n_steps=5)
    capturer.finish_requests(["req-a"])
    capturer.shutdown()

    path = deposit_path(tmp_path, "req-a")
    metadata, header = read_deposit(path)

    assert metadata["format"] == "sigcap-v1"
    assert metadata["tier"] == "residual_raw"
    assert metadata["num_captured_positions"] == "5"
    assert metadata["truncated"] == "false"

    # Default layer selector is "last": one row per token, from one layer.
    assert header["residual"]["shape"] == [5, HIDDEN]
    assert header["residual.index"]["shape"] == [5, 2]
    assert set(header) == {"residual", "residual.index"}


def test_residual_raw_all_layers(tmp_path):
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)
    run_steps(capturer, model, ["r0", "r1"], n_steps=3)
    capturer.finish_requests(["r0", "r1"])
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r0"))
    # 3 tokens x 4 layers.
    assert header["residual"]["shape"] == [3 * N_LAYERS, HIDDEN]

    # Index tensor carries the (token, layer) coordinate of every row.
    from safetensors.torch import load_file

    index = load_file(deposit_path(tmp_path, "r0"))["residual.index"]
    assert sorted(index[:, 0].tolist()) == sorted([t for t in range(3)] * N_LAYERS)
    assert sorted(set(index[:, 1].tolist())) == list(range(N_LAYERS))


def test_native_dtype_halves_the_deposit(tmp_path):
    sizes = {}
    for dtype in ("native", "float32"):
        out = tmp_path / dtype
        config = SignalCaptureConfig(
            tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(out), dtype=dtype
        )
        model = FakeModel().to(torch.bfloat16)
        capturer = SignalCapturer(config, model, model_name="fake")
        for _ in range(4):
            capturer.begin_step(["r"], torch.arange(1), 1)
            model(torch.randn(1, HIDDEN, dtype=torch.bfloat16))
            capturer.end_step()
        capturer.finish_requests(["r"])
        capturer.shutdown()
        _, header = read_deposit(deposit_path(out, "r"))
        start, end = header["residual"]["data_offsets"]
        sizes[dtype] = end - start

    assert sizes["native"] == 4 * HIDDEN * 2  # bf16, as the activations exist
    assert sizes["float32"] == 4 * HIDDEN * 4


def test_full_raw_captures_every_forward_signal(tmp_path):
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.FULL_RAW, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)
    run_steps(capturer, model, ["r"], n_steps=2)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    rows = 2 * N_LAYERS
    assert header["residual"]["shape"] == [rows, HIDDEN]
    assert header["attn_norm"]["shape"] == [rows, HIDDEN]
    assert header["ffn_norm"]["shape"] == [rows, HIDDEN]
    assert header["gate"]["shape"] == [rows, 2 * INTERMEDIATE]
    assert header["qcur"]["shape"] == [rows, N_HEAD * HEAD_DIM]
    assert header["kcur"]["shape"] == [rows, N_KV * HEAD_DIM]


def test_layer_stats_tier_writes_scalar_tuples(tmp_path):
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.LAYER_STATS, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)
    run_steps(capturer, model, ["r"], n_steps=3)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    rows = 3 * N_LAYERS
    assert header["stats.residual"]["shape"] == [rows, STATS_WIDTH]
    assert header["stats.gate"]["shape"] == [rows, STATS_WIDTH]
    # Stats are ~500x smaller than the raw vectors they summarize.
    assert all(not k.startswith("residual") for k in header)


def test_token_step_thins_the_capture(tmp_path):
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), token_step=3
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["r"], n_steps=9)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    assert header["residual"]["shape"] == [3, HIDDEN]  # tokens 0, 3, 6


def test_max_bytes_truncates_rather_than_growing(tmp_path):
    config = SignalCaptureConfig(
        tokens="all",
        tier=Tier.RESIDUAL_RAW,
        output_dir=str(tmp_path),
        max_bytes=HIDDEN * 4 * 3,
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["r"], n_steps=20)
    capturer.finish_requests(["r"])
    capturer.shutdown()

    metadata, header = read_deposit(deposit_path(tmp_path, "r"))
    assert metadata["truncated"] == "true"
    assert header["residual"]["shape"][0] == 3


def test_two_requests_get_separate_deposits(tmp_path):
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["alpha", "beta"], n_steps=4)
    capturer.finish_requests(["alpha"])
    run_steps(capturer, model, ["beta"], n_steps=2)
    capturer.finish_requests(["beta"])
    capturer.shutdown()

    _, alpha = read_deposit(deposit_path(tmp_path, "alpha"))
    _, beta = read_deposit(deposit_path(tmp_path, "beta"))
    assert alpha["residual"]["shape"] == [4, HIDDEN]
    assert beta["residual"]["shape"] == [6, HIDDEN]


def test_hooks_are_inert_outside_a_step(tmp_path):
    """A dummy/profile run between steps must not land in any deposit."""
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    model(torch.randn(16, HIDDEN))  # profile run, hooks disarmed
    run_steps(capturer, model, ["r"], n_steps=2)
    model(torch.randn(16, HIDDEN))  # another one
    capturer.finish_requests(["r"])
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
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
    config = SignalCaptureConfig(tokens="all", tier=Tier.RESIDUAL_RAW, layers=spec)
    assert config.resolve_layers(N_LAYERS) == expected


def test_residual_matches_vllms_own_aux_hidden_state():
    """The residual tap must equal what EAGLE-3's aux hidden state hook records.

    Both are meant to be the residual stream out of a layer; pinning them
    together keeps the deposit honest if either definition ever moves.
    """
    from vllm.model_executor.models.interfaces import EagleModelMixin

    mixin = EagleModelMixin()
    mixin._set_aux_hidden_state_layers((1,))
    hidden, residual = torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)
    (expected,) = mixin._maybe_add_hidden_state([], 1, hidden, residual)

    config = SignalCaptureConfig(tokens="all", tier=Tier.RESIDUAL_RAW, output_dir="")
    capturer = SignalCapturer(config, FakeModel())
    captured = {}
    capturer._stage = lambda signal, idx, tensor: captured.setdefault(signal, tensor)
    capturer._active = True
    capturer._make_residual_hook(0)(None, None, (hidden, residual))

    torch.testing.assert_close(captured["residual"], expected)


def test_residual_tap_handles_a_layer_without_a_fused_residual():
    """Models whose layers return a bare tensor still record the stream."""
    config = SignalCaptureConfig(tokens="all", tier=Tier.RESIDUAL_RAW, output_dir="")
    capturer = SignalCapturer(config, FakeModel())
    captured = {}
    capturer._stage = lambda signal, idx, tensor: captured.setdefault(signal, tensor)
    capturer._active = True

    stream = torch.randn(4, HIDDEN)
    capturer._make_residual_hook(0)(None, None, stream)
    torch.testing.assert_close(captured["residual"], stream)


def test_decoder_stack_discovery_ignores_a_drafter_stack():
    """A speculator's layers must not be mistaken for the target's."""

    class WithDrafter(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeInner()
            self.drafter = nn.Module()
            self.drafter.layers = nn.ModuleList(FakeLayer() for _ in range(2))

    layers = find_decoder_layers(WithDrafter())
    assert len(layers) == N_LAYERS


def test_spec_decode_rows_map_to_the_right_requests():
    """One request owns as many rows as it had draft positions verified."""
    from vllm.signals.capturer import counts_from_cumulative, rows_to_request_ids

    assert rows_to_request_ids(["a", "b"], [3, 1]) == ["a", "a", "a", "b"]
    assert rows_to_request_ids(["a", "b"], None) == ["a", "b"]
    # A mismatched count array is ignored rather than misattributing rows.
    assert rows_to_request_ids(["a", "b"], [3]) == ["a", "b"]
    assert counts_from_cumulative([0, 3, 4], 2) == [3, 1]
    assert counts_from_cumulative(None, 2) is None


def test_mismatched_row_mapping_skips_the_step(tmp_path, caplog):
    """Rather than misattribute activations, capture nothing for that step."""
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    capturer.begin_step(["a", "b"], torch.arange(3), 3)
    assert not capturer._active
    model(torch.randn(3, HIDDEN))
    capturer.end_step()
    capturer.finish_requests(["a", "b"])
    capturer.shutdown()

    assert not list(tmp_path.glob("*.safetensors"))


def test_shutdown_flushes_requests_that_never_got_a_final_step(tmp_path):
    """A request finishing on the last engine step still gets its deposit.

    `finish_requests` is driven by the scheduler's finished-id set, which only
    arrives on a *subsequent* step. Requests that finish last never see one, so
    shutdown is their only flush.
    """
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    run_steps(capturer, model, ["never-reported"], n_steps=3)
    assert not list(tmp_path.glob("*.safetensors"))

    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "never-reported"))
    assert header["residual"]["shape"] == [3, HIDDEN]


def test_shutdown_is_idempotent(tmp_path):
    """The exit backstop may fire after an explicit shutdown; that must be safe."""
    config = SignalCaptureConfig(
        tokens="all", tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["r"], n_steps=2)

    capturer.shutdown()
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    assert header["residual"]["shape"] == [2, HIDDEN]


# ── per-turn reduction: one residual for the whole generation ────────────────


def _turn_deposit(tmp_path, tokens, n_steps=5, layers="last"):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW,
        output_dir=str(tmp_path),
        tokens=tokens,
        layers=layers,
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    run_steps(capturer, model, ["turn"], n_steps=n_steps)
    capturer.shutdown()
    return read_deposit(deposit_path(tmp_path, "turn"))


@pytest.mark.parametrize("tokens", ["last", "first", "mean"])
def test_a_turn_is_one_vector_regardless_of_response_length(tmp_path, tokens):
    metadata, header = _turn_deposit(tmp_path, tokens, n_steps=40)
    assert header["residual"]["shape"] == [1, HIDDEN]
    assert metadata["token_reduce"] == tokens
    assert metadata["num_captured_positions"] == "40"


def test_deposit_size_is_flat_in_response_length(tmp_path):
    """The point of a per-turn deposit: 10 tokens and 200 tokens cost the same."""
    short = _turn_deposit(tmp_path / "a", "last", n_steps=10)[1]
    long = _turn_deposit(tmp_path / "b", "last", n_steps=200)[1]
    assert short["residual"]["shape"] == long["residual"]["shape"] == [1, HIDDEN]


def test_last_keeps_the_final_token(tmp_path):
    metadata, header = _turn_deposit(tmp_path, "last", n_steps=6)
    from safetensors.torch import load_file

    index = load_file(deposit_path(tmp_path, "turn"))["residual.index"]
    assert index[0, 0].item() == 5  # tokens 0..5, the last one


def test_first_keeps_the_opening_token(tmp_path):
    _turn_deposit(tmp_path, "first", n_steps=6)
    from safetensors.torch import load_file

    index = load_file(deposit_path(tmp_path, "turn"))["residual.index"]
    assert index[0, 0].item() == 0


def test_mean_averages_the_turn(tmp_path):
    """The stored vector is the mean of what the hook saw, not a sampled token."""
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), tokens="mean"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    seen = []
    original = capturer._stage

    def record(signal, layer_idx, tensor):
        original(signal, layer_idx, tensor)
        if signal == "residual" and capturer._active:
            seen.append(capturer._staged[signal][layer_idx].clone())

    capturer._stage = record
    run_steps(capturer, model, ["turn"], n_steps=4)
    capturer.shutdown()

    from safetensors.torch import load_file

    stored = load_file(deposit_path(tmp_path, "turn"))["residual"].float()
    torch.testing.assert_close(stored, torch.cat(seen).mean(0, keepdim=True))


def test_per_turn_with_all_layers_is_one_vector_per_layer(tmp_path):
    _, header = _turn_deposit(tmp_path, "last", n_steps=12, layers="all")
    assert header["residual"]["shape"] == [N_LAYERS, HIDDEN]


# ── runtime control: change what is recorded without a restart ───────────────


def _runtime_capturer(tmp_path, tier=Tier.OFF, max_tier=Tier.FULL_RAW):
    config = SignalCaptureConfig(
        tier=tier, max_tier=max_tier, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    return SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM), model


def test_hooks_install_for_the_ceiling_not_the_active_tier(tmp_path):
    """Launching at `off` still installs hooks, so capture can be switched on."""
    capturer, _ = _runtime_capturer(tmp_path, tier=Tier.OFF, max_tier=Tier.FULL_RAW)
    assert capturer.enabled
    assert capturer._handles, "no hooks installed"
    assert capturer.status()["recording"] is False


def test_off_tier_records_nothing_then_switching_on_records(tmp_path):
    capturer, model = _runtime_capturer(tmp_path, tier=Tier.OFF)

    run_steps(capturer, model, ["r"], n_steps=3)
    assert capturer._deposits == {}, "recorded while tier was off"

    capturer.set_runtime(tier="residual_raw", tokens="all")
    run_steps(capturer, model, ["r"], n_steps=4)
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    assert header["residual"]["shape"] == [4 * N_LAYERS, HIDDEN]


def test_switching_off_again_stops_recording(tmp_path):
    capturer, model = _runtime_capturer(tmp_path, tier=Tier.RESIDUAL_RAW)
    capturer.set_runtime(tokens="all")

    run_steps(capturer, model, ["r"], n_steps=2)
    capturer.set_runtime(tier="off")
    run_steps(capturer, model, ["r"], n_steps=10)
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    assert header["residual"]["shape"] == [2 * N_LAYERS, HIDDEN]


def test_tier_above_the_ceiling_is_refused(tmp_path):
    capturer, _ = _runtime_capturer(tmp_path, max_tier=Tier.RESIDUAL_RAW)
    with pytest.raises(ValueError, match="above this process's ceiling"):
        capturer.set_runtime(tier="full_raw")
    assert capturer.status()["tier"] == "off"


def test_runtime_tier_change_swaps_which_signals_are_recorded(tmp_path):
    capturer, model = _runtime_capturer(tmp_path, tier=Tier.RESIDUAL_RAW)
    capturer.set_runtime(tokens="all")
    run_steps(capturer, model, ["r"], n_steps=1)

    capturer.set_runtime(tier="full_raw")
    run_steps(capturer, model, ["r"], n_steps=1)
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    # residual spans both steps; the rest only appear after the tier went up.
    assert header["residual"]["shape"][0] == 2 * N_LAYERS
    assert header["gate"]["shape"][0] == 1 * N_LAYERS
    assert header["qcur"]["shape"][0] == 1 * N_LAYERS


def test_bad_runtime_values_are_refused(tmp_path):
    capturer, _ = _runtime_capturer(tmp_path)
    with pytest.raises(ValueError, match="Unknown capture tier"):
        capturer.set_runtime(tier="turbo")
    with pytest.raises(ValueError, match="unknown token reduction"):
        capturer.set_runtime(tokens="median")


def test_status_reports_the_live_configuration(tmp_path):
    capturer, _ = _runtime_capturer(tmp_path, tier=Tier.LOGIT, max_tier=Tier.FULL_RAW)
    status = capturer.status()
    assert status["tier"] == "logit"
    assert status["max_tier"] == "full_raw"
    assert status["num_layers"] == N_LAYERS
    assert status["layers"] == list(range(N_LAYERS))
    assert status["hooked_layers"] == list(range(N_LAYERS))


def test_token_metrics_follow_the_same_reduction(tmp_path):
    """A per-turn deposit must be fixed size, so logit rows reduce too."""
    sizes = {}
    for n_steps in (5, 100):
        out = tmp_path / str(n_steps)
        config = SignalCaptureConfig(
            tier=Tier.RESIDUAL_RAW, output_dir=str(out), tokens="last"
        )
        model = FakeModel()
        capturer = SignalCapturer(config, model, model_name="fake")
        for _ in range(n_steps):
            capturer.begin_step(["r"], torch.arange(1), 1)
            model(torch.randn(1, HIDDEN))
            capturer.end_step(logits=torch.randn(1, 512))
        capturer.shutdown()
        _, header = read_deposit(deposit_path(out, "r"))
        assert header["logit"]["shape"] == [1, 3]
        sizes[n_steps] = deposit_path(out, "r").stat().st_size

    assert sizes[5] == sizes[100], "deposit size still grows with turn length"


def test_all_reduction_keeps_every_token_metric(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), tokens="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")
    for _ in range(7):
        capturer.begin_step(["r"], torch.arange(1), 1)
        model(torch.randn(1, HIDDEN))
        capturer.end_step(logits=torch.randn(1, 512))
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    assert header["logit"]["shape"] == [7, 3]


def test_startup_warmup_requests_are_not_recorded(tmp_path):
    """vLLM drives synthetic warmup requests through the real execute path."""
    config = SignalCaptureConfig(tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path))
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    run_steps(capturer, model, ["_warmup_0_", "_warmup_1_"], n_steps=3)
    run_steps(capturer, model, ["_warmup_2_", "real-req"], n_steps=2)
    capturer.shutdown()

    written = sorted(p.name.split("-", 1)[1] for p in tmp_path.glob("*.safetensors"))
    assert written == ["real-req.safetensors"]


def test_every_turn_is_kept_as_its_own_deposit(tmp_path):
    """History, not a rolling file: N turns leave N deposits."""
    config = SignalCaptureConfig(tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path))
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    for turn in range(5):
        run_steps(capturer, model, [f"turn-{turn}"], n_steps=3)
        capturer.finish_requests([f"turn-{turn}"])
    capturer.shutdown()

    deposits = sorted(tmp_path.glob("*.safetensors"))
    assert len(deposits) == 5
    # Names sort chronologically, and each carries its own request id.
    assert [p.name.split("-", 1)[1] for p in deposits] == [
        f"turn-{i}.safetensors" for i in range(5)
    ]


def test_a_reused_request_id_does_not_clobber_an_earlier_turn(tmp_path):
    config = SignalCaptureConfig(tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path))
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake")

    for _ in range(3):
        run_steps(capturer, model, ["same-id"], n_steps=2)
        capturer.finish_requests(["same-id"])
    capturer.shutdown()

    assert len(list(tmp_path.glob("*.safetensors"))) == 3


def test_q_and_k_come_off_the_attention_call(tmp_path):
    """Post-RoPE q/k are read from the Attention call, not from rotary_emb.

    Architectures that fuse RoPE into a kernel never invoke rotary_emb, but
    every one of them still calls attention with the finished q, k, v.
    """

    class FusedRoPEAttn(nn.Module):
        """No rotary_emb module at all -- RoPE folded into the projection."""

        def __init__(self):
            super().__init__()
            self.qkv_proj = nn.Linear(HIDDEN, (N_HEAD + 2 * N_KV) * HEAD_DIM)
            self.attn = nn.Module()
            self.attn.forward = lambda q, k, v: q
            self.o_proj = nn.Linear(N_HEAD * HEAD_DIM, HIDDEN)

        def forward(self, x):
            q, k, v = self.qkv_proj(x).split(
                [N_HEAD * HEAD_DIM, N_KV * HEAD_DIM, N_KV * HEAD_DIM], dim=-1
            )
            return self.o_proj(self.attn(q, k, v))

    class FusedLayer(FakeLayer):
        def __init__(self):
            super().__init__()
            self.self_attn = FusedRoPEAttn()

    class FusedModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.model.layers = nn.ModuleList(FusedLayer() for _ in range(N_LAYERS))

    config = SignalCaptureConfig(
        tier=Tier.FULL_RAW, output_dir=str(tmp_path), layers="all", tokens="all"
    )
    model = FusedModel()
    capturer = SignalCapturer(config, model, model_name="fake", head_dim=HEAD_DIM)
    run_steps(capturer, model, ["r"], n_steps=2)
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    assert header["qcur"]["shape"] == [2 * N_LAYERS, N_HEAD * HEAD_DIM]
    assert header["kcur"]["shape"] == [2 * N_LAYERS, N_KV * HEAD_DIM]


# ── injection: put a captured residual back into the forward pass ────────────


def _captured_vector(tmp_path, layer=N_LAYERS - 1):
    """Capture one turn, and return its deposit path."""
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), layers=str(layer)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", hidden_size=HIDDEN)
    run_steps(capturer, model, ["donor"], n_steps=2)
    capturer.shutdown()
    return deposit_path(tmp_path, "donor")


def _injectable(tmp_path):
    config = SignalCaptureConfig(tier=Tier.OFF, output_dir=str(tmp_path))
    model = FakeModel()
    return (
        SignalCapturer(config, model, model_name="fake", hidden_size=HIDDEN),
        model,
    )


def test_injection_round_trips_a_captured_residual(tmp_path):
    source = _captured_vector(tmp_path)
    capturer, _ = _injectable(tmp_path)
    status = capturer.load_injection(str(source), alpha=2.0, mode="add")

    assert status["enabled"]
    assert status["layer"] == N_LAYERS - 1  # taken from the deposit
    assert status["alpha"] == 2.0
    assert status["hidden_size"] == HIDDEN


def test_injection_changes_the_stream_it_targets(tmp_path):
    source = _captured_vector(tmp_path, layer=1)
    capturer, model = _injectable(tmp_path)

    x = torch.randn(1, HIDDEN)
    torch.manual_seed(7)
    baseline = model(x).clone()

    capturer.load_injection(str(source), alpha=5.0, mode="add", positions="all")
    capturer.begin_step(["r"], torch.arange(1), 1)
    injected = model(x).clone()
    capturer.end_step()

    assert not torch.allclose(baseline, injected), "injection had no effect"
    assert capturer.injector.status()["applied"] == 1


def test_injection_is_inert_until_a_step_is_armed(tmp_path):
    source = _captured_vector(tmp_path, layer=1)
    capturer, model = _injectable(tmp_path)
    capturer.load_injection(str(source), alpha=5.0, positions="all")

    x = torch.randn(1, HIDDEN)
    before = model(x).clone()
    after = model(x).clone()
    torch.testing.assert_close(before, after)
    assert capturer.injector.status()["applied"] == 0


def test_first_seeds_each_request_once(tmp_path):
    source = _captured_vector(tmp_path, layer=1)
    capturer, model = _injectable(tmp_path)
    capturer.load_injection(str(source), positions="first")

    for _ in range(5):
        capturer.begin_step(["r"], torch.arange(1), 1)
        model(torch.randn(1, HIDDEN))
        capturer.end_step()

    assert capturer.injector.status()["applied"] == 1

    # A new request gets its own seed.
    capturer.begin_step(["other"], torch.arange(1), 1)
    model(torch.randn(1, HIDDEN))
    capturer.end_step()
    assert capturer.injector.status()["applied"] == 2


def test_all_positions_injects_every_step(tmp_path):
    source = _captured_vector(tmp_path, layer=1)
    capturer, model = _injectable(tmp_path)
    capturer.load_injection(str(source), positions="all")

    for _ in range(4):
        capturer.begin_step(["r"], torch.arange(1), 1)
        model(torch.randn(1, HIDDEN))
        capturer.end_step()

    assert capturer.injector.status()["applied"] == 4


def test_clearing_injection_removes_the_hook(tmp_path):
    source = _captured_vector(tmp_path, layer=1)
    capturer, model = _injectable(tmp_path)
    capturer.load_injection(str(source), alpha=5.0, positions="all")
    assert capturer.set_injection(None)["enabled"] is False

    x = torch.randn(1, HIDDEN)
    capturer.begin_step(["r"], torch.arange(1), 1)
    injected = model(x).clone()
    capturer.end_step()
    torch.testing.assert_close(model(x), injected)


def test_a_mismatched_vector_is_refused(tmp_path):
    from vllm.signals.inject import InjectionSpec

    capturer, _ = _injectable(tmp_path)
    with pytest.raises(ValueError, match="different model"):
        capturer.set_injection(InjectionSpec(vector=torch.randn(HIDDEN + 1), layer=0))
    with pytest.raises(ValueError, match="outside this model"):
        capturer.set_injection(InjectionSpec(vector=torch.randn(HIDDEN), layer=999))
    with pytest.raises(ValueError, match="unknown injection mode"):
        capturer.set_injection(
            InjectionSpec(vector=torch.randn(HIDDEN), layer=0, mode="blend")
        )


def test_replace_mode_sets_the_stream_to_the_captured_state(tmp_path):
    from safetensors.torch import load_file

    source = _captured_vector(tmp_path, layer=1)
    vector = load_file(source)["residual"][-1].float()

    capturer, model = _injectable(tmp_path)
    capturer.load_injection(str(source), mode="replace", positions="all")

    captured = {}
    layer = capturer.layers[1]
    layer.register_forward_hook(
        lambda m, a, out: captured.__setitem__("residual", out[1].clone())
    )
    capturer.begin_step(["r"], torch.arange(1), 1)
    model(torch.randn(1, HIDDEN))
    capturer.end_step()

    torch.testing.assert_close(captured["residual"][0].float(), vector)


# ── building an injectable vector from several turns ─────────────────────────


def _turns(tmp_path, n, layer=1):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), layers=str(layer)
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", hidden_size=HIDDEN)
    for i in range(n):
        run_steps(capturer, model, [f"t{i}"], n_steps=2)
        capturer.finish_requests([f"t{i}"])
    capturer.shutdown()
    return sorted(str(p) for p in tmp_path.glob("*.safetensors"))


def test_mean_averages_the_turns_it_is_given(tmp_path):
    from safetensors.torch import load_file

    from vllm.signals.build import collect, main

    paths = _turns(tmp_path, 4)
    out = str(tmp_path / "mean.safetensors")
    assert main(["mean", *paths, "-o", out]) == 0

    rows, layer = collect(paths, "residual", None)
    torch.testing.assert_close(
        load_file(out)["residual"][0], rows.mean(0), rtol=1e-5, atol=1e-5
    )
    assert layer == 1


def test_diff_is_the_direction_between_two_sets(tmp_path):
    from safetensors.torch import load_file

    from vllm.signals.build import collect, main

    a = _turns(tmp_path / "a", 3)
    b = _turns(tmp_path / "b", 3)
    out = str(tmp_path / "diff.safetensors")
    assert main(["diff", "--a", *a, "--b", *b, "-o", out]) == 0

    a_rows, _ = collect(a, "residual", None)
    b_rows, _ = collect(b, "residual", None)
    torch.testing.assert_close(
        load_file(out)["residual"][0],
        a_rows.mean(0) - b_rows.mean(0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_normalize_makes_alpha_the_whole_magnitude(tmp_path):
    from safetensors.torch import load_file

    from vllm.signals.build import main

    paths = _turns(tmp_path, 3)
    out = str(tmp_path / "unit.safetensors")
    main(["mean", *paths, "-o", out, "--normalize"])
    assert abs(float(load_file(out)["residual"][0].norm()) - 1.0) < 1e-5


def test_mixed_layers_are_refused_without_an_explicit_choice(tmp_path):
    from vllm.signals.build import collect

    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), layers="all"
    )
    model = FakeModel()
    capturer = SignalCapturer(config, model, model_name="fake", hidden_size=HIDDEN)
    run_steps(capturer, model, ["t"], n_steps=1)
    capturer.shutdown()
    paths = [str(p) for p in tmp_path.glob("*.safetensors")]

    with pytest.raises(ValueError, match="mix layers"):
        collect(paths, "residual", None)
    rows, layer = collect(paths, "residual", 2)
    assert rows.shape == (1, HIDDEN) and layer == 2


def test_a_built_vector_is_injectable(tmp_path):
    """Round trip: capture turns -> build a mean -> inject it."""
    from vllm.signals.build import main

    paths = _turns(tmp_path, 3)
    built = str(tmp_path / "built.safetensors")
    main(["mean", *paths, "-o", built])

    capturer, model = _injectable(tmp_path)
    status = capturer.load_injection(built, alpha=1.0, positions="all")
    assert status["enabled"] and status["layer"] == 1

    capturer.begin_step(["r"], torch.arange(1), 1)
    model(torch.randn(1, HIDDEN))
    capturer.end_step()
    assert capturer.injector.status()["applied"] == 1


# ── graph backend: residual as a model output, no hooks, no eager ────────────


class Eagle3Model(FakeModel):
    """A model exposing vLLM's EAGLE-3 interface, as the graph backend needs."""

    supports_eagle3 = True
    has_own_lm_head = False
    has_own_embed_tokens = False

    def __init__(self):
        super().__init__()
        self.aux_layers: tuple[int, ...] = ()

    def set_aux_hidden_state_layers(self, layers):
        self.aux_layers = layers

    def get_eagle3_default_aux_hidden_state_layers(self):
        return (1,)


def _graph_capturer(tmp_path, layers="last"):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW,
        output_dir=str(tmp_path),
        backend="graph",
        layers=layers,
        tokens="all",
    )
    model = Eagle3Model()
    return SignalCapturer(config, model, model_name="fake", hidden_size=HIDDEN), model


def test_graph_backend_installs_no_hooks(tmp_path):
    capturer, _ = _graph_capturer(tmp_path)
    assert capturer.backend == "graph"
    assert capturer._handles == []
    assert not capturer.config.needs_hooks


def test_graph_aux_indices_shift_by_one(tmp_path):
    """`_maybe_add_hidden_state` numbers 0 for embeddings, i+1 after layer i."""
    capturer, _ = _graph_capturer(tmp_path, layers="all")
    assert capturer.graph_aux_layers() == tuple(range(1, N_LAYERS + 1))
    capturer2, _ = _graph_capturer(tmp_path / "b", layers="0,3")
    assert capturer2.graph_aux_layers() == (1, 4)


def test_graph_backend_records_the_models_aux_output(tmp_path):
    capturer, model = _graph_capturer(tmp_path, layers="all")
    capturer.check_graph_backend(model, speculative_uses_aux=False)

    for _ in range(2):
        capturer.begin_step(["r"], torch.arange(1), 1)
        # Stand in for what the model returns alongside its hidden states.
        capturer.observe_aux([torch.randn(1, HIDDEN) for _ in range(N_LAYERS)])
        capturer.end_step()
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    assert header["residual"]["shape"] == [2 * N_LAYERS, HIDDEN]


def test_graph_backend_refuses_signals_it_cannot_reach(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.FULL_RAW, output_dir=str(tmp_path), backend="graph"
    )
    model = Eagle3Model()
    capturer = SignalCapturer(config, model, hidden_size=HIDDEN)
    with pytest.raises(ValueError, match="can only capture the residual"):
        capturer.check_graph_backend(model, speculative_uses_aux=False)


def test_graph_backend_refuses_a_model_without_the_interface(tmp_path):
    config = SignalCaptureConfig(
        tier=Tier.RESIDUAL_RAW, output_dir=str(tmp_path), backend="graph"
    )
    model = FakeModel()  # no EAGLE-3 interface
    capturer = SignalCapturer(config, model, hidden_size=HIDDEN)
    with pytest.raises(ValueError, match="EAGLE-3 interface"):
        capturer.check_graph_backend(model, speculative_uses_aux=False)


def test_graph_backend_reads_a_drafters_layers_for_free(tmp_path):
    """If a DFlash/EAGLE-3 drafter already publishes the layer, just read it."""
    capturer, model = _graph_capturer(tmp_path, layers="2")
    # The drafter publishes decoder layers 1 and 2 -> aux indices 2 and 3.
    owns = capturer.check_graph_backend(
        model, speculative_uses_aux=True, drafter_layers=(2, 3)
    )
    assert owns is False, "capture must not reconfigure the drafter's layers"

    capturer.begin_step(["r"], torch.arange(1), 1)
    capturer.observe_aux([torch.randn(1, HIDDEN), torch.randn(1, HIDDEN)])
    capturer.end_step()
    capturer.shutdown()

    _, header = read_deposit(deposit_path(tmp_path, "r"))
    # Only the requested layer is kept, not everything the drafter publishes.
    assert header["residual"]["shape"] == [1, HIDDEN]


def test_graph_backend_explains_itself_when_the_drafter_lacks_the_layer(tmp_path):
    capturer, model = _graph_capturer(tmp_path, layers="0")
    with pytest.raises(ValueError, match="publishes decoder layers \\[1, 2\\]"):
        capturer.check_graph_backend(
            model, speculative_uses_aux=True, drafter_layers=(2, 3)
        )
