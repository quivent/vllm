# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture-tier policy for inference-signal recording.

Port of the ``sigcap`` tier ladder from the signal-extraction repo
(``capture/src/signal_capture.cpp::Capture::mode_for``). The tier decides, per
named signal, whether to drop it, reduce it to scalars, or store the raw vector.
Volume grows roughly 10x per step:

======  ================  ==================================================
Tier    Name              Records
======  ================  ==================================================
T0      off               nothing
T1      logit             token scalars (entropy / perplexity / confidence)
T2      layer_stats       per-layer scalar reductions of each signal
T3      heads             per-head scalar reductions
T4      residual_raw      full residual (``l_out``) vector per layer
T5      full_raw          + gate / norms / q / k per layer
T6      full_raw_attn     + attention weights (needs an eager attn backend)
======  ================  ==================================================

``layer_step`` / ``token_step`` subsample every Nth layer/token to thin any tier
further.
"""

from enum import Enum, IntEnum

# Known signal names. The width comment is the per-layer element count.
SIG_RESIDUAL = "residual"  # n_embd -- the residual stream, l_out
SIG_ATTN_NORM = "attn_norm"  # n_embd -- input_layernorm output
SIG_FFN_NORM = "ffn_norm"  # n_embd -- post_attention_layernorm output
SIG_GATE = "gate"  # n_ff   -- post-activation FFN gate
SIG_QCUR = "qcur"  # n_head * head_dim
SIG_KCUR = "kcur"  # n_head_kv * head_dim
SIG_ATTN = "attn"  # n_kv * n_head (T6 only)

ALL_SIGNALS = (
    SIG_RESIDUAL,
    SIG_ATTN_NORM,
    SIG_FFN_NORM,
    SIG_GATE,
    SIG_QCUR,
    SIG_KCUR,
    SIG_ATTN,
)

# Field order of the T2/T3 stats tuple. Matches the C++ recorder exactly, so
# deposits from either engine deserialize with the same column meanings.
STATS_FIELDS = (
    "l2_norm",
    "mean_abs",
    "max_abs",
    "std",
    "sparsity",
    "cosine_sim",  # residual only
    "gate_entropy",  # gate only
    "q_k_cosine",  # kcur only
)
STATS_WIDTH = len(STATS_FIELDS)


class Tier(IntEnum):
    OFF = 0
    LOGIT = 1
    LAYER_STATS = 2
    HEADS = 3
    RESIDUAL_RAW = 4
    FULL_RAW = 5
    FULL_RAW_ATTN = 6

    @property
    def wire_name(self) -> str:
        """The name written into deposit metadata (matches ``sigcap::tier_name``)."""
        return _TIER_WIRE_NAMES[self]


_TIER_WIRE_NAMES = {
    Tier.OFF: "off",
    Tier.LOGIT: "logit",
    Tier.LAYER_STATS: "layer_stats",
    Tier.HEADS: "heads",
    Tier.RESIDUAL_RAW: "residual_raw",
    Tier.FULL_RAW: "full_raw",
    Tier.FULL_RAW_ATTN: "full_raw_attn",
}

# CLI spelling -> tier. Accepts the wire names and the bare T-numbers.
TIER_ALIASES = {name: tier for tier, name in _TIER_WIRE_NAMES.items()}
TIER_ALIASES.update({f"t{int(tier)}": tier for tier in Tier})


def tier_from_str(value: str) -> Tier:
    key = value.strip().lower()
    if key not in TIER_ALIASES:
        valid = ", ".join(sorted(TIER_ALIASES))
        raise ValueError(f"Unknown capture tier {value!r}. Expected one of: {valid}")
    return TIER_ALIASES[key]


class Mode(Enum):
    """How a given named signal is handled at the active tier."""

    DROP = "drop"
    STATS = "stats"
    RAW = "raw"


def mode_for(tier: Tier, signal: str) -> Mode:
    """Per-tier policy: what to do with each named signal."""
    if tier == Tier.OFF:
        return Mode.DROP
    if tier == Tier.LOGIT:
        # Logits are handled separately, via observe_logits().
        return Mode.DROP
    if tier in (Tier.LAYER_STATS, Tier.HEADS):
        return Mode.STATS
    if tier == Tier.RESIDUAL_RAW:
        return Mode.RAW if signal == SIG_RESIDUAL else Mode.DROP
    if tier == Tier.FULL_RAW:
        # Everything but attention weights, raw.
        return Mode.DROP if signal == SIG_ATTN else Mode.RAW
    if tier == Tier.FULL_RAW_ATTN:
        return Mode.RAW
    return Mode.DROP


def wants(
    tier: Tier,
    signal: str,
    layer: int,
    token: int,
    layer_step: int = 1,
    token_step: int = 1,
) -> bool:
    """Ask phase: should this (signal, layer, token) be copied off-device at all?

    This is the point of the design -- the engine never pays for a device->host
    transfer that the active tier would discard anyway.
    """
    if tier == Tier.OFF:
        return False
    if token_step > 1 and token % token_step != 0:
        return False
    if layer_step > 1 and layer % layer_step != 0:
        return False
    return mode_for(tier, signal) is not Mode.DROP


def signals_for(tier: Tier) -> tuple[str, ...]:
    """The signal names the given tier records (in any mode)."""
    return tuple(s for s in ALL_SIGNALS if mode_for(tier, s) is not Mode.DROP)
