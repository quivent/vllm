# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-device scalar reductions for the T1-T3 capture tiers.

Torch ports of the reference math in ``signal-extraction/signal_extraction/
compute.py`` and its C++ mirror in ``capture/src/signal_capture.cpp``. Every
function is batched over rows -- one row per (request, token) captured in the
current forward pass -- and returns a tensor the caller concatenates into a
deposit buffer.

The reference implementations accumulate in ``double``; these run in ``float32``,
which is the pragmatic choice on GPU. Agreement with the reference is therefore
to float32 precision, not bit-exact.
"""

import torch

from vllm.signals.tiers import (
    SIG_GATE,
    SIG_KCUR,
    SIG_QCUR,
    SIG_RESIDUAL,
    STATS_WIDTH,
)

# Sparsity thresholds. The gate signal uses a looser threshold than the rest,
# matching SPEC.md 3.2 (fraction |g| < 0.01) vs the generic 1e-6.
_SPARSITY_THRESHOLD = 1e-6
_GATE_SPARSITY_THRESHOLD = 0.01

_GATE_HIST_BINS = 32
_EPS = 1e-12


def shannon_entropy(p: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """H(p) = -sum p ln p over the last dim, ignoring entries below ``eps``."""
    safe = p.clamp_min(eps)
    return -torch.where(p > eps, p * safe.log(), torch.zeros_like(p)).sum(-1)


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine over the last dim, 0 where either row is degenerate."""
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    dot = (a * b).sum(-1)
    return torch.where(denom > _EPS, dot / denom.clamp_min(_EPS), torch.zeros_like(dot))


def logit_metrics(logits: torch.Tensor) -> torch.Tensor:
    """[rows, vocab] logits -> [rows, 3] of (entropy, perplexity, confidence)."""
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = shannon_entropy(probs)
    return torch.stack([entropy, entropy.exp(), probs.amax(-1)], dim=-1)


def _gate_entropy(x: torch.Tensor) -> torch.Tensor:
    """32-bin histogram entropy of each row (SPEC.md 3.2)."""
    rows, width = x.shape
    vmin = x.amin(-1, keepdim=True)
    vmax = x.amax(-1, keepdim=True)
    rng = vmax - vmin
    rng = torch.where(rng < 1e-10, torch.ones_like(rng), rng)
    # Truncation toward zero, matching the C++ (int) cast on a non-negative value.
    bins = (
        ((x - vmin) / rng * (_GATE_HIST_BINS - 1)).long().clamp_(0, _GATE_HIST_BINS - 1)
    )
    counts = torch.zeros(rows, _GATE_HIST_BINS, device=x.device, dtype=x.dtype)
    counts.scatter_add_(1, bins, torch.ones_like(x))
    return shannon_entropy(counts / width)


def _q_k_cosine(q: torch.Tensor, k: torch.Tensor, head_dim: int) -> torch.Tensor:
    """GQA-aware mean cosine between group-averaged queries and their key head.

    ``q`` is [rows, n_head * head_dim], ``k`` is [rows, n_kv * head_dim], both
    post-RoPE for the captured token.
    """
    rows = q.shape[0]
    n_head = q.shape[-1] // head_dim
    n_kv = k.shape[-1] // head_dim
    if n_kv == 0 or n_head < n_kv:
        return torch.zeros(rows, device=q.device, dtype=q.dtype)
    group = n_head // n_kv
    q_avg = q.view(rows, n_kv, group, head_dim).mean(dim=2)
    k_heads = k.view(rows, n_kv, head_dim)
    return cosine_rows(q_avg, k_heads).mean(-1)


def layer_stats(
    signal: str,
    x: torch.Tensor,
    *,
    prev_residual: torch.Tensor | None = None,
    pending_q: torch.Tensor | None = None,
    head_dim: int = 0,
) -> torch.Tensor:
    """Reduce [rows, width] activations to the [rows, 8] stats tuple.

    Column order is :data:`vllm.signals.tiers.STATS_FIELDS`, shared with the C++
    recorder so either engine's deposits decode identically. Columns that do not
    apply to ``signal`` are zero.

    Args:
        signal: one of the names in :data:`vllm.signals.tiers.ALL_SIGNALS`.
        x: [rows, width] float activations for one (layer, signal).
        prev_residual: the previous layer's residual rows, for the residual
            cosine column. ``None`` on the first captured layer.
        pending_q: the same layer's ``qcur`` rows, needed for the ``kcur``
            Q-K cosine column.
        head_dim: attention head dim, required for ``qcur``/``kcur`` columns.
    """
    x = x.float()
    rows, width = x.shape
    absx = x.abs()

    threshold = _GATE_SPARSITY_THRESHOLD if signal == SIG_GATE else _SPARSITY_THRESHOLD

    if signal == SIG_QCUR and head_dim > 0:
        # Mean per-head L2 norm rather than the whole-row norm (SPEC.md 3.4).
        n_head = width // head_dim
        l2 = x.view(rows, n_head, head_dim).norm(dim=-1).mean(-1)
    else:
        l2 = x.norm(dim=-1)

    mean = x.mean(-1)
    var = (x * x).mean(-1) - mean * mean

    out = torch.zeros(rows, STATS_WIDTH, device=x.device, dtype=torch.float32)
    out[:, 0] = l2
    out[:, 1] = absx.mean(-1)
    out[:, 2] = absx.amax(-1)
    out[:, 3] = var.clamp_min(0).sqrt()
    out[:, 4] = (absx < threshold).to(x.dtype).mean(-1)

    if signal == SIG_RESIDUAL and prev_residual is not None:
        out[:, 5] = cosine_rows(x, prev_residual.float())
    if signal == SIG_GATE:
        out[:, 6] = _gate_entropy(x)
    if signal == SIG_KCUR and pending_q is not None and head_dim > 0:
        out[:, 7] = _q_k_cosine(pending_q.float(), x, head_dim)

    return out
