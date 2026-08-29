# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved runtime configuration for inference-signal capture.

Kept separate from ``vllm.config`` so the worker-side capture code can depend on
it without pulling the whole config package (and torch) into config import time.
"""

from dataclasses import dataclass

from vllm.signals.tiers import Tier, tier_from_str


@dataclass(frozen=True)
class SignalCaptureConfig:
    """Everything the worker-side capturer needs, resolved from CLI flags."""

    tier: Tier = Tier.OFF
    """The tier active at startup. Changeable at runtime up to ``max_tier``."""

    max_tier: Tier | None = None
    """The highest tier this process can ever reach. Defaults to ``tier``.

    Determines which forward
    hooks get installed at load time, and therefore whether eager execution is
    forced. Launch with ``max_tier`` high and ``tier`` off to pay only a boolean
    check per layer until you switch capture on."""

    backend: str = "hook"
    """`hook` (any signal, forces eager) or `graph` (residual only, keeps CUDA
    graphs by reusing vLLM's EAGLE-3 auxiliary hidden states)."""

    output_dir: str = ""
    layer_step: int = 1
    token_step: int = 1
    layers: str = "last"
    """Which decoder layers to tap: ``all``, ``last``, or a comma-separated list
    of indices. Combined with ``layer_step`` (which thins whatever this selects).
    """
    tokens: str = "last"
    """How to reduce a turn's generated tokens: ``last`` (the turn's final
    state, one vector), ``first`` (the state at the first generated token),
    ``mean`` (averaged over the turn), or ``all`` (every token). Anything but
    ``all`` makes a deposit a fixed size regardless of response length."""

    dtype: str = "native"
    """``native`` keeps the model's activation dtype (bf16/fp16); ``float32``
    widens, matching the C++ recorder's output byte-for-byte."""
    max_bytes: int = 64 * (1 << 20)
    session: str = ""

    def __post_init__(self):
        # The ceiling defaults to the active tier: asking for a tier always
        # installs at least the hooks that tier needs.
        if self.max_tier is None:
            object.__setattr__(self, "max_tier", self.tier)

    @property
    def enabled(self) -> bool:
        return self.max_tier != Tier.OFF and bool(self.output_dir)

    @property
    def needs_hooks(self) -> bool:
        """Whether capture requires module forward hooks (and therefore eager)."""
        return self.backend == "hook" and self.max_tier >= Tier.LAYER_STATS

    def resolve_layers(self, num_layers: int) -> tuple[int, ...]:
        """Expand the ``layers`` selector against the model's layer count."""
        spec = self.layers.strip().lower()
        if spec in ("last", ""):
            selected: tuple[int, ...] = (num_layers - 1,)
        elif spec == "all":
            selected = tuple(range(num_layers))
        else:
            try:
                indices = [int(p) for p in spec.split(",") if p.strip()]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid --signal-capture-layers value {self.layers!r}: expected "
                    "'all', 'last', or a comma-separated list of layer indices."
                ) from exc
            selected = tuple(i % num_layers for i in indices)
        if self.layer_step > 1:
            selected = tuple(i for i in selected if i % self.layer_step == 0)
        return selected

    @classmethod
    def from_observability(cls, obs) -> "SignalCaptureConfig":
        """Build from an :class:`~vllm.config.ObservabilityConfig`."""
        return cls(
            backend=obs.signal_capture_backend,
            tier=tier_from_str(obs.signal_capture_tier),
            max_tier=tier_from_str(
                obs.signal_capture_max_tier or obs.signal_capture_tier
            ),
            output_dir=obs.signal_capture_dir or "",
            layer_step=max(1, obs.signal_capture_layer_step),
            token_step=max(1, obs.signal_capture_token_step),
            layers=obs.signal_capture_layers,
            tokens=obs.signal_capture_tokens,
            dtype=obs.signal_capture_dtype,
            max_bytes=obs.signal_capture_max_bytes,
            session=obs.signal_capture_session or "",
        )
