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
    output_dir: str = ""
    layer_step: int = 1
    token_step: int = 1
    layers: str = "last"
    """Which decoder layers to tap: ``all``, ``last``, or a comma-separated list
    of indices. Combined with ``layer_step`` (which thins whatever this selects).
    """
    dtype: str = "native"
    """``native`` keeps the model's activation dtype (bf16/fp16); ``float32``
    widens, matching the C++ recorder's output byte-for-byte."""
    max_bytes: int = 64 * (1 << 20)
    session: str = ""

    @property
    def enabled(self) -> bool:
        return self.tier != Tier.OFF and bool(self.output_dir)

    @property
    def needs_hooks(self) -> bool:
        """Whether capture requires module forward hooks (and therefore eager)."""
        return self.tier >= Tier.LAYER_STATS

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
            tier=tier_from_str(obs.signal_capture_tier),
            output_dir=obs.signal_capture_dir or "",
            layer_step=max(1, obs.signal_capture_layer_step),
            token_step=max(1, obs.signal_capture_token_step),
            layers=obs.signal_capture_layers,
            dtype=obs.signal_capture_dtype,
            max_bytes=obs.signal_capture_max_bytes,
            session=obs.signal_capture_session or "",
        )
