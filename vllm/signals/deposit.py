# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``sigcap-v1`` deposit format: one safetensors file per generation.

Raw signal vectors are stored as contiguous binary tensors -- never per-value
JSON -- keyed by signal name, with a companion ``<name>.index`` tensor giving the
``(token, layer)`` coordinate of each row. This is the same on-disk layout the
C++ recorder in ``signal-extraction/capture`` writes, so deposits from either
engine load with the same reader.

Layout::

    residual              [rows, n_embd]   raw vectors, model dtype by default
    residual.index        [rows, 2]        f32 (token, layer)
    stats.residual        [rows, 8]        f32, STATS_FIELDS columns
    stats.residual.index  [rows, 2]        f32 (token, layer)
    logit                 [rows, 3]        f32 (entropy, perplexity, confidence)
    logit.index           [rows, 1]        f32 (token)

The one deliberate divergence from the C++ writer: raw tensors keep the model's
native dtype (bf16/fp16) rather than being widened to f32, so a residual vector
lands at ``n_embd * 2`` bytes -- the activations as they exist in the forward
pass. Pass ``dtype="float32"`` for byte-identical output.
"""

import os
import time
from dataclasses import dataclass, field

import torch
from safetensors.torch import save_file

from vllm.logger import init_logger
from vllm.signals.tiers import Tier

logger = init_logger(__name__)

DEPOSIT_FORMAT = "sigcap-v1"


@dataclass
class _Buffer:
    """Accumulated rows for one signal, in capture order."""

    chunks: list[torch.Tensor] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    layers: list[int] = field(default_factory=list)
    width: int = 0
    nbytes: int = 0

    def append(self, rows: torch.Tensor, token: int, layer: int) -> int:
        """Append ``rows`` ([n, width]); returns the bytes added."""
        if self.width == 0:
            self.width = rows.shape[-1]
        elif rows.shape[-1] != self.width:
            # Ragged guard, matching the C++ recorder: silently skip.
            return 0
        self.chunks.append(rows)
        n = rows.shape[0]
        self.tokens.extend([token] * n)
        self.layers.extend([layer] * n)
        added = rows.numel() * rows.element_size()
        self.nbytes += added
        return added

    def stack(self) -> torch.Tensor:
        return torch.cat(self.chunks, dim=0)


class Deposit:
    """Accumulates one request's signals, then writes a safetensors deposit.

    Rows arrive per decode step. Everything is held in host memory until
    :meth:`write`, bounded by ``max_bytes``; once that budget is spent the
    deposit stops accepting rows and records ``truncated=true`` in its metadata
    rather than growing without limit.
    """

    def __init__(
        self,
        request_id: str,
        *,
        tier: Tier,
        layer_step: int = 1,
        token_step: int = 1,
        model: str = "",
        session: str = "",
        max_bytes: int = 0,
    ):
        self.request_id = request_id
        self.tier = tier
        self.layer_step = layer_step
        self.token_step = token_step
        self.model = model
        self.session = session
        self.max_bytes = max_bytes

        self._raw: dict[str, _Buffer] = {}
        self._stats: dict[str, _Buffer] = {}
        self._logit_rows: list[torch.Tensor] = []
        self._logit_tokens: list[int] = []

        self.nbytes = 0
        self.truncated = False
        self.num_tokens = 0
        self.created_at = time.time()

    def __len__(self) -> int:
        return self.nbytes

    @property
    def is_empty(self) -> bool:
        return not (self._raw or self._stats or self._logit_rows)

    def _budget_left(self, incoming: int) -> bool:
        if self.max_bytes <= 0:
            return True
        if self.nbytes + incoming > self.max_bytes:
            if not self.truncated:
                self.truncated = True
                logger.warning_once(
                    "Signal deposit for a request hit the %.1f MiB budget and "
                    "stopped recording. Raise --signal-capture-max-bytes, or thin "
                    "the capture with --signal-capture-token-step / "
                    "--signal-capture-layer-step.",
                    self.max_bytes / (1 << 20),
                )
            return False
        return True

    def add_raw(self, signal: str, rows: torch.Tensor, token: int, layer: int) -> None:
        incoming = rows.numel() * rows.element_size()
        if not self._budget_left(incoming):
            return
        buffer = self._raw.setdefault(signal, _Buffer())
        self.nbytes += buffer.append(rows, token, layer)

    def add_stats(
        self, signal: str, rows: torch.Tensor, token: int, layer: int
    ) -> None:
        incoming = rows.numel() * rows.element_size()
        if not self._budget_left(incoming):
            return
        self.nbytes += self._stats.setdefault(signal, _Buffer()).append(
            rows, token, layer
        )

    def add_logits(self, rows: torch.Tensor, token: int) -> None:
        incoming = rows.numel() * rows.element_size()
        if not self._budget_left(incoming):
            return
        self._logit_rows.append(rows)
        self._logit_tokens.extend([token] * rows.shape[0])
        self.nbytes += incoming

    def note_token(self) -> None:
        self.num_tokens += 1

    @staticmethod
    def _index(tokens: list[int], layers: list[int] | None) -> torch.Tensor:
        """(token, layer) pairs as f32 -- exact for these ranges, and portable."""
        if layers is None:
            return torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
        return torch.tensor([tokens, layers], dtype=torch.float32).T.contiguous()

    def build(self) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
        """Materialize the tensor dict and metadata for this deposit."""
        tensors: dict[str, torch.Tensor] = {}
        for name, buf in sorted(self._raw.items()):
            tensors[name] = buf.stack()
            tensors[f"{name}.index"] = self._index(buf.tokens, buf.layers)
        for name, buf in sorted(self._stats.items()):
            tensors[f"stats.{name}"] = buf.stack()
            tensors[f"stats.{name}.index"] = self._index(buf.tokens, buf.layers)
        if self._logit_rows:
            tensors["logit"] = torch.cat(self._logit_rows, dim=0)
            tensors["logit.index"] = self._index(self._logit_tokens, None)

        dtypes = {str(t.dtype).removeprefix("torch.") for t in tensors.values()}
        metadata = {
            "format": DEPOSIT_FORMAT,
            "engine": "vllm",
            "tier": self.tier.wire_name,
            "layer_step": str(self.layer_step),
            "token_step": str(self.token_step),
            "model": self.model,
            "session": self.session,
            "request_id": self.request_id,
            "num_tokens": str(self.num_tokens),
            "dtypes": ",".join(sorted(dtypes)),
            "truncated": "true" if self.truncated else "false",
            "created_at": f"{self.created_at:.3f}",
        }
        return tensors, metadata

    def write(self, path: str) -> str | None:
        """Write the deposit. Returns the path, or None if there was nothing."""
        if self.is_empty:
            return None
        tensors, metadata = self.build()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Write to a temp name and rename, so a reader tailing the directory
        # never observes a half-written deposit.
        tmp = f"{path}.partial"
        save_file({k: v.contiguous() for k, v in tensors.items()}, tmp, metadata)
        os.replace(tmp, path)
        return path
