# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Put a captured residual back into the forward pass.

The capture side reads the residual stream without touching it. This is the
other direction: take a vector a deposit recorded earlier and add it back in,
at a chosen layer, so a generation starts from -- or is steered by -- a state
the model was in before.

Two modes, because they answer different questions:

``add``
    ``stream += alpha * v``. Classic activation steering: the model keeps its
    own state and is nudged along the captured direction. ``alpha`` sets how
    hard.

``replace``
    ``stream = v`` at one layer. The captured state becomes the state at *that
    layer only*. Every layer below it still computed from the prompt, and the
    KV cache for the position was written by that computation, so later tokens
    attend to the original state and the generation drifts back. Steering, not
    resumption - see ``state``.

``state``
    ``stream = v`` at *every* layer, for the seeded position. Because each
    layer's attention computes that position's K/V from the stream it is
    handed, writing the vector at every layer makes the whole KV cache for that
    position derive from the captured state. Later tokens then attend to the
    state rather than to whatever the prompt computed, which is what "start a
    fresh discourse from this residual" actually requires. ``replace`` cannot
    do this at any single layer, and at the last layer it does almost nothing:
    63 of 64 layers still hold the prompt's K/V.

Two position policies:

``first``
    Only the first sampling position of each request -- the "starting point"
    reading. The model is seeded once and then runs free.

``all``
    Every sampling position, so the influence persists for the whole turn.

``zero``
    Sequence position 0, during prefill - the first thing in the context,
    *before* the prompt rather than after it.

    This is a different kind of thing from the other two, which are both
    sampling-row policies: they write where generation begins, which is the
    *end* of the prompt. A vector written there is downstream of everything the
    prompt already established, so it colours the next token and is then
    outvoted by the context already built. Written at position 0 the prompt
    itself is computed downstream of it, the model builds that position's K/V
    at every layer the way it does for any token, and every later token attends
    to it. What space the vector "came from" stops mattering, because 64 layers
    of the model's own computation run on top of it.

Injection rewrites a layer's output, which a CUDA graph would capture once and
replay forever, so it forces eager execution the same way the hook-based
capture tiers do.
"""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

INJECT_MODES = ("add", "replace", "state")
INJECT_POSITIONS = ("first", "all", "zero")


@dataclass
class InjectionSpec:
    """One configured injection."""

    vector: torch.Tensor
    """[hidden] the residual to inject."""

    layer: int
    """Where to inject. Ignored by ``state``, which writes every layer."""

    alpha: float = 1.0
    mode: str = "add"
    positions: str = "first"
    source: str = ""

    def validate(self, num_layers: int, hidden_size: int) -> None:
        if self.mode not in INJECT_MODES:
            raise ValueError(
                f"unknown injection mode {self.mode!r}; expected one of "
                f"{', '.join(INJECT_MODES)}"
            )
        if self.positions not in INJECT_POSITIONS:
            raise ValueError(
                f"unknown injection positions {self.positions!r}; expected one "
                f"of {', '.join(INJECT_POSITIONS)}"
            )
        if not 0 <= self.layer < num_layers:
            raise ValueError(
                f"injection layer {self.layer} is outside this model's "
                f"0..{num_layers - 1}"
            )
        if self.vector.numel() != hidden_size:
            raise ValueError(
                f"injection vector has {self.vector.numel()} elements but this "
                f"model's residual stream is {hidden_size} wide; the vector was "
                "almost certainly captured from a different model"
            )

    def describe(self) -> dict:
        return {
            "layer": self.layer,
            "alpha": self.alpha,
            "mode": self.mode,
            "positions": self.positions,
            "source": self.source,
            "hidden_size": int(self.vector.numel()),
        }


def load_vector(
    path: str,
    signal: str = "residual",
    layer: int | None = None,
    row: int = -1,
) -> tuple[torch.Tensor, dict]:
    """Read one residual vector out of a `sigcap-v1` deposit.

    Args:
        path: the deposit to read.
        signal: which stored signal to take (``residual`` by default).
        layer: take the row recorded at this layer; None takes whatever the
            chosen row holds.
        row: which row, when several match. -1 (default) is the last, which for
            a per-turn deposit is the turn's final state.

    Returns:
        The vector, and the deposit's metadata.

    Raises:
        ValueError: if the deposit has no such signal or layer.
    """
    from safetensors.torch import load_file

    from vllm.signals.view import read_metadata

    tensors = load_file(path)
    if signal not in tensors:
        available = sorted(k for k in tensors if not k.endswith(".index"))
        raise ValueError(
            f"deposit {path} has no {signal!r} tensor; it holds: "
            f"{', '.join(available) or '(nothing)'}"
        )

    values = tensors[signal]
    index = tensors.get(f"{signal}.index")
    if layer is not None and index is not None:
        matching = [i for i in range(index.shape[0]) if int(index[i, 1]) == layer]
        if not matching:
            recorded = sorted({int(index[i, 1]) for i in range(index.shape[0])})
            raise ValueError(
                f"deposit {path} has no {signal!r} row for layer {layer}; it "
                f"recorded layers {recorded}"
            )
        values = values[matching]

    metadata, _ = read_metadata(path)
    return values[row].clone(), metadata


class SignalInjector:
    """Applies an :class:`InjectionSpec` to the rows a step is sampling at.

    The model runner arms this with the same row indices the capturer uses, so
    injection lands on exactly the positions capture would have read.
    """

    def __init__(self, num_layers: int, hidden_size: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.spec: InjectionSpec | None = None
        self._row_index: torch.Tensor | None = None
        self._seeded: set[str] = set()
        self._row_req_ids: list[str] = []
        self._active = False
        self.applied = 0
        self._step_rows: torch.Tensor | None = None
        self._step_rows_ready = False
        self._num_tokens = 0

    @property
    def enabled(self) -> bool:
        return self.spec is not None

    def set_spec(self, spec: InjectionSpec | None) -> dict:
        """Install or clear the injection. Returns the resulting status."""
        if spec is not None:
            spec.validate(self.num_layers, self.hidden_size)
        self.spec = spec
        self._seeded.clear()
        self.applied = 0
        if spec is None:
            logger.info("Signal injection cleared")
        else:
            logger.info(
                "Signal injection set: %s alpha=%.3g at layer %d, positions=%s, "
                "from %s",
                spec.mode,
                spec.alpha,
                spec.layer,
                spec.positions,
                spec.source or "(inline)",
            )
        return self.status()

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "seeded_requests": len(self._seeded),
            **({} if self.spec is None else self.spec.describe()),
        }

    def begin_step(
        self,
        row_req_ids: list[str],
        row_index: torch.Tensor,
        num_tokens: int = 0,
    ) -> None:
        if not self.enabled:
            self._active = False
            return
        self._row_req_ids = row_req_ids
        self._row_index = row_index
        self._num_tokens = num_tokens
        self._active = True
        self._step_rows = None
        self._step_rows_ready = False

    def end_step(self) -> None:
        self._active = False
        self._row_index = None
        self._row_req_ids = []
        self._step_rows = None
        self._step_rows_ready = False

    def forget(self, request_ids) -> None:
        """Drop per-request seeding state when requests finish."""
        for request_id in request_ids:
            self._seeded.discard(request_id)

    def _target_rows(self) -> torch.Tensor | None:
        """Which of this step's sampling rows should be written.

        Memoized for the step. ``state`` calls this once per layer and every
        one of those calls must see the same rows; recomputing would mark the
        request seeded on layer 0 and skip layers 1..N-1.
        """
        if self._step_rows_ready:
            return self._step_rows
        rows = self._compute_target_rows()
        self._step_rows = rows
        self._step_rows_ready = True
        return rows

    def _compute_target_rows(self) -> torch.Tensor | None:
        spec = self.spec
        if spec is None or self._row_index is None:
            return None
        if spec.positions == "zero":
            # Prefill only: a decode step's batch is one token per request and
            # its row 0 is a *generated* position, not the context's start.
            if self._num_tokens <= self._row_index.numel():
                return None
            unseeded = [rid for rid in self._row_req_ids if rid not in self._seeded]
            if not unseeded:
                return None
            self._seeded.update(unseeded)
            return torch.zeros(1, device=self._row_index.device, dtype=torch.long)
        if spec.positions == "all":
            return self._row_index
        wanted = [
            i for i, rid in enumerate(self._row_req_ids) if rid not in self._seeded
        ]
        if not wanted:
            return None
        for i in wanted:
            self._seeded.add(self._row_req_ids[i])
        selector = torch.tensor(wanted, device=self._row_index.device, dtype=torch.long)
        return self._row_index.index_select(0, selector)

    def apply(self, layer_idx: int, stream: torch.Tensor) -> torch.Tensor | None:
        """Rewrite the residual stream at ``layer_idx``, or None to leave it.

        ``stream`` is [num_tokens, hidden]; only the sampling rows are touched.
        """
        spec = self.spec
        if not self._active or spec is None:
            return None
        # `state` writes the whole stack; the others write one layer.
        if spec.mode != "state" and layer_idx != spec.layer:
            return None
        rows = self._target_rows()
        if rows is None or rows.numel() == 0:
            return None

        vector = spec.vector.to(device=stream.device, dtype=stream.dtype)
        updated = stream.clone()
        if spec.mode == "add":
            updated[rows] = updated[rows] + spec.alpha * vector
        else:
            # replace and state both overwrite; they differ only in how many
            # layers they are invoked for.
            updated[rows] = spec.alpha * vector
        self.applied += int(rows.numel())
        return updated
