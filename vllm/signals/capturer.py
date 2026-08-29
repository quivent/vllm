# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side inference-signal capture.

Taps the model's forward pass through ``nn.Module`` forward hooks, gathers the
row belonging to each in-flight request's current token, and accumulates one
:class:`~vllm.signals.deposit.Deposit` per request. Deposits are written out --
off the engine thread -- when the request finishes.

Tap points, per SPEC.md 3:

======================  =====================================================
Signal                  Module tapped
======================  =====================================================
``residual``            each decoder layer; ``l_out = hidden_states + residual``
``attn_norm``           ``layer.input_layernorm``
``ffn_norm``            ``layer.post_attention_layernorm``
``gate``                ``layer.mlp.act_fn`` (post-activation, width n_ff)
``qcur`` / ``kcur``     ``layer.self_attn.rotary_emb`` (post-RoPE Q and K)
======================  =====================================================

``residual`` uses the same expression as vLLM's own EAGLE-3 auxiliary hidden
state hook (``EagleModelMixin._maybe_add_hidden_state``): decoder layers carry
the residual stream alongside the normed activations for fused add+norm, so the
stream itself is the sum of the two values the layer returns.

Attention weights (``attn``, tier T6) are not capturable behind a fused
attention kernel; the T6 tier is rejected at startup rather than silently
recording nothing.
"""

import atexit
import os
import re
import threading
import time
import weakref
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.signals import reductions
from vllm.signals.config import SignalCaptureConfig
from vllm.signals.deposit import Deposit
from vllm.signals.inject import InjectionSpec, SignalInjector
from vllm.signals.tiers import (
    ALL_SIGNALS,
    SIG_ATTN_NORM,
    SIG_FFN_NORM,
    SIG_GATE,
    SIG_KCUR,
    SIG_QCUR,
    SIG_RESIDUAL,
    Mode,
    Tier,
    is_internal_request,
    mode_for,
    tier_from_str,
)

logger = init_logger(__name__)

_LAYER_NAME_RE = re.compile(r"^(?P<prefix>.*\.layers)\.(?P<idx>\d+)$")


def _first_tensor(value) -> torch.Tensor | None:
    """Modules in the tap set return either a tensor or a (tensor, ...) tuple."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    return None


def find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Locate the decoder layer stack, without depending on a model class.

    Groups every ``*.layers.<n>`` module by its prefix and returns the largest
    group, which is the main decoder stack for every vLLM model layout (the
    smaller groups are drafters, vision towers, and the like).
    """
    groups: dict[str, dict[int, nn.Module]] = defaultdict(dict)
    for name, module in model.named_modules():
        match = _LAYER_NAME_RE.match(name)
        if match:
            groups[match.group("prefix")][int(match.group("idx"))] = module
    if not groups:
        return []
    best = max(groups.values(), key=len)
    return [best[i] for i in sorted(best)]


class SignalCapturer:
    """Owns the hooks, the per-step staging buffers, and the per-request deposits.

    Lifecycle, driven by the model runner:

    1. :meth:`begin_step` before the forward -- says which rows of the batch to
       gather and which request each row belongs to.
    2. hooks fire during the forward, staging gathered rows on device.
    3. :meth:`end_step` after the forward -- one device->host copy, then rows are
       filed into each request's deposit.
    4. :meth:`finish_requests` when requests complete -- deposits are written.
    """

    def __init__(
        self,
        config: SignalCaptureConfig,
        model: nn.Module,
        *,
        model_name: str = "",
        head_dim: int = 0,
        num_heads: int = 0,
        num_kv_heads: int = 0,
        hidden_size: int = 0,
    ):
        self.config = config
        self.model_name = model_name
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.tier = config.tier
        self.max_tier = config.max_tier
        self.backend = config.backend

        self.layers = find_decoder_layers(model)
        if not self.layers:
            logger.warning(
                "Signal capture is enabled but no decoder layer stack was found "
                "on this model; capture is inactive."
            )
            self.enabled = False
            return

        self.tapped_layers = set(config.resolve_layers(len(self.layers)))
        self._hooked_layers = set(self.tapped_layers)
        self._token_reduce = config.tokens
        self.injector = SignalInjector(len(self.layers), hidden_size)
        self._inject_handle = None
        self.enabled = bool(self.tapped_layers)

        self._deposits: dict[str, Deposit] = {}
        self._token_counter: dict[str, int] = defaultdict(int)
        # signal -> layer -> staged rows for the current step.
        self._staged: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        self._pending_q: dict[int, torch.Tensor] = {}

        self._active = False
        self._row_index: torch.Tensor | None = None
        self._row_req_ids: list[str] = []
        self._num_tokens = 0
        self._staged_logits: torch.Tensor | None = None
        self._silent_checked = False
        self._aux_layer_order: tuple[int, ...] = ()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="signal-deposit"
        )
        self._lock = threading.Lock()

        if self.enabled:
            if self.backend == "hook":
                self._install_hooks()
            self._register_exit_flush()
            logger.info(
                "Signal capture active: tier=%s layers=%s (of %d) token_step=%d "
                "dtype=%s -> %s",
                self.tier.wire_name,
                sorted(self.tapped_layers)
                if len(self.tapped_layers) <= 8
                else f"{len(self.tapped_layers)} layers",
                len(self.layers),
                self.config.token_step,
                self.config.dtype,
                self.config.output_dir,
            )

    # ── hook installation ────────────────────────────────────────────────────

    def _wants(self, signal: str) -> bool:
        """Hook installation is decided by the ceiling, not the live tier."""
        return mode_for(self.max_tier, signal) is not Mode.DROP

    def _install_hooks(self) -> None:
        want_residual = self._wants(SIG_RESIDUAL)
        want_attn_norm = self._wants(SIG_ATTN_NORM)
        want_ffn_norm = self._wants(SIG_FFN_NORM)
        want_gate = self._wants(SIG_GATE)
        want_qk = self._wants(SIG_QCUR) or self._wants(SIG_KCUR)

        missing: set[str] = set()
        for idx in sorted(self.tapped_layers):
            layer = self.layers[idx]
            if want_residual:
                self._hook(layer, self._make_residual_hook(idx))
            if want_attn_norm:
                self._hook_named(layer, "input_layernorm", SIG_ATTN_NORM, idx, missing)
            if want_ffn_norm:
                self._hook_named(
                    layer, "post_attention_layernorm", SIG_FFN_NORM, idx, missing
                )
            if want_gate:
                self._hook_named(layer, "mlp.act_fn", SIG_GATE, idx, missing)
            if want_qk:
                # vLLM's Attention module is *called with* post-RoPE q, k, v,
                # whatever the architecture did to produce them -- fused RoPE,
                # gated projections, QK-norm. A pre-hook reads them straight off
                # the call, which the rotary_emb tap cannot do when RoPE is
                # applied by a kernel that never invokes the module.
                attn = _resolve(layer, "self_attn.attn")
                if attn is not None:
                    self._hook_pre(attn, self._make_qk_pre_hook(idx))
                else:
                    rotary = _resolve(layer, "self_attn.rotary_emb")
                    if rotary is None:
                        missing.add("qcur/kcur")
                    else:
                        self._hook(rotary, self._make_qk_hook(idx))

        if missing:
            logger.warning(
                "Signal capture: no tap point on this architecture for %s; those "
                "signals will be absent from deposits.",
                ", ".join(sorted(missing)),
            )

    def _hook(self, module: nn.Module, fn) -> None:
        self._handles.append(module.register_forward_hook(fn))

    def _hook_pre(self, module: nn.Module, fn) -> None:
        self._handles.append(module.register_forward_pre_hook(fn))

    def _hook_named(
        self, layer: nn.Module, path: str, signal: str, idx: int, missing: set[str]
    ) -> None:
        module = _resolve(layer, path)
        if module is None:
            missing.add(signal)
            return
        self._hook(module, self._make_simple_hook(signal, idx))

    def _make_residual_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            if not self._active:
                return
            # Fused add+norm layers return (hidden_states, residual); the residual
            # stream is their sum. Non-fused layers return the stream directly.
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                hidden, residual = output[0], output[1]
                stream = hidden + residual if residual is not None else hidden
            else:
                stream = _first_tensor(output)
            if stream is not None:
                self._stage(SIG_RESIDUAL, layer_idx, stream)

        return hook

    def _make_simple_hook(self, signal: str, layer_idx: int):
        def hook(_module, _args, output):
            if not self._active:
                return
            tensor = _first_tensor(output)
            if tensor is not None:
                self._stage(signal, layer_idx, tensor)

        return hook

    def _make_qk_pre_hook(self, layer_idx: int):
        """Reads the (q, k, v) a decoder layer hands to its attention."""

        def hook(_module, args):
            if not self._active or len(args) < 2:
                return
            query, key = args[0], args[1]
            if torch.is_tensor(query):
                self._stage(SIG_QCUR, layer_idx, query)
            if torch.is_tensor(key):
                self._stage(SIG_KCUR, layer_idx, key)

        return hook

    def _make_qk_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            if not self._active:
                return
            if not (isinstance(output, (tuple, list)) and len(output) >= 2):
                return
            query, key = output[0], output[1]
            if isinstance(query, torch.Tensor):
                self._stage(SIG_QCUR, layer_idx, query)
            if isinstance(key, torch.Tensor):
                self._stage(SIG_KCUR, layer_idx, key)

        return hook

    # ── per-step capture ─────────────────────────────────────────────────────

    def _stage(self, signal: str, layer_idx: int, tensor: torch.Tensor) -> None:
        """Gather the rows of interest out of a [num_tokens, width] activation."""
        index = self._row_index
        if index is None or tensor.ndim < 2:
            return
        if mode_for(self.tier, signal) is Mode.DROP:
            return
        if layer_idx not in self.tapped_layers:
            return
        flat = tensor.reshape(tensor.shape[0], -1)
        if flat.shape[0] < self._num_tokens:
            # Not a per-token activation of this batch (or a dummy run slipping
            # through). Checked against a known count so no device sync is needed.
            return
        self._staged[signal][layer_idx] = flat.index_select(0, index)

    def begin_step(
        self, row_req_ids: list[str], row_index: torch.Tensor, num_tokens: int
    ) -> None:
        """Arm the hooks for one forward pass.

        Args:
            row_req_ids: the request id owning each captured row, in row order.
            row_index: indices into the flattened token batch to gather -- the
                sampling positions (``logits_indices``).
            num_tokens: rows in this batch's token dimension, used to tell
                per-token activations apart from other tensors the hooks see.
        """
        if not row_req_ids:
            self._active = False
            self.injector.begin_step([], row_index)
            return
        if not self.enabled or self.tier == Tier.OFF:
            self._active = False
            self.injector.begin_step(row_req_ids, row_index.to(torch.long))
            return
        if any(map(is_internal_request, row_req_ids)):
            keep = [
                i for i, rid in enumerate(row_req_ids) if not is_internal_request(rid)
            ]
            if not keep:
                self._active = False
                return
            sel = torch.tensor(keep, device=row_index.device, dtype=torch.long)
            row_index = row_index.index_select(0, sel)
            row_req_ids = [row_req_ids[i] for i in keep]
        if len(row_req_ids) != row_index.shape[0]:
            # The row->request mapping disagrees with the gather indices; record
            # nothing rather than misattribute activations to the wrong request.
            logger.warning_once(
                "Signal capture: %d sampling rows but %d request slots; skipping "
                "capture for this step.",
                row_index.shape[0],
                len(row_req_ids),
            )
            self._active = False
            return
        # token_step thins by each request's own generated-token counter.
        if self.config.token_step > 1:
            keep = [
                i
                for i, rid in enumerate(row_req_ids)
                if self._token_counter[rid] % self.config.token_step == 0
            ]
            if not keep:
                self._active = False
                for rid in row_req_ids:
                    self._token_counter[rid] += 1
                return
            if len(keep) != len(row_req_ids):
                sel = torch.tensor(keep, device=row_index.device, dtype=torch.long)
                row_index = row_index.index_select(0, sel)
                row_req_ids = [row_req_ids[i] for i in keep]

        self._staged.clear()
        self._pending_q.clear()
        self._staged_logits = None
        self._row_req_ids = row_req_ids
        self._row_index = row_index.to(torch.long)
        self._num_tokens = num_tokens
        self._active = True
        self.injector.begin_step(row_req_ids, self._row_index)

    def disarm(self) -> None:
        """Stop the hooks staging, without draining yet.

        Called as soon as the model forward returns, so that any other forward
        before :meth:`end_step` -- a drafter, a dummy run -- is not recorded.
        """
        self._active = False

    def record_logits(self, logits: torch.Tensor) -> None:
        """Hand over this step's sampling logits for the T1 token metrics."""
        if self._row_req_ids and self.tier >= Tier.LOGIT:
            self._staged_logits = logits

    def end_step(self, logits: torch.Tensor | None = None) -> None:
        """Drain the staged rows into per-request deposits."""
        if not self._row_req_ids:
            return
        self._active = False
        logits = logits if logits is not None else self._staged_logits
        row_req_ids = self._row_req_ids
        tokens = [self._token_counter[rid] for rid in row_req_ids]

        try:
            if self.tier >= Tier.LAYER_STATS:
                self._drain_signals(row_req_ids, tokens)
            if logits is not None:
                self._drain_logits(row_req_ids, tokens, logits)
        finally:
            for rid in row_req_ids:
                self._token_counter[rid] += 1
                self._deposit_for(rid).note_token()
            self._staged.clear()
            self._pending_q.clear()
            self._staged_logits = None
            self._row_index = None
            self._row_req_ids = []
            self.injector.end_step()

    def _warn_on_silent_signals(self) -> None:
        """Say so when a tier asks for a signal no tap produced.

        Without this a fused kernel or an unusual layer layout just yields a
        deposit that is quietly missing a column.
        """
        if self._silent_checked:
            return
        self._silent_checked = True
        wanted = {
            signal
            for signal in ALL_SIGNALS
            if mode_for(self.tier, signal) is not Mode.DROP
        }
        silent = sorted(wanted - set(self._staged))
        if silent:
            logger.warning(
                "Signal capture: tier %s asks for %s, but no tap on this "
                "architecture produced them; they will be absent from deposits.",
                self.tier.wire_name,
                ", ".join(silent),
            )

    def _drain_signals(self, row_req_ids: list[str], tokens: list[int]) -> None:
        raw_mode = {sig: mode_for(self.tier, sig) is Mode.RAW for sig in self._staged}
        host: dict[str, tuple[list[int], torch.Tensor]] = {}

        self._warn_on_silent_signals()

        # ALL_SIGNALS order puts qcur before kcur, which the Q-K cosine needs.
        for signal in [s for s in ALL_SIGNALS if s in self._staged]:
            per_layer = self._staged[signal]
            if not per_layer:
                continue
            layer_ids = sorted(per_layer)
            stacked = torch.stack([per_layer[i] for i in layer_ids])
            if raw_mode[signal]:
                if self.config.dtype == "float32":
                    stacked = stacked.float()
            else:
                stacked = self._reduce(signal, layer_ids, stacked)
            # One device->host copy per signal, for the whole step.
            host[signal] = (layer_ids, stacked.detach().to("cpu", copy=True))

        for signal, (layer_ids, values) in host.items():
            is_raw = raw_mode[signal]
            for li, layer_idx in enumerate(layer_ids):
                for ri, rid in enumerate(row_req_ids):
                    deposit = self._deposit_for(rid)
                    row = values[li, ri : ri + 1]
                    if is_raw:
                        deposit.add_raw(signal, row, tokens[ri], layer_idx)
                    else:
                        deposit.add_stats(signal, row, tokens[ri], layer_idx)

    def _reduce(
        self, signal: str, layer_ids: list[int], stacked: torch.Tensor
    ) -> torch.Tensor:
        """Apply the T2/T3 scalar reductions, keeping cross-layer/-tensor pairing."""
        out = []
        prev: torch.Tensor | None = None
        for li, layer_idx in enumerate(layer_ids):
            rows = stacked[li]
            pending_q = None
            if signal == SIG_KCUR:
                pending_q = self._pending_q.get(layer_idx)
            stats = reductions.layer_stats(
                signal,
                rows,
                prev_residual=prev if signal == SIG_RESIDUAL else None,
                pending_q=pending_q,
                head_dim=self.head_dim,
            )
            if signal == SIG_RESIDUAL:
                prev = rows
            if signal == SIG_QCUR:
                self._pending_q[layer_idx] = rows
            out.append(stats)
        return torch.stack(out)

    def _drain_logits(
        self, row_req_ids: list[str], tokens: list[int], logits: torch.Tensor
    ) -> None:
        if logits.ndim != 2 or logits.shape[0] < len(row_req_ids):
            return
        metrics = reductions.logit_metrics(logits[: len(row_req_ids)])
        host = metrics.detach().to("cpu", copy=True)
        for ri, rid in enumerate(row_req_ids):
            self._deposit_for(rid).add_logits(host[ri : ri + 1], tokens[ri])

    # ── request lifecycle ────────────────────────────────────────────────────

    def _deposit_for(self, request_id: str) -> Deposit:
        deposit = self._deposits.get(request_id)
        if deposit is None:
            deposit = Deposit(
                request_id,
                tier=self.tier,
                layer_step=self.config.layer_step,
                token_step=self.config.token_step,
                model=self.model_name,
                session=self.config.session,
                max_bytes=self.config.max_bytes,
                token_reduce=self._token_reduce,
            )
            self._deposits[request_id] = deposit
        return deposit

    def set_runtime(
        self,
        tier: str | None = None,
        tokens: str | None = None,
        layers: str | None = None,
    ) -> dict:
        """Change what is recorded, without a restart.

        Only what the hooks already produce can be recorded, so ``tier`` is
        clamped to the ceiling chosen at startup. Layer selection is likewise
        limited to the layers that were tapped. Returns the new status.

        Raises:
            ValueError: if ``tier`` exceeds the startup ceiling, or a value is
                not a known tier/reduction.
        """
        if tier is not None:
            requested = tier_from_str(tier)
            if requested > self.max_tier:
                raise ValueError(
                    f"tier {requested.wire_name!r} is above this process's "
                    f"ceiling {self.max_tier.wire_name!r}; the hooks for it were "
                    "never installed. Restart with --signal-capture-max-tier "
                    f"{requested.wire_name}."
                )
            self.tier = requested
            self._silent_checked = False
            logger.info("Signal capture tier set to %s", requested.wire_name)

        if tokens is not None:
            if tokens not in ("all", "first", "last", "mean"):
                raise ValueError(
                    f"unknown token reduction {tokens!r}; expected one of "
                    "all, first, last, mean"
                )
            self._token_reduce = tokens
            logger.info("Signal capture token reduction set to %s", tokens)

        if layers is not None:
            from dataclasses import replace

            selected = set(
                replace(self.config, layers=layers).resolve_layers(len(self.layers))
            )
            unavailable = selected - self._hooked_layers
            if unavailable:
                raise ValueError(
                    f"layers {sorted(unavailable)} were not tapped at startup; "
                    f"only {sorted(self._hooked_layers)} can be recorded. Restart "
                    "with a wider --signal-capture-layers."
                )
            self.tapped_layers = selected
            logger.info("Signal capture layers set to %s", sorted(selected))

        return self.status()

    def status(self) -> dict:
        """What capture is doing right now."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "tier": self.tier.wire_name,
            "max_tier": self.max_tier.wire_name,
            "tokens": self._token_reduce,
            "layers": sorted(self.tapped_layers),
            "hooked_layers": sorted(self._hooked_layers),
            "num_layers": len(self.layers),
            "output_dir": self.config.output_dir,
            "open_deposits": len(self._deposits),
            "recording": self.tier != Tier.OFF and self.enabled,
        }

    def graph_aux_layers(self) -> tuple[int, ...]:
        """Aux indices to request from the model, for the graph backend.

        `_maybe_add_hidden_state` numbers 0 for the embeddings and `i + 1` for
        the stream leaving decoder layer `i`, so the tapped layers shift by one.
        """
        return tuple(sorted(i + 1 for i in self.tapped_layers))

    def check_graph_backend(
        self,
        model: nn.Module,
        speculative_uses_aux: bool,
        drafter_layers: tuple[int, ...] = (),
    ) -> bool:
        """Decide whether the graph backend can work, and how.

        Returns True when capture should set the auxiliary layers itself, and
        False when a drafter already publishes everything wanted -- in which
        case capture reads what is already there and leaves the drafter's
        configuration untouched.

        Raises:
            ValueError: when the tier needs a signal only hooks can reach, the
                model has no EAGLE-3 interface, or a drafter owns the auxiliary
                hidden states and is not publishing the layers wanted.
        """
        from vllm.model_executor.models.interfaces import supports_eagle3

        extra = [
            signal
            for signal in ALL_SIGNALS
            if signal != SIG_RESIDUAL
            and mode_for(self.max_tier, signal) is not Mode.DROP
        ]
        if extra:
            raise ValueError(
                f"--signal-capture-backend graph can only capture the residual, "
                f"but tier {self.max_tier.wire_name!r} also wants "
                f"{', '.join(extra)}. Use --signal-capture-backend hook, or a "
                "tier of residual_raw or below."
            )
        if mode_for(self.max_tier, SIG_RESIDUAL) is Mode.STATS:
            raise ValueError(
                "--signal-capture-backend graph records raw residuals; tier "
                f"{self.max_tier.wire_name!r} asks for scalar reductions. Use "
                "residual_raw, or the hook backend."
            )
        if not supports_eagle3(model):
            raise ValueError(
                "--signal-capture-backend graph needs a model implementing the "
                "EAGLE-3 interface (it reuses that mechanism's auxiliary hidden "
                "states); this one does not. Use --signal-capture-backend hook."
            )
        if not speculative_uses_aux:
            return True

        # A drafter (EAGLE-3, DFlash, DSpark) already owns the auxiliary list
        # and indexes into it positionally, so its layer set cannot be widened.
        # It can still be *read*, if it happens to publish what capture wants.
        wanted = set(self.graph_aux_layers())
        published = set(drafter_layers)
        if wanted <= published:
            self._aux_layer_order = tuple(sorted(published))
            logger.info(
                "Signal capture: reading the residual out of the drafter's "
                "existing auxiliary hidden states (layers %s), so capture costs "
                "nothing and the drafter is untouched.",
                sorted(i - 1 for i in published),
            )
            return False
        raise ValueError(
            "--signal-capture-backend graph cannot widen the auxiliary hidden "
            "states while a drafter owns them. That drafter publishes decoder "
            f"layers {sorted(i - 1 for i in published)}; capture wants "
            f"{sorted(i - 1 for i in wanted)}. Either capture one of the layers "
            "it already publishes (--signal-capture-layers "
            f"{','.join(str(i - 1) for i in sorted(published))}), or use "
            "--signal-capture-backend hook."
        )

    def observe_aux(self, aux_hidden_states) -> None:
        """Take the residuals the model returned, in graph-backend mode.

        The list is in the order of :meth:`graph_aux_layers`, and every entry is
        an ordinary graph output, so nothing here has to be capture-safe.
        """
        if not self._active or self.backend != "graph" or not aux_hidden_states:
            return
        order = self._aux_layer_order or self.graph_aux_layers()
        for layer_idx, stream in zip(order, aux_hidden_states):
            # Back to decoder-layer numbering for the deposit.
            decoder_layer = layer_idx - 1
            if decoder_layer in self.tapped_layers:
                self._stage(SIG_RESIDUAL, decoder_layer, stream)

    def set_injection(self, spec: InjectionSpec | None) -> dict:
        """Install or clear a residual injection, hooking the target layer.

        The hook is registered on demand rather than planned at startup, so any
        layer can be injected into regardless of which ones capture taps.
        """
        if self._inject_handle is not None:
            self._inject_handle.remove()
            self._inject_handle = None
        status = self.injector.set_spec(spec)
        if spec is not None:
            layer = self.layers[spec.layer]
            self._inject_handle = layer.register_forward_hook(
                self._make_injection_hook(spec.layer)
            )
        return status

    def load_injection(
        self,
        source: str,
        layer: int | None = None,
        alpha: float = 1.0,
        mode: str = "add",
        positions: str = "first",
        signal: str = "residual",
        row: int = -1,
    ) -> dict:
        """Load a vector out of a deposit and install it as the injection.

        ``layer`` defaults to the layer the vector was recorded at, which is
        almost always what you want: a residual only means the same thing at
        the depth it came from.
        """
        from vllm.signals.inject import load_vector

        vector, metadata = load_vector(source, signal=signal, layer=layer, row=row)
        if layer is None:
            layer = _recorded_layer(source, signal, row)
            if layer is None:
                raise ValueError(
                    f"{source} does not record which layer its {signal!r} came "
                    "from; pass an explicit layer."
                )
        spec = InjectionSpec(
            vector=vector,
            layer=layer,
            alpha=alpha,
            mode=mode,
            positions=positions,
            source=source,
        )
        if (
            metadata.get("model")
            and self.model_name
            and metadata["model"] != self.model_name
        ):
            logger.warning(
                "Signal injection: vector was captured from %s but this server "
                "serves %s; residuals are not portable between models.",
                metadata["model"],
                self.model_name,
            )
        return self.set_injection(spec)

    def _make_injection_hook(self, layer_idx: int):
        """Rewrites a layer's residual stream on the way out."""

        def hook(_module, _args, output):
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                hidden, residual = output[0], output[1]
                if residual is None:
                    updated = self.injector.apply(layer_idx, hidden)
                    return None if updated is None else (updated, *output[1:])
                # The stream is hidden + residual; adding to the residual half
                # shifts the stream the next layer sees, leaving this layer's
                # own output untouched.
                updated = self.injector.apply(layer_idx, residual)
                return None if updated is None else (hidden, updated, *output[2:])
            if torch.is_tensor(output):
                return self.injector.apply(layer_idx, output)
            return None

        return hook

    def _register_exit_flush(self) -> None:
        """Last-resort flush, in case the runner is torn down without shutdown().

        Holds only a weak reference, so registering here never keeps the model
        alive past its natural lifetime.
        """
        ref = weakref.ref(self)

        def flush_on_exit() -> None:
            capturer = ref()
            if capturer is not None:
                capturer.shutdown()

        atexit.register(flush_on_exit)

    def finish_requests(self, request_ids) -> None:
        """Write and retire the deposits for finished requests."""
        if not self.enabled:
            return
        self.injector.forget(request_ids)
        for rid in request_ids:
            deposit = self._deposits.pop(rid, None)
            self._token_counter.pop(rid, None)
            if deposit is None or deposit.is_empty:
                continue
            # Timestamp first so the directory reads as a history in sort
            # order, and so a reused request id never overwrites an earlier
            # turn. The id stays in the name, and in the metadata.
            stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(deposit.created_at))
            millis = int(deposit.created_at * 1000) % 1000
            path = os.path.join(
                self.config.output_dir,
                f"{stamp}.{millis:03d}-{_safe_name(rid)}.safetensors",
            )
            self._writer.submit(self._write, deposit, path)

    def _write(self, deposit: Deposit, path: str) -> None:
        try:
            written = deposit.write(path)
            if written:
                logger.debug(
                    "Wrote signal deposit %s (%d tokens, %.1f KiB)",
                    written,
                    deposit.num_tokens,
                    deposit.nbytes / 1024,
                )
        except Exception:
            logger.exception("Failed to write signal deposit to %s", path)

    def shutdown(self) -> None:
        """Flush in-flight deposits and remove the hooks."""
        if not getattr(self, "enabled", False):
            return
        self.finish_requests(list(self._deposits))
        self._writer.shutdown(wait=True)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.enabled = False


def _resolve(root: nn.Module, path: str) -> nn.Module | None:
    node: nn.Module | None = root
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _recorded_layer(path: str, signal: str, row: int) -> int | None:
    """The layer index a deposit recorded a given row at."""
    from safetensors.torch import load_file

    index = load_file(path).get(f"{signal}.index")
    if index is None or index.ndim != 2 or index.shape[1] < 2:
        return None
    return int(index[row, 1])


def _safe_name(request_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", request_id)[:180]


def rows_to_request_ids(req_ids: list[str], counts=None) -> list[str]:
    """Map each sampling row back to the request that owns it.

    One row per request in the ordinary case. Speculative decoding verifies
    several draft positions per request per step, so a request then owns
    ``counts[i]`` consecutive rows.

    Args:
        req_ids: the batch's requests, in row order.
        counts: rows per request, or None for one row each.
    """
    if counts is None:
        return list(req_ids)
    if len(counts) != len(req_ids):
        return list(req_ids)
    out: list[str] = []
    for rid, count in zip(req_ids, counts):
        out.extend([rid] * int(count))
    return out


def counts_from_cumulative(cu_num_logits_np, num_reqs: int):
    """Per-request row counts from a cumulative-offset array, or None."""
    if cu_num_logits_np is None or len(cu_num_logits_np) < num_reqs + 1:
        return None
    return [int(cu_num_logits_np[i + 1] - cu_num_logits_np[i]) for i in range(num_reqs)]


def maybe_build_capturer(vllm_config, model: nn.Module) -> "SignalCapturer | None":
    """Construct a capturer if signal capture is configured and supported here.

    Returns None when capture is off, or when this process is not the rank that
    should be writing deposits.
    """
    observability = vllm_config.observability_config
    if not (observability.signal_capture_enabled or observability.signal_inject_from):
        return None

    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        logger.warning(
            "Signal capture is not supported with pipeline parallelism "
            "(each rank holds only part of the layer stack); capture is off."
        )
        return None

    from vllm.distributed import get_tensor_model_parallel_rank

    try:
        if get_tensor_model_parallel_rank() != 0:
            return None
    except Exception:
        pass

    model_config = vllm_config.model_config
    head_dim = num_heads = num_kv_heads = 0
    if model_config is not None:
        parallel_config = vllm_config.parallel_config
        try:
            head_dim = model_config.get_head_size()
            num_heads = model_config.get_num_attention_heads(parallel_config)
            num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        except Exception:
            # Head geometry is only needed to split a fused QKV projection;
            # without it that fallback is simply unavailable.
            logger.debug("Signal capture: head geometry unavailable", exc_info=True)

    capturer = SignalCapturer(
        SignalCaptureConfig.from_observability(observability),
        model,
        model_name=model_config.model if model_config is not None else "",
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=model_config.get_hidden_size() if model_config else 0,
    )

    if observability.signal_inject_from:
        try:
            capturer.load_injection(
                source=observability.signal_inject_from,
                layer=observability.signal_inject_layer,
                alpha=observability.signal_inject_alpha,
                mode=observability.signal_inject_mode,
                positions=observability.signal_inject_positions,
            )
        except Exception as exc:
            logger.error("Signal injection could not be configured: %s", exc)
            raise

    return capturer
