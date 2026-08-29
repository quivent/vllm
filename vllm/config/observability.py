# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cached_property
from typing import Any, Literal, cast

from packaging.version import parse
from pydantic import Field, field_validator, model_validator

from vllm import version
from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

DetailedTraceModules = Literal["model", "worker", "all"]


@config
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""

    show_hidden_metrics_for_version: str | None = None
    """Enable deprecated Prometheus metrics that have been hidden since the
    specified version. For example, if a previously deprecated metric has been
    hidden since the v0.7.0 release, you use
    `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while
    you migrate to new metrics. The metric is likely to be removed completely
    in an upcoming release."""

    @cached_property
    def show_hidden_metrics(self) -> bool:
        """Check if the hidden metrics should be shown."""
        if self.show_hidden_metrics_for_version is None:
            return False
        return version._prev_minor_version_was(self.show_hidden_metrics_for_version)

    otlp_traces_endpoint: str | None = None
    """Target URL to which OpenTelemetry traces will be sent."""

    collect_detailed_traces: list[DetailedTraceModules] | None = None
    """It makes sense to set this only if `--otlp-traces-endpoint` is set. If
    set, it will collect detailed traces for the specified modules. This
    involves use of possibly costly and or blocking operations and hence might
    have a performance impact.

    Note that collecting detailed timing information for each request can be
    expensive."""

    per_request_spec_decode_metrics: Literal["none", "summary", "detailed"] = "none"
    """Include per-request speculative-decoding acceptance metrics in the
    response under `metrics.speculative_decoding`. `none` disables; `summary` adds mean
    acceptance length, draft acceptance rate, and the step-by-draft-length
    histogram; `detailed` additionally records the ordered per-step
    accepted/proposed arrays (one entry per verify step). Only reported for
    single-sequence requests (`n == 1`), mirroring the timing metrics. No effect
    unless speculative decoding is enabled. Independent of `--disable-log-stats`.
    This is the per-request response-body counterpart of the aggregate
    `vllm:spec_decode_*` Prometheus metrics. The response field is experimental
    and its shape may change in a future release."""

    kv_cache_metrics: bool = False
    """Enable KV cache residency metrics (lifetime, idle time, reuse gaps).
    Uses sampling to minimize overhead.
    Requires log stats to be enabled (i.e., --disable-log-stats not set)."""

    kv_cache_metrics_sample: float = Field(default=0.01, gt=0, le=1)
    """Sampling rate for KV cache metrics (0.0, 1.0]. Default 0.01 = 1% of blocks."""

    cudagraph_metrics: bool = False
    """Enable CUDA graph metrics (number of padded/unpadded tokens, runtime cudagraph
    dispatch modes, and their observed frequencies at every logging interval)."""

    signal_capture_tier: str = "off"
    """Inference-signal capture tier -- what internal state to record for each
    request, written as one `sigcap-v1` safetensors deposit per generation.
    Volume grows roughly 10x per step:

    - `off`: no capture (default).
    - `logit`: per-token entropy / perplexity / confidence.
    - `layer_stats`: the above plus per-layer scalar reductions of every signal.
    - `heads`: per-head scalar reductions. Not implemented; rejected.
    - `residual_raw`: the full residual stream (`l_out`) vector per layer, raw.
    - `full_raw`: the above plus gate / norms / q / k vectors per layer.
    - `full_raw_attn`: the above plus attention weights (not supported behind a
      fused attention kernel; rejected at startup).

    Tiers `t0`-`t6` are accepted as aliases. Any tier above `logit` needs module
    forward hooks, which forces eager execution -- see `--enforce-eager`.
    Requires `--signal-capture-dir`."""

    signal_capture_max_tier: str | None = None
    """Highest tier this process may switch to at runtime. Defaults to
    `--signal-capture-tier`.

    The tier decides what gets *recorded*; this decides which forward hooks get
    *installed* at load time, and so whether eager execution is forced. Set this
    above `--signal-capture-tier` to keep capture switchable without a restart:

        --signal-capture-max-tier full_raw --signal-capture-tier off

    starts recording nothing (a boolean check per layer) but lets
    `POST /signals/control` raise the tier later. Because hooks force eager,
    this trades throughput for the ability to turn capture on in place."""

    signal_capture_dir: str | None = None
    """Directory to write signal deposits into, one `<request_id>.safetensors`
    per generation. Capture is inactive unless this is set."""

    signal_capture_layers: str = "last"
    """Which decoder layers to tap: `all`, `last`, or a comma-separated list of
    layer indices (negatives count from the end). Defaults to `last`, which at
    the `residual_raw` tier makes each generated token cost one hidden-size
    vector -- about 10 KiB for a 5120-wide model in bf16."""

    signal_capture_layer_step: int = Field(default=1, ge=1)
    """Record every Nth layer, thinning whatever `--signal-capture-layers`
    selects."""

    signal_capture_token_step: int = Field(default=1, ge=1)
    """Record every Nth generated token of each request."""

    signal_capture_tokens: Literal["last", "first", "mean", "all"] = "last"
    """How to reduce a turn's generated tokens into a deposit.

    - `last`: one vector per tapped layer, from the turn's final token. This is
      the default, and makes a turn cost one hidden-size vector no matter how
      long the response runs.
    - `first`: the state at the first generated token.
    - `mean`: averaged over every generated token of the turn.
    - `all`: every generated token, one row each. Grows with response length,
      so pair it with `--signal-capture-token-step`."""

    signal_capture_dtype: Literal["native", "float32"] = "native"
    """Storage dtype for raw vectors. `native` keeps the model's activation
    dtype, so values are stored exactly as they exist in the forward pass;
    `float32` widens them, matching the reference C++ recorder byte-for-byte."""

    signal_capture_max_bytes: int = Field(default=64 * (1 << 20), ge=0)
    """Per-request budget for buffered signal data. A request that exceeds it
    stops recording and is marked `truncated` in its deposit metadata, rather
    than growing without limit. 0 disables the cap."""

    signal_capture_session: str | None = None
    """Free-form session label written into deposit metadata, for grouping the
    deposits of one experiment."""

    @property
    def signal_capture_effective_tier(self) -> str:
        """The tier that determines hook installation: the ceiling if set."""
        return self.signal_capture_max_tier or self.signal_capture_tier

    @property
    def signal_capture_enabled(self) -> bool:
        return self.signal_capture_effective_tier != "off" and bool(
            self.signal_capture_dir
        )

    enable_layerwise_nvtx_tracing: bool = False
    """Enable layerwise NVTX tracing. This traces the execution of each layer or
    module in the model and attach information such as input/output shapes to
    nvtx range markers. Noted that this doesn't work with CUDA graphs enabled."""

    enable_mfu_metrics: bool = False
    """Enable Model FLOPs Utilization (MFU) metrics."""

    enable_mm_processor_stats: bool = False
    """Enable collection of timing statistics for multimodal processor operations.
    This is for internal use only (e.g., benchmarks) and is not exposed as a CLI
    argument."""

    enable_logging_iteration_details: bool = False
    """Enable detailed logging of iteration details.
    If set, vllm EngineCore will log iteration details
    This includes number of context/generation requests and tokens
    and the elapsed cpu time for the iteration."""

    jit_monitor_mode: Literal["warn", "error"] = "warn"
    """How to handle post-warmup JIT compilation events."""

    jit_monitor_verbose: bool = False
    """Log every monitored JIT compile with runtime details. This can emit many
    logs and add overhead, so it is intended for debugging."""

    @cached_property
    def collect_model_forward_time(self) -> bool:
        """Whether to collect model forward time for the request."""
        return self.collect_detailed_traces is not None and (
            "model" in self.collect_detailed_traces
            or "all" in self.collect_detailed_traces
        )

    @cached_property
    def collect_model_execute_time(self) -> bool:
        """Whether to collect model execute time for the request."""
        return self.collect_detailed_traces is not None and (
            "worker" in self.collect_detailed_traces
            or "all" in self.collect_detailed_traces
        )

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("show_hidden_metrics_for_version")
    @classmethod
    def _validate_show_hidden_metrics_for_version(cls, value: str | None) -> str | None:
        if value is not None:
            # Raises an exception if the string is not a valid version.
            parse(value)
        return value

    @field_validator("otlp_traces_endpoint")
    @classmethod
    def _validate_otlp_traces_endpoint(cls, value: str | None) -> str | None:
        if value is not None:
            from vllm.tracing import is_tracing_available, otel_import_error_traceback

            if not is_tracing_available():
                raise ValueError(
                    "OpenTelemetry is not available. Unable to configure "
                    "'otlp_traces_endpoint'. Ensure OpenTelemetry packages are "
                    f"installed. Original error:\n{otel_import_error_traceback}"
                )
        return value

    @field_validator("collect_detailed_traces")
    @classmethod
    def _validate_collect_detailed_traces(
        cls, value: list[DetailedTraceModules] | None
    ) -> list[DetailedTraceModules] | None:
        """Handle the legacy case where users might provide a comma-separated
        string instead of a list of strings."""
        if value is not None and len(value) == 1 and "," in value[0]:
            value = cast(list[DetailedTraceModules], value[0].split(","))
        return value

    @field_validator("signal_capture_tier", "signal_capture_max_tier")
    @classmethod
    def _validate_signal_capture_tier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from vllm.signals.tiers import Tier, tier_from_str

        tier = tier_from_str(value)
        if tier == Tier.HEADS:
            raise ValueError(
                "signal capture tier 'heads' (T3) is not implemented: the "
                "per-head reduction is a stub in the reference recorder too, "
                "and would silently record the same scalars as 'layer_stats'. "
                "Use 'layer_stats' (T2) for per-layer scalars."
            )
        if tier == Tier.FULL_RAW_ATTN:
            raise ValueError(
                "signal_capture_tier='full_raw_attn' (T6) records post-softmax "
                "attention weights, which fused attention kernels never "
                "materialize. Use 'full_raw' (T5) for every other forward-pass "
                "signal."
            )
        return tier.wire_name

    @model_validator(mode="after")
    def _validate_signal_capture(self):
        from vllm.signals.tiers import Tier, tier_from_str

        effective = tier_from_str(self.signal_capture_effective_tier)
        if effective != Tier.OFF and not self.signal_capture_dir:
            raise ValueError(
                "signal capture is configured but --signal-capture-dir is not; "
                "there is nowhere to write deposits."
            )
        if tier_from_str(self.signal_capture_tier) > effective:
            raise ValueError(
                f"--signal-capture-tier {self.signal_capture_tier!r} is above "
                f"--signal-capture-max-tier {self.signal_capture_max_tier!r}; the "
                "ceiling decides which hooks get installed, so the starting tier "
                "cannot exceed it."
            )
        return self

    @model_validator(mode="after")
    def _validate_tracing_config(self):
        if self.collect_detailed_traces and not self.otlp_traces_endpoint:
            raise ValueError(
                "collect_detailed_traces requires `--otlp-traces-endpoint` to be set."
            )
        return self
