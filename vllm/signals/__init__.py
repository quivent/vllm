# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-time signal capture.

A port of the ``signal-extraction`` mechanism: treat the transformer's internal
state during inference as structured, tiered records -- captured at the tap
point, written as compact binary deposits -- rather than shipping raw tensors
around by hand.

See :mod:`vllm.signals.tiers` for the capture-tier ladder, and
:mod:`vllm.signals.deposit` for the on-disk ``sigcap-v1`` format.
"""

from vllm.signals.config import SignalCaptureConfig
from vllm.signals.deposit import DEPOSIT_FORMAT, Deposit
from vllm.signals.tiers import Mode, Tier, mode_for, tier_from_str, wants

__all__ = [
    "DEPOSIT_FORMAT",
    "Deposit",
    "Mode",
    "SignalCaptureConfig",
    "Tier",
    "mode_for",
    "tier_from_str",
    "wants",
]
