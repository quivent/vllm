# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tap points resolve against a real vLLM model, not just a stand-in.

`SignalCapturer` reaches into decoder layers by attribute path. Those paths are
the most fragile assumption in the whole mechanism, so pin them against an
actual vLLM model built on the meta device -- no weights, no GPU.
"""

import os

import pytest
import torch

from vllm.signals.capturer import find_decoder_layers

NUM_LAYERS = 3


@pytest.fixture(scope="module")
def llama_on_meta():
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29594")

    model_config = ModelConfig(model="facebook/opt-125m", enforce_eager=True)
    vllm_config = VllmConfig(model_config=model_config)

    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    model_config.hf_config = hf_config
    model_config.hf_text_config = hf_config

    with set_current_vllm_config(vllm_config):
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)

        from vllm.model_executor.models.llama import LlamaForCausalLM

        with torch.device("meta"):
            yield LlamaForCausalLM(vllm_config=vllm_config)


def test_finds_the_real_decoder_stack(llama_on_meta):
    layers = find_decoder_layers(llama_on_meta)
    assert len(layers) == NUM_LAYERS
    assert type(layers[0]).__name__ == "LlamaDecoderLayer"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("input_layernorm", "RMSNorm"),
        ("post_attention_layernorm", "RMSNorm"),
        ("mlp.act_fn", "SiluAndMul"),
        ("self_attn.rotary_emb", "RotaryEmbedding"),
    ],
)
def test_tap_paths_resolve(llama_on_meta, path, expected):
    from vllm.signals.capturer import _resolve

    layer = find_decoder_layers(llama_on_meta)[0]
    module = _resolve(layer, path)
    assert module is not None, f"no tap point at {path}"
    assert type(module).__name__ == expected


def test_decoder_layer_returns_the_fused_residual_pair(llama_on_meta):
    """The residual tap assumes layers return (hidden_states, residual)."""
    import inspect

    source = inspect.getsource(type(find_decoder_layers(llama_on_meta)[0]).forward)
    assert "return hidden_states, residual" in source
