# Sample deposits

Real captures from live vLLM runs, checked in so the format can be inspected
without a GPU. Each is present twice: the **raw binary** exactly as capture
wrote it, and the **same data rendered for reading**.

| file | what |
|---|---|
| `*.safetensors` | the deposit as written — raw `sigcap-v1` binary |
| `*.md` | rendered: metadata, tensor table, per-layer stats, a norm profile |
| `*.stats.json` | machine-readable: metadata plus the SPEC scalar taxonomy |

## `qwen3.8-27b-residual-64layers`

Qwen3.8-27B-INT4 on a GH200, hook backend, all 64 layers of one turn.
`residual [64, 5120] BF16` = 655,360 B — **10,240 B per layer**, the
hidden_size × 2 arithmetic in the wild. The norm profile in the `.md` shows the
residual stream growing from ~12 at the embeddings to its final magnitude, with
`cos(prev layer)` near 1.0 throughout: consecutive layers nudge the stream
rather than redirect it.

Note this is a *hybrid* stack — some layers are GDN linear attention — and all
64 are present, which is the evidence that the residual tap does not depend on
a layer having an attention module.

## `qwen3-0.6b-residual-graph-backend`

Qwen3-0.6B, **graph backend**, last layer only. `residual [1, 1024] BF16` =
2,048 B. Captured with `ENFORCE_EAGER: False` and
`CUDAGRAPH_MODE: FULL_AND_PIECEWISE` — the proof that capture works with CUDA
graphs on, via the model's auxiliary hidden states rather than a forward hook.

This is the shape a per-turn deposit takes: one vector, regardless of how long
the turn ran.

## Reading them yourself

```bash
python -m vllm.signals.view vllm/signals/samples/qwen3.8-27b-residual-64layers.safetensors
python -m vllm.signals.view ...safetensors -f stats
python -m vllm.signals.view ...safetensors -f npz -o out.npz
```

They are also valid injection sources:

```bash
python -m vllm.signals.build mean 'vllm/signals/samples/*27b*.safetensors' \
  --layer 63 -o vec.safetensors
```
