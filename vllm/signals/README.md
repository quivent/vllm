# Inference-signal capture and injection

A private patch to vLLM that does two things:

- **Capture** — write the residual stream out of a live forward pass, one file
  per turn, without changing what the model produces.
- **Inject** — put a captured residual back in, so a generation starts from, or
  is steered by, a state the model was in before.

Ported from the [`signal-extraction`](https://github.com/quivent) taxonomy; the
on-disk format is the same `sigcap-v1` safetensors layout that repo's C++
recorder writes, so deposits from either engine read with the same tools.

## What you get

```bash
vllm serve --model <model> \
  --signal-capture-tier residual_raw \
  --signal-capture-dir  /var/signals
```

Every turn now leaves one file:

```
/var/signals/20260829T182233.537-chatcmpl-abe244925515b037-8d87ec41.safetensors
```

containing the residual stream at the last layer, in the model's own dtype:

| tensor | shape | dtype | bytes |
|---|---|---|---|
| `residual` | `[1, hidden_size]` | BF16 | `hidden_size × 2` |
| `residual.index` | `[1, 2]` | F32 | the row's `(token, layer)` |
| `logit` | `[1, 3]` | F32 | entropy, perplexity, confidence |

On a 5120-wide model that residual is **exactly 10,240 bytes — 10.0 KiB per
turn**, and it stays 10 KiB whether the turn ran 40 tokens or 400. Verified on
Qwen3.8-27B: a 40-token turn and a 400-token turn both produced 10,896-byte
deposits.

Quantization does not change this. `compressed-tensors` INT4 compresses
*weights*; the residual stream is activations, and stays BF16.

### Every turn is kept

Filenames are `<timestamp>-<request_id>`, so the directory reads as a history
in sort order and a reused request id never overwrites an earlier turn. There
is no rotation or retention policy — the directory grows, by design.

## Flags

| flag | default | what it does |
|---|---|---|
| `--signal-capture-tier` | `off` | what to record. `off`, `logit`, `layer_stats`, `residual_raw`, `full_raw` |
| `--signal-capture-max-tier` | = tier | ceiling for runtime changes; decides which hooks get installed |
| `--signal-capture-dir` | — | where deposits go. Required |
| `--signal-capture-layers` | `last` | `all`, `last`, or `0,31,63` (negatives count from the end) |
| `--signal-capture-tokens` | `last` | `last`, `first`, `mean`, `all` — how a turn's tokens reduce |
| `--signal-capture-layer-step` | `1` | thin the selected layers |
| `--signal-capture-token-step` | `1` | thin the recorded tokens |
| `--signal-capture-dtype` | `native` | `native` keeps BF16; `float32` widens |
| `--signal-capture-max-bytes` | 64 MiB | per-request buffer cap; over it, the deposit is marked `truncated` |
| `--signal-capture-session` | — | label written into deposit metadata |

### Every token instead of every turn

`--signal-capture-tokens all` records one vector per generated token rather
than one per turn. On a 5120-wide model that is 10 KiB × every token — a
400-token turn becomes ~4 MB. Everything else is unchanged.

### The tier ladder

| tier | records | ~per turn, 5120-wide |
|---|---|---|
| `off` | nothing | 0 |
| `logit` | entropy / perplexity / confidence | ~12 B |
| `layer_stats` | per-layer scalar reductions of every signal | ~KB |
| `residual_raw` | the residual stream, raw | 10 KiB |
| `full_raw` | + `attn_norm`, `ffn_norm`, `gate`, `qcur`, `kcur` | ~80 KiB |

`heads` (T3) and `full_raw_attn` (T6) are rejected at startup rather than
silently recording the wrong thing — T3's per-head reduction is unimplemented,
and T6 needs attention weights that fused attention kernels never materialize.

## Reading a deposit

```bash
python -m vllm.signals.view turn.safetensors              # what's inside
python -m vllm.signals.view turn.safetensors -f stats     # SPEC scalar taxonomy
python -m vllm.signals.view turn.safetensors -f json      # the vectors themselves
python -m vllm.signals.view turn.safetensors -f npz -o turn.npz
```

`-f stats` recomputes the per-layer `activation_norm` / `cosine_sim` taxonomy
from the stored raw vectors, so a `residual_raw` deposit can be read as though
it were `layer_stats` without re-running the model.

## Changing capture at runtime

Which hooks exist, and whether the model runs eager, are fixed at load. What
those hooks *record* is not. Launch with a ceiling and start at `off`:

```bash
vllm serve --model <model> \
  --signal-capture-max-tier full_raw \
  --signal-capture-tier off \
  --signal-capture-dir /var/signals
```

Nothing is recorded — the hooks cost one boolean check per layer — until:

```bash
curl -X POST localhost:8001/signals/control -d '{"tier":"residual_raw"}'
curl -X POST localhost:8001/signals/control -d '{"tokens":"all"}'
curl -X POST localhost:8001/signals/control -d '{"layers":"all"}'
curl -X POST localhost:8001/signals/control -d '{"tier":"off"}'
curl localhost:8001/signals/status
```

Requires `VLLM_SERVER_DEV_MODE=1`, alongside vLLM's other control endpoints.
A tier above the startup ceiling is refused, because its hooks were never
installed.

## Injection: choosing the vector to start from

This is the part with the most choices, so they are all explicit.

### 1. Which turn

Each deposit is one turn. Point at the file:

```bash
curl -X POST localhost:8001/signals/inject \
  -d '{"source": "/var/signals/20260829T182233.537-chatcmpl-abe244.safetensors"}'
```

or at startup, `--signal-inject-from <path>`.

### 2. Which vector inside it

| knob | default | meaning |
|---|---|---|
| `signal` | `residual` | which recorded signal to take |
| `layer` | where it was recorded | which layer's row |
| `row` | `-1` | which row, when the deposit holds several. `-1` is the last — for a per-turn deposit, the turn's final state; for a `tokens=all` deposit, the last token |

The layer defaults to the one the vector was recorded at, and you almost always
want that: a residual only means the same thing at the depth it came from.
Injecting a layer-63 vector at layer 12 is not wrong so much as meaningless.

### 3. How it is applied

| knob | values | meaning |
|---|---|---|
| `mode` | `add` (default), `replace` | `add`: `stream += alpha * v`, classic steering. `replace`: the captured state *becomes* the state at that layer |
| `alpha` | `1.0` | scale on the vector |
| `positions` | `first` (default), `all` | `first` seeds each request once — the "starting point" reading. `all` holds the influence for the whole turn |

```bash
# seed each new turn with a captured state, then let the model run free
curl -X POST localhost:8001/signals/inject \
  -d '{"source":"warm.safetensors","mode":"replace","positions":"first"}'

# steer the whole turn, gently
curl -X POST localhost:8001/signals/inject \
  -d '{"source":"polite.safetensors","mode":"add","alpha":0.6,"positions":"all"}'

curl localhost:8001/signals/injection      # what's installed
curl -X POST localhost:8001/signals/inject -d '{}'   # clear it
```

A vector whose width disagrees with the model is refused; one captured from a
different model logs a warning, because residuals are not portable across
models.

### 4. Building a vector rather than replaying one

A single turn's residual carries whatever that prompt happened to be about.
Useful directions are usually built from several:

```bash
# average a set of turns -- cancels what they did not share
python -m vllm.signals.build mean '/var/signals/2026*.safetensors' -o warm.safetensors

# contrast: mean(with) - mean(without), the direction between two sets
python -m vllm.signals.build diff \
  --a 'polite/*.safetensors' --b 'blunt/*.safetensors' \
  -o polite.safetensors --normalize
```

Both write the same `sigcap-v1` shape capture produces, so the output goes
straight into `--signal-inject-from`. `--normalize` scales to unit norm so
`alpha` becomes the whole magnitude.

## Costs

**Capture above `logit` forces eager execution.** Forward hooks are captured
into a CUDA graph and then never re-run on replay, so deposits would silently
go empty after warmup. The patch enables `enforce_eager` and says so in the
log. Injection forces it too, for the same reason.

Measured on Qwen3.8-27B INT4 (GH200, MTP spec decode, 48 prompts, concurrency
8, 256 in / 128 out):

| configuration | output tok/s | mean TPOT |
|---|---|---|
| eager, hooks installed, tier `off` | 181.1 | 39.9 ms |
| eager, recording, `layers=last` | 161.8 | 41.3 ms |
| eager, recording, `layers=all` (64) | 164.3 | 43.9 ms |

The two recording rows are within noise of each other, which is the useful
finding: recording all 64 layers costs about the same as recording one, because
the expense is the hook and the device→host copy per step, not the volume.

Eager also *removes* a large startup cost on this setup: the CUDA-graph path
spends ~17 minutes capturing 83 graph sizes (host-bound, one core pinned),
against ~83 seconds to serve in eager.

## Limitations

- **Pipeline parallelism** disables capture (each rank holds part of the stack).
- **Tensor parallelism** records on rank 0 only. `residual` and the norms are
  replicated and correct; `gate`, `qcur`, `kcur` are column-sharded and would
  be that rank's shard.
- **Prefill is not tapped** — only sampling positions, so the end-of-prompt
  state is not captured directly. `tokens=first` is the closest thing.
- **Capture is server-wide**, not per-request.
- On hybrid stacks, layers with no attention module (GDN linear attention)
  contribute no `qcur`/`kcur`. The residual is unaffected — verified capturing
  all 64 layers of Qwen3.8-27B, none missing.
- Not ported from the taxonomy: KV-cache signals, the speculative/drafter
  bundle, and band aggregation.

## Where the code lives

| file | what |
|---|---|
| `tiers.py` | the T0–T6 ladder and its per-signal drop/stats/raw policy |
| `reductions.py` | torch ports of the SPEC scalar formulas, validated against the numpy reference |
| `deposit.py` | the `sigcap-v1` safetensors writer and the per-turn reductions |
| `capturer.py` | forward hooks, per-step row gathering, per-request deposits |
| `inject.py` | putting a residual back in |
| `build.py` | `mean` / `diff` vector construction |
| `view.py` | reading a deposit back out |
| `config.py` | the resolved worker-side config |

Wired into both GPU model runners, `vllm/config/observability.py` for the
flags, and `vllm/entrypoints/serve/dev/signals/` for the runtime endpoints.
Tests in `tests/signals/`.
