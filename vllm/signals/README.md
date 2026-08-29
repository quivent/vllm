# Inference-signal capture and injection

> **Status, 2026-08-29 — graph backend + MTP speculative decoding: WORKING and
> MEASURED on Qwen3.8-27B-INT4 (H200).** Residual captured with CUDA graphs on,
> eager not forced, MTP untouched, at a **1.7% throughput cost**
> (723.8 vs 736.1 tok/s). Proof below.

A private patch to vLLM that does two things:

- **Capture** — write the residual stream out of a live forward pass, one file
  per turn, without changing what the model produces.
- **Inject** — put a captured residual back in, so a generation starts from, or
  is steered by, a state the model was in before.

Ported from the [`signal-extraction`](https://github.com/quivent) taxonomy; the
on-disk format is the same `sigcap-v1` safetensors layout that repo's C++
recorder writes, so deposits from either engine read with the same tools.

**Task-oriented guide: [USAGE.md](USAGE.md)** — get residuals, view them, build
and inject vectors. **Handover: [HANDOVER.md](HANDOVER.md)**. **Deploy:
[deploy/H200.md](deploy/H200.md)**.

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

## Shipping deposits to R2

A sidecar, outside the engine, so a slow upload can never slow a forward pass:

```bash
python -m vllm.signals.r2 --dir /var/signals sync     # upload the backlog, exit
python -m vllm.signals.r2 --dir /var/signals watch    # and keep going
python -m vllm.signals.r2 --dir /var/signals catalog  # what has been shipped
```

Uploads go through `gemstone store push` by default, reusing the credentials
and bucket that CLI already has, so this needs no secrets of its own. `--backend
s3` uses boto3 against the R2 endpoint from `~/.council-r2.env` instead, which
is faster for a large backlog and is the only backend that sets object
metadata.

Objects are grouped so a day's turns, or one experiment's, come back with a
single prefix:

```
signals/<model>/<session>/<YYYY-MM-DD>/<timestamp>-<request_id>.safetensors
signals/_catalog/<YYYY-MM-DD>.jsonl
```

Every upload appends a catalog row — key, sha256, bytes, model, session,
request id, tier, token reduction, captured positions, truncated flag, and both
timestamps — to a daily JSONL shard held locally and mirrored to the bucket. The
set of captured turns can then be queried without listing several hundred
thousand objects.

Uploads are idempotent: shipped basenames are recorded in `.r2-uploaded.json`
in the capture directory, so a restart resumes rather than re-uploading, and a
failed upload is simply left unrecorded for the next pass to retry. Deposits
are written atomically by the capture side, so the watcher never sees a
half-written file. `--delete-after` removes each local copy once it is safely
in the bucket; by default nothing is deleted.

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

## Verified configuration

Qwen3.8-27B-INT4 on an H200 NVL, graph backend, MTP speculative decoding:

```
$ curl localhost:8001/signals/status
{"enabled":true,"backend":"graph","tier":"residual_raw","max_tier":"residual_raw",
 "tokens":"last","layers":[63],"num_layers":64,"recording":true}

$ curl localhost:8001/server_info | grep -o "SpeculativeConfig(...)"
SpeculativeConfig(method='mtp', num_spec_tokens=3, ...)

grep -c "Cudagraph is disabled under eager mode"  ->  0
Capturing CUDA graphs (PIECEWISE): 100%
Capturing CUDA graphs (FULL):      100%
```

`python -m vllm.signals.bench --port 8001 --dir ~/signals`:

```
[  ok  ] patched vllm         /signals/status present
[  ok  ] capture configured   backend=graph tier=residual_raw tokens=last
[  ok  ] layers tapped        [63] of 64
[  ok  ] recording            yes
[  ok  ] cuda graphs          compiled
[  ok  ] inference            64 tokens in 0.85s = 75.4 tok/s (single stream)
[  ok  ] residual saved       [1, 5120] BF16 = 10,240 B (10.0 KiB)
[  ok  ] deposit metadata     tier=residual_raw reduce=last positions=85
[  ok  ] residual is real     L2 norm 349.9, all finite=True
```

Two things to read carefully in that output.

**`positions=85` for 64 completion tokens.** That ratio is speculative decoding
at work: MTP verifies draft positions, so capture sees more sampling positions
than the request emitted tokens. It is the reason the metadata field is called
`num_captured_positions` and not `num_tokens`.

**The bench's own "speculative decoding: not enabled" is a false negative.** Its
`/server_info` parser misses vLLM's `SpeculativeConfig(...)` repr; the config is
there, confirmed directly above. The check needs fixing; the deployment does not.

### Getting there took three fixes, each hidden behind the last

`use_aux_hidden_state_outputs` is overloaded — three components read it and want
different things:

1. the runner's EAGLE-3 setup re-derived the auxiliary layers straight after
   capture set them, silently replacing layer 63 with EAGLE-3's defaults;
2. the **speculator** branches on that flag and takes the EAGLE-3 path, so MTP
   died on `assert self.method == "eagle3"`;
3. the **cudagraph manager** allocates output buffers with
   `empty_like(model_output)` and only unpacks a tuple when *its* copy of the
   flag is set, so it choked on the tuple capture had asked the model for.

Capture now owns its own flag, tells the graph manager to expect the tuple, and
leaves the speculator's view untouched.

## Costs — measured

**With the `graph` backend, capture costs about 1.7%.** With the `hook` backend
it costs about 4x. That is the whole reason the graph backend exists.

Qwen3.8-27B-INT4, MTP with 3 draft tokens, 48 prompts, concurrency 8,
256 in / 128 out.

**H200 NVL — graph backend, CUDA graphs on throughout:**

| configuration | output tok/s | mean TPOT | cost |
|---|---|---|---|
| graph backend, capture **off** | 736.1 | 9.25 ms | — |
| graph backend, **recording** | **723.8** | 9.31 ms | **1.7%** |

**GH200 — hook backend, which forces eager:**

| configuration | output tok/s | mean TPOT | vs graphs |
|---|---|---|---|
| CUDA graphs, no capture | 737.6 | 9.3 ms | — |
| eager, hooks idle | 181.1 | 39.9 ms | 0.25x |
| eager, recording | 161.8 | 41.3 ms | 0.22x |

Read the two tables together. Nearly all of the hook backend's loss is *eager*,
not the recording — hooks idle already cost 4x. The graph backend removes that
entirely by reading the residual as a model output rather than reaching in with
a hook, leaving only the aux computation and one device→host copy per step:
1.7%.

Note the tok/s figures are aggregate across 8 concurrent streams. Single-stream
decode is the TPOT column — ~108 tok/s per stream at 9.3 ms.

Practical consequences:

- Use `graph` in front of traffic. Capture is effectively free.
- Use `hook` only for signals other than the residual, and treat it as a
  capture *window* rather than a steady state.
- The `logit` tier installs no hooks and never forces eager, whichever backend.
- Startup differs sharply by host, not by capture: 83 CUDA graphs capture at
  ~1.3/s on the H200's x86 cores and ~1 per 12.3 s on Grace — ~4 minutes versus
  ~17. It is host-bound single-thread work.

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
