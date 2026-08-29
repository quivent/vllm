# Usage: getting residuals out of Qwen, reading them, and putting them back

Assumes a serving Qwen3.8-27B with the patch. If you need to stand one up, see
`deploy/H200.md`. Everything below is copy-pasteable.

---

## 1. Get residuals

Capture is a server flag, not a per-request one. Add these to `vllm serve`:

```bash
  --signal-capture-backend graph \        # keeps CUDA graphs; costs 1.7%
  --signal-capture-tier residual_raw \    # record the residual, raw
  --signal-capture-layers last \          # layer 63 of 64
  --signal-capture-tokens last \          # one vector per turn
  --signal-capture-dir  $HOME/signals
```

Then just use the model normally:

```bash
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen38",
  "messages": [{"role": "user", "content": "Explain entropy in one sentence."}],
  "max_tokens": 64
}' | jq -r '.choices[0].message.content'
```

Every turn leaves one file:

```bash
ls -l ~/signals/
# 10904  20260829T194551.261-chatcmpl-a0a15b1accf9a731-874fdcaf.safetensors
```

`residual [1, 5120] BF16` = **10,240 bytes**, and it stays 10 KiB whether the
turn ran 40 tokens or 400.

**One catch:** a deposit is flushed when the *next* engine step runs, or at
shutdown. Send a second request, or stop the server, and the last turn appears.

### Variations

| you want | change |
|---|---|
| every token, not one per turn | `--signal-capture-tokens all` (~10 KiB × tokens) |
| the turn's average | `--signal-capture-tokens mean` |
| all 64 layers | `--signal-capture-layers all` (655,360 B/turn) |
| specific layers | `--signal-capture-layers 0,31,63` |
| everything, not just residual | `--signal-capture-tier full_raw` + `--signal-capture-backend hook` (**~4x slower**) |

### Turning capture on and off without a restart

Launch with a ceiling and start off:

```bash
  --signal-capture-max-tier residual_raw --signal-capture-tier off
```

Then (needs `VLLM_SERVER_DEV_MODE=1`):

```bash
curl -X POST localhost:8001/signals/control -d '{"tier":"residual_raw"}'
curl -X POST localhost:8001/signals/control -d '{"tokens":"all"}'
curl -X POST localhost:8001/signals/control -d '{"tier":"off"}'
curl localhost:8001/signals/status
```

---

## 2. View them

```bash
python -m vllm.signals.view ~/signals/<file>.safetensors
```

```
residual        shape=[1, 5120]   BF16   10,240 B  (2.00 B/value)
residual.index  shape=[1, 2]      F32         8 B
metadata: tier=residual_raw token_reduce=last model=...
```

Other formats:

```bash
python -m vllm.signals.view <file> -f stats    # per-layer norms and cosines
python -m vllm.signals.view <file> -f json     # the vectors themselves
python -m vllm.signals.view <file> -f npz -o out.npz
```

`-f stats` gives the SPEC taxonomy recomputed from the raw vector:

```json
{"layer_idx": 63, "activation_norm": 404.14, "cosine_sim": 0.0,
 "mean_abs": 3.92, "max_abs": 95.0}
```

In Python:

```python
from safetensors.torch import load_file
t = load_file("turn.safetensors")
v = t["residual"][0]              # [5120] bf16 - the residual stream
idx = t["residual.index"][0]      # (token, layer) it came from
```

Checked-in examples with rendered `.md` and `.stats.json` alongside the raw
binary are in `samples/`.

---

## 3. Work with Qwen using them

### Build a direction from many turns

One turn's residual carries whatever that prompt was about. Useful vectors come
from several:

```bash
# average a set of turns
python -m vllm.signals.build mean '~/signals/*.safetensors' --layer 63 -o warm.safetensors

# contrast: mean(A) - mean(B), the direction between two behaviours
python -m vllm.signals.build diff \
  --a 'runs/polite/*.safetensors' --b 'runs/blunt/*.safetensors' \
  --layer 63 --normalize -o polite.safetensors
```

`--normalize` makes the vector unit-length so `alpha` below is the whole
magnitude.

### Inject it back

Injection **forces eager** (it rewrites a layer's output, which a CUDA graph
would bake in), so it costs the ~4x. Start the server with
`--signal-inject-from`, or set it live:

```bash
# seed each new turn with a captured state, then let the model run free
curl -X POST localhost:8001/signals/inject -d '{
  "source": "/home/ubuntu/warm.safetensors",
  "mode": "replace", "positions": "first"}'

# or steer the whole turn, gently
curl -X POST localhost:8001/signals/inject -d '{
  "source": "/home/ubuntu/polite.safetensors",
  "mode": "add", "alpha": 0.6, "positions": "all"}'

curl localhost:8001/signals/injection        # what is installed
curl -X POST localhost:8001/signals/inject -d '{}'   # clear it
```

| knob | values | meaning |
|---|---|---|
| `mode` | `add` / `replace` | `add`: `stream += alpha·v`, steering. `replace`: the captured state *becomes* the state at that layer |
| `positions` | `first` / `all` | `first` seeds each request once; `all` holds it for the turn |
| `alpha` | float | scale |
| `layer` | int | defaults to where the vector was recorded — usually leave it |

Then generate normally and compare against the same prompt with injection
cleared. **Injection is unit-tested but has never been run against a real
model** — treat the first results with suspicion, and sanity-check that `alpha:
0` behaves identically to no injection.

### What this is not

This grafts a vector into one layer's output; it does **not** start the model
from that state. Layers below still computed from the real prompt, and the KV
cache for that position was written by that real computation - so later tokens
attend to the original state and the generation drifts back toward it.

Use it as steering along a direction (`build diff`), not as state resumption.
Resuming from a captured state coherently would need the KV cache to agree with
the injected residual, which is not built.

### A caution

A residual only means the same thing at the layer and model it came from.
Injecting a layer-63 vector at layer 12, or a vector from another model, is not
an error the code can always catch — width mismatches are refused, but a
same-width vector from a different model only logs a warning.

---

## 4. Ship them somewhere

```bash
python -m vllm.signals.r2 --dir ~/signals watch     # continuous upload
python -m vllm.signals.r2 --dir ~/signals catalog   # what has been shipped
```

Objects land at `signals/<model>/<session>/<day>/`, with a queryable JSONL
catalog at `signals/_catalog/<day>.jsonl`. Set `--signal-capture-session` per
machine to keep corpora separable.

---

## 5. Check it is really working

```bash
python -m vllm.signals.bench --port 8001 --dir ~/signals
```

Runs a real turn, finds *that request's* deposit, opens it, and reports the
vector's shape, bytes and L2 norm — because every other indicator can read green
while capture silently writes nothing. Expect `11 ok, 1 warn, 0 failed`; the
warn is a known false negative in its speculative-decoding detector.
