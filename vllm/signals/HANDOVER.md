# Handover

Written 2026-08-29 for whoever picks this up next, assuming no context.

This branch adds **inference-signal capture** to vLLM: it writes the residual
stream out of a live forward pass, one file per turn, without changing what the
model produces. It also injects a saved residual back in. Everything is on
`quivent/vllm@signal-extraction-capture`, additive to upstream `cacc429f62`.

Read `README.md` for how to use it. This file is what you need to *trust* it:
what is proven, what is not, and every trap that cost time.

---

## 1. Status — proven vs assumed

| claim | status | evidence |
|---|---|---|
| Residual captured, 10,240 B/turn on a 5120-wide model | **proven** | Qwen3.8-27B, both backends |
| Deposit size flat in turn length | **proven** | 40-token and 400-token turns both 10,896 B |
| All 64 layers captured on a hybrid stack | **proven** | none missing, GDN layers included |
| `graph` backend keeps CUDA graphs (no eager) | **proven** | Qwen3-0.6B and 27B |
| `graph` backend + MTP speculative decoding | **proven** | H200, see §2 |
| Hook backend costs ~4x throughput | **proven** | 737.6 → 161.8 tok/s, GH200 |
| `graph` backend's throughput cost | **measured: 1.7%** | 723.8 vs 736.1 tok/s, H200 |
| Injection changes generation | **unit-tested only** | never run on a real model |
| `graph` backend with EAGLE-3/DFlash drafters | **not tried** | code path exists, see §4 |
| Tensor parallelism | **not tried** | rank 0 only; sharded signals would be partial |

**That gap is now closed.** The graph backend costs **1.7%** (723.8 tok/s
recording vs 736.1 with capture off, both with CUDA graphs on). The hook
backend costs ~4x. Capture in front of traffic is a solved problem; use
`--signal-capture-backend graph`.

---

## 2. The verified configuration

Qwen3.8-27B-INT4, H200 NVL, graph backend, MTP with 3 draft tokens:

```
{"backend":"graph","tier":"residual_raw","layers":[63],"recording":true}
SpeculativeConfig(method='mtp', num_spec_tokens=3, ...)
grep -c "Cudagraph is disabled under eager mode"  ->  0
Capturing CUDA graphs (PIECEWISE): 100%   (FULL): 100%

residual saved     [1, 5120] BF16 = 10,240 B
residual is real   L2 norm 349.9, all finite=True
```

Reproduce with one command:

```bash
python -m vllm.signals.bench --port 8001 --dir ~/signals
```

It exits non-zero on failure, so it can gate a deploy. **Two of its lines lie —
see §5.**

---

## 3. The deepest thing to understand: one overloaded flag

`use_aux_hidden_state_outputs` is read by **three** components that want
different things from it. Getting graph capture working alongside speculative
decoding meant separating them, and each fix was hidden behind the previous one:

1. **The runner's EAGLE-3 setup** re-derives the auxiliary layers right after
   capture sets them, silently replacing layer 63 with EAGLE-3's defaults.
   Symptom: capture records a layer you did not ask for, or nothing.
2. **The speculator** branches on the flag and takes the EAGLE-3 path.
   Symptom: `assert self.method == "eagle3"` — MTP dies.
3. **The cudagraph manager** allocates output buffers with
   `empty_like(model_output)` and only unpacks a tuple when *its* copy of the
   flag is set. Symptom:
   `TypeError: empty_like(): argument 'input' must be Tensor, not tuple`.

Capture now owns a separate `_signal_aux_outputs` flag, tells the graph manager
to expect the tuple, and leaves the speculator's view untouched. **If you touch
the aux path, re-read this section** — the three consumers are in different
files and nothing links them.

### Why the graph backend exists at all

A CUDA graph is a *recording of GPU work*. On replay, no Python executes. A
forward hook is Python, so it never fires — the residual is right there in
memory and nothing copies it out. Two ways round it: put the copy inside the
recording, or have the model *return* the value so it is a graph output by
construction. The second is what vLLM's EAGLE-3 aux hidden states already are,
which is why the graph backend reuses them rather than inventing a mechanism.

The corollary: **capture cannot be switched on at runtime under CUDA graphs
using hooks**, because a recorded graph is immutable. That is why
`--signal-capture-max-tier` (the hook ceiling) forces eager even when the tier
is `off`.

---

## 4. Speculative decoding compatibility

| drafter | graph backend |
|---|---|
| **MTP** | works — MTP never claims the aux list |
| none | works |
| **EAGLE-3 / DFlash / DSpark** | only if the drafter already publishes the layer you want |

Those drafters own the aux list and index into it positionally, so it cannot be
widened underneath them. If your layer is one they already emit, capture reads
it for free and leaves them alone; otherwise it refuses and names the layers
they do publish. **DFlash2 is the case to watch** — it is in that list.

---

## 5. Known-wrong things

- **`bench` reports "speculative decoding: not enabled" when it is.** Its
  `/server_info` parser misses vLLM's `SpeculativeConfig(...)` repr. Every other
  line of that tool is trustworthy. Fix the parser.
- **`num_captured_positions` exceeds completion tokens under spec decode** (85
  vs 64). That is correct behaviour — capture sees verified draft positions —
  but it reads like a bug if you do not know.
- **The R2 `models/` prefix is content-addressed** (`blobs/refs/trees`), not a
  servable directory. Pulling it yields 19 GB with no `config.json`. `refs/main`
  is a 40-byte pointer into `trees/`; a materializer is ~20 lines and nobody
  wrote it. Use `hf download` or `gemstone models download --local-dir`.
- **`artifacts/vllm-wheels/` is empty on purpose.** A wheel was published keyed
  under the wrong commit — `VLLM_USE_PRECOMPILED` bakes the repo's Python into
  the wheel, so its filename said `g1047d7fbc` while the key said otherwise, and
  it would have installed code without the graph backend. Deleted rather than
  left wrong. Wheels are per-*architecture* too: aarch64 will not run on x86_64.
- **Tier `heads` (T3) and `full_raw_attn` (T6) are rejected at startup**, not
  silently degraded. T3's per-head reduction is unimplemented; T6 needs
  attention weights a fused kernel never materializes.

---

## 6. Machine state as left

### GH200 (this box, aarch64, driver 570 — needs the compat shim)
- Idle, no server. Patched vLLM editable at `~/vllm`.
- `~/start-qwen38-patched.sh` — hook backend
- `~/restore-qwen38-original.sh` — the original root/system vLLM, byte-identical

### H200 NVL `154.54.100.100` (x86_64, driver 580 — CUDA 13 native)
- **Serving** `qwen38` on :8001, graph backend + MTP, capture recording to
  `~/signals`. Launcher: `~/start-qwen-graph.sh`.
- **Benchmarked:** `bash ~/h200bench.sh` → recording 723.8 tok/s vs 736.1 with
  capture off, both with CUDA graphs on. 1.7%.
- **A Gemma-4-31B governor stack was stopped to free the GPU.** Restore:
  ```bash
  sudo docker start vllm-gemma4-fp8-h200
  sudo systemctl start gemstone-governor-agentic-8000.service
  ```

---

## 7. Environment traps, by symptom

Run `deploy/preflight.sh` (`--fix` to remediate). Every row below is a check in
it, because every one cost real time. They all surface *indirectly*, minutes
into a startup, as something that looks unrelated.

| you see | it is |
|---|---|
| `driver too old (found version 12080)` | CUDA 13 build on an r570 driver; install the arch-correct `cuda-compat-13-0` (**sbsa** for aarch64, **x86_64** otherwise) and put it on `LD_LIBRARY_PATH` |
| `Could not find nvcc and cuda_home doesn't exist` | no CUDA toolkit; the venv's pip CUDA package has one — set `CUDA_HOME` to `site-packages/nvidia/cu13` |
| `cannot execute 'cc1plus'` → `nvcc fatal` | no `g++`. nvcc shells out to the host compiler; gcc alone is not enough |
| still `cc1plus` after installing g++ | poisoned flashinfer build cache — `rm -rf ~/.cache/flashinfer`; if it persists, `VLLM_USE_FLASHINFER_SAMPLER=0` (what the H200 needed) |
| `FileNotFoundError: 'ninja'` | flashinfer JIT needs it |
| `PermissionError` under `~/.cache/vllm` | root-owned dirs from a previous root run; set `VLLM_CACHE_ROOT` |
| `metadata not found for cu129` | pin `VLLM_PRECOMPILED_WHEEL_VARIANT=cu130`; auto-detection probes a variant that is not published |
| version reads `0.26` not `0.28` | clone lacks tags; setuptools_scm derives from them. Cosmetic. `git fetch --tags` |
| `/signals/status` 404s | `VLLM_SERVER_DEV_MODE=1` unset |
| GPU reclaimed under you | a docker restart policy; `sudo docker ps` / `stop`, and the systemd gateway unit |
| 19 GB model with no `config.json` | the R2 CAS layout — see §5 |
| graph capture taking ~17 minutes | host-bound single-thread work. 12.3 s/graph on Grace vs 1.33 graphs/s on the x86 H200 — a ~16x difference. Not a hang |

---

## 8. What to do next, in order

1. **Fix the bench spec detector** (§5). A verification tool that lies about one
   field undermines the eleven it gets right.
2. **Injection as built is a graft, not a starting state — this is the gap.**
   `mode=replace` swaps layer 63's output while layers 0-62 still computed from
   the real prompt. The first token is coherent (final norm and lm_head see the
   injected vector) but **the KV cache for that position was written by the real
   computation at every layer**, so every later token attends to the original
   state, not the injected one. The generation drifts back.

   What is actually wanted is the residual as a genuine *starting state*: the
   model begins from it and everything downstream is consistent. That needs the
   KV cache to agree with the injected state, not just one layer's output. It is
   a deeper change than what is here, and nobody has designed it yet.

   Treat the current `add`/`replace` as activation *steering* - useful for
   nudging along a contrast direction (`build diff`), not for resuming a state.

3. **Try the existing injection on a real model anyway.** It is unit-tested and has never touched a
   GPU. `POST /signals/inject` with a sample from `samples/`.
4. **Materialize the R2 model copy** (§5) so provisioning stops depending on HF.
5. Then, if wanted: capture the MTP drafter's own residual (its
   `Qwen3_5MultiTokenPredictor.layers` returns the same `(hidden, residual)`
   pair, and `find_decoder_layers` deliberately skips it by taking the largest
   stack); per-request capture control; prefill positions.

## 9. Where things live

| path | what |
|---|---|
| `vllm/signals/README.md` | usage, flags, format, costs |
| `vllm/signals/HANDOVER.md` | this file |
| `vllm/signals/deploy/preflight.sh` | the environment profile, `--fix` capable |
| `vllm/signals/deploy/README.md` | H100/H200 redeploy spec |
| `vllm/signals/samples/` | two real deposits, raw + rendered |
| `vllm/signals/bench.py` | the one-command verifier |
| `tests/signals/` | 94 tests |
