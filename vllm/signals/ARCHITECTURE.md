# Residual-as-carrier: what is proven, why it fails, and three ways forward

2026-08-30. H200 NVL, Qwen3.8-27B-INT4, `quivent/vllm@signal-extraction-capture`.

Written to be implemented from. Every claim is marked **measured** or
**inferred**. Two of my inferences today were wrong, so treat the unmarked
reasoning as a hypothesis to falsify, not a design to trust.

---

## 1. What is measured

| claim | evidence |
|---|---|
| Capture works, graph backend, 1.7% cost | 64 layers available, layer 63 tapped, ~670 deposits |
| A deposit is 10,240 B, flat in turn length | `[1,5120]` BF16, `reduce="last"` replaces in place |
| Runtime injection on a **graph-mode** engine is a silent no-op | `applied:0` after real traffic; output byte-identical |
| `--signal-inject-from` at boot forces eager and injection fires | `applied:1` per request, output changes |
| Eager costs ~1.6x, not the 4x the docs imply | 140.9→85.3 (c1), 950.3→563.4 (c8), 2315→1678.6 (c32) tok/s |
| MTP is unaffected by any of this | `SpeculativeConfig(mtp, 3)` alive in every run |
| A seeded state moves the **first token** and nothing after | `" autom"` deposit → first token `"automaton"` |
| `positions:all` + all-layer write is degenerate | 25,408 applications → `" may may may may…"` forever |
| A layer-0 write is legible and in-distribution | `applied:1`, output diverges, no collapse |
| **One position does not restore a discourse** | 5-turn discourse, seeded probe: *"references earlier conversation not provided"*, zero referents |

**Capacity is not the constraint.** 5120 × BF16 = 81,920 bits. At
log2(248,320) = 17.92 bits/token, a 200-token turn is 3,584 bits. The vector
holds **22.9x** the raw entropy of the thing it failed to carry, and could in
principle address ~4,570 tokens. Any explanation of the failure that appeals to
"not enough room" is wrong.

## 2. Why it fails

Three facts that together are sufficient, and none of which is about the KV
cache being sacred:

1. **RMSNorm discards magnitude.** Every layer normalises the stream before it
   computes. An injected vector with L2 349.9 does not overpower a stream with
   L2 ~5. Only *direction* survives, so a seeded position contributes one
   direction, not a loud one. (**inferred**, but consistent with every run:
   the effect is always visible and always small.)
2. **One position is one position.** The seeded row sits among ~20 real prompt
   positions that literally say *"user says Continue."* Attention weights it
   like any other token. It moved the first token and lost the argument.
   (**measured**: first token always changes; nothing downstream does.)
3. **Nothing was ever trained to unpack a state.** The model can read a
   *token's* embedding at layer 0 because it has seen billions of them. It has
   never seen a compressed conversation at layer 0 and has no learned operator
   for expanding one. Information being present is not the same as being
   legible. (**inferred**, and the crux.)

Point 3 is why "the residual has 23x the entropy" and "the residual doesn't
work" are both true at once. The bits are there. The decoder isn't.

## 3. Three designs

### A. Residual trace injection (span)

**The idea.** Stop injecting one vector. `--signal-capture-tokens all` already
records *every* position of a turn — the `graphcache` deposit holds **341
rows**. Inject row *i* at position *i* across a span of prefill positions. The
model then computes K/V for all of them itself, at every layer,
in-distribution, with no cache surgery. The residual is genuinely the carrier
and nothing is unrolled into tokens.

**Why it might work where a single vector cannot.** It removes failure 2
entirely: the state is now N positions against N, not 1 against 20. It does not
address failure 3 — the model still has no trained operator for reading
layer-63 states as layer-0 inputs — but a *sequence* of them is far closer to
the manifold the model actually produced than any single point on it.

**Why it might not.** A layer-63 residual re-entered at layer 0 is
out-of-distribution *as an input*, and doing it 341 times may compound rather
than accumulate. This is the honest risk and the reason to build it as an
experiment with a falsifier, not as a feature.

**Implementation.**

| where | change |
|---|---|
| `vllm/signals/inject.py` | `InjectionSpec.vector: Tensor` → `[N, hidden]`. `validate` checks `shape[-1]`, not `numel`. |
| `vllm/signals/inject.py` | `load_vector` → `load_matrix(path, signal, layer, rows=slice)`, returning the ordered rows for one layer. Rows are already indexed by `(token, layer)` in `residual.index`. |
| `vllm/signals/inject.py` | `SignalInjector.apply` writes `updated[rows[i]] = v[i]` — a row-to-position map, not a broadcast. |
| `vllm/signals/capturer.py` | The injector is armed with **sampling** rows (`begin_step(row_req_ids, row_index)`, called at `capturer.py:350/354/401`). A span needs **prefill** positions. Arm it from the runner's input positions for the request's first forward, i.e. add a `begin_prefill(req_id, positions)` alongside `begin_step`. |
| `vllm/v1/worker/gpu_model_runner.py` | Call that new arm point where the prefill batch's positions are known. This is the only file outside `signals/` that needs real work. |
| `api_router.py`, `gpu_worker.py` | Pass `rows` (a slice or count) the way `at` is passed now. |

**Falsifier.** `./discourse.py probe` must name the committed topic. Nothing
less counts — not "sounds thematically related", not "mentions dinosaurs".

**Effort.** A day. The `gpu_model_runner` arming is the only unknown.

### B. Learned state → prefix projector

**The idea.** Accept failure 3 and fix it directly. Train a small module
`P: R^5120 → R^{k × 5120}` that maps one captured residual to *k* soft prompt
embeddings prepended at layer 0. Freeze the model; train only `P`, on pairs of
(conversation, its final residual) with the objective that generation from
`P(v)` reproduces generation from the real conversation.

**Why this is the one that actually works.** This is prefix-tuning with a
state-derived prefix. It is the only design here that creates the missing
decoder rather than hoping one exists. The bits are already there (23x
headroom); `P` is what reads them.

**Why you may not want it.** It requires training, a dataset of captured
conversations (you have ~670 deposits and the tooling to mint more), and `P` is
model- and layer-specific — a new checkpoint means retraining.

**Implementation.** Out-of-tree first: capture `(residual, transcript)` pairs
with `./residual capture --tokens all`, train `P` offline, then serve it as a
`--signal-inject-projector <path>` that expands the vector before injection.
The injection path itself needs no change beyond accepting `[k, hidden]`, which
design A already gives you. **A and B compose: build A, and B becomes a weight
file.**

### C. Compressed KV restore

**The idea.** The baseline you asked me to stop treating as god, stated
honestly so the comparison is fair. Persist the conversation's KV, evict/compress
it (H2O, SnapKV, low-rank), restore on load.

**Why it is in this document.** It is the only design that is *known* to
reproduce a conversation, because it restores the thing the model actually
reads. Whatever A or B achieve should be measured against it, or you will not
know whether the residual path is good or merely non-zero.

**Why it is not what you want.** Storage is O(positions × layers × kv_heads ×
head_dim) — for this model, ~262 KB per position in fp8 against 10 KB for a
whole turn's residual. That ratio is the entire case for the residual path.

## 4. Recommendation

Build **A**, measure with `discourse.py probe`, and if the trace alone does not
name the committed topic, go to **B** — A is B's serving path, so nothing is
wasted. Keep **C** as the control that tells you what "working" looks like.

Do not build A expecting it to work. Build it because it is cheap, it is the
natural completion of what the capture side already records, and its failure
would be informative: if 341 positions of real trajectory cannot carry the
conversation, then failure 3 is the whole story and only a trained decoder will
do.

## 5. Tooling

`~/qwen38/residual` — engine-direct, no gemstone dependency:

```bash
./residual capture "prompt" --name warm --tokens all --max-tokens 256
./residual rows warm [--all]          # every (row, token, layer) - 341 for a 200-token turn
./residual list [filter]
./residual seed warm --mode state|replace|add --at 0 --row -1
./residual ab "prompt" --residual warm --mode replace --at 0
./residual status                     # warns if the engine is graph-mode (injection is a no-op)
./residual clear
```

`--layer` picks which recorded row to load; `--at` picks which layer to write
it into. They are separate questions and were conflated until today.

`~/qwen38/discourse.py` — the honest test:

```bash
./discourse.py run --name dino          # 5-turn discourse, captures the end state
./discourse.py probe --residual dino --at 0 --mode replace
```

The probe asks the model to name the topic it committed to defending. It
cannot be passed by agreement, and the answer is checked for discourse
referents. Control output is `"references earlier conversation not provided"` —
that is the bar to beat.

**Engine posture.** Injection requires `--signal-inject-from` at boot (forces
eager, ~1.6x). `~/qwen38/profile-inject-state.sh` serves that;
`profile-fp8-triton.sh` is the fast graph build with no injection.
