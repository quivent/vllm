# Redeploying the signals patch + Qwen3.8-27B on H100 / H200

What transfers from the GH200 build, what does not, and what to pull from R2.

## What does not transfer

**The wheel, across architectures.** The artifact built on the GH200 is
`vllm-…-cp312-cp312-linux_aarch64.whl`. H100 and H200 boards sit in **x86_64**
hosts, so that wheel is unusable there — the GPU is the same Hopper `sm_90`, but
the *host* architecture is not.

Wheels *are* cached in R2, keyed by commit and host architecture:

```
artifacts/vllm-wheels/<git-sha>/<uname -m>/vllm-….whl
```

The script pulls a matching wheel when one exists and installs that, which pins
the exact artifact instead of re-resolving against `wheels.vllm.ai` — and after
a build on a new architecture it publishes the result, so the next box of that
kind skips the build. A cached wheel gives a non-editable install: right for a
deployment, wrong for development, where you want `pip install -e .` so repo
edits take effect.

**The CUDA forward-compat shim.** The GH200 needed
`cuda-compat-13-0_…_arm64.deb` from the **sbsa** repo. On x86_64 the equivalent
lives under `ubuntu2204/x86_64`. Only needed at all if the driver is older than
the CUDA this vLLM was built against — see below.

## What does transfer

| R2 prefix | what | why pull it |
|---|---|---|
| `artifacts/vllm-wheels/<sha>/<arch>/` | the built wheel | pins the artifact, skips the build |
| `artifacts/vllm-compile-cache/<arch>/` | torch.compile artifacts | skips ~35 s of compile per boot |
| `signals/<model>/<session>/<day>/` | captured residuals | the corpus, for building injection vectors |
| `signals/_catalog/*.jsonl` | the catalog | query the corpus without listing it |

The model is **not** pulled from `models/` as a prefix: the bucket stores it
content-addressed (`blobs/refs/trees`), which is not a servable directory —
pulling it yields 19 GB with no `config.json`. Acquire it through
`gemstone models download --local-dir`, which materializes an exact revision and
registers the placement.

The compile cache is keyed by GPU arch — `h100-sm90a` already exists. H100,
H200 and GH200 are all `sm_90`, but the cache also keys on the CUDA and torch
build, so treat a mismatch as a cache miss rather than an error: a wrong cache
costs a recompile, never a wrong result.

## The CUDA version trap

This vLLM commit is a **CUDA 13.0** build (`VLLM_MAIN_CUDA_VERSION = 13.0`,
`torch==2.13.0+cu130`). CUDA 13 needs an **r580+** driver.

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

- **≥ 580** — nothing to do.
- **< 580** (e.g. the 570.148.08 on the GH200) — CUDA will not initialise at all.
  Either update the driver, or install the forward-compatibility libraries.
  Hopper is a datacenter part, so forward compat is supported:

  ```bash
  curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb
  dpkg -x cuda-compat-13-0_*.deb /tmp/compat
  mv /tmp/compat/usr/local/cuda-13.0/compat ~/cuda-compat-13.0
  export LD_LIBRARY_PATH="$HOME/cuda-compat-13.0:$LD_LIBRARY_PATH"
  ```

  Every process that touches CUDA needs that on `LD_LIBRARY_PATH`, the server
  included. `deploy.sh` handles this.

## Memory sizing

`--gpu-memory-utilization` is a fraction of **total** board memory. The 27B INT4
needs ~19.6 GiB of weights plus ~1.9 GiB of peak activation; the rest is KV
cache.

| board | HBM | at 0.85 | KV cache left | notes |
|---|---|---|---|---|
| GH200 96GB | 95.6 GiB | 81 GiB | ~59 GiB | the reference config |
| **H200** | 141 GiB | 120 GiB | ~98 GiB | keep 0.85; more KV than the GH200 |
| **H100 80GB** | 79.6 GiB | 68 GiB | ~46 GiB | 0.85 is fine; `--max-num-seqs 512` will simply cap out earlier |

`--max-model-len 131072` is unchanged in all three — it bounds a single
sequence, not the cache. On the H100 the smaller cache means fewer concurrent
long sequences, not shorter ones.

## Capture backend on the target

| backend | throughput | works on Qwen3.5 today |
|---|---|---|
| `hook` | **~0.22x** — forces eager | yes |
| `graph` | full speed, keeps CUDA graphs | **yes, verified with MTP** |

Deploy with `graph`: verified on Qwen3.8-27B-INT4 with MTP on an H200, CUDA
graphs on and eager not forced. `hook` is the fallback for signals other than
the residual, and costs ~4x.

One environment note from that box: flashinfer JIT-compiles its sampler through
nvcc into the host gcc, which failed with `cannot execute 'cc1plus'` even after
installing g++. `VLLM_USE_FLASHINFER_SAMPLER=0` sidesteps it and is set in the
launcher; the sampler is not what capture touches.

Measured on the GH200 (48 prompts, concurrency 8, 256 in / 128 out, MTP 3):

```
CUDA graphs, no capture      737.6 tok/s     9.3 ms TPOT
eager, hooks idle            181.1 tok/s    39.9 ms TPOT
eager, recording             161.8 tok/s    41.3 ms TPOT
```

Expect the same *ratio* on H100/H200; the absolute numbers will differ.

## Running it

```bash
./deploy.sh --gpu h200                 # provision, serve, verify
./deploy.sh --gpu h100 --capture off   # serve without capture
./deploy.sh --gpu h200 --pull-signals  # also pull the existing corpus
```

The script is idempotent: each step checks before doing. See `--help`.

## Afterwards

Ship new captures back to the same bucket:

```bash
python -m vllm.signals.r2 --dir /var/signals watch &
```

Objects land under `signals/<model>/<session>/<day>/`, so give each machine its
own `--signal-capture-session` (the script defaults it to the hostname) and the
corpus stays separable per box.
