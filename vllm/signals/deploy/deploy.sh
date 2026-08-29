#!/usr/bin/env bash
# Redeploy the signals patch + Qwen3.8-27B on an H100 or H200.
# Idempotent: every step checks before doing. See README.md for the reasoning.
set -euo pipefail

GPU=""
CAPTURE="hook"
TIER="residual_raw"
PULL_SIGNALS=0
PORT=8001
BRANCH="signal-extraction-capture"
REPO="${REPO:-$HOME/vllm}"
MODELS="${MODELS:-$HOME/models}"
SIGNALS="${SIGNALS:-$HOME/signals}"
SESSION="${SESSION:-$(hostname -s)}"
SERVE=1

usage() {
  sed -n '2,4p' "$0"
  cat <<'USAGE'

  --gpu h100|h200        required; sets memory sizing
  --capture hook|graph|off   capture backend (default hook; off = full speed)
  --tier TIER            capture tier (default residual_raw)
  --pull-signals         also pull the existing corpus from R2
  --port N               serve port (default 8001)
  --no-serve             provision only, do not start the server
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --capture) CAPTURE="$2"; shift 2;;
    --tier) TIER="$2"; shift 2;;
    --pull-signals) PULL_SIGNALS=1; shift;;
    --port) PORT="$2"; shift 2;;
    --no-serve) SERVE=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage; exit 2;;
  esac
done
[[ -n "$GPU" ]] || { echo "--gpu is required" >&2; usage; exit 2; }

say() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mfatal:\033[0m %s\n' "$*" >&2; exit 1; }

case "$GPU" in
  h100) GPU_UTIL=0.85; ARCH_TAG="h100-sm90a";;
  h200) GPU_UTIL=0.85; ARCH_TAG="h200-sm90a";;
  *) die "--gpu must be h100 or h200";;
esac

# ── 1. host checks ───────────────────────────────────────────────────────────
say "host"
HOST_ARCH="$(uname -m)"
[[ "$HOST_ARCH" == "x86_64" ]] || echo "  note: host is $HOST_ARCH, not x86_64 - the CUDA-compat package below assumes x86_64"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"

# ── 2. CUDA 13 forward compatibility ─────────────────────────────────────────
# This vLLM is a CUDA 13.0 build (torch 2.13.0+cu130), which needs an r580+
# driver. Hopper is a datacenter part, so an older driver can be bridged with
# the forward-compat libraries rather than upgraded.
COMPAT="$HOME/cuda-compat-13.0"
if (( DRIVER >= 580 )); then
  say "driver $DRIVER supports CUDA 13 natively"
else
  say "driver $DRIVER is older than CUDA 13 needs; using forward-compat libraries"
  if [[ ! -f "$COMPAT/libcuda.so.1" ]]; then
    DEB=cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb
    BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/$DEB" "$BASE/$DEB" || die "could not fetch $DEB"
    dpkg -x "$tmp/$DEB" "$tmp/x"
    mv "$tmp/x/usr/local/cuda-13.0/compat" "$COMPAT"
    rm -rf "$tmp"
  fi
  export LD_LIBRARY_PATH="$COMPAT:${LD_LIBRARY_PATH:-}"
fi

# ── 3. repo + patched vLLM ───────────────────────────────────────────────────
say "vllm ($BRANCH)"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/quivent/vllm "$REPO"
fi
git -C "$REPO" fetch origin "$BRANCH" 2>/dev/null || git -C "$REPO" fetch fork "$BRANCH"
git -C "$REPO" checkout "$BRANCH"
git -C "$REPO" pull --ff-only 2>/dev/null || true

cd "$REPO"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[[ -d .venv ]] || uv venv --python 3.12
export PATH="$REPO/.venv/bin:$PATH"

if ! .venv/bin/python -c "import vllm, torch" 2>/dev/null; then
  # The precompiled variant must match this commit's CUDA (13.0). Auto-detection
  # probes for a cu129 build that is not published, so pin it.
  VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 uv pip install -e .
fi
uv pip install ninja >/dev/null 2>&1 || true   # flashinfer JITs at startup
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('  torch', torch.__version__, torch.cuda.get_device_name(0))"

# ── 4. artifacts from R2 ─────────────────────────────────────────────────────
say "artifacts from R2"
# gemstone is the preferred path, but a bare box may only have the credentials
# file, so fall back to boto3 against the R2 endpoint.
if command -v gemstone >/dev/null && gemstone r2 check --json >/dev/null 2>&1; then
  R2_MODE=gemstone
elif [[ -f "$HOME/.council-r2.env" ]]; then
  R2_MODE=s3
  uv pip install boto3 >/dev/null 2>&1 || true
else
  die "no R2 access: install gemstone, or provide ~/.council-r2.env"
fi
echo "  using the $R2_MODE backend"

r2_pull() {  # r2_pull <prefix> <dest>
  local prefix="$1" dest="$2"
  mkdir -p "$dest"
  if [[ "$R2_MODE" == "gemstone" ]]; then
    gemstone store pull "$prefix" "$dest"
  else
    "$REPO/.venv/bin/python" - "$prefix" "$dest" <<'PY'
import os, sys, boto3
prefix, dest = sys.argv[1], sys.argv[2]
env = {}
with open(os.path.expanduser("~/.council-r2.env")) as fh:
    for line in fh:
        line = line.strip().removeprefix("export ")
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip("'\"")
s3 = boto3.client("s3", endpoint_url=env["COUNCIL_R2_ENDPOINT"],
                  aws_access_key_id=env["COUNCIL_R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=env["COUNCIL_R2_SECRET_ACCESS_KEY"],
                  region_name="auto")
bucket = env["COUNCIL_R2_BUCKET"]
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        rel = obj["Key"][len(prefix):].lstrip("/")
        if not rel:
            continue
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == obj["Size"]:
            continue
        s3.download_file(bucket, obj["Key"], out)
        n += 1
        print(f"    {rel} ({obj['Size']/1e6:.0f} MB)", flush=True)
print(f"    {n} object(s) fetched")
PY
  fi
}

MODEL_DIR="$MODELS/RedHatAI/Qwen3.8-27B-INT4"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "  pulling the model (~19 GB)"
  mkdir -p "$MODEL_DIR"
  r2_pull models/RedHatAI/Qwen3.8-27B-INT4/ "$MODEL_DIR"
else
  echo "  model already present"
fi

# The compile cache is keyed by GPU arch and by the CUDA/torch build. A miss
# only costs a recompile, so a wrong or absent cache is never fatal.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm-signals}"
mkdir -p "$VLLM_CACHE_ROOT"
r2_pull "artifacts/vllm-compile-cache/$ARCH_TAG/" "$VLLM_CACHE_ROOT" 2>/dev/null \
  && echo "  compile cache restored ($ARCH_TAG)" \
  || echo "  no compile cache for $ARCH_TAG - first boot will compile (~35 s)"

mkdir -p "$SIGNALS"
if (( PULL_SIGNALS )); then
  echo "  pulling the signals corpus"
  r2_pull signals/ "$SIGNALS/_r2" || true
fi

# ── 5. serve ─────────────────────────────────────────────────────────────────
CAPTURE_ARGS=()
if [[ "$CAPTURE" != "off" ]]; then
  CAPTURE_ARGS=(
    --signal-capture-backend "$CAPTURE"
    --signal-capture-tier "$TIER"
    --signal-capture-tokens last
    --signal-capture-layers last
    --signal-capture-session "$SESSION"
    --signal-capture-dir "$SIGNALS"
  )
  [[ "$CAPTURE" == "hook" ]] && cat <<'WARN'

  NOTE: the hook backend forces eager execution. On the reference GH200 that
  cost ~4x throughput (737 -> 162 tok/s). Use --capture off for full-speed
  serving, and turn capture on for a window via POST /signals/control.

WARN
fi

if (( ! SERVE )); then say "provisioned; not serving (--no-serve)"; exit 0; fi

say "serving qwen38 on :$PORT"
export VLLM_SERVER_DEV_MODE=1     # enables /signals/* control endpoints
exec .venv/bin/vllm serve \
  --model "$MODEL_DIR" \
  --served-model-name qwen38 --port "$PORT" --host 0.0.0.0 \
  --kv-cache-dtype fp8 --attention-backend TRITON_ATTN \
  --max-model-len 131072 --max-num-seqs 512 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  "${CAPTURE_ARGS[@]}"
