#!/usr/bin/env bash
# Environment profile for the signals patch: every trap this deployment has
# actually hit, as a check that can also fix itself.
#
#   ./preflight.sh          report only
#   ./preflight.sh --fix    remediate what can be remediated
#
# Each check names the symptom it prevents, because the failures are mostly
# indirect: a missing nvcc surfaces as an engine-core crash 90 seconds in, a
# root-owned cache directory as a PermissionError inside flashinfer, an
# unpinned wheel variant as "metadata not found for cu129".
set -uo pipefail

FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1
REPO="${REPO:-$HOME/vllm}"
VENV="$REPO/.venv"
PASS=0; FIXED=0; FAIL=0

ok()   { printf '  \033[32mok\033[0m    %-34s %s\n' "$1" "${2:-}"; PASS=$((PASS+1)); }
fixed(){ printf '  \033[36mfixed\033[0m %-34s %s\n' "$1" "${2:-}"; FIXED=$((FIXED+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %-34s %s\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); }
note() { printf '        %s\n' "$1"; }

echo "=== host ==="
ARCH="$(uname -m)"
ok "architecture" "$ARCH"
if command -v nvidia-smi >/dev/null; then
  ok "nvidia-smi" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  bad "nvidia-smi" "no NVIDIA driver tooling"
fi

# 1. PATH. uv and gemstone live in ~/.local/bin, which a non-login shell omits;
#    the symptom is a deploy silently falling back to a worse code path.
echo "=== PATH ==="
export PATH="$HOME/.local/bin:$PATH"
for tool in uv gemstone; do
  command -v "$tool" >/dev/null && ok "$tool" "$(command -v $tool)" || bad "$tool" "not found even with ~/.local/bin on PATH"
done

# 2. Driver vs CUDA 13. This vLLM is built for CUDA 13.0 (torch 2.13.0+cu130),
#    which needs r580+. Below that, CUDA does not initialise at all:
#    "The NVIDIA driver on your system is too old (found version 12080)".
#    Hopper/Grace are datacenter parts, so forward-compat libraries bridge it.
echo "=== CUDA 13 vs driver ==="
DRIVER_FULL="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
DRIVER="${DRIVER_FULL%%.*}"
COMPAT="$HOME/cuda-compat-13.0"
if [[ -z "$DRIVER" ]]; then
  bad "driver version" "could not read"
elif (( DRIVER >= 580 )); then
  ok "driver $DRIVER_FULL" "supports CUDA 13 natively"
elif [[ -f "$COMPAT/libcuda.so.1" ]]; then
  ok "forward-compat libs" "$COMPAT (driver $DRIVER_FULL needs them)"
  note "every CUDA process needs LD_LIBRARY_PATH=$COMPAT:\$LD_LIBRARY_PATH"
elif (( FIX )); then
  case "$ARCH" in
    aarch64) REPOARCH=sbsa;   SUFFIX=arm64;;
    *)       REPOARCH=x86_64; SUFFIX=amd64;;
  esac
  DEB="cuda-compat-13-0_580.178.04-1ubuntu1_${SUFFIX}.deb"
  URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$REPOARCH/$DEB"
  tmp="$(mktemp -d)"
  if curl -fsSL -o "$tmp/$DEB" "$URL" && dpkg -x "$tmp/$DEB" "$tmp/x"; then
    mv "$tmp/x/usr/local/cuda-13.0/compat" "$COMPAT" && fixed "forward-compat libs" "$COMPAT"
  else
    bad "forward-compat libs" "could not fetch $URL"
  fi
  rm -rf "$tmp"
else
  bad "driver $DRIVER_FULL" "older than CUDA 13 needs; --fix installs compat libs"
fi

# 3. nvcc. flashinfer JIT-compiles its sampler on first use. Without a toolkit
#    the engine dies mid-startup with:
#      RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
#    The pip CUDA package inside the venv provides one; no system install needed.
echo "=== flashinfer JIT toolchain ==="
PIP_CUDA="$VENV/lib/python3.12/site-packages/nvidia/cu13"
if command -v nvcc >/dev/null; then
  ok "nvcc" "$(command -v nvcc)"
elif [[ -x "$PIP_CUDA/bin/nvcc" ]]; then
  ok "nvcc (pip)" "$PIP_CUDA/bin/nvcc"
  note "export CUDA_HOME=$PIP_CUDA and add \$CUDA_HOME/bin to PATH"
elif (( FIX )) && [[ -d "$VENV" ]]; then
  (cd "$REPO" && uv pip install nvidia-cuda-nvcc-cu13 >/dev/null 2>&1)
  [[ -x "$PIP_CUDA/bin/nvcc" ]] && fixed "nvcc (pip)" "$PIP_CUDA/bin/nvcc" \
    || bad "nvcc" "install a CUDA toolkit, or nvidia-cuda-nvcc-cu13"
else
  bad "nvcc" "flashinfer will fail at startup; --fix installs the pip package"
fi

# 4. ninja. Same JIT path, different missing piece:
#      FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
if [[ -x "$VENV/bin/ninja" ]] || command -v ninja >/dev/null; then
  ok "ninja" "present"
elif (( FIX )) && [[ -d "$VENV" ]]; then
  (cd "$REPO" && uv pip install ninja >/dev/null 2>&1) && fixed "ninja" "installed" \
    || bad "ninja" "uv pip install ninja"
else
  bad "ninja" "flashinfer JIT needs it; --fix installs it"
fi

# 5. Writable cache. A previous root-run leaves root-owned subdirectories under
#    ~/.cache/vllm; the symptom is PermissionError on modelinfos or
#    flashinfer_autotune_cache, again mid-startup.
echo "=== caches ==="
CACHE="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
if [[ -d "$CACHE" ]] && find "$CACHE" -maxdepth 1 ! -writable -print -quit 2>/dev/null | grep -q .; then
  if (( FIX )); then
    export VLLM_CACHE_ROOT="$HOME/.cache/vllm-signals"
    mkdir -p "$VLLM_CACHE_ROOT" && fixed "cache root" "moved to $VLLM_CACHE_ROOT"
  else
    bad "cache root" "$CACHE has unwritable entries; set VLLM_CACHE_ROOT"
  fi
else
  ok "cache root" "$CACHE writable"
fi

# 6. Wheels are per-architecture. An aarch64 build is useless on x86_64, and the
#    reverse; the key must carry uname -m.
echo "=== vllm build ==="
if [[ -d "$REPO/.git" ]]; then
  ok "repo" "$(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  # setuptools_scm derives the version from tags; a clone without them reports a
  # wildly older version for identical code.
  if [[ -z "$(git -C "$REPO" tag --list 'v0.2[0-9]*' | head -1)" ]]; then
    if (( FIX )); then
      git -C "$REPO" fetch --tags --quiet 2>/dev/null && fixed "git tags" "fetched"
    else
      bad "git tags" "missing; the reported vllm version will be wrong (cosmetic)"
    fi
  else
    ok "git tags" "present"
  fi
else
  bad "repo" "$REPO is not a git checkout"
fi
note "wheel variant must be pinned: VLLM_PRECOMPILED_WHEEL_VARIANT=cu130"
note "auto-detection probes cu129, which is not published for this commit"

if [[ -x "$VENV/bin/python" ]]; then
  V="$("$VENV/bin/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
  [[ -n "$V" ]] && ok "vllm importable" "$V" || bad "vllm importable" "import failed"
  LD_LIBRARY_PATH="${COMPAT}:${LD_LIBRARY_PATH:-}" "$VENV/bin/python" - <<'PY' 2>/dev/null && ok "torch CUDA" "available" || bad "torch CUDA" "unavailable - check the driver/compat step"
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
else
  bad "venv" "$VENV missing"
fi

# 7. Competing tenants. A container with a restart policy will respawn a vLLM
#    and take the GPU back from under you.
echo "=== gpu tenancy ==="
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
OTHERS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
if (( OTHERS > 0 )); then
  bad "gpu free" "${USED} MiB in use by $OTHERS process(es)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/        /'
  note "a docker restart policy may respawn these: sudo docker ps; sudo docker stop <id>"
  note "and a systemd unit may supervise the gateway: systemctl stop gemstone-governor-agentic-8000"
else
  ok "gpu free" "${USED} MiB in use"
fi

# 8. The control endpoints only exist in dev mode.
echo "=== serving prerequisites ==="
[[ "${VLLM_SERVER_DEV_MODE:-0}" == "1" ]] \
  && ok "VLLM_SERVER_DEV_MODE" "set - /signals/* will be mounted" \
  || bad "VLLM_SERVER_DEV_MODE" "unset - /signals/status and /signals/control will 404"

MODEL_DIR="${MODEL_DIR:-$HOME/models/RedHatAI/Qwen3.8-27B-INT4}"
if [[ -f "$MODEL_DIR/config.json" ]]; then
  ok "model" "$MODEL_DIR"
elif [[ -d "$MODEL_DIR/blobs" ]]; then
  bad "model" "$MODEL_DIR is content-addressed (blobs/refs/trees), not servable"
  note "the R2 models/ prefix stores CAS, not a directory; use hf download or"
  note "gemstone models download --local-dir"
else
  bad "model" "$MODEL_DIR has no config.json"
fi

echo
echo "$PASS ok, $FIXED fixed, $FAIL outstanding"
(( FAIL == 0 )) || {
  echo "re-run with --fix to remediate what can be remediated"
  exit 1
}
