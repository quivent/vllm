# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prove a signals deployment is actually doing what it claims.

One command against a running server, answering the questions that decide
whether a capture run is trustworthy:

- is this the patched vLLM, and which commit?
- is speculative decoding on, and which method?
- are CUDA graphs compiled, or was eager forced?
- what is capture configured to extract, and is it recording?
- what throughput does this configuration actually get?
- and, end to end: does a real turn leave a real residual on disk?

The last one is the point. Everything else can look right while capture
silently records nothing, so the check runs a generation and then goes and
finds that request's deposit, opens it, and reports the vector it holds.

::

    python -m vllm.signals.bench --port 8001 --dir ~/signals
    python -m vllm.signals.bench --port 8001 --dir ~/signals --json report.json
    python -m vllm.signals.bench --port 8001 --skip-throughput
"""

import argparse
import glob
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

OK = "PASS"
BAD = "FAIL"
MEH = "warn"


def _get(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _post(url: str, payload: dict, timeout: float = 300.0):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode()[:300]}
    except Exception as exc:
        return {"error": str(exc)[:300]}


class Report:
    """Collects checks, prints them, and decides the exit status."""

    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []
        self.data: dict = {}

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def render(self) -> str:
        width = max(len(name) for _, name, _ in self.rows) + 2
        lines = []
        for status, name, detail in self.rows:
            mark = {OK: "  ok  ", BAD: " FAIL ", MEH: " warn "}[status]
            lines.append(f"[{mark}] {name:<{width}} {detail}")
        failed = sum(1 for s, _, _ in self.rows if s == BAD)
        warned = sum(1 for s, _, _ in self.rows if s == MEH)
        lines.append("")
        lines.append(
            f"{len(self.rows) - failed - warned} passed, {warned} warned, "
            f"{failed} failed"
        )
        return "\n".join(lines)

    @property
    def failed(self) -> bool:
        return any(s == BAD for s, _, _ in self.rows)


def check_server(base: str, report: Report) -> str | None:
    models = _get(f"{base}/v1/models")
    if not models or not models.get("data"):
        report.add(BAD, "server reachable", f"no /v1/models at {base}")
        return None
    served = models["data"][0]["id"]
    root = models["data"][0].get("root", "")
    report.add(OK, "server reachable", f"{served}  ({os.path.basename(root)})")
    report.data["model"] = served
    report.data["model_path"] = root
    return served


def check_version(base: str, report: Report) -> None:
    version = _get(f"{base}/version")
    if not version:
        report.add(MEH, "vllm version", "/version not exposed")
        return
    text = version.get("version", "?")
    report.data["vllm_version"] = text
    report.add(OK, "vllm version", text)


def check_signals_build(base: str, report: Report) -> dict | None:
    """The patched build is the one that answers /signals/status at all."""
    status = _get(f"{base}/signals/status")
    if status is None:
        report.add(
            BAD,
            "patched vllm",
            "/signals/status missing - unpatched build, or VLLM_SERVER_DEV_MODE unset",
        )
        return None
    report.add(OK, "patched vllm", "/signals/status present")
    report.data["signals"] = status

    if not status.get("enabled"):
        report.add(BAD, "capture enabled", status.get("detail", "capture is off"))
        return status

    backend = status.get("backend", "?")
    detail = (
        f"backend={backend} tier={status.get('tier')} tokens={status.get('tokens')}"
    )
    report.add(OK, "capture configured", detail)

    layers = status.get("layers", [])
    report.add(
        OK,
        "layers tapped",
        f"{layers if len(layers) <= 6 else f'{len(layers)} layers'} "
        f"of {status.get('num_layers')}",
    )
    report.add(
        OK if status.get("recording") else MEH,
        "recording",
        "yes" if status.get("recording") else "tier is off - nothing is being written",
    )
    return status


def check_engine(base: str, report: Report, signals: dict | None) -> None:
    """Speculative decoding and CUDA graphs, from the server's own config."""
    info = _get(f"{base}/server_info")
    if not info:
        report.add(MEH, "engine config", "/server_info not exposed (dev mode off?)")
        return

    text = json.dumps(info)
    report.data["server_info_keys"] = sorted(info)[:40]

    spec = info.get("speculative_config") or {}
    if isinstance(spec, str):
        spec = {"raw": spec}
    if spec and spec != {}:
        method = spec.get("method") or ("mtp" if "mtp" in text else "?")
        n = spec.get("num_speculative_tokens", "?")
        report.add(OK, "speculative decoding", f"{method}, {n} draft tokens")
        report.data["spec"] = {"method": method, "num_speculative_tokens": n}
    else:
        report.add(MEH, "speculative decoding", "not enabled")

    model_config = info.get("model_config") or {}
    eager = model_config.get("enforce_eager")
    if eager is None:
        eager = '"enforce_eager": true' in text.lower()
    compilation = info.get("compilation_config") or {}
    cg_mode = compilation.get("cudagraph_mode", "?")
    report.data["enforce_eager"] = bool(eager)
    report.data["cudagraph_mode"] = cg_mode

    if eager:
        note = "eager - CUDA graphs OFF"
        if signals and signals.get("backend") == "hook":
            note += " (expected: the hook backend forces it, ~4x throughput cost)"
        report.add(MEH, "cuda graphs", note)
    else:
        report.add(OK, "cuda graphs", f"compiled, mode={cg_mode}")


def measure(base: str, model: str, report: Report, tokens: int) -> str | None:
    """One real generation, timed. Returns its request id."""
    started = time.perf_counter()
    result = _post(
        f"{base}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count slowly from one to ten."}],
            "max_tokens": tokens,
            "temperature": 0,
        },
    )
    elapsed = time.perf_counter() - started
    if not result or "error" in result or "choices" not in result:
        report.add(BAD, "inference", str(result)[:120])
        return None

    generated = result["usage"]["completion_tokens"]
    tps = generated / elapsed if elapsed else 0.0
    report.data["throughput"] = {
        "completion_tokens": generated,
        "seconds": round(elapsed, 3),
        "tokens_per_second": round(tps, 1),
    }
    report.add(
        OK,
        "inference",
        f"{generated} tokens in {elapsed:.2f}s = {tps:.1f} tok/s (single stream)",
    )
    return result.get("id")


def read_header(path: str) -> tuple[dict, dict]:
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header.pop("__metadata__", {}), header


def check_deposit(
    directory: str, request_id: str | None, report: Report, wait: float
) -> None:
    """The end-to-end check: did that turn actually leave a residual on disk?"""
    if request_id is None:
        report.add(BAD, "residual saved", "no request id to look for")
        return
    if not directory:
        report.add(MEH, "residual saved", "no --dir given, cannot verify")
        return

    deadline = time.time() + wait
    match = None
    while time.time() < deadline:
        # Deposits flush when the next step runs or at shutdown, so a lone
        # trailing request may need a nudge; poll rather than assume.
        hits = [
            p
            for p in glob.glob(os.path.join(directory, "*.safetensors"))
            if request_id in os.path.basename(p)
        ]
        if hits:
            match = sorted(hits)[-1]
            break
        time.sleep(1.0)

    if match is None:
        report.add(
            BAD,
            "residual saved",
            f"no deposit for {request_id} in {directory} after {wait:.0f}s "
            "(a trailing request flushes on the next step or at shutdown)",
        )
        return

    metadata, header = read_header(match)
    if "residual" not in header:
        report.add(BAD, "residual saved", f"{os.path.basename(match)} has no residual")
        return

    spec = header["residual"]
    lo, hi = spec["data_offsets"]
    nbytes = hi - lo
    report.data["deposit"] = {
        "path": match,
        "shape": spec["shape"],
        "dtype": spec["dtype"],
        "residual_bytes": nbytes,
        "file_bytes": os.path.getsize(match),
        "metadata": metadata,
    }
    report.add(
        OK,
        "residual saved",
        f"{spec['shape']} {spec['dtype']} = {nbytes:,} B "
        f"({nbytes / 1024:.1f} KiB), file {os.path.getsize(match):,} B",
    )
    report.add(
        OK,
        "deposit metadata",
        f"tier={metadata.get('tier')} reduce={metadata.get('token_reduce')} "
        f"positions={metadata.get('num_captured_positions')} "
        f"truncated={metadata.get('truncated')}",
    )

    # Open the vector itself: a deposit of the right shape can still be junk.
    try:
        from safetensors.torch import load_file

        vector = load_file(match)["residual"].float()
        norm = float(vector.norm())
        finite = bool(vector.isfinite().all())
        report.data["deposit"]["l2_norm"] = round(norm, 4)
        report.add(
            OK if finite and norm > 0 else BAD,
            "residual is real",
            f"L2 norm {norm:.4g}, all finite={finite}",
        )
    except Exception as exc:
        report.add(MEH, "residual is real", f"could not load: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.signals.bench",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dir", default="", help="the capture directory to verify")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--wait", type=float, default=25.0, help="seconds to wait for the deposit"
    )
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--json", help="also write the full report here")
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    report = Report()

    model = check_server(base, report)
    if model is None:
        print(report.render())
        return 1

    check_version(base, report)
    signals = check_signals_build(base, report)
    check_engine(base, report, signals)

    request_id = None
    if not args.skip_throughput:
        request_id = measure(base, model, report, args.tokens)
        # A single trailing request flushes on the next step, so give it one.
        _post(
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 4,
                "temperature": 0,
            },
        )
        check_deposit(args.dir, request_id, report, args.wait)

    print(report.render())
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report.data, handle, indent=2, sort_keys=True)
        print(f"\nfull report: {args.json}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
