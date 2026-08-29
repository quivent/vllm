# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ship deposits to R2 as they land, and keep a catalog of what went where.

A sidecar, not part of serving: it watches the capture directory from outside
the engine, so an upload that stalls or a bucket that rejects a write can never
slow a forward pass down or lose a turn. Deposits are written atomically
(`.partial` then rename), so anything matching `*.safetensors` is complete.

Uploads go through whatever backend is available:

``gemstone``
    Shells out to ``gemstone store push``, reusing the credentials and bucket
    that CLI is already configured with. The default, and the reason this needs
    no secrets of its own.

``s3``
    boto3 straight at the R2 endpoint, reading ``~/.council-r2.env``. Faster for
    a large backlog, and the only backend that can set object metadata.

Every upload appends a row to a catalog: one JSONL shard per day, held locally
and mirrored to the bucket, so the set of captured turns can be queried without
listing several hundred thousand objects.

::

    python -m vllm.signals.r2 sync    --dir /var/signals     # upload the backlog
    python -m vllm.signals.r2 watch   --dir /var/signals     # and keep watching
    python -m vllm.signals.r2 catalog --dir /var/signals     # what has been shipped
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

logger_prefix = "[signals-r2]"

DEFAULT_PREFIX = "signals"
STATE_FILE = ".r2-uploaded.json"
CATALOG_DIR = ".r2-catalog"


def _log(message: str) -> None:
    print(f"{logger_prefix} {message}", flush=True)


def _slug(value: str, fallback: str = "unknown") -> str:
    """A path-safe fragment for a model name or session label."""
    cleaned = "".join(c if c.isalnum() or c in "._-" else "-" for c in value.strip())
    cleaned = cleaned.strip("-.")
    return cleaned or fallback


@dataclass
class Deposit:
    """One capture file, and what the catalog needs to know about it."""

    path: str
    size: int
    metadata: dict = field(default_factory=dict)

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)

    @property
    def model(self) -> str:
        return _slug(os.path.basename(self.metadata.get("model", "")), "unknown-model")

    @property
    def session(self) -> str:
        return _slug(self.metadata.get("session", ""), "default")

    @property
    def day(self) -> str:
        created = self.metadata.get("created_at")
        stamp = float(created) if created else os.path.getmtime(self.path)
        return time.strftime("%Y-%m-%d", time.gmtime(stamp))

    def key(self, prefix: str) -> str:
        """Where this lands in the bucket.

        Grouped model / session / day so a day's turns, or one experiment's,
        can be pulled with a single prefix rather than a filtered listing.
        """
        return f"{prefix}/{self.model}/{self.session}/{self.day}/{self.basename}"

    def digest(self) -> str:
        sha = hashlib.sha256()
        with open(self.path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def catalog_row(self, key: str, digest: str) -> dict:
        meta = self.metadata
        return {
            "key": key,
            "basename": self.basename,
            "bytes": self.size,
            "sha256": digest,
            "model": meta.get("model", ""),
            "session": meta.get("session", ""),
            "request_id": meta.get("request_id", ""),
            "tier": meta.get("tier", ""),
            "token_reduce": meta.get("token_reduce", ""),
            "positions": meta.get("num_captured_positions", ""),
            "truncated": meta.get("truncated", ""),
            "dtypes": meta.get("dtypes", ""),
            "created_at": meta.get("created_at", ""),
            "uploaded_at": f"{time.time():.3f}",
        }


def read_header(path: str) -> dict:
    """The safetensors metadata block, without loading any tensor data."""
    import struct

    try:
        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        return header.get("__metadata__", {})
    except Exception as exc:  # a file mid-write, or not a deposit at all
        _log(f"skipping {os.path.basename(path)}: unreadable header ({exc})")
        return {}


# ── backends ─────────────────────────────────────────────────────────────────


class GemstoneBackend:
    """Uploads via `gemstone store push`, reusing its configured credentials."""

    name = "gemstone"

    def __init__(self, binary: str = "gemstone"):
        self.binary = binary

    def available(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [self.binary, "r2", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            return False, f"{self.binary} not on PATH"
        except subprocess.TimeoutExpired:
            return False, "gemstone r2 check timed out"
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(
                        result.stdout[
                            result.stdout.index("{") : result.stdout.rindex("}") + 1
                        ]
                    )
                except Exception:
                    break
                if payload.get("ok"):
                    return True, f"bucket {payload.get('bucket', '?')}"
                return False, "gemstone r2 check reported not ok"
        return False, (result.stderr or result.stdout or "unknown").strip()[:200]

    def put(self, local: str, key: str) -> None:
        result = subprocess.run(
            [self.binary, "store", "push", local, key],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                (
                    result.stderr or result.stdout or "gemstone store push failed"
                ).strip()[:300]
            )


class S3Backend:
    """Uploads straight to the R2 endpoint with boto3."""

    name = "s3"

    def __init__(self, env_file: str = "~/.council-r2.env"):
        self.env_file = os.path.expanduser(env_file)
        self._client = None
        self.bucket = ""

    def _env(self) -> dict:
        values = {}
        if os.path.exists(self.env_file):
            with open(self.env_file) as handle:
                for line in handle:
                    line = line.strip().removeprefix("export ").strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    values[key.strip()] = value.strip().strip("'\"")
        for key in list(values):
            values.setdefault(key, os.environ.get(key, ""))
        return {**values, **{k: v for k, v in os.environ.items() if k in values}}

    def available(self) -> tuple[bool, str]:
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False, "boto3 is not installed"
        env = self._env()
        required = (
            "COUNCIL_R2_ENDPOINT",
            "COUNCIL_R2_ACCESS_KEY_ID",
            "COUNCIL_R2_SECRET_ACCESS_KEY",
            "COUNCIL_R2_BUCKET",
        )
        missing = [key for key in required if not env.get(key)]
        if missing:
            return False, f"{self.env_file} is missing {', '.join(missing)}"
        self.bucket = env["COUNCIL_R2_BUCKET"]
        return True, f"bucket {self.bucket}"

    def client(self):
        if self._client is None:
            import boto3

            env = self._env()
            self.bucket = env["COUNCIL_R2_BUCKET"]
            self._client = boto3.client(
                "s3",
                endpoint_url=env["COUNCIL_R2_ENDPOINT"],
                aws_access_key_id=env["COUNCIL_R2_ACCESS_KEY_ID"],
                aws_secret_access_key=env["COUNCIL_R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
        return self._client

    def put(self, local: str, key: str, metadata: dict | None = None) -> None:
        extra = {"Metadata": {k: str(v)[:1024] for k, v in (metadata or {}).items()}}
        self.client().upload_file(local, self.bucket, key, ExtraArgs=extra)


def pick_backend(name: str):
    """Resolve a backend by name, or auto-select the first that works."""
    candidates = {"gemstone": GemstoneBackend, "s3": S3Backend}
    if name != "auto":
        backend = candidates[name]()
        ok, detail = backend.available()
        if not ok:
            raise SystemExit(f"{logger_prefix} backend {name!r} unusable: {detail}")
        _log(f"backend {name} ready ({detail})")
        return backend
    for candidate in (GemstoneBackend(), S3Backend()):
        ok, detail = candidate.available()
        if ok:
            _log(f"backend {candidate.name} ready ({detail})")
            return candidate
        _log(f"backend {candidate.name} unavailable: {detail}")
    raise SystemExit(f"{logger_prefix} no usable R2 backend")


# ── uploader ─────────────────────────────────────────────────────────────────


class Uploader:
    """Uploads new deposits and records them in a catalog.

    State is a local file of already-shipped basenames, so a restart resumes
    instead of re-uploading, and a failed upload is simply left unrecorded for
    the next pass to retry.
    """

    def __init__(
        self,
        directory: str,
        backend,
        prefix: str = DEFAULT_PREFIX,
        delete_after: bool = False,
    ):
        self.directory = directory
        self.backend = backend
        self.prefix = prefix.strip("/")
        self.delete_after = delete_after
        self.state_path = os.path.join(directory, STATE_FILE)
        self.catalog_dir = os.path.join(directory, CATALOG_DIR)
        os.makedirs(self.catalog_dir, exist_ok=True)
        self.done: set[str] = self._load_state()
        self.uploaded = 0
        self.failed = 0

    def _load_state(self) -> set[str]:
        try:
            with open(self.state_path) as handle:
                return set(json.load(handle))
        except Exception:
            return set()

    def _save_state(self) -> None:
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(sorted(self.done), handle)
        os.replace(tmp, self.state_path)

    def pending(self) -> list[str]:
        """Complete deposits not yet shipped, oldest first."""
        names = [
            name
            for name in os.listdir(self.directory)
            if name.endswith(".safetensors") and name not in self.done
        ]
        return sorted(names)

    def upload(self, name: str) -> bool:
        path = os.path.join(self.directory, name)
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            return False
        deposit = Deposit(path=path, size=size, metadata=read_header(path))
        key = deposit.key(self.prefix)
        try:
            digest = deposit.digest()
            row = deposit.catalog_row(key, digest)
            if isinstance(self.backend, S3Backend):
                self.backend.put(path, key, metadata=row)
            else:
                self.backend.put(path, key)
        except Exception as exc:
            self.failed += 1
            _log(f"FAILED {name}: {exc}")
            return False

        self._append_catalog(deposit.day, row)
        self.done.add(name)
        self.uploaded += 1
        if self.delete_after:
            with contextlib.suppress(OSError):
                os.remove(path)
        return True

    def _append_catalog(self, day: str, row: dict) -> None:
        shard = os.path.join(self.catalog_dir, f"{day}.jsonl")
        with open(shard, "a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def push_catalog(self) -> None:
        """Mirror the catalog shards to the bucket."""
        for name in sorted(os.listdir(self.catalog_dir)):
            if not name.endswith(".jsonl"):
                continue
            local = os.path.join(self.catalog_dir, name)
            try:
                self.backend.put(local, f"{self.prefix}/_catalog/{name}")
            except Exception as exc:
                _log(f"catalog {name} not mirrored: {exc}")

    def drain(self) -> int:
        names = self.pending()
        for name in names:
            self.upload(name)
        if names:
            self._save_state()
        return len(names)


def cmd_sync(args) -> int:
    backend = pick_backend(args.backend)
    uploader = Uploader(args.dir, backend, args.prefix, args.delete_after)
    count = uploader.drain()
    uploader.push_catalog()
    _log(f"{uploader.uploaded} uploaded, {uploader.failed} failed, {count} considered")
    return 1 if uploader.failed else 0


def cmd_watch(args) -> int:
    backend = pick_backend(args.backend)
    uploader = Uploader(args.dir, backend, args.prefix, args.delete_after)
    _log(f"watching {args.dir} -> {backend.name}:{uploader.prefix}/ (ctrl-c to stop)")
    last_catalog = 0.0
    try:
        while True:
            moved = uploader.drain()
            now = time.time()
            if moved or now - last_catalog > args.catalog_interval:
                uploader.push_catalog()
                last_catalog = now
            if moved:
                _log(f"shipped {uploader.uploaded} total, {uploader.failed} failed")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        uploader.push_catalog()
        _log(f"stopped: {uploader.uploaded} uploaded, {uploader.failed} failed")
    return 0


def cmd_catalog(args) -> int:
    catalog_dir = os.path.join(args.dir, CATALOG_DIR)
    if not os.path.isdir(catalog_dir):
        _log("no catalog yet")
        return 0
    rows: list[dict] = []
    for name in sorted(os.listdir(catalog_dir)):
        if name.endswith(".jsonl"):
            with open(os.path.join(catalog_dir, name)) as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    total = sum(r["bytes"] for r in rows)
    _log(f"{len(rows)} deposits, {total / (1 << 20):.1f} MiB")
    by_model: dict[str, int] = {}
    for row in rows:
        by_model[row["model"] or "?"] = by_model.get(row["model"] or "?", 0) + 1
    for model, count in sorted(by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>7}  {model}")
    for row in rows[-args.tail :]:
        print(f"  {row['created_at']}  {row['bytes']:>8} B  {row['key']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.signals.r2",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--dir", required=True, help="the capture directory")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="bucket key prefix")
    parser.add_argument("--backend", default="auto", choices=("auto", "gemstone", "s3"))
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="remove each deposit once it is safely in the bucket",
    )
    sub = parser.add_subparsers(dest="op", required=True)

    sub.add_parser("sync", help="upload everything outstanding, then exit")

    watch = sub.add_parser("watch", help="upload continuously")
    watch.add_argument("--interval", type=float, default=5.0)
    watch.add_argument("--catalog-interval", type=float, default=60.0)

    cat = sub.add_parser("catalog", help="summarize what has been shipped")
    cat.add_argument("--json", action="store_true")
    cat.add_argument("--tail", type=int, default=10)

    args = parser.parse_args(argv)
    return {"sync": cmd_sync, "watch": cmd_watch, "catalog": cmd_catalog}[args.op](args)


if __name__ == "__main__":
    sys.exit(main())
