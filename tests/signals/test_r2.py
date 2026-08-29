# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The R2 sidecar: key layout, catalog rows, idempotence, and failure handling."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from vllm.signals.r2 import Deposit, Uploader, read_header


class FakeBackend:
    """Records what it was asked to upload; can be told to fail."""

    name = "fake"

    def __init__(self, fail: set[str] | None = None):
        self.put_keys: list[str] = []
        self.fail = fail or set()

    def put(self, local: str, key: str) -> None:
        if os.path.basename(local) in self.fail:
            raise RuntimeError("bucket said no")
        self.put_keys.append(key)


def write_deposit(
    directory,
    name,
    *,
    model="/models/Qwen3.8-27B-INT4",
    session="",
    created="1788028682.0",
    request_id="req-1",
):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    save_file(
        {"residual": torch.zeros(1, 8), "residual.index": torch.zeros(1, 2)},
        str(path),
        {
            "format": "sigcap-v1",
            "model": model,
            "session": session,
            "request_id": request_id,
            "tier": "residual_raw",
            "token_reduce": "last",
            "num_captured_positions": "24",
            "truncated": "false",
            "created_at": created,
        },
    )
    return path


def test_key_groups_by_model_session_and_day(tmp_path):
    path = write_deposit(
        tmp_path, "20260829T183802.266-req-1.safetensors", session="sweep-a"
    )
    deposit = Deposit(str(path), path.stat().st_size, read_header(str(path)))
    assert deposit.key("signals") == (
        "signals/Qwen3.8-27B-INT4/sweep-a/2026-08-29/"
        "20260829T183802.266-req-1.safetensors"
    )


def test_missing_session_becomes_default(tmp_path):
    path = write_deposit(tmp_path, "a.safetensors")
    deposit = Deposit(str(path), path.stat().st_size, read_header(str(path)))
    assert "/default/" in deposit.key("signals")


def test_upload_writes_a_catalog_row(tmp_path):
    write_deposit(tmp_path, "a.safetensors", request_id="chatcmpl-xyz")
    backend = FakeBackend()
    uploader = Uploader(str(tmp_path), backend, prefix="signals")
    assert uploader.drain() == 1

    shard = tmp_path / ".r2-catalog" / "2026-08-29.jsonl"
    row = json.loads(shard.read_text().strip())
    assert row["request_id"] == "chatcmpl-xyz"
    assert row["tier"] == "residual_raw"
    assert row["positions"] == "24"
    assert row["bytes"] > 0
    assert len(row["sha256"]) == 64
    assert row["key"] == backend.put_keys[0]


def test_uploads_are_idempotent_across_restarts(tmp_path):
    write_deposit(tmp_path, "a.safetensors")
    backend = FakeBackend()
    Uploader(str(tmp_path), backend).drain()
    assert len(backend.put_keys) == 1

    # A fresh uploader reads the state file rather than re-shipping.
    again = FakeBackend()
    assert Uploader(str(tmp_path), again).drain() == 0
    assert again.put_keys == []


def test_a_failed_upload_is_retried_next_pass(tmp_path):
    write_deposit(tmp_path, "a.safetensors")
    failing = FakeBackend(fail={"a.safetensors"})
    uploader = Uploader(str(tmp_path), failing)
    uploader.drain()
    assert uploader.failed == 1
    assert not (tmp_path / ".r2-catalog" / "2026-08-29.jsonl").exists()

    working = FakeBackend()
    retry = Uploader(str(tmp_path), working)
    assert retry.drain() == 1
    assert len(working.put_keys) == 1


def test_partial_files_are_never_uploaded(tmp_path):
    write_deposit(tmp_path, "done.safetensors")
    (tmp_path / "half.safetensors.partial").write_bytes(b"not finished")
    backend = FakeBackend()
    Uploader(str(tmp_path), backend).drain()
    assert [k.rsplit("/", 1)[-1] for k in backend.put_keys] == ["done.safetensors"]


def test_delete_after_frees_the_local_copy(tmp_path):
    path = write_deposit(tmp_path, "a.safetensors")
    Uploader(str(tmp_path), FakeBackend(), delete_after=True).drain()
    assert not path.exists()


def test_local_copy_is_kept_by_default(tmp_path):
    path = write_deposit(tmp_path, "a.safetensors")
    Uploader(str(tmp_path), FakeBackend()).drain()
    assert path.exists()


def test_catalog_shards_by_day(tmp_path):
    write_deposit(tmp_path, "a.safetensors", created="1788028682.0")  # 2026-08-29
    write_deposit(tmp_path, "b.safetensors", created="1788115082.0")  # 2026-08-30
    Uploader(str(tmp_path), FakeBackend()).drain()
    shards = sorted(p.name for p in (tmp_path / ".r2-catalog").glob("*.jsonl"))
    assert shards == ["2026-08-29.jsonl", "2026-08-30.jsonl"]


@pytest.mark.parametrize("raw,expected", [("a b", "a-b"), ("../etc", "etc"), ("", "x")])
def test_slugs_are_path_safe(raw, expected):
    from vllm.signals.r2 import _slug

    assert _slug(raw, "x") == expected
