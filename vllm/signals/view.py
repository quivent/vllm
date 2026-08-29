# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read a ``sigcap-v1`` deposit back out.

Deposits store activations as they exist in the forward pass -- raw binary
vectors in the model's own dtype. This module is the other end: it decodes a
deposit into whatever shape is useful for looking at.

::

    python -m vllm.signals.view gen.safetensors                  # what's inside
    python -m vllm.signals.view gen.safetensors -f stats         # scalar taxonomy
    python -m vllm.signals.view gen.safetensors -f json --token 0
    python -m vllm.signals.view gen.safetensors -f npz -o gen.npz

The ``stats`` view is the interesting one: it applies the SPEC.md reductions to
the stored raw vectors, so a T4 deposit can be read as though it were a T2 one
(per-layer ``activation_norm`` / ``cosine_sim``) without re-running the model.
"""

import argparse
import json
import os
import struct
import sys

from safetensors.torch import load_file

from vllm.signals.reductions import cosine_rows
from vllm.signals.tiers import STATS_FIELDS


def read_metadata(path: str) -> tuple[dict, dict]:
    """Parse the safetensors header without loading any tensor data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    metadata = header.pop("__metadata__", {})
    return metadata, header


def _signal_names(header: dict) -> list[str]:
    """Raw per-(token, layer) signal tensors -- not indices, stats, or logits."""
    return sorted(
        name
        for name in header
        if not name.endswith(".index")
        and not name.startswith("stats.")
        and name != "logit"
    )


def summary(path: str) -> str:
    """Metadata, tensor shapes, and the raw-binary density of the file."""
    size = os.path.getsize(path)
    metadata, header = read_metadata(path)
    lines = [f"file: {path}  ({size:,} bytes)"]
    lines.append("metadata:")
    for key in sorted(metadata):
        lines.append(f"  {key:12s} {metadata[key]}")

    header_bytes = 8 + (size - max(t["data_offsets"][1] for t in header.values()))
    lines.append("tensors:")
    total_values = 0
    for name in sorted(header):
        spec = header[name]
        start, end = spec["data_offsets"]
        shape = spec["shape"]
        values = 1
        for dim in shape:
            values *= dim
        total_values += values
        width = (end - start) / max(values, 1)
        lines.append(
            f"  {name:26s} shape={str(shape):16s} {spec['dtype']:6s} "
            f"{end - start:>12,} B  ({width:.2f} B/value)"
        )
    lines.append(f"total values: {total_values:,}")
    lines.append(
        f"json header: {header_bytes:,} B "
        f"({100 * header_bytes / max(size, 1):.3f}% of file)"
    )
    return "\n".join(lines)


def to_stats(path: str) -> dict:
    """Reduce every stored raw vector to the SPEC.md per-layer scalars.

    Returns ``{signal: [{token, layer_idx, activation_norm, cosine_sim, ...}]}``.
    Any ``stats.*`` tensors already in the deposit are passed through under their
    :data:`~vllm.signals.tiers.STATS_FIELDS` names.
    """
    tensors = load_file(path)
    out: dict[str, list[dict]] = {}

    for name in _signal_names(tensors):
        index = tensors.get(f"{name}.index")
        if index is None:
            continue
        values = tensors[name].float()
        rows = []
        # Cosine chains across layers within one token, per SPEC.md 3.1.
        by_token: dict[int, list[int]] = {}
        for i, (token, _layer) in enumerate(index.tolist()):
            by_token.setdefault(int(token), []).append(i)
        for token in sorted(by_token):
            order = sorted(by_token[token], key=lambda i: index[i, 1].item())
            prev = None
            for i in order:
                vector = values[i : i + 1]
                cosine = (
                    0.0 if prev is None else float(cosine_rows(vector, prev).item())
                )
                rows.append(
                    {
                        "token": token,
                        "layer_idx": int(index[i, 1].item()),
                        "activation_norm": float(vector.norm().item()),
                        "cosine_sim": cosine,
                        "mean_abs": float(vector.abs().mean().item()),
                        "max_abs": float(vector.abs().amax().item()),
                    }
                )
                prev = vector
        out[name] = rows

    for name in sorted(k for k in tensors if k.startswith("stats.")):
        if name.endswith(".index"):
            continue
        index = tensors.get(f"{name}.index")
        if index is None:
            continue
        values = tensors[name]
        signal = name.removeprefix("stats.")
        out[signal] = [
            {
                "token": int(index[i, 0].item()),
                "layer_idx": int(index[i, 1].item()),
                **{
                    field: float(values[i, j].item())
                    for j, field in enumerate(STATS_FIELDS)
                },
            }
            for i in range(values.shape[0])
        ]

    if "logit" in tensors:
        index = tensors["logit.index"]
        values = tensors["logit"]
        out["logit"] = [
            {
                "token": int(index[i, 0].item()),
                "entropy": float(values[i, 0].item()),
                "perplexity": float(values[i, 1].item()),
                "confidence": float(values[i, 2].item()),
            }
            for i in range(values.shape[0])
        ]
    return out


def to_json(
    path: str,
    signal: str | None = None,
    token: int | None = None,
    layer: int | None = None,
) -> dict:
    """Decode the raw vectors themselves to JSON, optionally filtered."""
    tensors = load_file(path)
    metadata, _ = read_metadata(path)
    out: dict = {"metadata": metadata, "vectors": []}
    for name in _signal_names(tensors):
        if signal and name != signal:
            continue
        index = tensors.get(f"{name}.index")
        if index is None:
            continue
        values = tensors[name].float()
        for i in range(values.shape[0]):
            row_token, row_layer = (int(v) for v in index[i].tolist())
            if token is not None and row_token != token:
                continue
            if layer is not None and row_layer != layer:
                continue
            out["vectors"].append(
                {
                    "signal": name,
                    "token": row_token,
                    "layer_idx": row_layer,
                    "values": values[i].tolist(),
                }
            )
    return out


def to_npz(path: str, out_path: str) -> str:
    """Export every tensor to a numpy ``.npz``."""
    import numpy as np

    tensors = load_file(path)
    np.savez_compressed(out_path, **{k: v.float().numpy() for k, v in tensors.items()})
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.signals.view",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("deposit", help="path to a .safetensors deposit")
    parser.add_argument(
        "-f",
        "--format",
        default="summary",
        choices=("summary", "stats", "json", "npz"),
        help="summary: what's inside; stats: SPEC.md scalar reductions; "
        "json: the raw vectors decoded; npz: numpy export",
    )
    parser.add_argument("--signal", help="restrict to one signal name")
    parser.add_argument("--token", type=int, help="restrict to one token index")
    parser.add_argument("--layer", type=int, help="restrict to one layer index")
    parser.add_argument("-o", "--output", help="write to this file instead of stdout")
    args = parser.parse_args(argv)

    if args.format == "npz":
        out = args.output or f"{os.path.splitext(args.deposit)[0]}.npz"
        print(to_npz(args.deposit, out))
        return 0

    if args.format == "summary":
        text = summary(args.deposit)
    elif args.format == "stats":
        rows = to_stats(args.deposit)
        if args.signal:
            rows = {k: v for k, v in rows.items() if k == args.signal}
        text = json.dumps(rows, indent=2)
    else:
        text = json.dumps(
            to_json(args.deposit, args.signal, args.token, args.layer), indent=2
        )

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
