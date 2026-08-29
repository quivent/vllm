# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an injectable vector out of captured turns.

A single turn's residual is one sample, and one sample carries whatever the
prompt happened to be about. The useful vectors are usually built from several:

``mean``
    Average a set of turns. Averaging cancels what the turns did not share and
    keeps what they did.

``diff``
    ``mean(A) - mean(B)``. The contrastive construction: collect turns that
    have the property you want and turns that do not, and the difference is the
    direction between them, with the shared content subtracted out.

Both write a deposit in the same `sigcap-v1` shape capture produces, so the
result goes straight into ``--signal-inject-from``.

::

    python -m vllm.signals.build mean  signals/2026*.safetensors -o warm.safetensors
    python -m vllm.signals.build diff  --a 'polite/*.safetensors' \\
                                       --b 'blunt/*.safetensors' -o polite.safetensors
"""

import argparse
import glob
import sys

import torch
from safetensors.torch import load_file, save_file

from vllm.signals.deposit import DEPOSIT_FORMAT


def _expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched or ([pattern] if "*" not in pattern else []))
    if not paths:
        raise ValueError(f"no deposits matched {patterns}")
    return paths


def collect(
    paths: list[str], signal: str = "residual", layer: int | None = None
) -> tuple[torch.Tensor, int]:
    """Stack the chosen rows from every deposit into [n, hidden].

    Raises:
        ValueError: if the deposits disagree on width, or none has the signal.
    """
    rows: list[torch.Tensor] = []
    layers: set[int] = set()
    for path in paths:
        tensors = load_file(path)
        if signal not in tensors:
            continue
        values = tensors[signal].float()
        index = tensors.get(f"{signal}.index")
        if index is not None:
            keep = [
                i
                for i in range(index.shape[0])
                if layer is None or int(index[i, 1]) == layer
            ]
            if not keep:
                continue
            values = values[keep]
            layers.update(int(index[i, 1]) for i in keep)
        rows.append(values)

    if not rows:
        raise ValueError(
            f"none of the {len(paths)} deposits hold a {signal!r} row"
            + (f" for layer {layer}" if layer is not None else "")
        )
    widths = {r.shape[-1] for r in rows}
    if len(widths) != 1:
        raise ValueError(f"deposits disagree on width: {sorted(widths)}")
    if len(layers) > 1 and layer is None:
        raise ValueError(
            f"deposits mix layers {sorted(layers)}; pass --layer to pick one, "
            "since a residual only means the same thing at one depth"
        )
    stacked = torch.cat(rows, dim=0)
    return stacked, (layers.pop() if layers else -1)


def write_vector(
    vector: torch.Tensor,
    layer: int,
    path: str,
    signal: str = "residual",
    note: str = "",
    dtype: torch.dtype = torch.float32,
) -> str:
    """Write a one-row deposit that ``--signal-inject-from`` accepts."""
    tensors = {
        signal: vector.reshape(1, -1).to(dtype),
        f"{signal}.index": torch.tensor([[0.0, float(layer)]], dtype=torch.float32),
    }
    save_file(
        tensors,
        path,
        {
            "format": DEPOSIT_FORMAT,
            "engine": "vllm",
            "tier": "residual_raw",
            "token_reduce": "built",
            "layer_step": "1",
            "token_step": "1",
            "model": "",
            "session": "",
            "request_id": "",
            "built_from": note,
        },
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.signals.build",
        description=__doc__.split("\n\n")[0],
    )
    sub = parser.add_subparsers(dest="op", required=True)

    p_mean = sub.add_parser("mean", help="average a set of turns")
    p_mean.add_argument("deposits", nargs="+", help="paths or globs")

    p_diff = sub.add_parser("diff", help="mean(A) - mean(B), a contrast direction")
    p_diff.add_argument("--a", nargs="+", required=True, help="the 'with' set")
    p_diff.add_argument("--b", nargs="+", required=True, help="the 'without' set")

    for p in (p_mean, p_diff):
        p.add_argument("-o", "--output", required=True)
        p.add_argument("--signal", default="residual")
        p.add_argument("--layer", type=int, help="required if deposits mix layers")
        p.add_argument(
            "--normalize",
            action="store_true",
            help="scale the result to unit norm, so --signal-inject-alpha is "
            "the whole magnitude",
        )
    args = parser.parse_args(argv)

    if args.op == "mean":
        rows, layer = collect(_expand(args.deposits), args.signal, args.layer)
        vector = rows.mean(0)
        note = f"mean of {rows.shape[0]} rows"
    else:
        a_rows, a_layer = collect(_expand(args.a), args.signal, args.layer)
        b_rows, b_layer = collect(_expand(args.b), args.signal, args.layer)
        if a_layer != b_layer:
            raise SystemExit(
                f"the two sets were captured at different layers "
                f"({a_layer} vs {b_layer}); pass --layer"
            )
        layer = a_layer
        vector = a_rows.mean(0) - b_rows.mean(0)
        note = f"mean({a_rows.shape[0]}) - mean({b_rows.shape[0]})"

    norm = float(vector.norm())
    if args.normalize and norm > 0:
        vector = vector / norm

    path = write_vector(
        vector,
        layer if args.layer is None else args.layer,
        args.output,
        args.signal,
        note,
    )
    print(
        f"{path}: {args.signal} [1, {vector.numel()}] at layer "
        f"{layer if args.layer is None else args.layer} ({note}, norm {norm:.4g})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
