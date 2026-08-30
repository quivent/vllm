#!/home/ubuntu/vllm/.venv/bin/python3
"""Tools for working with residual states. No restarts, no gemstone dependency.

    ./residual capture "prompt" --name warm --tokens all   mint a deposit
    ./residual rows warm                                   what's inside it
    ./residual seed warm --mode state --row -1             mount it
    ./residual ab "prompt" --residual warm --mode state    control vs seeded
    ./residual status | clear

Everything talks to the engine directly at $QWEN_ENDPOINT (default
http://127.0.0.1:9000).

Two things worth knowing before you use it:

- `state` and `replace` need an engine booted with --signal-inject-from, which
  forces eager. On a graph-mode engine every mount is a silent no-op: the
  injector's hook is Python and a captured CUDA graph never calls it. `status`
  tells you which you are on.
- `--tokens last` (the serving default) keeps only the FINAL position of a
  turn, so every deposit is either "this turn is ending" or, from a
  max_tokens=1 mint, "start answering". Neither is a state worth resuming.
  `capture --tokens all` keeps every position; `rows` then shows you which one
  to seed from with `--row`.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("QWEN_ENDPOINT", "http://127.0.0.1:9000").rstrip("/")
SIGNALS_DIR = os.path.expanduser(os.environ.get("QWEN_SIGNALS", "~/signals"))
ALIASES = os.path.expanduser("~/.gemstone/residuals.json")
MODEL = os.environ.get("QWEN_MODEL", "qwen38")


def call(path, payload=None, method=None):
    url = f"{ENDPOINT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"{path} -> {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"{path} unreachable at {ENDPOINT}: {e.reason}")


def aliases():
    try:
        return json.load(open(ALIASES)).get("aliases", {})
    except Exception:
        return {}


def save_alias(name, filename):
    store = {"aliases": aliases()}
    store["aliases"][name] = filename
    os.makedirs(os.path.dirname(ALIASES), exist_ok=True)
    json.dump(store, open(ALIASES, "w"), indent=2, sort_keys=True)


def resolve(ref):
    """alias | filename | path -> absolute path."""
    if os.path.exists(ref):
        return os.path.abspath(ref)
    mapped = aliases().get(ref, ref)
    path = os.path.join(SIGNALS_DIR, mapped)
    if os.path.exists(path):
        return path
    sys.exit(f"no residual named {ref!r} (try: ./residual list)")


def complete(prompt, max_tokens, temperature=0.0):
    return call("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    })


def deposit_for(request_id, timeout=20.0):
    """Match a deposit to its completion. The engine embeds the id in the name.

    Never take "the newest deposit": ordinary background traffic lands several
    per minute and you will file someone else's turn under your name.
    """
    if "governor-" in request_id:
        sys.exit(f"completion id {request_id} was rewritten by the agentic "
                 "gateway; point QWEN_ENDPOINT at the engine, not the gateway")
    deadline = time.time() + timeout
    while time.time() < deadline:
        for name in os.listdir(SIGNALS_DIR):
            if request_id in name and name.endswith(".safetensors"):
                return os.path.join(SIGNALS_DIR, name)
        time.sleep(0.15)
    sys.exit(f"no deposit for {request_id} after {timeout}s (is capture recording?)")


def inject_body(a):
    """`--layer` picks the recorded row; `--at` picks where to write it.

    They are different questions. The residual stream is one bus, so a state
    recorded at the top of the stack can be written in near the bottom, where
    every layer above then computes that position's K/V from it itself.
    """
    body = {"source": resolve(a.residual), "mode": a.mode,
            "positions": a.positions, "alpha": a.alpha, "row": a.row}
    if a.layer is not None:
        body["layer"] = a.layer
    if getattr(a, "at", None) is not None:
        body["at"] = a.at
    return body


# ---------------------------------------------------------------- commands

def cmd_capture(a):
    was = call("/signals/status").get("tokens", "last")
    if a.tokens != was:
        call("/signals/control", {"tokens": a.tokens})
    try:
        call("/signals/inject", {})          # a capture needs a blank engine
        resp = complete(a.prompt, a.max_tokens)
        path = deposit_for(resp["id"])
    finally:
        if a.tokens != was:
            call("/signals/control", {"tokens": was})
    name = a.name or resp["id"]
    save_alias(name, os.path.basename(path))
    print(f"{name}  ->  {os.path.basename(path)}")
    print(f"  completion: {resp['usage']['completion_tokens']} tokens, tokens={a.tokens}")
    if a.show:
        print("\n" + resp["choices"][0]["message"]["content"][:800])


def cmd_rows(a):
    from safetensors.torch import load_file
    path = resolve(a.residual)
    t = load_file(path)
    index = t.get("residual.index")
    print(f"{os.path.basename(path)}")
    if index is None:
        print("  no index; single row")
        return
    print(f"  {index.shape[0]} rows  (row, token, layer)")
    for i in range(index.shape[0]):
        if a.all or i < 10 or i >= index.shape[0] - 10:
            print(f"    {i:>5}  token={int(index[i,0]):<6} layer={int(index[i,1])}")
        elif i == 10:
            print(f"    ... {index.shape[0]-20} more (--all)")


def cmd_seed(a):
    print(json.dumps(call("/signals/inject", inject_body(a)), indent=2))


def cmd_clear(_a):
    print(json.dumps(call("/signals/inject", {}), indent=2))


def cmd_status(_a):
    cap, inj = call("/signals/status"), call("/signals/injection")
    print(json.dumps({"capture": cap, "injection": inj}, indent=2))
    if inj.get("enabled") and inj.get("applied", 0) == 0:
        print("\n! mounted but applied=0 after traffic means the engine is in "
              "graph mode and injection is a no-op.\n"
              "  Restart with --signal-inject-from to force eager.", file=sys.stderr)


def cmd_ab(a):
    """Control vs seeded, same prompt, greedy. The only honest test."""
    call("/signals/inject", {})
    control = complete(a.prompt, a.max_tokens)["choices"][0]["message"]["content"]

    call("/signals/inject", inject_body(a))
    before = call("/signals/injection").get("applied", 0)
    seeded = complete(a.prompt, a.max_tokens)["choices"][0]["message"]["content"]
    applied = call("/signals/injection").get("applied", 0) - before
    if not a.keep:
        call("/signals/inject", {})

    width = a.width
    print(f"=== CONTROL ===\n{control[:width]}\n")
    print(f"=== SEEDED ({a.mode}, row {a.row}, {a.residual}) ===\n{seeded[:width]}\n")
    print(f"applied this turn: {applied}"
          f"{'   <-- ZERO: injection is not firing' if applied == 0 else ''}")
    print(f"identical: {control == seeded}"
          f"{'   <-- the seed changed nothing' if control == seeded else ''}")


def cmd_list(a):
    known = aliases()
    for name in sorted(known):
        if not a.filter or a.filter in name:
            print(f"{name:<40} {known[name]}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="mint a deposit from a prompt")
    c.add_argument("prompt")
    c.add_argument("--name")
    c.add_argument("--tokens", default="all", choices=["all", "last", "first", "mean"],
                   help="what the deposit keeps (default all, so you can pick a row)")
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--show", action="store_true")
    c.set_defaults(fn=cmd_capture)

    r = sub.add_parser("rows", help="what positions/layers a deposit holds")
    r.add_argument("residual")
    r.add_argument("--all", action="store_true")
    r.set_defaults(fn=cmd_rows)

    for name, fn, helptext in (("seed", cmd_seed, "mount a residual"),
                               ("ab", cmd_ab, "control vs seeded, same prompt")):
        s = sub.add_parser(name, help=helptext)
        if name == "ab":
            s.add_argument("prompt")
            s.add_argument("--residual", required=True)
            s.add_argument("--max-tokens", type=int, default=150)
            s.add_argument("--width", type=int, default=600)
            s.add_argument("--keep", action="store_true", help="leave it mounted")
        else:
            s.add_argument("residual")
        s.add_argument("--mode", default="state", choices=["add", "replace", "state"])
        s.add_argument("--positions", default="first", choices=["first", "all", "zero"])
        s.add_argument("--layer", type=int, default=None,
                       help="which recorded layer's row to load")
        s.add_argument("--at", type=int, default=None,
                       help="which layer to write it into (default: same as --layer)")
        s.add_argument("--row", type=int, default=-1,
                       help="which captured row; -1 is the turn's last position")
        s.add_argument("--alpha", type=float, default=1.0)
        s.set_defaults(fn=fn)

    l = sub.add_parser("list", help="named residuals")
    l.add_argument("filter", nargs="?")
    l.set_defaults(fn=cmd_list)

    sub.add_parser("clear", help="unmount").set_defaults(fn=cmd_clear)
    sub.add_parser("status", help="capture + injection state").set_defaults(fn=cmd_status)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
