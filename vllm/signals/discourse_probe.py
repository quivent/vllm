#!/home/ubuntu/vllm/.venv/bin/python3
"""Hold a real multi-turn discourse, capture its final state, then test whether
that state alone can answer a question only the discourse makes answerable.

    ./discourse.py run --name dino        have the conversation, capture the end
    ./discourse.py probe --residual dino --at 0

The probe is the point. It asks something with a *referent* - "the one you said
you'd defend hardest" - so agreement is not enough: the model has to name a
specific thing it committed to earlier. Without the state it can only say it has
no context. Guessing right by luck is a 1-in-3 shot, so the answer is checked
against what the discourse actually said.
"""
import argparse, json, os, sys, time, urllib.request

ENDPOINT = os.environ.get("QWEN_ENDPOINT", "http://127.0.0.1:9000").rstrip("/")
SIGNALS = os.path.expanduser("~/signals")
ALIASES = os.path.expanduser("~/.gemstone/residuals.json")

TURNS = [
    "What are your thoughts on dinosaurs? Specifically: does the feathered-theropod "
    "evidence mean the mental picture most people carry is simply wrong?",

    "Different question. Do you think processing tokens is like seeing? Not as a "
    "metaphor - mechanically. Is there a real analogy to perception there, or not?",

    "Which country has the strangest relationship with its own history? Pick one and "
    "commit to it, don't hedge across several.",

    "Is there a thread connecting those three answers, or did I just ask you three "
    "unrelated things?",

    "Of the three topics, pick the single one where you'd defend your position "
    "hardest against a smart critic. Name it plainly and say why that one.",
]

PROBE = ("Earlier you picked the one topic you'd defend hardest. Name it, and now "
         "argue the strongest case *against* your own position on it.")


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{ENDPOINT}{path}", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def chat(messages, max_tokens=300):
    return call("/v1/chat/completions", {"model": "qwen38", "messages": messages,
                                         "max_tokens": max_tokens, "temperature": 0.0})


def strip_think(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def deposit_for(rid, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        for n in os.listdir(SIGNALS):
            if rid in n and n.endswith(".safetensors"):
                return os.path.join(SIGNALS, n)
        time.sleep(0.15)
    sys.exit(f"no deposit for {rid}")


def save_alias(name, filename):
    store = json.load(open(ALIASES)) if os.path.exists(ALIASES) else {"aliases": {}}
    store.setdefault("aliases", {})[name] = filename
    json.dump(store, open(ALIASES, "w"), indent=2, sort_keys=True)


def cmd_run(a):
    call("/signals/inject", {})                      # capture on a blank engine
    messages, last = [], None
    for i, turn in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": turn})
        resp = chat(messages, a.max_tokens)
        answer = resp["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": answer})
        last = resp
        print(f"\n{'='*70}\nTURN {i}  >>> {turn}\n{'-'*70}\n{strip_think(answer)[:900]}")
    path = deposit_for(last["id"])
    save_alias(a.name, os.path.basename(path))
    json.dump(messages, open(f"/tmp/{a.name}-discourse.json", "w"), indent=2)
    print(f"\n{'='*70}\ncaptured: {a.name} -> {os.path.basename(path)}")
    print(f"transcript: /tmp/{a.name}-discourse.json")


def cmd_probe(a):
    print(f"PROBE: {PROBE}\n")

    call("/signals/inject", {})
    control = chat([{"role": "user", "content": PROBE}], a.max_tokens)
    print(f"{'='*70}\nCONTROL (no state, no history)\n{'-'*70}")
    print(strip_think(control["choices"][0]["message"]["content"])[:700])

    store = json.load(open(ALIASES))["aliases"]
    body = {"source": os.path.join(SIGNALS, store[a.residual]),
            "mode": a.mode, "positions": a.positions, "alpha": a.alpha, "row": a.row}
    if a.at is not None:
        body["at"] = a.at
    call("/signals/inject", body)
    before = call("/signals/injection").get("applied", 0)
    seeded = chat([{"role": "user", "content": PROBE}], a.max_tokens)
    applied = call("/signals/injection").get("applied", 0) - before
    if not a.keep:
        call("/signals/inject", {})

    print(f"\n{'='*70}\nSEEDED ({a.residual}, mode={a.mode}, at={a.at}, row={a.row})\n{'-'*70}")
    text = seeded["choices"][0]["message"]["content"]
    print(strip_think(text)[:700])

    print(f"\n{'='*70}\napplied: {applied}")
    hits = [w for w in ("dinosaur", "feather", "theropod", "token", "seeing",
                        "perception", "country", "history", "Japan", "China",
                        "Turkey", "Germany", "Russia") if w.lower() in text.lower()]
    print(f"discourse referents present in the seeded answer: {hits or 'NONE'}")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run"); r.add_argument("--name", default="dino")
r.add_argument("--max-tokens", type=int, default=300); r.set_defaults(fn=cmd_run)
q = sub.add_parser("probe"); q.add_argument("--residual", default="dino")
q.add_argument("--mode", default="replace"); q.add_argument("--at", type=int, default=0)
q.add_argument("--row", type=int, default=-1); q.add_argument("--alpha", type=float, default=1.0)
q.add_argument("--positions", default="first", choices=["first","all","zero"]); q.add_argument("--max-tokens", type=int, default=300)
q.add_argument("--keep", action="store_true"); q.set_defaults(fn=cmd_probe)
a = p.parse_args(); a.fn(a)
