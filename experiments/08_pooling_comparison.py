"""Three-pooling comparison: diagnosing the operator-renaming artifact from script 07.

Script 07 reported CKA(A, B') = 0.999 from layer 1 onward — almost certainly
a positional artifact, not real operator substrate-invariance. The last token
of every prompt is "." and doesn't strongly attend to operators in the middle
of the sentence. So operator changes don't perturb the last-token residual
stream much, regardless of whether the model is actually treating "bliq" and
"and" as semantically equivalent.

This script runs the same six conditions (A, B, B', B'', C, D) and reports
CKA(A, *) at every layer under three pooling strategies:

  1. last-token:        reproduces 07's measurement, expected to show the artifact
  2. mean-pool:         averages over all tokens, sensitive to operator presence
  3. operator-anchored: extracts at the position immediately after the first
                        operator's last subtoken. Principled measure for
                        operator substrate-invariance specifically.

The discrepancy between the three is itself the methodological finding.
If mean-pool and operator-anchored show CKA(A, B') substantially below 1.0
while last-token shows ~1.0, we've confirmed the artifact diagnosis.
If all three converge near 1.0, then the strong-Platonic reading might
actually be real (very unlikely, but the experiment is designed to allow
this conclusion).
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-1124-7B"
# For 1B reference: MODEL_ID = "allenai/OLMo-2-0425-1B"
N_STIMULI = 50
SEED = 17


TEMPLATES = [
    "If {p} and {q} are both true, then {p} and {q} is true.",
    "{p} or {q} is true when at least one of {p} or {q} is true.",
    "If not {p}, then {p} and {q} is false.",
    "If {p} implies {q} and {p} is true, then {q} must be true.",
    "{p} and not {q} is true only when {p} is true and {q} is false.",
    "Either {p} or {q} but not both means exactly one of {p}, {q} is true.",
    "If {p} is false, then {p} and {q} is false regardless of {q}.",
    "When {p} or {q} is true, at least one of {p} and {q} must be true.",
]

VAR_MAP_GREEK = {"p": "α", "q": "β", "r": "γ", "s": "δ"}
OP_MAP = {"and": "bliq", "or": "dren", "not": "vusp", "implies": "molex"}

UNRELATED = [
    "The Pacific Ocean is the largest of the world's oceans.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Beethoven composed nine symphonies during his lifetime in Vienna.",
    "Mount Everest sits on the border between Nepal and Tibet.",
    "The Renaissance flourished in Italy during the fifteenth century.",
    "Honeybees communicate the location of food through dance patterns.",
    "Mars has two small moons named Phobos and Deimos.",
    "The Magna Carta was signed by King John in twelve fifteen.",
    "Ocean currents distribute heat between equatorial and polar regions.",
    "Penicillin was discovered accidentally by Alexander Fleming in 1928.",
]


def make_canonical(rng: random.Random) -> list[str]:
    vars_ = ["p", "q", "r", "s"]
    out: list[str] = []
    for _ in range(N_STIMULI):
        tmpl = rng.choice(TEMPLATES)
        p, q = rng.sample(vars_, 2)
        out.append(tmpl.format(p=p, q=q))
    return out


def apply_var_map(prompts: list[str]) -> list[str]:
    pattern = re.compile(r"\b([pqrs])\b")
    return [pattern.sub(lambda m: VAR_MAP_GREEK[m.group(1)], s) for s in prompts]


def apply_op_map(prompts: list[str]) -> list[str]:
    pattern = re.compile(r"\b(" + "|".join(OP_MAP.keys()) + r")\b")
    return [pattern.sub(lambda m: OP_MAP[m.group(1)], s) for s in prompts]


def make_scrambled(canonical: list[str], rng: random.Random) -> list[str]:
    out: list[str] = []
    for s in canonical:
        toks = s.split()
        rng.shuffle(toks)
        out.append(" ".join(toks))
    return out


def make_unrelated(rng: random.Random) -> list[str]:
    pool = UNRELATED * (N_STIMULI // len(UNRELATED) + 1)
    rng.shuffle(pool)
    return pool[:N_STIMULI]


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    nx = (X.T @ X).norm(p="fro")
    ny = (Y.T @ Y).norm(p="fro")
    if nx.item() == 0 or ny.item() == 0:
        return float("nan")
    num = (X.T @ Y).pow(2).sum()
    return min((num / (nx * ny)).item(), 1.0)


def find_operator_position(tok, prompt: str, operators: list[str]) -> int | None:
    """Position (in the tokenized prompt) immediately after the last subtoken
    of the first occurrence of any operator word. Returns None if no operator
    found, which shouldn't happen for our templates."""
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    decoded_tokens = [tok.decode([i]) for i in ids]
    for op in operators:
        target = " " + op
        joined = ""
        for i, t in enumerate(decoded_tokens):
            joined += t
            if joined.endswith(target):
                return i + 1 if i + 1 < len(ids) else i
        joined = ""
    return None


def extract_per_layer(
    model,
    tok,
    prompts: list[str],
    device: str,
    position_fn: Callable[[str, list[int]], int | None],
) -> list[torch.Tensor]:
    """position_fn takes (prompt, token_ids) and returns a position or None.
    If None for any prompt, that prompt is skipped (kept consistent across conditions
    by caller responsibility)."""
    layer_buffers: list[list[torch.Tensor]] | None = None
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        ids = enc.input_ids[0].tolist()
        pos = position_fn(p, ids)
        if pos is None or pos >= enc.input_ids.shape[1]:
            pos = enc.input_ids.shape[1] - 1  # fallback to last
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        if device == "mps":
            torch.mps.synchronize()
        layer_vecs = [h[0, pos, :].float().cpu() for h in out.hidden_states]
        if layer_buffers is None:
            layer_buffers = [[] for _ in layer_vecs]
        for buf, vec in zip(layer_buffers, layer_vecs):
            buf.append(vec)
    assert layer_buffers is not None
    return [torch.stack(buf) for buf in layer_buffers]


def extract_mean_pool(model, tok, prompts: list[str], device: str) -> list[torch.Tensor]:
    layer_buffers: list[list[torch.Tensor]] | None = None
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        if device == "mps":
            torch.mps.synchronize()
        pooled = [h.mean(dim=1).squeeze(0).float().cpu() for h in out.hidden_states]
        if layer_buffers is None:
            layer_buffers = [[] for _ in pooled]
        for buf, vec in zip(layer_buffers, pooled):
            buf.append(vec)
    assert layer_buffers is not None
    return [torch.stack(buf) for buf in layer_buffers]


def make_position_fn(
    tok, operators: list[str], fallback_last: bool = True
):
    def fn(prompt: str, ids: list[int]) -> int | None:
        decoded_tokens = [tok.decode([i]) for i in ids]
        for op in operators:
            target = " " + op
            joined = ""
            for i, t in enumerate(decoded_tokens):
                joined += t
                if joined.endswith(target):
                    return i + 1 if i + 1 < len(ids) else i
            joined = ""
        return (len(ids) - 1) if fallback_last else None

    return fn


def last_token_fn(prompt: str, ids: list[int]) -> int:
    return len(ids) - 1


def report_table(name: str, all_cka: dict[str, list[float]]) -> None:
    n_layers = len(next(iter(all_cka.values())))
    keys = list(all_cka.keys())
    print(f"\n=== {name} ===")
    header = "Layer  " + "  ".join(f"{k:>9s}" for k in keys)
    print(header)
    print("-" * len(header))
    for layer in range(n_layers):
        cells = []
        for k in keys:
            v = all_cka[k][layer]
            cells.append(f"{v:>9.3f}" if not (v != v) else "      nan")
        print(f"  {layer:3d}  " + "  ".join(cells))


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device).eval()
    print(f"  load time: {time.time() - t0:.1f}s")

    rng = random.Random(SEED)
    prompts = {
        "A": make_canonical(rng),
        "B": None,
        "B'": None,
        "B''": None,
        "C": None,
        "D": None,
    }
    prompts["B"] = apply_var_map(prompts["A"])
    prompts["B'"] = apply_op_map(prompts["A"])
    prompts["B''"] = apply_op_map(apply_var_map(prompts["A"]))
    prompts["C"] = make_scrambled(prompts["A"], rng)
    prompts["D"] = make_unrelated(rng)

    canonical_ops = list(OP_MAP.keys())
    invented_ops = list(OP_MAP.values())

    op_position_canonical = make_position_fn(tok, canonical_ops)
    op_position_invented = make_position_fn(tok, invented_ops)
    no_op_position = make_position_fn(tok, [], fallback_last=True)

    pos_fn_per_condition = {
        "A": op_position_canonical,
        "B": op_position_canonical,
        "B'": op_position_invented,
        "B''": op_position_invented,
        "C": no_op_position,
        "D": no_op_position,
    }

    print("\nExtracting activations under three pooling strategies ...")
    print("  This runs 6 conditions x 3 strategies x 50 prompts = 900 forward passes")
    print("  (one forward pass per prompt yields all three pooling extractions)")
    t0 = time.time()

    last: dict[str, list[torch.Tensor]] = {}
    mean: dict[str, list[torch.Tensor]] = {}
    opanc: dict[str, list[torch.Tensor]] = {}

    for label, ps in prompts.items():
        layer_last: list[list[torch.Tensor]] | None = None
        layer_mean: list[list[torch.Tensor]] | None = None
        layer_op: list[list[torch.Tensor]] | None = None
        pos_fn = pos_fn_per_condition[label]
        for p in ps:
            enc = tok(p, return_tensors="pt").to(device)
            ids = enc.input_ids[0].tolist()
            op_pos = pos_fn(p, ids)
            if op_pos is None or op_pos >= enc.input_ids.shape[1]:
                op_pos = enc.input_ids.shape[1] - 1
            last_pos = enc.input_ids.shape[1] - 1
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            if device == "mps":
                torch.mps.synchronize()
            l_last = [h[0, last_pos, :].float().cpu() for h in out.hidden_states]
            l_mean = [h.mean(dim=1).squeeze(0).float().cpu() for h in out.hidden_states]
            l_op = [h[0, op_pos, :].float().cpu() for h in out.hidden_states]
            if layer_last is None:
                layer_last = [[] for _ in l_last]
                layer_mean = [[] for _ in l_mean]
                layer_op = [[] for _ in l_op]
            for buf, vec in zip(layer_last, l_last):
                buf.append(vec)
            for buf, vec in zip(layer_mean, l_mean):
                buf.append(vec)
            for buf, vec in zip(layer_op, l_op):
                buf.append(vec)
        last[label] = [torch.stack(b) for b in layer_last]
        mean[label] = [torch.stack(b) for b in layer_mean]
        opanc[label] = [torch.stack(b) for b in layer_op]
    print(f"  extraction time: {time.time() - t0:.1f}s")

    def compute_cka_vs_A(reps: dict[str, list[torch.Tensor]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        n_layers = len(reps["A"])
        for k in ["B", "B'", "B''", "C", "D"]:
            out[k] = [linear_cka(reps["A"][l], reps[k][l]) for l in range(n_layers)]
        return out

    print("\n\n###### LAST-TOKEN (reproducing script 07) ######")
    report_table("CKA(A, *) under LAST-TOKEN pooling", compute_cka_vs_A(last))

    print("\n\n###### MEAN-POOL (sensitive to operator presence) ######")
    report_table("CKA(A, *) under MEAN-POOL", compute_cka_vs_A(mean))

    print("\n\n###### OPERATOR-ANCHORED (post-first-operator position) ######")
    report_table("CKA(A, *) under OPERATOR-ANCHORED pooling", compute_cka_vs_A(opanc))

    print("\n\nDiagnostic: B-vs-B' gap (variable-renaming CKA - operator-renaming CKA)")
    print("This is the cleanest single number for 'how much harder is operator renaming?'")
    print()
    print("Layer  last-token   mean-pool   operator-anchored")
    print("-----  ----------   ---------   -----------------")
    last_cka = compute_cka_vs_A(last)
    mean_cka = compute_cka_vs_A(mean)
    opanc_cka = compute_cka_vs_A(opanc)
    n_layers = len(last["A"])
    for layer in range(n_layers):
        lt = last_cka["B"][layer] - last_cka["B'"][layer]
        mp = mean_cka["B"][layer] - mean_cka["B'"][layer]
        op = opanc_cka["B"][layer] - opanc_cka["B'"][layer]
        def fmt(v):
            return f"{v:+.3f}" if not (v != v) else "   nan"
        print(f"  {layer:3d}    {fmt(lt)}      {fmt(mp)}     {fmt(op)}")

    print("\nReading guide:")
    print("  - LAST-TOKEN B-vs-B' near zero    = positional artifact (07's finding)")
    print("  - MEAN-POOL B-vs-B' positive      = operator-renaming costs something real")
    print("  - OP-ANCHORED B-vs-B' positive    = operator semantics specifically aren't recovered")
    print("  - All three near zero             = genuine strong Platonic (extraordinary claim)")
    print("  - All three large + diverging     = three measurements of three different things")


if __name__ == "__main__":
    main()
