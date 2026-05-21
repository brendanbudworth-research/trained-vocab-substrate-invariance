"""Operator renaming: the next-hardest substrate-invariance test.

Builds on 06 by adding two new conditions where the *logical operators*
(and, or, not, implies) are replaced by invented Tier 2 words while
keeping the surrounding natural-language framing intact.

Predictions worth pre-registering before reading the output:

  (1) Strong Platonic: CKA(A, B') ≈ CKA(A, B) ≈ 1.0 in late layers.
      The model figures out from context that BLIQ plays AND's role
      and treats them as equivalent in representation.

  (2) Partial Platonic: CKA(A, B') starts low, climbs with depth as
      the model accumulates contextual evidence for what BLIQ means,
      but plateaus below CKA(A, B).

  (3) Operator-semantics-bound: CKA(A, B') stays in the 0.3-0.6 range,
      well separated from both CKA(A, B) (substrate-invariant) and
      CKA(A, C) (structureless). The model recognizes a binary infix
      operator syntactically but doesn't recover its semantics.

  (4) Tokenization disruption: CKA(A, B') drops near CKA(A, C). The
      byte-fallback tokens for BLIQ etc. are noise the model can't
      see past.

The depth-of-emergence curve is more informative than the final-layer
value. A late-rising CKA(A, B') would be evidence of in-context
operator binding; a flat low curve would be evidence against.

Conditions:
  A   canonical: English operators, p/q/r/s variables
  B   variable-renamed only: English operators, Greek variables (control)
  B'  operator-renamed only: invented operators, English variables
  B'' both renamed: invented operators, Greek variables
  C   scrambled token-bag control
  D   unrelated natural language calibration baseline
"""

from __future__ import annotations

import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"
N_STIMULI = 50
SEED = 17
GAP_THRESHOLD = 0.10


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

OP_MAP = {
    "and": "bliq",
    "or": "dren",
    "not": "vusp",
    "implies": "molex",
}

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
    out: list[str] = []
    vars_ = ["p", "q", "r", "s"]
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


def last_token_per_layer(
    model, tok, prompts: list[str], device: str
) -> list[torch.Tensor]:
    layer_buffers: list[list[torch.Tensor]] | None = None
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        if device == "mps":
            torch.mps.synchronize()
        last_idx = enc.input_ids.shape[1] - 1
        last = [h[0, last_idx, :].float().cpu() for h in out.hidden_states]
        if layer_buffers is None:
            layer_buffers = [[] for _ in last]
        for buf, vec in zip(layer_buffers, last):
            buf.append(vec)
    assert layer_buffers is not None
    return [torch.stack(buf) for buf in layer_buffers]


def screen_operator_tokenization(tok) -> None:
    print("\nOperator-word tokenization screening:")
    print(f"  {'replacement':<10} {'tokens':<6}  decoded")
    for orig, repl in OP_MAP.items():
        ids = tok.encode(" " + repl, add_special_tokens=False)
        decoded = [tok.decode([i]) for i in ids]
        print(f"  {repl:<10} {len(ids):<6}  {decoded}    (replaces {orig!r})")


def report(name: str, refs: dict, ref_key: str) -> None:
    """refs: dict of label -> list-of-tensors (one per layer)."""
    other_keys = [k for k in refs if k != ref_key]
    n_layers = len(refs[ref_key])

    header = f"\n=== {name} ===\nLayer  " + "  ".join(f"{ref_key}-{k}" for k in other_keys)
    print(header)
    print("-" * len(header))

    for layer in range(n_layers):
        cells = []
        for k in other_keys:
            c = linear_cka(refs[ref_key][layer], refs[k][layer])
            cells.append(f"{c:.3f}" if not (c != c) else "  nan")
        print(f"  {layer:3d}    " + "    ".join(cells))


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

    screen_operator_tokenization(tok)

    rng = random.Random(SEED)
    A = make_canonical(rng)
    B = apply_var_map(A)
    Bp = apply_op_map(A)
    Bpp = apply_op_map(apply_var_map(A))
    C = make_scrambled(A, rng)
    D = make_unrelated(rng)

    print("\nExample stimuli (same logical content across A/B/B'/B''):")
    print(f"  A   canonical:           {A[0]}")
    print(f"  B   var-renamed:         {B[0]}")
    print(f"  B'  op-renamed:          {Bp[0]}")
    print(f"  B'' var+op renamed:      {Bpp[0]}")
    print(f"  C   scrambled:           {C[0]}")
    print(f"  D   unrelated:           {D[0]}")

    print(f"\nExtracting last-token activations (6 conditions x {N_STIMULI}) ...")
    t0 = time.time()
    H = {
        "A": last_token_per_layer(model, tok, A, device),
        "B": last_token_per_layer(model, tok, B, device),
        "B'": last_token_per_layer(model, tok, Bp, device),
        "B''": last_token_per_layer(model, tok, Bpp, device),
        "C": last_token_per_layer(model, tok, C, device),
        "D": last_token_per_layer(model, tok, D, device),
    }
    print(f"  extraction time: {time.time() - t0:.1f}s")

    print("\n\nCKA against canonical (A) at each layer:")
    print(f"\nLayer  CKA(A,B)  CKA(A,B')  CKA(A,B'')  CKA(A,C)  CKA(A,D)")
    print("-----  --------  ---------  ----------  --------  --------")
    n_layers = len(H["A"])
    rows = []
    for layer in range(n_layers):
        ab = linear_cka(H["A"][layer], H["B"][layer])
        abp = linear_cka(H["A"][layer], H["B'"][layer])
        abpp = linear_cka(H["A"][layer], H["B''"][layer])
        ac = linear_cka(H["A"][layer], H["C"][layer])
        ad = linear_cka(H["A"][layer], H["D"][layer])
        rows.append((layer, ab, abp, abpp, ac, ad))
        print(f"  {layer:3d}    {ab:.3f}     {abp:.3f}      {abpp:.3f}       {ac:.3f}     {ad:.3f}")

    print("\nKey diagnostic curves:")
    print("\nLayer  var-gap=B-C   op-gap=B'-C   both-gap=B''-C   B-vs-B' (op cost)")
    print("-----  -----------   -----------   --------------   -----------------")
    for layer, ab, abp, abpp, ac, ad in rows:
        if ab != ab:
            continue
        print(f"  {layer:3d}    {ab - ac:+.3f}        {abp - ac:+.3f}        {abpp - ac:+.3f}            {ab - abp:+.3f}")

    print("\nInterpretation cheat sheet:")
    print("  var-gap large + op-gap large + small B-vs-B' cost  =>  strong Platonic (1)")
    print("  var-gap large + op-gap rising with depth          =>  partial Platonic (2)")
    print("  var-gap large + op-gap flat & modest              =>  syntactic-only (3)")
    print("  var-gap large + op-gap near zero                  =>  tokenization-bound (4)")


if __name__ == "__main__":
    main()
