"""Substrate-invariance v2: corrected metric and pooling.

Phase 0 finding from 04_toy_substrate_invariance.py:
  - Mean-pool CKA on largely-overlapping token bags is dominated by the
    shared tokens; absolute CKA(A,B) stays ~0.95 at all layers and isn't
    informative as a threshold.
  - The real signal is the *gap* CKA(A,B) - CKA(A,C), which grew
    monotonically from -0.05 to +0.46 across depth.

This script tests two methodological fixes:
  1. Last-token pooling instead of mean-pool. The last token must integrate
     the whole sequence, so its representation is sensitive to compositional
     structure rather than to the token bag.
  2. The substrate-invariance gap (CKA(A,B) - CKA(A,C)) reported as the
     primary metric, with the depth-of-emergence as a quantitative summary.

Also adds a fourth condition (D) — unrelated natural-language sentences —
as a calibration point for what "completely structurally distinct, fluent"
representations look like in CKA terms.
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
    "The negation of {p} holds only when {p} is false.",
    "If {p} implies {q} and {p} is true, then {q} must be true.",
    "{p} and not {q} is true only when {p} is true and {q} is false.",
    "Either {p} or {q} but not both means exactly one of {p}, {q} is true.",
    "If {p} is false, then {p} and {q} is false regardless of {q}.",
    "When {p} equals {q}, both {p} and {q} share the same truth value.",
]

VAR_MAP_GREEK = {"p": "α", "q": "β", "r": "γ", "s": "δ"}

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


def make_renamed(canonical: list[str]) -> list[str]:
    pattern = re.compile(r"\b([pqrs])\b")
    return [pattern.sub(lambda m: VAR_MAP_GREEK[m.group(1)], s) for s in canonical]


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
    num = (X.T @ Y).pow(2).sum()
    denom = (X.T @ X).norm(p="fro") * (Y.T @ Y).norm(p="fro")
    if denom.item() == 0:
        return float("nan")
    return (num / denom).item()


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


def mean_pool_per_layer(
    model, tok, prompts: list[str], device: str
) -> list[torch.Tensor]:
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


def report(name: str, HA: list, HB: list, HC: list, HD: list) -> None:
    print(f"\n=== Pooling strategy: {name} ===")
    print(f"\nLayer  CKA(A,B)  CKA(A,C)  CKA(A,D)   gap=AB-AC   AB-AD")
    print(f"-----  --------  --------  --------   ---------   -----")
    crossed = None
    for layer in range(len(HA)):
        ab = linear_cka(HA[layer], HB[layer])
        ac = linear_cka(HA[layer], HC[layer])
        ad = linear_cka(HA[layer], HD[layer])
        gap = ab - ac
        print(f"  {layer:3d}    {ab:.3f}     {ac:.3f}     {ad:.3f}     {gap:+.3f}      {ab - ad:+.3f}")
        if crossed is None and gap > GAP_THRESHOLD:
            crossed = layer

    if crossed is not None:
        print(f"\nGap > {GAP_THRESHOLD} first reached at layer {crossed}.")
    else:
        print(f"\nGap never exceeded {GAP_THRESHOLD}.")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rng = random.Random(SEED)
    A = make_canonical(rng)
    B = make_renamed(A)
    C = make_scrambled(A, rng)
    D = make_unrelated(rng)

    print("\nExample stimuli:")
    for label, s in [("A canonical", A[0]), ("B renamed  ", B[0]), ("C scrambled", C[0]), ("D unrelated", D[0])]:
        print(f"  {label}: {s}")

    print(f"\nLoading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device).eval()
    print(f"  load time: {time.time() - t0:.1f}s")

    print(f"\nExtracting last-token activations ({4 * N_STIMULI} forward passes) ...")
    t0 = time.time()
    HA_last = last_token_per_layer(model, tok, A, device)
    HB_last = last_token_per_layer(model, tok, B, device)
    HC_last = last_token_per_layer(model, tok, C, device)
    HD_last = last_token_per_layer(model, tok, D, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")

    print(f"\nExtracting mean-pool activations (for comparison) ...")
    t0 = time.time()
    HA_mean = mean_pool_per_layer(model, tok, A, device)
    HB_mean = mean_pool_per_layer(model, tok, B, device)
    HC_mean = mean_pool_per_layer(model, tok, C, device)
    HD_mean = mean_pool_per_layer(model, tok, D, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")

    report("LAST-TOKEN (primary)", HA_last, HB_last, HC_last, HD_last)
    report("MEAN-POOL (for comparison)", HA_mean, HB_mean, HC_mean, HD_mean)

    print("\nInterpretation guide:")
    print("  gap = CKA(A,B) - CKA(A,C) is the substrate-invariance signal.")
    print("  A monotonically increasing gap across depth = structural sensitivity")
    print("  emerges as depth integrates compositional information.")
    print("  CKA(A,D) is the calibration baseline — unrelated fluent text, same model.")
    print("  Look for: CKA(A,B) high AND well-separated from BOTH CKA(A,C) and CKA(A,D)")
    print("  in the deeper layers.")


if __name__ == "__main__":
    main()
