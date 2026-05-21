"""Toy substrate-invariance experiment.

Generates 50 propositional-logic sentences in three forms:
  A — canonical:   "... p and q ..."
  B — renamed:     "... α and β ..."  (Greek-letter variable substitution)
  C — scrambled:   tokens of A shuffled (negative control)

Extracts mean-pooled residual-stream activations at every layer of
OLMo 2 1B and reports linear CKA between (A, B) and (A, C) per layer.

The methodology works if:
  - CKA(A, B) climbs above ~0.7 at some intermediate or late layer.
  - CKA(A, C) stays well below CKA(A, B) at every layer beyond embedding.

The "structural recovery depth" is the first layer where CKA(A, B)
crosses a threshold (default 0.7). A shallow recovery depth is the
signature of substrate-invariant structural encoding.

Intentionally small, intentionally toy. Goal is intuition, not a result.
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
RECOVERY_THRESHOLD = 0.7


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

VAR_MAP_GREEK = {
    "p": "α", "q": "β", "r": "γ", "s": "δ",
}


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


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA per Kornblith et al. (2019). X, Y: [n, d]."""
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    num = (X.T @ Y).pow(2).sum()
    denom = (X.T @ X).norm(p="fro") * (Y.T @ Y).norm(p="fro")
    if denom.item() == 0:
        return float("nan")
    return (num / denom).item()


def pooled_per_layer(
    model, tok, prompts: list[str], device: str
) -> list[torch.Tensor]:
    """Returns list of [n_prompts, hidden] tensors, one per layer (incl. embedding)."""
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


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rng = random.Random(SEED)
    A = make_canonical(rng)
    B = make_renamed(A)
    C = make_scrambled(A, rng)

    print("\nExample stimuli:")
    for label, s in [("A (canonical)", A[0]), ("B (renamed) ", B[0]), ("C (scrambled)", C[0])]:
        print(f"  {label}: {s}")

    print(f"\nLoading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device).eval()
    print(f"  load time: {time.time() - t0:.1f}s")

    print(f"\nExtracting activations for {N_STIMULI} stimuli x 3 forms ...")
    t0 = time.time()
    HA = pooled_per_layer(model, tok, A, device)
    HB = pooled_per_layer(model, tok, B, device)
    HC = pooled_per_layer(model, tok, C, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers (incl. embedding): {len(HA)}, hidden_dim: {HA[0].shape[1]}")

    print(f"\nLayer  CKA(A,B)  CKA(A,C)   gap")
    print(f"-----  --------  --------   -----")
    recovery = None
    for layer in range(len(HA)):
        cab = linear_cka(HA[layer], HB[layer])
        cac = linear_cka(HA[layer], HC[layer])
        print(f"  {layer:3d}    {cab:.3f}     {cac:.3f}   {cab - cac:+.3f}")
        if recovery is None and cab > RECOVERY_THRESHOLD:
            recovery = layer

    print()
    if recovery is not None:
        print(f"Structural recovery depth (first layer with CKA(A,B) > {RECOVERY_THRESHOLD}): layer {recovery}")
    else:
        max_layer = max(range(len(HA)), key=lambda l: linear_cka(HA[l], HB[l]))
        max_val = linear_cka(HA[max_layer], HB[max_layer])
        print(f"No layer crossed CKA(A,B) > {RECOVERY_THRESHOLD}. Peak was {max_val:.3f} at layer {max_layer}.")
        print("Interpretation: either the toy is too small, the variable substitution")
        print("is too disruptive, or substrate-invariance isn't holding for this stimulus class.")


if __name__ == "__main__":
    main()
