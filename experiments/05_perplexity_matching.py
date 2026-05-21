"""Perplexity-matching exploration.

For a structured target prompt, sample random token sequences from
OLMo 2's vocabulary and find the ones whose perplexity (under OLMo
itself) matches the target within a tolerance band.

This builds intuition for whether the perplexity-matched-control
infrastructure described in research_plan.md Section 6 is feasible
to scale, and how "expensive" (in samples drawn) it is to find a
match for various target perplexities.

If matching is cheap and stable, the perplexity-matched control is a
viable mandatory baseline. If it's brittle or expensive, we need a
fallback strategy.
"""

from __future__ import annotations

import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"
TARGETS = [
    "If p and q are both true, then p and q is true.",
    "α ⊕ β is true exactly when α and β disagree.",
    "𐀀 𐀁 𐀂 𐀃 𐀄 𐀅 𐀆 𐀇",
]
N_SAMPLES = 200
SEQ_LEN = 16
TOLERANCE = 0.10


def perplexity(model, tok, text: str, device: str) -> float:
    enc = tok(text, return_tensors="pt").to(device)
    if enc.input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(**enc, labels=enc.input_ids)
    if device == "mps":
        torch.mps.synchronize()
    return float(math.exp(out.loss.item()))


def perplexity_from_ids(model, ids: torch.Tensor, device: str) -> float:
    if ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    if device == "mps":
        torch.mps.synchronize()
    return float(math.exp(out.loss.item()))


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

    vocab_size = model.config.vocab_size
    gen = torch.Generator()
    gen.manual_seed(0)

    print(f"\nSampling {N_SAMPLES} random {SEQ_LEN}-token sequences ...")
    t0 = time.time()
    random_ids = torch.randint(0, vocab_size, (N_SAMPLES, SEQ_LEN), generator=gen).to(device)
    random_ppls = []
    for i in range(N_SAMPLES):
        ppl = perplexity_from_ids(model, random_ids[i : i + 1], device)
        if math.isfinite(ppl) and ppl < 1e9:
            random_ppls.append(ppl)
    random_ppls.sort()
    print(f"  scored {len(random_ppls)} in {time.time() - t0:.1f}s")
    if not random_ppls:
        print("  no usable random samples; aborting")
        return
    print(f"  random ppl percentiles: "
          f"p10={random_ppls[len(random_ppls) // 10]:.1f}  "
          f"p50={random_ppls[len(random_ppls) // 2]:.1f}  "
          f"p90={random_ppls[len(random_ppls) * 9 // 10]:.1f}")

    print(f"\nTarget prompts and matching coverage (tolerance ±{TOLERANCE * 100:.0f}%):")
    for target in TARGETS:
        tgt_ppl = perplexity(model, tok, target, device)
        if not math.isfinite(tgt_ppl):
            print(f"\n  target: {target!r}\n    ppl unavailable")
            continue
        lo, hi = tgt_ppl * (1 - TOLERANCE), tgt_ppl * (1 + TOLERANCE)
        n_match = sum(1 for p in random_ppls if lo <= p <= hi)
        rate = n_match / len(random_ppls)
        print(f"\n  target: {target!r}")
        print(f"    target ppl: {tgt_ppl:.2f}")
        print(f"    band [{lo:.1f}, {hi:.1f}]")
        print(f"    matching random samples: {n_match}/{len(random_ppls)}  ({rate * 100:.1f}%)")
        if rate < 0.02:
            print("    -> rare. Will need biased sampling or a longer search budget.")
        elif rate < 0.10:
            print("    -> usable but expensive. Plan for ~50-200x oversampling.")
        else:
            print("    -> matching is cheap for this target ppl.")


if __name__ == "__main__":
    main()
