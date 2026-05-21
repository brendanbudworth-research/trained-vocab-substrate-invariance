"""Tokenization screening for the substrate-invariance probe.

For a battery of candidate "alien" symbol categories, report how
OLMo 2's tokenizer handles them. The output is the Tier 1 / Tier 2 /
Tier 3 classification described in research_plan.md Section 6:

  Tier 1 — single-token glyphs. Cleanest substrate-invariance test material.
  Tier 2 — consistent multi-token glyphs. Usable with anchor-aggregated comparison.
  Tier 3 — byte-fallback fragmentation. Most "alien" but most expensive to compare.

Pure CPU; no model load. Finishes in seconds.
"""

from __future__ import annotations

from transformers import AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"


CATEGORIES: dict[str, list[str]] = {
    "Common ASCII (control)": list("abcxyzAB"),
    "Standard logic ops": ["∧", "∨", "¬", "→", "↔", "⊕", "⊻"],
    "Standard quantifiers": ["∀", "∃", "∈", "∉", "⊆", "⊇", "⊢"],
    "Greek lowercase": list("αβγδεζηθικλμ"),
    "Greek uppercase rare": list("ΞΨΩΓΔΛΦΣ"),
    "Cyrillic rare": list("ЁЪЫЭЯ"),
    "Math script / fraktur": ["𝒜", "𝓑", "𝔅", "𝕂", "𝖀"],
    "Tifinagh (Berber)": ["ⴰ", "ⴱ", "ⴲ", "ⴳ", "ⴴ"],
    "Linear B (Mycenaean)": ["𐀀", "𐀁", "𐀂", "𐀃"],
    "Alchemical": ["🜀", "🜁", "🜂", "🜃", "🜄"],
    "Private Use Area": ["\uE000", "\uE001", "\uE100", "\uF8FF"],
}


def classify(n_tokens: int) -> str:
    if n_tokens == 1:
        return "Tier 1"
    if n_tokens <= 3:
        return "Tier 2"
    return "Tier 3"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Tokenizer:  {MODEL_ID}")
    print(f"Vocab size: {tok.vocab_size}")

    totals = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0}

    for cat, syms in CATEGORIES.items():
        print(f"\n[{cat}]")
        print(f"  {'glyph':<8} {'codepoint':<12} {'n_tok':>5}  {'tier':<7}  decoded")
        for s in syms:
            ids = tok.encode(s, add_special_tokens=False)
            n = len(ids)
            tier = classify(n)
            totals[tier] += 1
            cp = f"U+{ord(s):04X}" if len(s) == 1 else "(multi-char)"
            decoded = [tok.decode([i]) for i in ids]
            display = repr(s)[1:-1]  # strip outer quotes for alignment
            print(f"  {display:<8} {cp:<12} {n:>5d}  {tier:<7}  {decoded}")

    print("\n=== Totals ===")
    for tier, count in totals.items():
        print(f"  {tier}: {count}")
    print("\nUse Tier 1 symbols as the first substrate-invariance stimulus set.")
    print("Treat Tier 2 as a fallback when Tier 1 vocabulary is exhausted.")
    print("Reserve Tier 3 for stress-testing the anchor-aggregation methodology.")


if __name__ == "__main__":
    main()
