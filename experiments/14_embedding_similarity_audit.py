"""Embedding-similarity audit.

Script 13 surfaced an unexpected within-unary heterogeneity:

  bliq   -> 88% to `not`         dren  -> 74% to `not`
  vusp   -> 34% to `not`, 46% to `necessarily`  (split)
  molex  -> 71% to `necessarily`
  perph  -> 72% to `necessarily`

The reinterpreted H1 says invented operators land in the unary-class region,
with the not vs necessarily landing modulated by H3 (word-embedding) and
H4 (template-context).

This script tests the H3 within-unary claim directly: do the per-word
landings in script 13 correlate with each invented word's embedding-layer
cosine similarity to `not` vs `necessarily`?

Predictions, if H3 (within-unary regime) is correct:

  - bliq, dren should have higher cosine similarity to `not` than to
    `necessarily` (predicting their `not`-lean)
  - molex, perph should have higher cosine similarity to `necessarily`
    than to `not` (predicting their `necessarily`-lean)
  - vusp should be roughly equal (predicting the split)

If the embedding similarities don't predict the script-13 landings, H3
is not a layer-0 phenomenon and must be constructed by later layers
(potentially overturning the simple H3 story).

We also include the L=1 sets from script 11 to test the cross-class H3
regime: `bar` should have unusually high similarity to `or` compared
to other invented words.

For multi-subword invented words, we report three pooling strategies:
mean, first-subword-only, last-subword-only. The probe operates on
residual-stream activations at the operator-anchored position, which
is the position immediately after the last subword. Last-subword
embedding is therefore the most architecturally-relevant baseline; mean
is the conventional measurement.

Method:
  - Load OLMo 2 7B's embedding layer.
  - For each canonical and each invented word, tokenize with leading
    space (matching probe-extraction conventions) and report subword
    decomposition.
  - Compute cosine similarity between each invented word's embedding
    and each canonical's embedding under three pooling strategies.
  - Report the predicted "winner" canonical per invented word.
  - Compare with script-13 actual outcomes.

This is a cheap test (no forward passes, just embedding-matrix lookup)
that either reinforces or refutes the H3-within-unary mechanism story.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-1124-7B"

CANONICALS: list[str] = ["and", "or", "not", "implies", "necessarily"]

INVENTED_GROUPS: dict[str, list[str]] = {
    "L=2 set (scripts 9, 10, 12, 13)": ["bliq", "dren", "vusp", "molex", "perph"],
    "L=1 set A (script 10, 11)": ["foo", "bar", "baz", "fred"],
    "L=1 set C (script 11)": ["qux", "quux", "thud", "pop"],
    "L=1 set C-extra (script 11)": ["zap", "ping", "huh"],
    "L=3 set (script 10)": ["bligrex", "drentup", "vuspect", "molecule"],
    "L=4 set (script 10)": ["bliquenter", "drentopals", "vusperinder", "molexicode"],
}

# Reference: script 13's actual landings (per-word totals across all 5 templates,
# 250 stimuli per word). Used to compare predicted-by-embedding vs observed.
SCRIPT13_LANDINGS_PCT: dict[str, dict[str, float]] = {
    "bliq":  {"and": 0.0, "or": 1.2, "not": 88.0, "implies": 3.2, "necessarily": 9.2},
    "dren":  {"and": 0.0, "or": 0.0, "not": 74.4, "implies": 0.0, "necessarily": 25.6},
    "vusp":  {"and": 0.0, "or": 6.8, "not": 34.0, "implies": 13.2, "necessarily": 46.0},
    "molex": {"and": 0.0, "or": 0.0, "not": 28.8, "implies": 0.0, "necessarily": 71.2},
    "perph": {"and": 0.0, "or": 1.2, "not": 26.4, "implies": 1.2, "necessarily": 71.6},
}


def get_word_embeddings(
    model, tok, word: str, device: str
) -> tuple[torch.Tensor, list[str]]:
    """Return (subword_embeddings, decoded_subwords) for the leading-space form
    of `word`. The probe extraction uses operator-anchored positions on prompts
    where the operator follows a space, so we use the leading-space tokenization
    here for consistency."""
    target = " " + word
    ids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids[0]
    decoded = [tok.decode([i]) for i in ids.tolist()]
    embed_layer = model.get_input_embeddings()
    vecs = embed_layer(ids.to(device)).float().detach().cpu()
    return vecs, decoded


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item())


def predicted_winner(
    word_vec: torch.Tensor, canonical_vecs: dict[str, torch.Tensor]
) -> tuple[str, float]:
    sims = {c: cosine(word_vec, v) for c, v in canonical_vecs.items()}
    winner = max(sims.items(), key=lambda kv: kv[1])
    return winner[0], winner[1]


def pool_embedding(vecs: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "mean":
        return vecs.mean(dim=0)
    if strategy == "first":
        return vecs[0]
    if strategy == "last":
        return vecs[-1]
    raise ValueError(f"unknown pooling: {strategy}")


def main() -> None:
    device = (
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device).eval()
    print(f"  load time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print("Canonical operator tokenizations (with leading space)")
    print("=" * 80)
    canonical_subwords: dict[str, list[str]] = {}
    canonical_vecs_raw: dict[str, torch.Tensor] = {}
    for c in CANONICALS:
        vecs, decoded = get_word_embeddings(model, tok, c, device)
        canonical_subwords[c] = decoded
        canonical_vecs_raw[c] = vecs
        print(f"  {c:<14s} -> {len(decoded)} subword(s): {decoded}")

    print("\n" + "=" * 80)
    print("Canonical-canonical baseline cosine similarities (mean pooling)")
    print("(For multi-subword canonicals this differs from single-subword cosine.)")
    print("=" * 80)
    canonical_mean = {c: pool_embedding(canonical_vecs_raw[c], "mean") for c in CANONICALS}
    print()
    header = "  " + "       " + "".join(f"{c:>14s}" for c in CANONICALS)
    print(header)
    for c1 in CANONICALS:
        row = f"  {c1:<7s}"
        for c2 in CANONICALS:
            sim = cosine(canonical_mean[c1], canonical_mean[c2])
            row += f"{sim:>14.4f}"
        print(row)
    print()
    print("Note: diagonal == 1.000 confirms self-similarity. Off-diagonals give")
    print("the baseline pairwise canonical separation in embedding space; a")
    print("similarity of 0.20 between distinct canonicals is the typical magnitude.")

    for group_name, words in INVENTED_GROUPS.items():
        print("\n" + "=" * 80)
        print(f"Invented words: {group_name}")
        print("=" * 80)

        for word in words:
            vecs, decoded = get_word_embeddings(model, tok, word, device)
            print(f"\n  {word:<14s} -> {len(decoded)} subword(s): {decoded}")

            for strategy in ["mean", "first", "last"]:
                if len(vecs) == 1 and strategy != "mean":
                    continue
                pooled = pool_embedding(vecs, strategy)
                sims = {c: cosine(pooled, canonical_mean[c]) for c in CANONICALS}
                winner = max(sims.items(), key=lambda kv: kv[1])
                strat_label = f"{strategy} pool" if len(vecs) > 1 else "single subword"
                sims_str = "  ".join(f"{c}: {sims[c]:+.4f}" for c in CANONICALS)
                marker = "  <- winner: " + winner[0]
                print(f"    [{strat_label:>11s}]  {sims_str}{marker}")

    # Detailed comparison: script-13 words vs predicted-by-embedding rankings.
    print("\n" + "=" * 80)
    print("H3 prediction check: does embedding similarity predict script-13 landings?")
    print("=" * 80)
    print()
    print(f"  For each script-13 invented word, we rank canonicals by embedding")
    print(f"  cosine similarity (mean pooling) and compare to script 13's observed")
    print(f"  landing distribution. A correct H3-within-unary prediction means the")
    print(f"  top-by-similarity canonical matches the top-by-landings canonical.")
    print()

    print(
        f"  {'Word':<8s} "
        f"{'Predicted (mean)':<24s}  "
        f"{'Predicted (last)':<24s}  "
        f"{'Observed top':<14s}  "
        f"{'Match?':<8s}"
    )
    print(f"  {'-' * 8} {'-' * 24}  {'-' * 24}  {'-' * 14}  {'-' * 8}")

    n_match_mean = 0
    n_match_last = 0
    script13_words = list(SCRIPT13_LANDINGS_PCT.keys())
    for word in script13_words:
        vecs, _ = get_word_embeddings(model, tok, word, device)
        mean_pooled = pool_embedding(vecs, "mean")
        last_pooled = pool_embedding(vecs, "last")

        sims_mean = {c: cosine(mean_pooled, canonical_mean[c]) for c in CANONICALS}
        sims_last = {c: cosine(last_pooled, canonical_mean[c]) for c in CANONICALS}

        pred_mean = max(sims_mean.items(), key=lambda kv: kv[1])
        pred_last = max(sims_last.items(), key=lambda kv: kv[1])

        landings = SCRIPT13_LANDINGS_PCT[word]
        observed_top = max(landings.items(), key=lambda kv: kv[1])

        mean_match = pred_mean[0] == observed_top[0]
        last_match = pred_last[0] == observed_top[0]
        n_match_mean += int(mean_match)
        n_match_last += int(last_match)

        pred_mean_str = f"{pred_mean[0]} ({pred_mean[1]:+.3f})"
        pred_last_str = f"{pred_last[0]} ({pred_last[1]:+.3f})"
        observed_str = f"{observed_top[0]} ({observed_top[1]:.1f}%)"
        match_str = (
            f"{'M' if mean_match else '.'}"
            f"{'L' if last_match else '.'}"
        )

        print(
            f"  {word:<8s} "
            f"{pred_mean_str:<24s}  "
            f"{pred_last_str:<24s}  "
            f"{observed_str:<14s}  "
            f"{match_str:<8s}"
        )
    print()
    print(f"  Mean-pool predicts top canonical correctly: {n_match_mean}/{len(script13_words)}")
    print(f"  Last-subword predicts top canonical correctly: {n_match_last}/{len(script13_words)}")

    # Within-unary specific test: for each script-13 word, is the not-vs-necessarily
    # similarity ranking consistent with the observed not-vs-necessarily landing ratio?
    print("\n" + "=" * 80)
    print("Within-unary H3 check: not vs necessarily similarity vs landings")
    print("=" * 80)
    print()
    print(f"  Tests whether each word's relative similarity to `not` vs `necessarily`")
    print(f"  predicts its observed not-vs-necessarily landing ratio in script 13.")
    print()

    print(
        f"  {'Word':<8s} "
        f"{'sim(not)':>10s} {'sim(necc)':>10s} {'diff':>10s} | "
        f"{'pct(not)':>10s} {'pct(necc)':>11s} {'ratio':>10s} | "
        f"{'agree?':<8s}"
    )
    print(f"  {'-' * 8} " + "-" * 33 + " | " + "-" * 35 + " | " + "-" * 8)

    n_agree = 0
    for word in script13_words:
        vecs, _ = get_word_embeddings(model, tok, word, device)
        pooled = pool_embedding(vecs, "mean")
        sim_not = cosine(pooled, canonical_mean["not"])
        sim_necc = cosine(pooled, canonical_mean["necessarily"])
        diff_sim = sim_not - sim_necc

        landings = SCRIPT13_LANDINGS_PCT[word]
        pct_not = landings["not"]
        pct_necc = landings["necessarily"]
        ratio_landings = pct_not - pct_necc

        agree = (diff_sim > 0) == (ratio_landings > 0)
        n_agree += int(agree)

        print(
            f"  {word:<8s} "
            f"{sim_not:>+10.4f} {sim_necc:>+10.4f} {diff_sim:>+10.4f} | "
            f"{pct_not:>9.1f}% {pct_necc:>10.1f}% {ratio_landings:>+10.1f} | "
            f"{'YES' if agree else 'NO':<8s}"
        )
    print()
    print(f"  Sign-agreement: {n_agree}/{len(script13_words)} words have consistent")
    print(f"  not-vs-necessarily ordering between embedding similarity and landings.")

    # Cross-class H3 check: bar's similarity to or, compared to other L=1 words.
    print("\n" + "=" * 80)
    print("Cross-class H3 check: `bar` vs `or` (script 11)")
    print("=" * 80)
    print()
    print(f"  Script 11 found bar -> or at 74-82% regardless of slot. H3 (cross-class)")
    print(f"  predicts: bar's embedding has unusually high similarity to `or` compared")
    print(f"  to other L=1 invented words.")
    print()
    bar_check_words = ["foo", "bar", "baz", "fred", "qux", "quux", "thud", "pop", "zap", "ping", "huh"]
    print(f"  {'Word':<8s} {'sim(or)':>10s} {'sim(not)':>10s} {'sim(or)-sim(not)':>20s}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 20}")
    for word in bar_check_words:
        vecs, _ = get_word_embeddings(model, tok, word, device)
        pooled = pool_embedding(vecs, "mean")
        sim_or = cosine(pooled, canonical_mean["or"])
        sim_not = cosine(pooled, canonical_mean["not"])
        marker = "  <- bar (script 11: ~74-82% -> or)" if word == "bar" else ""
        print(
            f"  {word:<8s} {sim_or:>+10.4f} {sim_not:>+10.4f} "
            f"{sim_or - sim_not:>+20.4f}{marker}"
        )

    print()
    print("Reading guide:")
    print("  - If bar's sim(or) - sim(not) is the largest positive value in this list,")
    print("    H3 (cross-class regime) is supported as a layer-0 embedding phenomenon.")
    print("  - If bar is not anomalous in embedding space, H3 (cross-class) must be")
    print("    constructed by later layers — interesting in itself.")


if __name__ == "__main__":
    main()
