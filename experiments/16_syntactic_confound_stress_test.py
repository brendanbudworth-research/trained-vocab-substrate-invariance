"""Syntactic-confound stress test (the critical Phase 0 falsification test).

In all of scripts 7-15, canonical binary operators (and, or, implies) appear
in INFIX position ("p and q") and canonical unary operators (not, necessarily)
appear in PREFIX position ("not p"). Arity and syntactic position are
perfectly confounded.

When an invented operator is substituted into an infix slot ("p bliq q") and
the probe maps it to `not`, two hypotheses are observationally equivalent:

  H_arity     The model encodes arity (unary vs binary). Invented operators
              that fail to bind to a specific canonical default to the
              unary-class region in activation space.

  H_syntactic The model encodes syntactic position (prefix-modifier vs
              infix-connective). Invented operators in infix slots fail to
              parse as a recognised binary canonical and the model retreats
              to whatever cluster of canonicals appears in a comparable
              "standalone modifier" role -- which happens to be {not,
              necessarily}.

These hypotheses are dissociable by using functional-prefix notation, in
which every operator appears as `op(arg1, ...)` regardless of arity. The
model has substantial training-data exposure to this notation (code, math).
With uniform preceding context "The function {op}", every canonical sits in
the same syntactic role at the operator-anchored position, and the only
remaining structural variable is the operator's intrinsic representation
plus its post-operator argument count (which is NOT visible at the
operator-anchored position itself, due to causal masking).

Experimental conditions:

  Condition 1 (REFERENCE):     script-15 neutral metalinguistic templates
                                ("Consider the word {op} in this sentence.")
                                Replicates the script-15 99.6% unary-mass
                                finding within the same script for direct
                                comparison.

  Condition 2 (CRITICAL):      functional-prefix notation
                                ("The function {op}(p, q) returns a boolean
                                  when called.")
                                Binaries get 2-arg templates; unaries get
                                1-arg templates. All canonicals share the
                                identical preceding context "The function ".

Outcome interpretation:

  Condition 2 unary-mass approximately equal to Condition 1 (~99%):
    - The unary-region attractor is robust to syntactic-position change.
    - H_arity survives; the Phase 1 hierarchical-arity claim holds.

  Condition 2 unary-mass materially lower than Condition 1 (40-60%):
    - The attractor was at least partly syntactic-position-driven.
    - H_arity is partially refuted; the headline needs rephrasing.

  Condition 2 invented binary replacements (bliq, dren, molex) classified
    as their arity-matched canonicals (and, or, implies) more often than
    as unaries:
    - H_syntactic wins. The "arity attractor" was a "this token doesn't fit
      the syntactic position" artifact. Different paper.

This script does not write to disk; it prints all results to stdout for
direct integration into paper_notes.md.

Diagnostic layer 7 throughout (matches scripts 9-15 for direct comparison).
"""

from __future__ import annotations

import random
import time
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-1124-7B"
DIAGNOSTIC_LAYER = 7

N_PER_CLASS = 50
SEED = 17

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
CANONICAL_ARITY = {
    "and": 2, "or": 2, "implies": 2,
    "not": 1, "necessarily": 1,
}

INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]
# Same script-13 mapping: each invented word replaces a specific canonical.
W_TO_CANONICAL = dict(zip(INVENTED_WORDS, CANONICALS))


# ==============================================================================
# Condition 1: script-15 neutral metalinguistic templates
# ==============================================================================
NEUTRAL_TEMPLATES = [
    "Consider the word {op} in this sentence.",
    "We see the word {op} written here.",
    "The word {op} appears in the text below.",
    "Look at the word {op} on the page.",
    "Note the word {op} in the example.",
    "The word {op} is shown in the figure.",
    "Examine the word {op} carefully.",
    "The token {op} is given in the source.",
    "Find the word {op} in the document.",
    "Read the word {op} aloud.",
    "Mark the word {op} with a circle.",
    "The string {op} occurs in the line.",
    "I write the word {op} on the board.",
    "The word {op} is highlighted in red.",
    "Underline the word {op} please.",
    "Copy the word {op} to the next line.",
    "Print the word {op} on screen.",
    "The label {op} is attached to the box.",
    "She typed the word {op} quickly.",
    "The word {op} is part of the passage.",
    "Repeat the word {op} once more.",
    "Spell the word {op} aloud.",
    "The word {op} is on the list.",
    "I noticed the word {op} as I read.",
    "The word {op} stands out in the text.",
    "He pointed at the word {op}.",
    "The word {op} is circled below.",
    "We discussed the word {op} in class.",
    "The word {op} was used in the lecture.",
    "She wrote the word {op} on paper.",
    "Translate the word {op} into another language.",
    "The word {op} appears twice in the file.",
    "I underlined the word {op} for emphasis.",
    "The word {op} was the topic of the talk.",
    "Pronounce the word {op} slowly.",
    "The word {op} can be found in the index.",
    "Quote the word {op} verbatim.",
    "Add the word {op} to your notes.",
    "The word {op} is missing from the list.",
    "Remove the word {op} from the line.",
    "The word {op} is written in italics.",
    "Search for the word {op} in the file.",
    "The word {op} is followed by a comma.",
    "I typed the word {op} into the search box.",
    "The word {op} was spoken first.",
    "Insert the word {op} after the noun.",
    "The word {op} is part of the title.",
    "She circled the word {op} on the test.",
    "The word {op} caught my attention.",
    "Replace the word {op} with a synonym.",
]
assert len(NEUTRAL_TEMPLATES) >= N_PER_CLASS


# ==============================================================================
# Condition 2: functional-prefix notation, uniform preceding context.
# All five canonicals appear in the same prefix-function syntactic role.
# Binaries get 2 arguments; unaries get 1 argument.
# Preceding context up to the operator is IDENTICAL across canonicals
# ("The function "), so per-operator differences in the operator-anchored
# residual stream are driven by the operator's own intrinsic representation.
# ==============================================================================
FUNCTIONAL_TEMPLATE_FRAMES = [
    "The function {call} returns a boolean when invoked.",
    "The function {call} is used in the algorithm.",
    "The function {call} computes a value from its inputs.",
    "The function {call} produces a result.",
    "The function {call} evaluates to true or false.",
    "The function {call} accepts inputs and returns a boolean output.",
    "The function {call} is defined in the standard library.",
    "The function {call} appears in this code listing.",
    "The function {call} is called within the loop.",
    "The function {call} executes during program runtime.",
]
assert len(FUNCTIONAL_TEMPLATE_FRAMES) * 5 >= N_PER_CLASS


def build_functional_stimulus(op: str, frame: str, p: str, q: str | None) -> str:
    """Build a single functional-prefix stimulus. If the operator is unary
    (q is None), uses a single-argument call; otherwise uses two-argument."""
    if q is None:
        call = f"{op}({p})"
    else:
        call = f"{op}({p}, {q})"
    return frame.format(call=call)


def make_neutral_stimuli(op: str, rng: random.Random, n: int) -> list[str]:
    templates = NEUTRAL_TEMPLATES[:]
    rng.shuffle(templates)
    return [templates[i % len(templates)].format(op=op) for i in range(n)]


def make_functional_stimuli(op: str, arity: int, rng: random.Random, n: int) -> list[str]:
    """Generate functional-prefix stimuli for the operator at given arity.
    The arity is taken from the OPERATOR'S role (the canonical it represents),
    not from any property of the operator surface form itself."""
    vars_ = ["p", "q", "r", "s", "x", "y"]
    frames = FUNCTIONAL_TEMPLATE_FRAMES[:]
    stimuli: list[str] = []
    for _ in range(n):
        frame = rng.choice(frames)
        if arity == 1:
            arg_p = rng.choice(vars_)
            stimuli.append(build_functional_stimulus(op, frame, arg_p, None))
        else:
            arg_p, arg_q = rng.sample(vars_, 2)
            stimuli.append(build_functional_stimulus(op, frame, arg_p, arg_q))
    return stimuli


def find_operator_anchor(tok, prompt: str, operators: list[str]) -> int | None:
    """Find the position immediately after the operator's last subword."""
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    decoded_tokens = [tok.decode([i]) for i in ids]
    best_pos: int | None = None
    best_idx = len(ids)
    for op in operators:
        target = " " + op
        joined = ""
        for i, t in enumerate(decoded_tokens):
            joined += t
            if joined.endswith(target):
                pos = i + 1 if i + 1 < len(ids) else i
                if i < best_idx:
                    best_idx = i
                    best_pos = pos
                break
        joined = ""
    return best_pos


def extract_anchored_activations(
    model, tok, prompts: list[str], operators: list[str], device: str
) -> list[np.ndarray]:
    layer_buffers: list[list[np.ndarray]] | None = None
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        pos = find_operator_anchor(tok, p, operators)
        if pos is None or pos >= enc.input_ids.shape[1]:
            pos = enc.input_ids.shape[1] - 1
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        if device == "mps":
            torch.mps.synchronize()
        layer_vecs = [h[0, pos, :].float().cpu().numpy() for h in out.hidden_states]
        if layer_buffers is None:
            layer_buffers = [[] for _ in layer_vecs]
        for buf, vec in zip(layer_buffers, layer_vecs):
            buf.append(vec)
    assert layer_buffers is not None
    return [np.stack(buf) for buf in layer_buffers]


def make_classifier() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")


def cv_accuracy(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = make_classifier()
        clf.fit(X_tr, y[train_idx])
        scores.append(clf.score(X_te, y[test_idx]))
    return float(np.mean(scores))


def fit_probe(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = make_classifier()
    clf.fit(X_scaled, y)
    return clf, scaler


def predict(clf, scaler, X: np.ndarray) -> np.ndarray:
    return clf.predict(scaler.transform(X))


def run_condition(
    name: str,
    canonical_stimuli_fn,
    invented_stimuli_fn,
    model,
    tok,
    device: str,
) -> dict:
    """Run a single condition: build A, train probe, evaluate B'."""
    print(f"\n{'=' * 80}")
    print(f"Condition: {name}")
    print(f"{'=' * 80}")

    print(f"\nBuilding canonical A ({N_PER_CLASS} stimuli per canonical, 5 canonicals)...")
    A: list[str] = []
    A_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(SEED + hash((name, op)) % 100000)
        stimuli = canonical_stimuli_fn(op, op_rng, N_PER_CLASS)
        A.extend(stimuli)
        A_labels.extend([op] * N_PER_CLASS)

    print(f"  Sample stimuli:")
    for op in CANONICALS:
        idx = A_labels.index(op)
        print(f"    {op:<14s}: {A[idx]!r}")

    print("\nExtracting canonical A activations ...")
    t0 = time.time()
    X_A = extract_anchored_activations(model, tok, A, CANONICALS, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers: {len(X_A)}, hidden_dim: {X_A[0].shape[1]}")

    y_A = np.array(A_labels)
    diag_layer = DIAGNOSTIC_LAYER if DIAGNOSTIC_LAYER < len(X_A) else len(X_A) - 1

    # Probe sanity check.
    print("\nPer-layer CV accuracy on canonical A:")
    for layer in [0, 1, 4, 7, 12, 16]:
        if layer < len(X_A):
            acc = cv_accuracy(X_A[layer], y_A, seed=SEED)
            print(f"  layer {layer:3d}: {acc:.3f}")

    cv_at_diag = cv_accuracy(X_A[diag_layer], y_A, seed=SEED)
    print(f"\n  CV accuracy at diagnostic layer {diag_layer}: {cv_at_diag:.3f}")
    print(f"  (chance: {1.0/len(CANONICALS):.3f})")

    print(f"\nFitting probe at layer {diag_layer} ...")
    clf, scaler = fit_probe(X_A[diag_layer], y_A)

    # Evaluate invented operators.
    print("\nExtracting B' (invented operators) activations ...")
    t0 = time.time()
    pred_counts: dict[str, Counter] = {}
    for w in INVENTED_WORDS:
        w_rng = random.Random(SEED + hash((name, w, "invent")) % 100000)
        stimuli = invented_stimuli_fn(w, w_rng, N_PER_CLASS)
        X_w = extract_anchored_activations(model, tok, stimuli, [w], device)
        y_pred = predict(clf, scaler, X_w[diag_layer])
        pred_counts[w] = Counter(y_pred.tolist())
    print(f"  extraction time: {time.time() - t0:.1f}s")

    # Per-word distribution.
    print(f"\nB' prediction distribution at layer {diag_layer} (n={N_PER_CLASS} per word)")
    print(f"  {'invented (replaces)':<28s} | " + " | ".join(
        f"{c:>13s}" for c in CANONICALS
    ) + f" | {'unary %':>9s}")
    print("  " + "-" * (28 + 5 * 16 + 12))

    unary_masses: list[float] = []
    for w in INVENTED_WORDS:
        c = pred_counts[w]
        total = sum(c.values())
        unary_pct = 100 * (c.get("not", 0) + c.get("necessarily", 0)) / total
        unary_masses.append(unary_pct)
        rep = W_TO_CANONICAL[w]
        arity_str = "unary" if CANONICAL_ARITY[rep] == 1 else "binary"
        label = f"{w} (~{rep}, {arity_str})"
        row = f"  {label:<28s} | " + " | ".join(
            f"{c.get(cl, 0):>5d} ({100*c.get(cl, 0)/total:>5.1f}%)"
            for cl in CANONICALS
        ) + f" | {unary_pct:>8.1f}%"
        print(row)

    mean_unary_mass = float(np.mean(unary_masses))
    print(f"\n  Mean unary-region mass across invented words: {mean_unary_mass:.1f}%")

    # Discriminative metric: do binary-replacement invented words go to
    # their arity-matched canonicals or to unary canonicals?
    binary_replacements = [w for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 2]
    unary_replacements = [w for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 1]

    print(f"\n  Arity-aware breakdown:")
    print(f"    Binary-replacement words ({', '.join(binary_replacements)}):")
    for w in binary_replacements:
        c = pred_counts[w]
        total = sum(c.values())
        unary_pct = 100 * (c.get("not", 0) + c.get("necessarily", 0)) / total
        binary_pct = 100 * (c.get("and", 0) + c.get("or", 0) + c.get("implies", 0)) / total
        print(f"      {w:<8s}: unary {unary_pct:5.1f}%, binary {binary_pct:5.1f}%")

    print(f"    Unary-replacement words ({', '.join(unary_replacements)}):")
    for w in unary_replacements:
        c = pred_counts[w]
        total = sum(c.values())
        unary_pct = 100 * (c.get("not", 0) + c.get("necessarily", 0)) / total
        binary_pct = 100 * (c.get("and", 0) + c.get("or", 0) + c.get("implies", 0)) / total
        print(f"      {w:<8s}: unary {unary_pct:5.1f}%, binary {binary_pct:5.1f}%")

    binary_repl_unary_mass = float(np.mean([
        100 * (pred_counts[w].get("not", 0) + pred_counts[w].get("necessarily", 0))
        / sum(pred_counts[w].values())
        for w in binary_replacements
    ]))
    binary_repl_binary_mass = float(np.mean([
        100 * (
            pred_counts[w].get("and", 0) + pred_counts[w].get("or", 0)
            + pred_counts[w].get("implies", 0)
        ) / sum(pred_counts[w].values())
        for w in binary_replacements
    ]))
    print(f"\n  Binary-replacement words: mean unary mass = {binary_repl_unary_mass:.1f}%,"
          f" mean binary mass = {binary_repl_binary_mass:.1f}%")

    return {
        "cv_acc": cv_at_diag,
        "pred_counts": pred_counts,
        "mean_unary_mass": mean_unary_mass,
        "binary_repl_unary_mass": binary_repl_unary_mass,
        "binary_repl_binary_mass": binary_repl_binary_mass,
    }


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

    # =========================================================================
    # Condition 1: REFERENCE -- script-15 neutral metalinguistic templates
    # =========================================================================

    def cond1_canonical_fn(op, rng, n):
        return make_neutral_stimuli(op, rng, n)

    def cond1_invented_fn(w, rng, n):
        return make_neutral_stimuli(w, rng, n)

    cond1_result = run_condition(
        name="1 (REFERENCE: script-15 neutral metalinguistic templates)",
        canonical_stimuli_fn=cond1_canonical_fn,
        invented_stimuli_fn=cond1_invented_fn,
        model=model, tok=tok, device=device,
    )

    # =========================================================================
    # Condition 2: CRITICAL -- functional-prefix notation, uniform syntactic position
    # =========================================================================

    def cond2_canonical_fn(op, rng, n):
        arity = CANONICAL_ARITY[op]
        return make_functional_stimuli(op, arity, rng, n)

    def cond2_invented_fn(w, rng, n):
        # Invented word inherits the arity of the canonical it replaces.
        arity = CANONICAL_ARITY[W_TO_CANONICAL[w]]
        return make_functional_stimuli(w, arity, rng, n)

    cond2_result = run_condition(
        name="2 (CRITICAL: functional-prefix notation, uniform syntactic position)",
        canonical_stimuli_fn=cond2_canonical_fn,
        invented_stimuli_fn=cond2_invented_fn,
        model=model, tok=tok, device=device,
    )

    # =========================================================================
    # Side-by-side comparison + verdict
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()
    print(f"  {'':<60s} {'Cond 1':>10s} {'Cond 2':>10s} {'Delta':>10s}")
    print(f"  {'-' * 60} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(
        f"  {'Probe CV accuracy on canonicals at L7':<60s} "
        f"{cond1_result['cv_acc']:>10.3f} "
        f"{cond2_result['cv_acc']:>10.3f} "
        f"{cond2_result['cv_acc']-cond1_result['cv_acc']:>+10.3f}"
    )
    print(
        f"  {'Mean unary-region mass on invented operators':<60s} "
        f"{cond1_result['mean_unary_mass']:>9.1f}% "
        f"{cond2_result['mean_unary_mass']:>9.1f}% "
        f"{cond2_result['mean_unary_mass']-cond1_result['mean_unary_mass']:>+9.1f}pp"
    )
    print(
        f"  {'  ...for binary-replacement words only':<60s} "
        f"{cond1_result['binary_repl_unary_mass']:>9.1f}% "
        f"{cond2_result['binary_repl_unary_mass']:>9.1f}% "
        f"{cond2_result['binary_repl_unary_mass']-cond1_result['binary_repl_unary_mass']:>+9.1f}pp"
    )
    print(
        f"  {'Mean binary-canonical mass on binary-repl words':<60s} "
        f"{cond1_result['binary_repl_binary_mass']:>9.1f}% "
        f"{cond2_result['binary_repl_binary_mass']:>9.1f}% "
        f"{cond2_result['binary_repl_binary_mass']-cond1_result['binary_repl_binary_mass']:>+9.1f}pp"
    )
    print()

    print("Interpretation guide:")
    print()
    print("  ARITY ATTRACTOR SURVIVES the syntactic-position confound if:")
    print("    - Condition 2 mean unary mass on binary-replacement words is")
    print("      within ~10 pp of Condition 1 (both should be >= 80%).")
    print("    - Probe CV accuracy on Condition 2 canonicals is high (>= 0.8),")
    print("      indicating canonicals are still distinguishable in functional")
    print("      prefix notation.")
    print()
    print("  SYNTACTIC-POSITION CONFOUND WINS if:")
    print("    - Condition 2 mean unary mass on binary-replacement words drops")
    print("      below 60%.")
    print("    - Condition 2 binary-canonical mass on binary-replacement words")
    print("      rises above 40% (binary-replacement words route to their")
    print("      arity-matched canonicals when the syntactic position permits).")
    print()
    print("  INCONCLUSIVE / NEW MECHANISM if:")
    print("    - Probe CV accuracy on Condition 2 canonicals is low (< 0.5),")
    print("      meaning canonicals are not distinguishable in functional prefix")
    print("      notation -- the test is uninformative because the model treats")
    print("      all functional-prefix calls as similar.")
    print()
    print("Final flag for the writeup: report whether the binary-replacement")
    print("words' unary mass changes between conditions. That is the cleanest")
    print("single number for the syntactic-confound question.")


if __name__ == "__main__":
    main()
