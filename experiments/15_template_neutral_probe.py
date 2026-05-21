"""Template-neutral probe (final Phase 0 test).

Scripts 9-13 left one open question: how much of the H1 unary-region
attractor is a property of the *probe* (because we trained on
canonical-rich templates that have lots of canonical-specific lexical
context), versus a property of the *operator-anchored representation*
itself?

This script ablates the probe's training-template lexical content. We
build a set of canonical-neutral templates that treat the operator as a
quoted word: "Consider the word {op}.", "We see the word {op}.", etc.
These templates have minimal canonical-specific lexical signal: the
same surrounding text works for any of the five canonicals. Train the
probe on canonical A in these neutral templates, then evaluate on three
test conditions:

  Test 1: B'_neutral - invented operators in the same neutral templates
          (no H4 channel, minimal template-context lexical signal)
  Test 2: B'_rich    - invented operators in script-13's lexically-rich
          single-operator templates (H4 channel maximized)
  Test 3: A_rich     - canonical operators in script-13's rich templates,
          evaluated by the neutral-trained probe (does the probe still
          recognize canonicals when surface form is canonical but
          context is unfamiliar?)

Predictions per hypothesis:

  H1 robust (unary-region attractor is intrinsic to operator-position
  representation):
    - Probe achieves > 0.9 CV accuracy on canonical A_neutral
    - Invented operators in B'_neutral are classified mostly as
      {not, necessarily} (90%+ unary-region mass), confirming H1
      survives template-context ablation
    - Cross-template generalization: probe trained on neutral templates
      also generalizes to canonical A_rich (the probe direction is
      the operator's intrinsic signature, not a template artifact)

  H1 probe-instrument-dependent (the unary direction was constructed by
  the rich-template training data):
    - Probe achieves < 0.5 CV accuracy on canonical A_neutral (the
      canonicals are themselves indistinguishable without rich context)
    - Invented operators in B'_neutral are classified ~chance
    - Cross-template generalization fails on A_rich

  Intermediate:
    - Probe achieves moderate CV accuracy on A_neutral (0.6-0.9)
    - Invented operators in B'_neutral show partial unary-region
      attractor (60-80% unary-region mass)
    - The H1 mechanism exists but is amplified by rich training data

This is the cleanest single-experiment test of the H1 mechanism's
robustness. If it passes (Test 1 shows ≥80% unary-region mass on
invented operators with neutral training and testing), the hierarchical-
arity finding is the centerpiece of the Phase 1 paper.

Diagnostic layer 7 throughout (consistent with scripts 9-13).
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-1124-7B"
DIAGNOSTIC_LAYER = 7

N_PER_CLASS = 50
SEED = 17


# Canonical-neutral templates: treat the operator as a quoted word.
# Each template works syntactically for any of the 5 canonicals
# (and, or, not, implies, necessarily) and contains minimal
# canonical-specific lexical signal. The operator slot is the
# operator-anchored position; what comes before is roughly the
# same regardless of canonical.
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

assert len(NEUTRAL_TEMPLATES) >= N_PER_CLASS, "Not enough neutral templates"

OPERATORS = ["and", "or", "not", "implies", "necessarily"]

# 5 invented words from script 13, holding constant the {bliq, dren, vusp,
# molex, perph} set so we can directly compare neutral-template vs
# rich-template (script 13) results.
INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]


# Rich (script-13) templates for the cross-template generalization test.
RICH_TEMPLATES = {
    "and": [
        "If {p} and {q} are both true, the conjunction holds.",
        "Both {p} and {q} must be true for the conjunction to be asserted.",
        "The conjunction {p} and {q} is true only with both inputs true.",
        "Whenever {p} and {q} hold together, the conjunction is satisfied.",
        "{p} and {q} is a conjunction true exactly when both are true.",
    ],
    "or": [
        "If {p} or {q} is true, the disjunction holds.",
        "Either {p} or {q} suffices for the disjunction to be asserted.",
        "The disjunction {p} or {q} is true when at least one input is true.",
        "Whenever {p} or {q} holds, the disjunction is satisfied.",
        "{p} or {q} is a disjunction true exactly when at least one is true.",
    ],
    "not": [
        "If not {p}, the negation of {p} holds.",
        "The negation not {p} is true when the input proposition is false.",
        "Whenever not {p} holds, the proposition {p} is false.",
        "Not {p} is the negation of {p}, false only when {p} is true.",
        "The expression not {p} asserts the negation of the proposition.",
    ],
    "implies": [
        "If {p} implies {q}, then whenever {p} is true {q} must be true.",
        "The implication {p} implies {q} fails only when {p} holds but {q} does not.",
        "Whenever {p} implies {q} holds, {q} follows from {p}.",
        "The conditional {p} implies {q} encodes the inference from {p} to {q}.",
        "{p} implies {q} is true unless {p} is true and {q} is false.",
    ],
    "necessarily": [
        "If necessarily {p}, then {p} holds in every situation.",
        "The modal claim necessarily {p} asserts {p} must always hold.",
        "Whenever necessarily {p} is asserted, {p} is true without exception.",
        "Necessarily {p} means {p} is true in every case under consideration.",
        "The proposition necessarily {p} is the modal-box claim about {p}.",
    ],
}


def count_subwords(tok, word: str) -> int:
    return len(tok(" " + word, add_special_tokens=False).input_ids)


def make_neutral_stimuli(operator: str, rng: random.Random, n: int) -> list[str]:
    """Generate neutral-template stimuli with the operator in the {op} slot."""
    stimuli: list[str] = []
    templates = NEUTRAL_TEMPLATES[:]
    rng.shuffle(templates)
    for i in range(n):
        tmpl = templates[i % len(templates)]
        stimuli.append(tmpl.format(op=operator))
    return stimuli


def make_rich_stimuli(operator: str, rng: random.Random, n: int) -> list[str]:
    """Generate rich-template stimuli with the operator filled in."""
    templates = RICH_TEMPLATES[operator]
    vars_ = ["p", "q", "r", "s"]
    stimuli: list[str] = []
    for _ in range(n):
        tmpl = rng.choice(templates)
        p, q = rng.sample(vars_, 2)
        stimuli.append(tmpl.format(p=p, q=q))
    return stimuli


def make_rich_invented_stimuli(
    template_canonical: str, invented: str, rng: random.Random, n: int
) -> list[str]:
    """Generate rich-template stimuli where `invented` replaces `template_canonical`."""
    templates = RICH_TEMPLATES[template_canonical]
    vars_ = ["p", "q", "r", "s"]
    pattern = re.compile(r"\b" + re.escape(template_canonical) + r"\b")
    stimuli: list[str] = []
    for _ in range(n):
        tmpl = rng.choice(templates)
        p, q = rng.sample(vars_, 2)
        canonical_prompt = tmpl.format(p=p, q=q)
        stimuli.append(pattern.sub(invented, canonical_prompt))
    return stimuli


def find_operator_anchor(tok, prompt: str, operators: list[str]) -> int | None:
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


def print_confusion(y_true_labels, y_pred_labels, class_names, title):
    print(f"\n{title}")
    print("-" * 80)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
    print(
        f"  {'true \\ pred':<15s}"
        + "".join(f"{c:>14s}" for c in class_names)
        + f"  {'n':>5s}"
    )
    for i, c_true in enumerate(class_names):
        row = f"  {c_true:<15s}"
        n = cm[i].sum()
        for j in range(len(class_names)):
            row += f"{cm[i, j]:>14d}"
        row += f"  {n:>5d}"
        print(row)


def print_invented_distribution(
    invented_words: list[str],
    canonical_classes: list[str],
    pred_counts: dict[str, Counter],
    title: str,
) -> None:
    print(f"\n{title}")
    print("-" * 80)
    header = f"  {'invented':<10s} | " + " | ".join(
        f"{c:>13s}" for c in canonical_classes
    ) + f" | {'unary %':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for w in invented_words:
        counts = pred_counts[w]
        total = sum(counts.values())
        row = f"  {w:<10s} | " + " | ".join(
            f"{counts.get(c, 0):>5d} ({100*counts.get(c, 0)/total:>5.1f}%)"
            for c in canonical_classes
        )
        unary_pct = 100 * (counts.get("not", 0) + counts.get("necessarily", 0)) / total
        row += f" | {unary_pct:>8.1f}%"
        print(row)


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

    print("\nValidating tokenization:")
    for op in OPERATORS:
        print(f"  canonical {op:<13s} -> {count_subwords(tok, op)} subword(s)")
    for w in INVENTED_WORDS:
        print(f"  invented  {w:<13s} -> {count_subwords(tok, w)} subword(s)")

    # ===== Step 1: Build canonical A in neutral templates =====
    print("\n" + "=" * 80)
    print("Step 1: Canonical A in neutral templates")
    print("=" * 80)
    rng = random.Random(SEED)
    A_neutral: list[str] = []
    A_neutral_labels: list[str] = []
    for op in OPERATORS:
        op_rng = random.Random(SEED + hash(op) % 10000)
        A_neutral.extend(make_neutral_stimuli(op, op_rng, N_PER_CLASS))
        A_neutral_labels.extend([op] * N_PER_CLASS)

    print(f"  {len(A_neutral)} stimuli ({N_PER_CLASS} per class x {len(OPERATORS)} classes)")
    print("  Sample stimuli:")
    for op in OPERATORS:
        idx = A_neutral_labels.index(op)
        print(f"    {op:<13s}: {A_neutral[idx]!r}")

    print("\nExtracting canonical A (neutral templates) activations ...")
    t0 = time.time()
    X_A_neutral = extract_anchored_activations(model, tok, A_neutral, OPERATORS, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers: {len(X_A_neutral)}, hidden_dim: {X_A_neutral[0].shape[1]}")

    y_neutral = np.array(A_neutral_labels)

    print("\nPer-layer CV accuracy on canonical A (neutral templates):")
    for layer in [0, 1, 4, 7, 12, 16, 20, 24, 28, 32]:
        if layer < len(X_A_neutral):
            acc = cv_accuracy(X_A_neutral[layer], y_neutral, seed=SEED)
            print(f"  layer {layer:3d}: {acc:.3f}")

    diag_layer = DIAGNOSTIC_LAYER if DIAGNOSTIC_LAYER < len(X_A_neutral) else len(X_A_neutral) - 1
    print(f"\nFitting neutral-template probe at layer {diag_layer} ...")
    clf, scaler = fit_probe(X_A_neutral[diag_layer], y_neutral)

    # ===== Step 2: Cross-template generalization test on canonical A_rich =====
    print("\n" + "=" * 80)
    print("Step 2: Cross-template generalization on canonical A_rich")
    print("(Neutral-trained probe applied to script-13 lexically-rich templates)")
    print("=" * 80)
    A_rich: list[str] = []
    A_rich_labels: list[str] = []
    for op in OPERATORS:
        op_rng = random.Random(SEED + 100 + hash(op) % 10000)
        A_rich.extend(make_rich_stimuli(op, op_rng, N_PER_CLASS))
        A_rich_labels.extend([op] * N_PER_CLASS)

    print(f"\n  {len(A_rich)} stimuli")
    print("\nExtracting canonical A (rich templates) activations ...")
    t0 = time.time()
    X_A_rich = extract_anchored_activations(model, tok, A_rich, OPERATORS, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")

    y_pred_A_rich = predict(clf, scaler, X_A_rich[diag_layer])
    acc_xfer = float(np.mean(y_pred_A_rich == np.array(A_rich_labels)))
    print(f"\n  Cross-template (neutral-train -> rich-test) accuracy: {acc_xfer:.3f}")
    print(f"  (Chance: {1.0/len(OPERATORS):.3f})")
    print_confusion(
        A_rich_labels, y_pred_A_rich.tolist(), OPERATORS,
        "Confusion matrix: canonical A_rich, neutral-trained probe (layer 7)"
    )

    # ===== Step 3: Test 1 - invented operators in neutral templates =====
    print("\n" + "=" * 80)
    print("Step 3 (Test 1): Invented operators in neutral templates")
    print("(Direct test of H1 robustness with template-context ablated)")
    print("=" * 80)
    print("\nExtracting B'_neutral activations ...")
    t0 = time.time()
    pred_counts_neutral: dict[str, Counter] = {}
    for w in INVENTED_WORDS:
        w_rng = random.Random(SEED + 200 + hash(w) % 10000)
        stimuli = make_neutral_stimuli(w, w_rng, N_PER_CLASS)
        X_w = extract_anchored_activations(model, tok, stimuli, [w], device)
        y_pred = predict(clf, scaler, X_w[diag_layer])
        pred_counts_neutral[w] = Counter(y_pred.tolist())
    print(f"  extraction time: {time.time() - t0:.1f}s")

    print_invented_distribution(
        INVENTED_WORDS, OPERATORS, pred_counts_neutral,
        f"B'_neutral predictions at layer {diag_layer} (n={N_PER_CLASS} per word)"
    )

    # ===== Step 4: Test 2 - invented operators in rich templates =====
    # Maps each invented word to its rich-template-family from script 13:
    # bliq replaces and; dren replaces or; vusp replaces not; molex replaces implies; perph replaces necessarily.
    W_TO_CANONICAL = dict(zip(INVENTED_WORDS, OPERATORS))

    print("\n" + "=" * 80)
    print("Step 4 (Test 2): Invented operators in rich (script-13) templates")
    print("(Neutral-trained probe applied to maximally H4-bearing templates)")
    print("=" * 80)
    print("\nExtracting B'_rich activations ...")
    print("  (each invented word in its script-13 'canonical of W' template family)")
    t0 = time.time()
    pred_counts_rich: dict[str, Counter] = {}
    for w in INVENTED_WORDS:
        w_rng = random.Random(SEED + 300 + hash(w) % 10000)
        canonical_family = W_TO_CANONICAL[w]
        stimuli = make_rich_invented_stimuli(canonical_family, w, w_rng, N_PER_CLASS)
        X_w = extract_anchored_activations(model, tok, stimuli, [w], device)
        y_pred = predict(clf, scaler, X_w[diag_layer])
        pred_counts_rich[w] = Counter(y_pred.tolist())
    print(f"  extraction time: {time.time() - t0:.1f}s")

    print_invented_distribution(
        INVENTED_WORDS, OPERATORS, pred_counts_rich,
        f"B'_rich predictions at layer {diag_layer} (each W in its script-13 canonical-family template)"
    )

    # ===== Step 5: Summary verdict =====
    print("\n" + "=" * 80)
    print("Summary: H1 robustness verdict")
    print("=" * 80)
    print()

    cv_acc_neutral = cv_accuracy(X_A_neutral[diag_layer], y_neutral, seed=SEED)
    print(f"  Probe CV accuracy on A_neutral at layer {diag_layer}: {cv_acc_neutral:.3f}")
    print(f"    (chance: {1.0/len(OPERATORS):.3f}; if < 0.5, canonicals are")
    print(f"     indistinguishable in neutral templates -> probe is unreliable)")

    print(f"\n  Cross-template generalization (A_rich, neutral-trained): {acc_xfer:.3f}")
    print(f"    (if > 0.9, probe direction is template-invariant)")

    # Compute unary-region mass for both tests.
    unary_neutral = []
    for w in INVENTED_WORDS:
        c = pred_counts_neutral[w]
        total = sum(c.values())
        unary_neutral.append(100 * (c.get("not", 0) + c.get("necessarily", 0)) / total)
    unary_rich = []
    for w in INVENTED_WORDS:
        c = pred_counts_rich[w]
        total = sum(c.values())
        unary_rich.append(100 * (c.get("not", 0) + c.get("necessarily", 0)) / total)

    print(f"\n  Test 1 (B'_neutral, neutral-trained):")
    print(f"    Mean unary-region mass: {np.mean(unary_neutral):.1f}%")
    print(f"    Per word: " + ", ".join(
        f"{w}={u:.1f}%" for w, u in zip(INVENTED_WORDS, unary_neutral)
    ))

    print(f"\n  Test 2 (B'_rich, neutral-trained):")
    print(f"    Mean unary-region mass: {np.mean(unary_rich):.1f}%")
    print(f"    Per word: " + ", ".join(
        f"{w}={u:.1f}%" for w, u in zip(INVENTED_WORDS, unary_rich)
    ))

    print(f"\n  Script 13 reference (B'_rich, rich-trained, full 5x5 design):")
    print(f"    Mean unary-region mass: 94.6%")

    print()
    print("  Reading guide:")
    print("    H1 ROBUST:")
    print("      - CV accuracy on A_neutral > 0.8 (canonicals are distinguishable")
    print("        even in neutral templates)")
    print("      - Test 1 mean unary mass > 80% (invented operators default to")
    print("        unary even when no template-context bias exists)")
    print("      - Cross-template generalization > 0.85 (probe direction is")
    print("        template-invariant)")
    print()
    print("    H1 PARTIALLY DEPENDENT ON TEMPLATE-CONTEXT (intermediate):")
    print("      - CV accuracy on A_neutral 0.5-0.8")
    print("      - Test 1 mean unary mass 50-80%")
    print("      - Test 2 unary mass significantly higher than Test 1 (rich")
    print("        templates amplify the effect)")
    print()
    print("    H1 PROBE-INSTRUMENT-DEPENDENT (overturning):")
    print("      - CV accuracy on A_neutral < 0.5 (canonicals indistinguishable")
    print("        without rich-template context)")
    print("      - Test 1 mean unary mass at chance (40% for 5-class)")


if __name__ == "__main__":
    main()
