"""Template-context (H4) quantification.

Scripts 11 and 12 surfaced a fourth channel modulating operator-renaming probe
predictions: template-lexical-context pull (H4). The hypothesis is that the
probe trained on canonical operators learns a direction defined partly by the
surrounding template's lexical content rather than purely by the post-operator
semantic load. When B' substitutes invented operators into the same templates,
attention from the operator-anchored position still flows to template-context
words, and the probe partially predicts the canonical that the template
"encodes" — independent of the invented operator's identity.

This script quantifies H4 directly with a factorial design:

  - 5 invented operators (W):    {bliq, dren, vusp, molex, perph}
       chosen because scripts 9-12 indicate these have minimal embedding
       similarity to any non-`not` canonical (i.e., H3-quiet).
  - 5 template families (T):     {and, or, not, implies, necessarily}
  - 50 stimuli per (W, T) cell

For each cell, the canonical operator that "owns" T is replaced by W in the
template. The probe anchors on W (operators=[W]) so co-occurring canonicals
elsewhere in the sentence don't capture the anchor.

Templates are deliberately single-operator and lexically rich. That is the
condition under which H4 is strongest and is what gives us the cleanest
per-template H4 magnitude.

H4 per template C, averaged across W:
  H4_pull(C) = mean_W [ P(predicted=C | W, T=C) ]
             - mean_{W, T' != C} [ P(predicted=C | W, T=T') ]

A positive value indicates the template-C lexical context biases the probe
toward predicting C beyond what the H1 default-to-`not` baseline would give.
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

N_PER_CELL = 50
SEED = 17


# Single-operator templates: exactly one canonical operator per template.
# Each template family is lexically rich in its canonical-relevant context
# (conjunction / disjunction / negation / implication / modal terms), which
# is the H4-bearing content we want to measure the pull of.
SINGLE_OP_TEMPLATES = {
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
        "The statement not {p} flips the truth value of {p}.",
        "If not {p} is asserted, {p} cannot be true.",
    ],
    "implies": [
        "If {p} implies {q}, the implication relates {p} and {q}.",
        "The implication {p} implies {q} is a conditional statement.",
        "Whenever {p} implies {q} holds, asserting {p} forces {q}.",
        "{p} implies {q} states that {p} being true requires {q} being true.",
        "The conditional {p} implies {q} expresses material implication.",
    ],
    "necessarily": [
        "If necessarily {p}, the modal claim about {p} holds in every world.",
        "The modal statement necessarily {p} asserts {p} is true always.",
        "Whenever necessarily {p} holds, {p} must hold without exception.",
        "The modality necessarily {p} expresses that {p} is unconditional.",
        "If necessarily {p}, then {p} cannot fail in any situation.",
    ],
}

OPERATORS = list(SINGLE_OP_TEMPLATES.keys())

INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]


def count_subwords(tok, word: str) -> int:
    ids = tok(" " + word, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    return len(ids)


def make_canonical(rng: random.Random) -> tuple[list[str], list[str]]:
    """Canonical A: each template family with its canonical operator."""
    stimuli: list[str] = []
    labels: list[str] = []
    vars_ = ["p", "q", "r", "s"]
    for op in OPERATORS:
        for _ in range(N_PER_CELL):
            tmpl = rng.choice(SINGLE_OP_TEMPLATES[op])
            p, q = rng.sample(vars_, 2)
            stimuli.append(tmpl.format(p=p, q=q))
            labels.append(op)
    order = list(range(len(stimuli)))
    rng.shuffle(order)
    return [stimuli[i] for i in order], [labels[i] for i in order]


def make_cell_stimuli(
    template_family: str, invented: str, rng: random.Random, n: int
) -> list[str]:
    """Stimuli for one (W, T) cell: template family T's templates with the
    canonical operator (which is also the family's name) replaced by W."""
    templates = SINGLE_OP_TEMPLATES[template_family]
    vars_ = ["p", "q", "r", "s"]
    pattern = re.compile(r"\b" + re.escape(template_family) + r"\b")
    stimuli: list[str] = []
    for _ in range(n):
        tmpl = rng.choice(templates)
        p, q = rng.sample(vars_, 2)
        canonical_prompt = tmpl.format(p=p, q=q)
        invented_prompt = pattern.sub(invented, canonical_prompt)
        stimuli.append(invented_prompt)
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


def fit_probe_at_layer(X_A_layer: np.ndarray, y: np.ndarray):
    """Train a probe on canonical A at the given layer. Return (clf, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_A_layer)
    clf = make_classifier()
    clf.fit(X_scaled, y)
    return clf, scaler


def probe_predict(clf, scaler, X_layer: np.ndarray) -> np.ndarray:
    return clf.predict(scaler.transform(X_layer))


def print_cell_matrix(
    pred_counts_by_cell: dict[tuple[str, str], Counter],
    invented_words: list[str],
    template_families: list[str],
    canonical_classes: list[str],
    title: str,
) -> None:
    """Print the (W, T) -> prediction-counts matrix, for each canonical class
    aggregate across W to show per-template predicted-class shares."""
    print(f"\n{title}")
    print("=" * 80)
    print()
    print(f"For each (invented_word, template_family) cell:")
    print(f"  prediction counts out of {N_PER_CELL}, by canonical class")
    print()
    header = "  " + "W \\ T".ljust(8) + " | " + " | ".join(
        f"{t:>13s}" for t in template_families
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for W in invented_words:
        for c_idx, c in enumerate(canonical_classes):
            label = f"  {W if c_idx == 0 else '':<8s} | "
            cells = " | ".join(
                f"{pred_counts_by_cell[(W, T)].get(c, 0):>5d} -> {c:<5s}"
                for T in template_families
            )
            print(label + cells)
        print("  " + "-" * (len(header) - 2))


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

    print("\nValidating tokenization:")
    for op in OPERATORS:
        print(f"  canonical {op:<13s} -> {count_subwords(tok, op)} subword(s)")
    for w in INVENTED_WORDS:
        print(f"  invented  {w:<13s} -> {count_subwords(tok, w)} subword(s)")

    rng = random.Random(SEED)
    A, A_labels = make_canonical(rng)
    print(f"\nCanonical A: {len(A)} stimuli ({N_PER_CELL} per class x {len(OPERATORS)} classes)")

    print("\nExtracting canonical A activations ...")
    t0 = time.time()
    X_A = extract_anchored_activations(model, tok, A, OPERATORS, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers: {len(X_A)}, hidden_dim: {X_A[0].shape[1]}")

    y_A = np.array(A_labels)
    chance = 1.0 / len(OPERATORS)
    print(f"\nChance accuracy: {chance:.3f} ({len(OPERATORS)}-class)")

    print("\nSanity probe on canonical A at layer 1:")
    acc_1 = cv_accuracy(X_A[1], y_A, seed=SEED)
    print(f"  layer-1 CV accuracy on A: {acc_1:.3f}")
    print(f"  (Note: single-op templates are linguistically simpler than the multi-op")
    print(f"   templates of script 12; CV accuracy here may differ slightly.)")

    print("\nPer-layer CV accuracy on canonical A:")
    for layer in [0, 1, 4, 7, 12, 16, 20, 24, 28, 32]:
        if layer < len(X_A):
            acc = cv_accuracy(X_A[layer], y_A, seed=SEED)
            print(f"  layer {layer:3d}: {acc:.3f}")

    diag_layer = DIAGNOSTIC_LAYER if DIAGNOSTIC_LAYER < len(X_A) else len(X_A) - 1
    print(f"\nFitting probe at diagnostic layer {diag_layer} ...")
    clf, scaler = fit_probe_at_layer(X_A[diag_layer], y_A)

    print("\nExtracting (W, T) cell activations ...")
    print(f"  Cells: {len(INVENTED_WORDS)} W x {len(OPERATORS)} T = "
          f"{len(INVENTED_WORDS) * len(OPERATORS)} cells x {N_PER_CELL} stimuli = "
          f"{len(INVENTED_WORDS) * len(OPERATORS) * N_PER_CELL} prompts")
    t0 = time.time()
    pred_counts_by_cell: dict[tuple[str, str], Counter] = {}
    for W in INVENTED_WORDS:
        for T in OPERATORS:
            cell_rng = random.Random(SEED + hash((W, T)) % 10000)
            stimuli = make_cell_stimuli(T, W, cell_rng, n=N_PER_CELL)
            X_cell = extract_anchored_activations(model, tok, stimuli, [W], device)
            y_pred = probe_predict(clf, scaler, X_cell[diag_layer])
            pred_counts_by_cell[(W, T)] = Counter(y_pred.tolist())
    print(f"  total cell extraction time: {time.time() - t0:.1f}s")

    print_cell_matrix(
        pred_counts_by_cell, INVENTED_WORDS, OPERATORS, OPERATORS,
        f"Cell prediction-count matrix at layer {diag_layer}",
    )

    # H4 quantification: for each canonical C, compute the in-template vs
    # out-of-template predicted-as-C share, averaged across invented words.
    print("\n\nH4 quantification (per-canonical):")
    print("=" * 80)
    print()
    print(f"  H4_pull(C) = avg_W [ P(pred = C | W in T=C-template) ]")
    print(f"             - avg_{{W, T' != C}} [ P(pred = C | W in T'-template) ]")
    print()
    print(f"  A positive H4_pull(C) means the C-template lexical context biases the")
    print(f"  probe toward predicting C beyond the off-template baseline.")
    print()

    print(f"  {'Canonical C':<14s} {'in-template':>13s} {'out-of-template':>17s} {'H4_pull':>10s}")
    print(f"  {'-' * 14}  {'-' * 12} {'-' * 17} {'-' * 9}")
    h4_pull_by_canonical: dict[str, float] = {}
    for C in OPERATORS:
        in_template_counts = [
            pred_counts_by_cell[(W, C)].get(C, 0) for W in INVENTED_WORDS
        ]
        in_template_rate = float(np.mean(in_template_counts)) / N_PER_CELL

        out_template_counts: list[int] = []
        for W in INVENTED_WORDS:
            for T in OPERATORS:
                if T == C:
                    continue
                out_template_counts.append(pred_counts_by_cell[(W, T)].get(C, 0))
        out_template_rate = float(np.mean(out_template_counts)) / N_PER_CELL

        h4_pull = in_template_rate - out_template_rate
        h4_pull_by_canonical[C] = h4_pull
        print(
            f"  {C:<14s} {in_template_rate:>12.3f}  "
            f"{out_template_rate:>16.3f}  {h4_pull:>+9.3f}"
        )

    # Show the marginal distribution of predictions in each (W, T) cell as
    # percentages, aggregated over W (to expose template effect per canonical).
    print("\n\nPredicted-class share by template, averaged over invented words:")
    print("=" * 80)
    print()
    header = f"  {'template':<14s} | " + " | ".join(
        f"{c:>13s}" for c in OPERATORS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for T in OPERATORS:
        avg_by_c = {
            c: float(np.mean([pred_counts_by_cell[(W, T)].get(c, 0) for W in INVENTED_WORDS])) / N_PER_CELL
            for c in OPERATORS
        }
        row = f"  {T:<14s} | " + " | ".join(
            f"{avg_by_c[c]:>12.1%}" + ("*" if c == T else " ")
            for c in OPERATORS
        )
        print(row)
    print()
    print("  (* indicates the canonical that the template family 'owns'.)")
    print("  H1 reads off the `not` column: should be high across all rows.")
    print("  H4 reads off the diagonal: should be elevated above the column off-diagonal.")

    # Per-template `not` share (the H1 dominance check across template families)
    print("\n\nH1 stability across templates (should be similar across template families if H1 is structural):")
    print("=" * 80)
    print(f"  {'template':<14s}  {'avg(pred=not)':>15s}")
    print(f"  {'-' * 14}  {'-' * 15}")
    for T in OPERATORS:
        rates = [pred_counts_by_cell[(W, T)].get("not", 0) for W in INVENTED_WORDS]
        avg_rate = float(np.mean(rates)) / N_PER_CELL
        print(f"  {T:<14s}  {avg_rate:>14.1%}")

    print("\nReading guide:")
    print(f"  Strong H4:      diagonal (template-T -> predicted-T) is well above the")
    print(f"                  off-diagonal. H4_pull(C) > 0.20 for at least three canonicals.")
    print(f"  Weak H4:        H4_pull(C) < 0.05 for all canonicals. Template-context")
    print(f"                  contribution is near zero; what we saw in scripts 11-12 was")
    print(f"                  embedding-driven, not template-driven.")
    print(f"  H1 robust:      avg(pred=not) is similar (within ~10 pp) across all five")
    print(f"                  template families. The default attractor doesn't depend on")
    print(f"                  template.")
    print(f"  H1 modulated:   avg(pred=not) is materially lower in templates whose canonical")
    print(f"                  has a strong H4 channel (because H4 'steals' predictions from")
    print(f"                  the H1 attractor).")


if __name__ == "__main__":
    main()
