"""L=1 bar-anomaly follow-up.

Script 10 found that in the L=1 condition (foo/bar/baz/fred), `bar` is
recovered as `or` by the canonical-operator probe at 82% (41/50 at 7B
layer 7), against a backdrop of ~92% default-to-`not` for every other
invented word at every other L condition. This is the only non-trivial
diagonal recovery in the entire subword-length experiment.

We hypothesized H3 (embedding-similarity channel): `bar` has accidental
embedding proximity to `or`, allowing the probe to recover it.

This script tests H3 directly with three L=1 invented-word sets:

  Set A (original):   {and:foo, or:bar,  not:baz, implies:fred}
  Set B (bar moved):  {and:bar, or:foo,  not:baz, implies:fred}
  Set C (no bar):     {and:qux, or:quux, not:thud, implies:pop}  -- validated

Predictions per hypothesis:

  H3 confirmed:
    - Set A reproduces bar -> or (41/50-ish).
    - Set B shows bar -> or (predicted) even when bar is in the `and` slot.
      That is, the recovery follows bar regardless of which canonical it
      is "supposed to" represent. This is the cleanest signature.
    - Set C shows default-to-`not` for all four operators (no recovery).

  H3 rejected (positional artifact):
    - Set A reproduces bar -> or.
    - Set B shows bar -> default-to-`not` (whatever's in the `or` slot
      gets the recovery instead, suggesting the effect is template-position).
    - Set C result depends on which "or" slot word was picked.

  H3 rejected (probe artifact):
    - Set A fails to reproduce bar -> or on a fresh seed.

Diagnostic layer 7 (matching scripts 09 and 10 for direct comparability).
"""

from __future__ import annotations

import random
import re
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-1124-7B"
DIAGNOSTIC_LAYER = 7

N_STIMULI = 200
SEED = 17


TEMPLATES_BY_OPERATOR = {
    "and": [
        "If {p} and {q} are both true, then {p} and {q} is true.",
        "{p} and {q} is true only when both {p} and {q} are true.",
        "If {p} is false, then {p} and {q} is false regardless of {q}.",
        "Whenever {p} and {q} hold, both {p} and {q} must be true.",
        "The conjunction {p} and {q} requires {p} and {q} to be true.",
    ],
    "or": [
        "{p} or {q} is true when at least one of {p} or {q} is true.",
        "Either {p} or {q} but not both means exactly one of {p}, {q} is true.",
        "When {p} or {q} is true, at least one of {p} and {q} must be true.",
        "The disjunction {p} or {q} holds if {p} or {q} holds.",
        "If neither {p} nor {q} is true, then {p} or {q} is false.",
    ],
    "not": [
        "If not {p}, then {p} and {q} is false.",
        "The statement not {p} is true exactly when {p} is false.",
        "When not {p} holds, {p} cannot be true.",
        "{p} and not {q} is true only when {p} is true and {q} is false.",
        "If not {p} is false, then {p} must be true.",
    ],
    "implies": [
        "If {p} implies {q} and {p} is true, then {q} must be true.",
        "The statement {p} implies {q} is false only when {p} is true and {q} is false.",
        "Whenever {p} implies {q} holds and {p} is true, {q} follows.",
        "If {p} implies {q}, then not {p} or {q} must be true.",
        "The implication {p} implies {q} holds when {p} is false or {q} is true.",
    ],
}

OPERATORS = list(TEMPLATES_BY_OPERATOR.keys())


SET_A_ORIGINAL = {"and": "foo", "or": "bar", "not": "baz", "implies": "fred"}
SET_B_BAR_MOVED = {"and": "bar", "or": "foo", "not": "baz", "implies": "fred"}
SET_C_NO_BAR_CANDIDATES = [
    "qux", "quux", "thud", "pop", "zap", "ping", "boop", "huh", "hmm", "yay",
    "oof", "ack", "ick", "eep", "thwack", "ergo", "viz", "yon", "anon",
]


def count_subwords(tok, word: str) -> int:
    ids = tok(" " + word, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    return len(ids)


def select_l1_words_excluding(tok, candidates: list[str], excluded: set[str], n: int) -> list[str]:
    """Pick `n` distinct L=1 words from candidates, skipping `excluded`."""
    picked: list[str] = []
    for w in candidates:
        if w in excluded or w in picked:
            continue
        if count_subwords(tok, w) == 1:
            picked.append(w)
        if len(picked) == n:
            break
    if len(picked) < n:
        raise RuntimeError(
            f"Only found {len(picked)} L=1 candidates from "
            f"{len(candidates)} options excluding {excluded}; "
            f"need {n}. Expand SET_C_NO_BAR_CANDIDATES."
        )
    return picked


def make_canonical(rng: random.Random) -> tuple[list[str], list[str]]:
    stimuli: list[str] = []
    labels: list[str] = []
    vars_ = ["p", "q", "r", "s"]
    per_op = N_STIMULI // len(OPERATORS)
    for op in OPERATORS:
        for _ in range(per_op):
            tmpl = rng.choice(TEMPLATES_BY_OPERATOR[op])
            p, q = rng.sample(vars_, 2)
            stimuli.append(tmpl.format(p=p, q=q))
            labels.append(op)
    order = list(range(len(stimuli)))
    rng.shuffle(order)
    return [stimuli[i] for i in order], [labels[i] for i in order]


def apply_op_map(prompts: list[str], op_map: dict[str, str]) -> list[str]:
    pattern = re.compile(r"\b(" + "|".join(op_map.keys()) + r")\b")
    return [pattern.sub(lambda m: op_map[m.group(1)], s) for s in prompts]


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


def heldout_accuracy(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = make_classifier()
    clf.fit(X_tr, y_train)
    return float(clf.score(X_te, y_test))


def heldout_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = make_classifier()
    clf.fit(X_tr, y_train)
    return clf.predict(X_te)


def print_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], title: str,
    invented_lookup: dict[str, str] | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"\n  {title}")
    print(f"    rows = true canonical operator, cols = predicted")
    header = "    " + " " * 10 + "".join(f"{c:>9s}" for c in classes) + "    row total"
    print(header)
    for i, c in enumerate(classes):
        row_total = int(cm[i].sum())
        cells = "".join(f"{cm[i][j]:>9d}" for j in range(len(classes)))
        suffix = ""
        if invented_lookup is not None and c in invented_lookup:
            suffix = f"    ({c} <- {invented_lookup[c]!r})"
        print(f"    {c:<10s}{cells}    {row_total:>4d}{suffix}")
    diag = sum(cm[i][i] for i in range(len(classes)))
    total = cm.sum()
    print(f"    overall accuracy: {diag}/{total} = {diag/total:.3f}")


def fmt(v: float) -> str:
    return f"{v:.3f}" if not (v != v) else "  nan"


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

    print("\nValidating invented-word subword counts (all must be L=1) ...")
    for name, op_map in [("A_original", SET_A_ORIGINAL), ("B_bar_moved", SET_B_BAR_MOVED)]:
        for canon, inv in op_map.items():
            n = count_subwords(tok, inv)
            print(f"  {name}: {canon}->{inv}({n})")
            if n != 1:
                raise RuntimeError(f"In set {name}, {inv!r} tokenizes to {n} subwords, expected 1")

    print("  Selecting Set C from no-bar candidate pool ...")
    set_c_words = select_l1_words_excluding(
        tok, SET_C_NO_BAR_CANDIDATES, excluded={"foo", "bar", "baz", "fred"}, n=4
    )
    SET_C = dict(zip(OPERATORS, set_c_words))
    for canon, inv in SET_C.items():
        print(f"  C_no_bar:  {canon}->{inv}({count_subwords(tok, inv)})")

    SETS = {
        "A_original": SET_A_ORIGINAL,
        "B_bar_moved": SET_B_BAR_MOVED,
        "C_no_bar": SET_C,
    }

    rng = random.Random(SEED)
    A, labels = make_canonical(rng)
    y = np.array(labels)

    print(f"\nN stimuli per condition: {len(A)}")
    print(f"Generating B'_set for each of the three sets ...")
    Bp_by_set: dict[str, list[str]] = {
        name: apply_op_map(A, op_map) for name, op_map in SETS.items()
    }

    print(f"\nExtracting operator-anchored activations ...")
    print(f"  Total prompts: {(1 + len(SETS)) * len(A)} = "
          f"1 (A) + {len(SETS)} (B') x {len(A)}")
    t0 = time.time()
    X_A = extract_anchored_activations(model, tok, A, OPERATORS, device)
    X_Bp_by_set: dict[str, list[np.ndarray]] = {}
    for name, op_map in SETS.items():
        invented = list(op_map.values())
        print(f"    extracting B'_{name} ({invented}) ...")
        X_Bp_by_set[name] = extract_anchored_activations(
            model, tok, Bp_by_set[name], invented, device
        )
    print(f"  total extraction time: {time.time() - t0:.1f}s")

    n_layers = len(X_A)
    chance = 1.0 / len(OPERATORS)
    print(f"\nChance accuracy: {chance:.3f} ({len(OPERATORS)}-class)")

    print("\nSanity probe at layer 1 (probe trained on canonical A):")
    acc_test = cv_accuracy(X_A[1], y, seed=SEED)
    print(f"  layer-1 CV accuracy on A: {acc_test:.3f}")

    print("\nPer-layer probe accuracy (probe trained on A):")
    print()
    set_names = list(SETS.keys())
    header = "Layer  A(CV)   " + "".join(f"{n:>14s}  " for n in set_names)
    print(header)
    print("-" * len(header))
    for layer in range(n_layers):
        acc_A = cv_accuracy(X_A[layer], y, seed=SEED)
        accs = {
            name: heldout_accuracy(X_A[layer], y, X_Bp_by_set[name][layer], y)
            for name in set_names
        }
        row = "  " + f"{layer:3d}    {fmt(acc_A)}  " + "".join(
            f"{fmt(accs[n]):>14s}  " for n in set_names
        )
        print(row)

    diag_layer = DIAGNOSTIC_LAYER if DIAGNOSTIC_LAYER < n_layers else n_layers - 1
    print(f"\n\nConfusion matrices at the fixed diagnostic layer {diag_layer}:")
    print("=" * 70)

    y_pred_A = heldout_predict(X_A[diag_layer], y, X_A[diag_layer])
    print_confusion(y, y_pred_A, OPERATORS, f"A (canonical, layer {diag_layer}) — self-prediction sanity")

    for name, op_map in SETS.items():
        y_pred = heldout_predict(X_A[diag_layer], y, X_Bp_by_set[name][diag_layer])
        print_confusion(
            y, y_pred, OPERATORS,
            f"B' (set {name}, layer {diag_layer})",
            invented_lookup={canon: inv for canon, inv in op_map.items()},
        )

    print("\n\nKey diagnostic comparisons:")
    print("=" * 70)
    # Where does the bar-input land across sets A and B?
    print("\nWhere do `bar`-input rows predict, across sets A and B?")
    print("  (Set A: bar replaces `or`; Set B: bar replaces `and`)")
    for name in ("A_original", "B_bar_moved"):
        op_map = SETS[name]
        bar_canon = next((canon for canon, inv in op_map.items() if inv == "bar"), None)
        if bar_canon is None:
            continue
        y_pred = heldout_predict(X_A[diag_layer], y, X_Bp_by_set[name][diag_layer])
        bar_rows = (y == bar_canon)
        bar_preds = y_pred[bar_rows]
        counts = {c: int((bar_preds == c).sum()) for c in OPERATORS}
        total = int(bar_rows.sum())
        print(f"  {name}: bar is in the {bar_canon!r} slot. "
              f"Predictions on those rows -> {counts} (n={total})")
        if total > 0:
            argmax = max(counts.items(), key=lambda kv: kv[1])
            print(f"    dominant prediction: {argmax[0]!r} ({argmax[1]}/{total} = "
                  f"{argmax[1]/total:.2%})")

    print("\nReading guide:")
    print("  H3 confirmed: in BOTH Set A and Set B, the bar-input rows are dominantly")
    print("    predicted as 'or'. The recovery follows bar regardless of slot.")
    print("    Set C should show default-to-`not` for all four (no escape hatch).")
    print("  H3 rejected (positional artifact): bar-input rows in Set B revert to")
    print("    default-to-`not`; whatever invented word is in the 'or' slot in Set B")
    print("    (i.e. `foo`) gets a partial 'or' recovery instead.")
    print("  H3 rejected (probe artifact): Set A fails to reproduce bar -> or with")
    print("    similar magnitude as script 10's 41/50 (0.82).")


if __name__ == "__main__":
    main()
