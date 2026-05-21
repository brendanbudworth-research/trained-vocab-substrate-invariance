"""Second-canonical-unary probe.

The uniform default-to-`not` failure mode (script 09: 186/200 B' → not at
7B layer 7; script 10: 91-93% across L ∈ {1,2,3,4}) leaves one binary
hypothesis open:

  H1a: default to `not` specifically. The model has a `not`-shaped
       attractor at the operator-anchored position that pulls every
       unresolvable operator toward it.
  H1b: default to UNARY CLASS generically. The model has a "post-unary"
       cluster encompassing all unary canonicals; `not` was the only
       unary in our previous probe sets, so we couldn't separate H1a
       from H1b. With two unaries in the canonical set, an unresolvable
       operator should split between them.

We add `necessarily` as a second canonical unary (modal-logic box
operator, semantically distinct from `not`, structurally unary in the
template position). The probe becomes 5-class: {and, or, not, implies,
necessarily}. All five canonicals are replaced in B' with Tier-2
invented words.

Predictions:

  H1a confirmed:
    - Invented binary rows (and, or, implies) → predicted as `not`,
      with `necessarily` predictions near zero (<10%).
    - Invented `not`-replacement row → predicted as `not`.
    - Invented `necessarily`-replacement row → ALSO predicted as `not`
      (the H1a "trap" — invented unary still goes to `not`).

  H1b confirmed:
    - Invented binary rows split between `not` and `necessarily`,
      with combined unary-class share ≥ 80% but neither dominating.
    - Invented `not`-replacement and `necessarily`-replacement rows
      both split roughly evenly between the two unary classes.

  Intermediate:
    - `not` still dominates but `necessarily` gets a non-trivial share
      (10-40%). Suggests a partial-class-generic mechanism with `not`
      retaining a privileged position (e.g., due to higher token
      frequency in training).

Diagnostic layer 7 (matching scripts 09, 10, 11 for direct comparison).
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

N_PER_CLASS = 50
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
    # Modal-logic box operator. Templates contain only `necessarily` and no
    # other canonical operator, so find_operator_anchor reliably anchors here.
    "necessarily": [
        "If necessarily {p}, then {p} is true in every situation.",
        "The statement necessarily {p} means {p} is true always.",
        "When necessarily {p} holds, {p} cannot fail.",
        "Whenever necessarily {p} is asserted, {p} must hold.",
        "If necessarily {p} is false, then {p} is sometimes false.",
    ],
}

OPERATORS = list(TEMPLATES_BY_OPERATOR.keys())
VAR_MAP_GREEK = {"p": "α", "q": "β", "r": "γ", "s": "δ"}

# Original 4 invented operators (from scripts 09 and 10, L=2).
# `necessarily` replacement is selected at runtime from the candidate list.
BASE_OP_MAP = {"and": "bliq", "or": "dren", "not": "vusp", "implies": "molex"}
NECESSARILY_INVENTED_CANDIDATES = [
    "xelph", "noxim", "perph", "frob", "snurk", "klep", "drox", "vepth",
    "moltz", "qwib", "yexis", "blarsh", "fronix", "morbisk", "splerg",
]


def count_subwords(tok, word: str) -> int:
    ids = tok(" " + word, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    return len(ids)


def select_necessarily_replacement(tok, target_count: int = 2) -> str:
    """Pick the first candidate that tokenizes as exactly target_count subwords
    and isn't already used in BASE_OP_MAP."""
    used = set(BASE_OP_MAP.values())
    for w in NECESSARILY_INVENTED_CANDIDATES:
        if w in used:
            continue
        if count_subwords(tok, w) == target_count:
            return w
    raise RuntimeError(
        f"No {target_count}-subword candidate found for `necessarily` "
        f"in {NECESSARILY_INVENTED_CANDIDATES}. Extend the pool."
    )


def make_canonical(rng: random.Random) -> tuple[list[str], list[str]]:
    stimuli: list[str] = []
    labels: list[str] = []
    vars_ = ["p", "q", "r", "s"]
    for op in OPERATORS:
        for _ in range(N_PER_CLASS):
            tmpl = rng.choice(TEMPLATES_BY_OPERATOR[op])
            p, q = rng.sample(vars_, 2)
            stimuli.append(tmpl.format(p=p, q=q))
            labels.append(op)
    order = list(range(len(stimuli)))
    rng.shuffle(order)
    return [stimuli[i] for i in order], [labels[i] for i in order]


def apply_var_map(prompts: list[str]) -> list[str]:
    pattern = re.compile(r"\b([pqrs])\b")
    return [pattern.sub(lambda m: VAR_MAP_GREEK[m.group(1)], s) for s in prompts]


def apply_op_map(prompts: list[str], op_map: dict[str, str]) -> list[str]:
    # Sort keys by length (descending) so `necessarily` matches before any
    # accidental substring overlap with another canonical.
    keys = sorted(op_map.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(keys) + r")\b")
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
    header = "    " + " " * 14 + "".join(f"{c:>13s}" for c in classes) + "    row total"
    print(header)
    for i, c in enumerate(classes):
        row_total = int(cm[i].sum())
        cells = "".join(f"{cm[i][j]:>13d}" for j in range(len(classes)))
        suffix = ""
        if invented_lookup is not None and c in invented_lookup:
            suffix = f"    ({c} <- {invented_lookup[c]!r})"
        print(f"    {c:<14s}{cells}    {row_total:>4d}{suffix}")
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

    print("\nValidating tokenization of canonical operators:")
    for op in OPERATORS:
        n = count_subwords(tok, op)
        print(f"  {op:<13s} -> {n} subword(s)")
    print("  (note: `necessarily` is multi-subword; operator-anchored position\n"
          "   is taken immediately after its last subword, same logic as bliq.)")

    nec_inv = select_necessarily_replacement(tok, target_count=2)
    op_map = dict(BASE_OP_MAP)
    op_map["necessarily"] = nec_inv

    print("\nInvented-operator map (B'):")
    for canon, inv in op_map.items():
        print(f"  {canon:<13s} -> {inv} ({count_subwords(tok, inv)} subwords)")

    rng = random.Random(SEED)
    A, labels = make_canonical(rng)
    B = apply_var_map(A)
    Bp = apply_op_map(A, op_map)
    Bpp = apply_op_map(apply_var_map(A), op_map)
    y = np.array(labels)

    print(f"\nN stimuli: {len(A)} ({N_PER_CLASS} per class x {len(OPERATORS)} classes)")
    print(f"Class distribution:")
    for op in OPERATORS:
        n = labels.count(op)
        print(f"  {op:<13s} {n:>4d}")

    print(f"\nExtracting operator-anchored activations (4 conditions x {len(A)} prompts) ...")
    t0 = time.time()
    canonical_ops = OPERATORS
    invented_ops = list(op_map.values())
    X_A = extract_anchored_activations(model, tok, A, canonical_ops, device)
    X_B = extract_anchored_activations(model, tok, B, canonical_ops, device)
    X_Bp = extract_anchored_activations(model, tok, Bp, invented_ops, device)
    X_Bpp = extract_anchored_activations(model, tok, Bpp, invented_ops, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers: {len(X_A)}, hidden_dim: {X_A[0].shape[1]}")

    n_layers = len(X_A)
    chance = 1.0 / len(OPERATORS)
    print(f"\nChance accuracy: {chance:.3f} ({len(OPERATORS)}-class)")

    print("\nSanity probe at layer 1:")
    acc_test = cv_accuracy(X_A[1], y, seed=SEED)
    print(f"  layer-1 CV accuracy on A: {acc_test:.3f}")

    print("\nPer-layer probe accuracy (probe trained on A):")
    print()
    print(f"Layer  A (CV)   B       B'      B''     gap(A-B')")
    print(f"-----  -------  ------  ------  ------  ---------")
    rows: list[tuple[int, float, float, float, float, float]] = []
    for layer in range(n_layers):
        acc_A = cv_accuracy(X_A[layer], y, seed=SEED)
        acc_B = heldout_accuracy(X_A[layer], y, X_B[layer], y)
        acc_Bp = heldout_accuracy(X_A[layer], y, X_Bp[layer], y)
        acc_Bpp = heldout_accuracy(X_A[layer], y, X_Bpp[layer], y)
        gap = acc_A - acc_Bp
        rows.append((layer, acc_A, acc_B, acc_Bp, acc_Bpp, gap))
        print(f"  {layer:3d}    {fmt(acc_A)}    {fmt(acc_B)}   "
              f"{fmt(acc_Bp)}   {fmt(acc_Bpp)}   {fmt(gap)}")

    valid_rows = [r for r in rows if not (r[1] != r[1])]
    if valid_rows:
        max_gap_row = max(valid_rows, key=lambda r: r[5])
        print(f"\nSummary:")
        print(f"  Peak A accuracy:    {max(valid_rows, key=lambda r: r[1])[1]:.3f}")
        print(f"  Min B' accuracy:    {min(valid_rows, key=lambda r: r[3])[3]:.3f}")
        print(f"  Max gap (A-B'):     {max_gap_row[5]:.3f} at layer {max_gap_row[0]}")

    diag_layer = DIAGNOSTIC_LAYER if DIAGNOSTIC_LAYER < n_layers else n_layers - 1
    print(f"\n\nConfusion matrices at layer {diag_layer} (5-class probe):")
    print("=" * 80)

    y_pred_A = heldout_predict(X_A[diag_layer], y, X_A[diag_layer])
    print_confusion(y, y_pred_A, OPERATORS, f"A (canonical, layer {diag_layer}) — self-prediction sanity")

    y_pred_B = heldout_predict(X_A[diag_layer], y, X_B[diag_layer])
    print_confusion(y, y_pred_B, OPERATORS, f"B (var-renamed, layer {diag_layer})")

    y_pred_Bp = heldout_predict(X_A[diag_layer], y, X_Bp[diag_layer])
    print_confusion(
        y, y_pred_Bp, OPERATORS,
        f"B' (op-renamed, layer {diag_layer}) — all 5 invented",
        invented_lookup=op_map,
    )

    y_pred_Bpp = heldout_predict(X_A[diag_layer], y, X_Bpp[diag_layer])
    print_confusion(
        y, y_pred_Bpp, OPERATORS,
        f"B'' (both renamed, layer {diag_layer})",
        invented_lookup=op_map,
    )

    # H1a vs H1b headline numbers
    print("\n\nH1a vs H1b headline numbers (from B' confusion at layer {}):".format(diag_layer))
    print("=" * 80)
    cm_Bp = confusion_matrix(y, y_pred_Bp, labels=OPERATORS)
    op_to_idx = {op: i for i, op in enumerate(OPERATORS)}
    not_idx = op_to_idx["not"]
    nec_idx = op_to_idx["necessarily"]

    for true_op in OPERATORS:
        row = cm_Bp[op_to_idx[true_op]]
        total = int(row.sum())
        n_to_not = int(row[not_idx])
        n_to_nec = int(row[nec_idx])
        n_unary_total = n_to_not + n_to_nec
        invented = op_map[true_op]
        print(f"  true={true_op!r:<14s} (input={invented!r:<8s}): "
              f"{n_to_not}/{total} -> not, "
              f"{n_to_nec}/{total} -> necessarily, "
              f"unary-class total {n_unary_total}/{total} = "
              f"{n_unary_total/total:.0%}")

    print("\nReading guide:")
    print("  H1a (not-specific):")
    print("    invented binaries -> almost all 'not'; 'necessarily' near 0%")
    print("    invented 'necessarily' (xelph/etc) -> still mostly 'not'")
    print("  H1b (unary-class-generic):")
    print("    invented binaries split between 'not' and 'necessarily'")
    print("    invented 'necessarily' -> roughly equal between 'not' and 'necessarily'")
    print("    combined unary-class share >= 80% for invented binaries")
    print("  Intermediate:")
    print("    'not' still dominates but 'necessarily' captures 10-40% of invented")
    print("    binaries; suggests partial-class-generic with `not` privileged")
    print("    (possibly by training-corpus frequency).")


if __name__ == "__main__":
    main()
