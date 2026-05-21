"""Operator-identity linear probe.

A direct, interpretable measurement of operator substrate-invariance: train
a linear classifier on the operator-anchored activations of canonical
prompts (A) to predict which canonical operator (and/or/not/implies) was
at that position, then test the same probe on activations from:

  - B   variable-renamed (operators unchanged): positive control, accuracy
        should match A's baseline
  - B'  operator-renamed (bliq/dren/vusp/molex): the substrate-invariance
        question — does the probe still recover operator identity from
        invented-operator activations?
  - B'' both renamed: tests the B'' > B' counterintuition from script 08
        at the level of operator semantics rather than manifold similarity

Probe accuracy per layer gives a single interpretable curve per condition.
The accuracy gap (probe_acc_on_A − probe_acc_on_B') is the cleanest single
number for "how much does the model fail to map bliq -> AND at this layer."

Predictions worth pre-registering:

  (1) Strong Platonic: probe_acc(B') ≈ probe_acc(B) ≈ probe_acc(A) at
      middle-to-late layers. Model treats invented operators as fully
      equivalent to canonical ones.
  (2) Partial recovery: probe_acc(B') climbs with depth, reaching maybe
      60-80% in late layers but never matching B. The U-shape in 08's
      CKA gets a quantitative semantic interpretation.
  (3) Syntactic-only: probe_acc(B') stays near chance (25%) at all layers.
      Operator-anchored position has compositional structure but no
      operator-specific semantics.
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
# For 1B reference: MODEL_ID = "allenai/OLMo-2-0425-1B"

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
VAR_MAP_GREEK = {"p": "α", "q": "β", "r": "γ", "s": "δ"}
OP_MAP = {"and": "bliq", "or": "dren", "not": "vusp", "implies": "molex"}
INVERSE_OP_MAP = {v: k for k, v in OP_MAP.items()}


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


def apply_var_map(prompts: list[str]) -> list[str]:
    pattern = re.compile(r"\b([pqrs])\b")
    return [pattern.sub(lambda m: VAR_MAP_GREEK[m.group(1)], s) for s in prompts]


def apply_op_map(prompts: list[str]) -> list[str]:
    pattern = re.compile(r"\b(" + "|".join(OP_MAP.keys()) + r")\b")
    return [pattern.sub(lambda m: OP_MAP[m.group(1)], s) for s in prompts]


def find_operator_anchor(tok, prompt: str, operators: list[str]) -> int | None:
    """Position immediately after the first operator's last subtoken.
    Returns None if no operator found."""
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
    """Returns list of [n_prompts, hidden] arrays, one per layer."""
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
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], title: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"\n  {title}")
    print(f"    rows = true canonical operator, cols = predicted")
    header = "    " + " " * 10 + "".join(f"{c:>9s}" for c in classes) + "    row total"
    print(header)
    for i, c in enumerate(classes):
        row_total = int(cm[i].sum())
        cells = "".join(f"{cm[i][j]:>9d}" for j in range(len(classes)))
        print(f"    {c:<10s}{cells}    {row_total:>4d}")
    diag = sum(cm[i][i] for i in range(len(classes)))
    total = cm.sum()
    print(f"    overall accuracy: {diag}/{total} = {diag/total:.3f}")


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

    rng = random.Random(SEED)
    A, labels = make_canonical(rng)
    B = apply_var_map(A)
    Bp = apply_op_map(A)
    Bpp = apply_op_map(apply_var_map(A))

    print(f"\nN stimuli per condition: {len(A)}")
    print(f"Class distribution:")
    for op in OPERATORS:
        n = labels.count(op)
        print(f"  {op:<8} {n:>4d}")

    print(f"\nExtracting operator-anchored activations (4 conditions x {len(A)} prompts) ...")
    t0 = time.time()
    canonical_ops = list(OP_MAP.keys())
    invented_ops = list(OP_MAP.values())
    X_A = extract_anchored_activations(model, tok, A, canonical_ops, device)
    X_B = extract_anchored_activations(model, tok, B, canonical_ops, device)
    X_Bp = extract_anchored_activations(model, tok, Bp, invented_ops, device)
    X_Bpp = extract_anchored_activations(model, tok, Bpp, invented_ops, device)
    print(f"  extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers: {len(X_A)}, hidden_dim: {X_A[0].shape[1]}")

    y = np.array(labels)
    n_layers = len(X_A)
    chance = 1.0 / len(OPERATORS)
    print(f"\nChance accuracy: {chance:.3f} ({len(OPERATORS)}-class)")

    print("\nSanity probe at layer 1 (raises on failure so we see the error):")
    acc_test = cv_accuracy(X_A[1], y, seed=SEED)
    print(f"  layer-1 CV accuracy on A: {acc_test:.3f}")

    print("\nPer-layer probe accuracy (probe trained on A, evaluated on each condition):")
    print()
    print(f"Layer  A (CV)   B       B'      B''     gap(A-B')")
    print(f"-----  -------  ------  ------  ------  ---------")
    rows = []
    first_err: Exception | None = None
    for layer in range(n_layers):
        try:
            acc_A = cv_accuracy(X_A[layer], y, seed=SEED)
            acc_B = heldout_accuracy(X_A[layer], y, X_B[layer], y)
            acc_Bp = heldout_accuracy(X_A[layer], y, X_Bp[layer], y)
            acc_Bpp = heldout_accuracy(X_A[layer], y, X_Bpp[layer], y)
            gap = acc_A - acc_Bp
        except Exception as e:
            if first_err is None:
                first_err = e
                print(f"\n  !! layer {layer} failed: {type(e).__name__}: {e}")
            acc_A = acc_B = acc_Bp = acc_Bpp = gap = float("nan")
        rows.append((layer, acc_A, acc_B, acc_Bp, acc_Bpp, gap))
        def fmt(v):
            return f"{v:.3f}" if not (v != v) else "  nan"
        print(f"  {layer:3d}    {fmt(acc_A)}    {fmt(acc_B)}   {fmt(acc_Bp)}   {fmt(acc_Bpp)}   {fmt(gap)}")

    valid_rows = [r for r in rows if not (r[1] != r[1] or r[3] != r[3])]
    if valid_rows:
        max_A = max(valid_rows, key=lambda r: r[1])
        min_Bp = min(valid_rows, key=lambda r: r[3])
        max_gap = max(valid_rows, key=lambda r: r[5])
        print(f"\nSummary:")
        print(f"  Peak A accuracy: {max_A[1]:.3f} at layer {max_A[0]}")
        print(f"  Min B' accuracy: {min_Bp[3]:.3f} at layer {min_Bp[0]}")
        print(f"  Max gap (A-B'): {max_gap[5]:.3f} at layer {max_gap[0]}")

        diagnostic_layer = max_gap[0]
        print(f"\n\nConfusion matrices at layer {diagnostic_layer} (peak A-B' gap):")
        print("=" * 70)

        y_pred_B = heldout_predict(X_A[diagnostic_layer], y, X_B[diagnostic_layer])
        print_confusion(y, y_pred_B, OPERATORS, f"B (var-renamed, layer {diagnostic_layer}) — operators unchanged")

        y_pred_Bp = heldout_predict(X_A[diagnostic_layer], y, X_Bp[diagnostic_layer])
        print_confusion(y, y_pred_Bp, OPERATORS, f"B' (op-renamed, layer {diagnostic_layer}) — invented operator words")
        print("    Mapping of invented words to canonical labels:")
        for canon, inv in OP_MAP.items():
            print(f"      true={canon!r:<12s} -> input contains {inv!r}")

        y_pred_Bpp = heldout_predict(X_A[diagnostic_layer], y, X_Bpp[diagnostic_layer])
        print_confusion(y, y_pred_Bpp, OPERATORS, f"B'' (both renamed, layer {diagnostic_layer})")

        if valid_rows[1:]:
            mid_layer = len(valid_rows) // 2
            mid_row = valid_rows[mid_layer]
            if mid_row[0] != diagnostic_layer:
                print(f"\n\nFor comparison — confusion at layer {mid_row[0]} (mid-network):")
                print("=" * 70)
                y_pred_Bp_mid = heldout_predict(X_A[mid_row[0]], y, X_Bp[mid_row[0]])
                print_confusion(y, y_pred_Bp_mid, OPERATORS, f"B' at layer {mid_row[0]}")

    print("\nReading guide:")
    print(f"  acc(B') ≈ acc(B) ≈ acc(A)     => strong substrate-invariance for operators")
    print(f"  acc(B') climbing with depth   => in-context operator binding emerges with depth")
    print(f"  acc(B') stuck near chance ({chance:.2f})  => operator semantics not recovered")
    print(f"  acc(B') < acc(B) by >0.2      => meaningful substrate-invariance gap")
    print(f"\n  Confusion matrix interpretation:")
    print(f"    diagonal dominant on B'   => substrate-invariance present")
    print(f"    single column dominant    => model defaults all invented ops to one canonical")
    print(f"    per-word distinct columns => byte-structure of invented words drives mapping")


if __name__ == "__main__":
    main()
