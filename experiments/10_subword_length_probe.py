"""Subword-length variation probe.

Vary BPE subword count of invented operators to distinguish two hypotheses
for the uniform-default-to-'not' failure observed at OLMo 2 7B layer 7 in
script 09:

  H1 (structural defaulting): the unique unary status of `not` in the canonical
       set draws unresolvable operator representations, regardless of how the
       invented operator tokenizes. Prediction: the default-to-`not` pattern
       persists across all four subword-count conditions, with similar
       magnitude (almost everything classified as `not`).

  H2 (tokenization-position): 2-subword invented operators land geometrically
       one token further into the sentence than 1-subword canonical operators,
       and that geometric shift happens to land near the canonical `not`-position
       cluster. Prediction: 1-subword invented operators (sharing canonical
       position-geometry) recover much more substrate-invariance than 4-subword
       ones; the default-mapping pattern strengthens monotonically with L.

Method:
  - Keep canonical operator set fixed: {and, or, not, implies}.
  - For each L in {1, 2, 3, 4}, select 4 invented words from a candidate pool
    such that each word tokenizes as exactly L BPE subwords when preceded by a
    space, per OLMo 2's live tokenizer. The L=2 set is constrained to match
    script 09's {bliq, dren, vusp, molex} where possible, for direct
    comparability.
  - Generate stimuli for A (canonical), B (var-renamed), and B'_L (operator-
    renamed at each length).
  - Train the canonical-operator linear probe on A's operator-anchored
    activations; evaluate on each B'_L.
  - Report per-layer probe accuracy per L; confusion matrices at the
    cross-condition diagnostic layer (default layer 7 on 7B, configurable).

Outputs let us read the dependence of operator substrate-invariance on
tokenization length directly.
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
DIAGNOSTIC_LAYER = 7  # peak A-B' gap layer at 7B from script 09

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

# Candidate pool for invented operator words. The script validates each
# candidate's actual subword count against the live tokenizer and selects 4
# matches per target L. The pool is deliberately oversized so that even if
# many candidates collide with common tokens (and tokenize unexpectedly), we
# can still fill all four length conditions. Words are kept semantically
# inert (nonsense or programmer-placeholder style) and visually diverse.
CANDIDATE_POOL = [
    # programmer placeholders (often 1 token in code-trained vocabs)
    "foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply",
    "waldo", "fred", "plugh", "xyzzy", "thud",
    # short Latin / archaic English
    "viz", "qua", "ergo", "ipso", "yon", "anon", "thee", "thou",
    "hither", "thither", "whither", "henceforth",
    # onomatopoeia / interjections (often 1-2 tokens)
    "huh", "hmm", "oof", "yay", "ack", "ick", "eep", "urk",
    "blip", "boop", "ping", "pong", "ding", "dong", "splat", "pop",
    "zap", "kaboom", "thwack", "fwip", "shplork",
    # script-09's L=2 set (preserve for cross-comparison)
    "bliq", "dren", "vusp", "molex",
    # invented short
    "frob", "snurk", "klep", "drox", "vepth", "moltz", "qwib", "yexis",
    "blarsh", "fronix", "morbisk", "splerg", "tronkle", "vorpish",
    # invented medium
    "qibblist", "drennox", "vusplark", "molexite", "frobnix", "snurklon",
    "klepvort", "droxitone", "blaroxxen", "froniexis", "morbisken",
    "splergoff", "tronklenix", "vorpishly",
    # invented long
    "qibblistor", "drennoxxic", "vusplarktic", "molexitium",
    "frobnixotur", "snurklonity", "klepvortexus", "droxitonely",
    "blaroxxenium", "froniexisotur", "morbiskenity", "splergoffian",
    # invented very long
    "qibblistorium", "drennoxxicality", "vusplarktically", "molexitiumous",
    "frobnixoturian", "snurklonityful", "klepvortexusish", "droxitonelyism",
]


def count_subwords(tok, word: str) -> int:
    """Count BPE subwords for ' ' + word (operator position in our templates)."""
    ids = tok(" " + word, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    return len(ids)


def select_invented_words_by_length(
    tok, target_lengths: list[int], candidate_pool: list[str],
    preferred_by_length: dict[int, list[str]] | None = None,
) -> dict[int, list[str]]:
    """For each target length, pick 4 distinct invented words from the pool
    that tokenize to exactly that many subwords (preceded by a space).

    `preferred_by_length` lets the caller pin specific words at specific
    lengths (used to keep the L=2 set aligned with script 09 when possible).
    """
    preferred = preferred_by_length or {}
    selected: dict[int, list[str]] = {}
    used: set[str] = set()

    # First, satisfy preferred picks where they validate.
    for L, prefs in preferred.items():
        bucket: list[str] = []
        for w in prefs:
            if w in used:
                continue
            if count_subwords(tok, w) == L:
                bucket.append(w)
                used.add(w)
            if len(bucket) == 4:
                break
        selected[L] = bucket

    # Fill remaining slots from the general pool.
    for L in target_lengths:
        bucket = selected.get(L, [])
        if len(bucket) == 4:
            continue
        for w in candidate_pool:
            if w in used:
                continue
            if count_subwords(tok, w) == L:
                bucket.append(w)
                used.add(w)
            if len(bucket) == 4:
                break
        selected[L] = bucket
        if len(bucket) < 4:
            raise RuntimeError(
                f"Only {len(bucket)} candidates tokenize to L={L} subwords; "
                f"expand CANDIDATE_POOL with more length-{L} options."
            )

    return selected


def build_op_map(invented_words: list[str]) -> dict[str, str]:
    """Map canonical operator -> invented replacement. Order matches OPERATORS."""
    if len(invented_words) != len(OPERATORS):
        raise ValueError(
            f"Need {len(OPERATORS)} invented words, got {len(invented_words)}"
        )
    return dict(zip(OPERATORS, invented_words))


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


def apply_op_map(prompts: list[str], op_map: dict[str, str]) -> list[str]:
    pattern = re.compile(r"\b(" + "|".join(op_map.keys()) + r")\b")
    return [pattern.sub(lambda m: op_map[m.group(1)], s) for s in prompts]


def find_operator_anchor(tok, prompt: str, operators: list[str]) -> int | None:
    """Position immediately after the first operator's last subtoken."""
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

    target_lengths = [1, 2, 3, 4]
    print(f"\nSelecting invented-operator words by BPE subword count ...")
    preferred = {2: ["bliq", "dren", "vusp", "molex"]}
    invented_by_L = select_invented_words_by_length(
        tok, target_lengths, CANDIDATE_POOL, preferred_by_length=preferred
    )
    op_maps_by_L: dict[int, dict[str, str]] = {}
    for L in target_lengths:
        op_map = build_op_map(invented_by_L[L])
        op_maps_by_L[L] = op_map
        print(f"  L={L}: " + ", ".join(
            f"{canon}->{inv}({count_subwords(tok, inv)})"
            for canon, inv in op_map.items()
        ))

    rng = random.Random(SEED)
    A, labels = make_canonical(rng)
    B = apply_var_map(A)
    Bp_by_L: dict[int, list[str]] = {
        L: apply_op_map(A, op_maps_by_L[L]) for L in target_lengths
    }

    print(f"\nN stimuli per condition: {len(A)}")
    print(f"Class distribution:")
    for op in OPERATORS:
        n = labels.count(op)
        print(f"  {op:<8} {n:>4d}")

    print(f"\nExtracting operator-anchored activations ...")
    print(f"  Conditions: A, B, B'_L for L in {target_lengths}")
    print(f"  Total prompts: {(2 + len(target_lengths)) * len(A)}")
    t0 = time.time()
    X_A = extract_anchored_activations(model, tok, A, OPERATORS, device)
    X_B = extract_anchored_activations(model, tok, B, OPERATORS, device)
    X_Bp_by_L: dict[int, list[np.ndarray]] = {}
    for L in target_lengths:
        invented = list(op_maps_by_L[L].values())
        print(f"    extracting B'_L={L} ({invented}) ...")
        X_Bp_by_L[L] = extract_anchored_activations(
            model, tok, Bp_by_L[L], invented, device
        )
    print(f"  total extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers: {len(X_A)}, hidden_dim: {X_A[0].shape[1]}")

    y = np.array(labels)
    n_layers = len(X_A)
    chance = 1.0 / len(OPERATORS)
    print(f"\nChance accuracy: {chance:.3f} ({len(OPERATORS)}-class)")

    print("\nSanity probe at layer 1:")
    acc_test = cv_accuracy(X_A[1], y, seed=SEED)
    print(f"  layer-1 CV accuracy on A: {acc_test:.3f}")

    print("\nPer-layer probe accuracy (probe trained on A, evaluated on each condition):")
    print()
    header = "Layer  A(CV)    B       " + "".join(f"B'_L{L}   " for L in target_lengths)
    print(header)
    print("-" * len(header))

    rows: list[tuple[int, float, float, dict[int, float]]] = []
    first_err: Exception | None = None
    for layer in range(n_layers):
        try:
            acc_A = cv_accuracy(X_A[layer], y, seed=SEED)
            acc_B = heldout_accuracy(X_A[layer], y, X_B[layer], y)
            acc_Bp_by_L = {
                L: heldout_accuracy(X_A[layer], y, X_Bp_by_L[L][layer], y)
                for L in target_lengths
            }
        except Exception as e:
            if first_err is None:
                first_err = e
                print(f"\n  !! layer {layer} failed: {type(e).__name__}: {e}")
            acc_A = acc_B = float("nan")
            acc_Bp_by_L = {L: float("nan") for L in target_lengths}
        rows.append((layer, acc_A, acc_B, acc_Bp_by_L))
        bp_str = "".join(f"{fmt(acc_Bp_by_L[L]):>6s}  " for L in target_lengths)
        print(f"  {layer:3d}    {fmt(acc_A)}    {fmt(acc_B)}   {bp_str}")

    print("\nGap (A - B'_L) per layer (positive = substrate-invariance failure):")
    print()
    header = "Layer  " + "".join(f"gap_L{L}   " for L in target_lengths)
    print(header)
    print("-" * len(header))
    for layer, acc_A, _, acc_Bp_by_L in rows:
        gap_str = "".join(
            f"{fmt(acc_A - acc_Bp_by_L[L]):>6s}   " for L in target_lengths
        )
        print(f"  {layer:3d}    {gap_str}")

    valid_rows = [
        (L_idx, layer, acc_A, acc_Bp_by_L)
        for L_idx, (layer, acc_A, _, acc_Bp_by_L) in enumerate(rows)
        if not (acc_A != acc_A)
        and not any(acc_Bp_by_L[L] != acc_Bp_by_L[L] for L in target_lengths)
    ]
    if valid_rows:
        print("\nSummary per length condition:")
        for L in target_lengths:
            min_layer, min_acc = min(
                ((layer, acc_Bp_by_L[L]) for _, layer, _, acc_Bp_by_L in valid_rows),
                key=lambda x: x[1],
            )
            max_gap_layer, max_gap = max(
                (
                    (layer, acc_A - acc_Bp_by_L[L])
                    for _, layer, acc_A, acc_Bp_by_L in valid_rows
                ),
                key=lambda x: x[1],
            )
            print(
                f"  L={L}: min B'_L = {min_acc:.3f} at layer {min_layer}; "
                f"max gap = {max_gap:.3f} at layer {max_gap_layer}"
            )

    # Confusion matrices at the cross-condition diagnostic layer.
    diag_layer = DIAGNOSTIC_LAYER if DIAGNOSTIC_LAYER < n_layers else n_layers - 1
    print(f"\n\nConfusion matrices at the fixed diagnostic layer {diag_layer}:")
    print(f"(Layer {diag_layer} was the peak A-B' gap layer for L=2 at 7B in script 09.)")
    print("=" * 70)

    y_pred_A = heldout_predict(X_A[diag_layer], y, X_A[diag_layer])
    print_confusion(y, y_pred_A, OPERATORS, f"A (canonical, layer {diag_layer}) — self-prediction sanity")

    y_pred_B = heldout_predict(X_A[diag_layer], y, X_B[diag_layer])
    print_confusion(y, y_pred_B, OPERATORS, f"B (var-renamed, layer {diag_layer}) — operators unchanged")

    for L in target_lengths:
        invented = list(op_maps_by_L[L].values())
        y_pred = heldout_predict(X_A[diag_layer], y, X_Bp_by_L[L][diag_layer])
        title = (
            f"B'_L={L} (op-renamed, {L}-subword invented, layer {diag_layer})"
        )
        print_confusion(y, y_pred, OPERATORS, title)
        print("    Mapping of invented words to canonical labels:")
        for canon, inv in op_maps_by_L[L].items():
            print(f"      true={canon!r:<12s} -> input contains {inv!r}")

    print("\nReading guide:")
    print(f"  H1 (structural defaulting)  => all four B'_L confusion matrices")
    print(f"     show a single column dominant on `not`, with similar magnitude.")
    print(f"     Probe accuracy on B'_L flat across L.")
    print(f"  H2 (tokenization-position)  => B'_L=1 confusion matrix is close to")
    print(f"     diagonal; default-to-`not` strengthens monotonically with L.")
    print(f"     Probe accuracy on B'_L decreases with L (B'_L=1 high, B'_L=4 low).")
    print(f"  Intermediate / mixed        => some structural and some positional")
    print(f"     contribution; further teasing apart needed.")


if __name__ == "__main__":
    main()
