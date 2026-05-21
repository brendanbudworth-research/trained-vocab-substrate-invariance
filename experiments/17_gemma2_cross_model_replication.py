"""Cross-model replication on Gemma 2 9B (the Phase 1 entry test).

Phase 0 established the arity-region attractor on OLMo 2 1B and 7B across four
probe instruments. The last remaining vulnerability to the headline claim is
model-specific quirkiness: maybe the arity attractor is an OLMo-family artifact
rather than a property of large language models in this size range.

This script replicates the two cleanest Phase 0 probe instruments on Gemma 2 9B
(google/gemma-2-9b, base — NOT the instruction-tuned variant). Gemma 2 is a
deliberately different choice from OLMo 2:

  - Different lab (Google DeepMind vs AI2)
  - Different training data (not Dolma)
  - Different architecture (Grouped Query Attention; soft-capping on logits and
    attention scores; alternating local/global attention)
  - Different tokenizer (256k SentencePiece vocab vs OLMo's BPE)
  - Different layer count (42 hidden layers vs OLMo 2 7B's 32)
  - Different hidden dim (3584 vs 4096)
  - Native bfloat16 (Gemma 2 9B is known to overflow in fp16 due to the
    soft-cap mechanism; we use bf16 throughout)

If the arity-region attractor reproduces here, that strongly supports the
"property of large LMs" reading. If it does not reproduce (or reproduces
weakly), we have to scope the claim to OLMo-family models and investigate why.

Two conditions, mirroring the Phase 0 cleanest probes:

  Condition 1 (NEUTRAL): script-15 canonical-neutral metalinguistic templates
                          ("Consider the word {op} in this sentence.")
                          Reference OLMo 2 7B result: 99.6% unary-region mass.

  Condition 2 (FUNCTIONAL-PREFIX): script-16 Condition 2 functional-prefix
                          notation ("The function {op}(p, q) returns a
                          boolean.") with binaries getting 2-arg calls and
                          unaries 1-arg calls. Reference OLMo 2 7B result:
                          100.0% unary-region mass (all 250 stimuli to `not`).

Per-condition we also compute:

  - Per-layer probe CV accuracy across all 42 Gemma 2 layers, to locate the
    Gemma-2 equivalent of OLMo 2's diagnostic layer 7 (the peak-gap layer for
    operator-identity classification).
  - Per-layer unary-region mass curve on invented operators, training a
    fresh probe at each layer. This is the mechanism trace: at what depth in
    Gemma 2's residual stream does the arity attractor get built? Phase 0 left
    this unanswered for OLMo 2; running it for Gemma 2 begins to map the
    timing question.

Gemma 2 is a gated model. The first run requires:

  1. Accept the Gemma 2 license at https://huggingface.co/google/gemma-2-9b
  2. `huggingface-cli login` with a personal access token
  3. The script will then download ~18 GB of weights to ~/.cache/huggingface

Memory: bf16 weights are ~18 GB; with activation buffers and overhead, expect
~25-30 GB resident. M4 with 48 GB unified memory is sufficient.

Diagnostic strategy: we run the full per-layer probe sweep on each condition,
then report invented-operator predictions at three layers per condition:
  - Layer with peak CV accuracy
  - Layer where unary-region mass peaks (may or may not be the same)
  - A fixed reference layer (~20% of total depth, matching OLMo 2 7B's layer 7
    out of 32, i.e. Gemma 2 9B layer 8 of 42)
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-9b"

# Fixed reference layer: 20% of Gemma 2 9B's 42 layers (matches OLMo 2 7B layer
# 7 / 32 ≈ 22%). Used as one of three diagnostic layers.
FIXED_REFERENCE_LAYER = 8

N_PER_CLASS = 50
SEED = 17

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
CANONICAL_ARITY = {
    "and": 2, "or": 2, "implies": 2,
    "not": 1, "necessarily": 1,
}

INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL = dict(zip(INVENTED_WORDS, CANONICALS))


# OLMo 2 7B reference results (from scripts 15 and 16 in this repo).
# Used only for the side-by-side comparison printout at the end.
OLMO2_7B_REFERENCE = {
    "neutral": {
        "diagnostic_layer": 7,
        "cv_accuracy": 0.996,
        "mean_unary_mass": 99.6,
        "binary_canonical_mass": 0.4,
        "or_mass": 0.0,
    },
    "functional_prefix": {
        "diagnostic_layer": 7,
        "cv_accuracy": 1.000,
        "mean_unary_mass": 100.0,
        "binary_canonical_mass": 0.0,
        "or_mass": 0.0,
        "binary_replacement_unary_mass": 100.0,
    },
}


# ==============================================================================
# Condition 1: script-15 canonical-neutral metalinguistic templates
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
# Condition 2: functional-prefix notation
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


def build_functional_stimulus(op: str, frame: str, p: str, q: str | None) -> str:
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


def count_subwords(tok, word: str) -> int:
    return len(tok(" " + word, add_special_tokens=False).input_ids)


def subwords(tok, word: str) -> list[str]:
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return [tok.decode([i]) for i in ids]


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
    """Return a list of (n_prompts, hidden_dim) arrays, one per hidden-state
    layer (including the embedding layer at index 0)."""
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


def compute_unary_mass(pred_counts: dict[str, Counter]) -> float:
    """Mean unary-region mass across all invented words."""
    masses = []
    for w, c in pred_counts.items():
        total = sum(c.values())
        if total == 0:
            continue
        masses.append(100 * (c.get("not", 0) + c.get("necessarily", 0)) / total)
    return float(np.mean(masses))


def compute_binary_replacement_unary_mass(pred_counts: dict[str, Counter]) -> float:
    """Mean unary-region mass restricted to binary-replacement invented words."""
    binary_replacements = [
        w for w in pred_counts if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 2
    ]
    masses = []
    for w in binary_replacements:
        c = pred_counts[w]
        total = sum(c.values())
        if total == 0:
            continue
        masses.append(100 * (c.get("not", 0) + c.get("necessarily", 0)) / total)
    if not masses:
        return float("nan")
    return float(np.mean(masses))


def per_layer_invented_predictions(
    clf_per_layer, scaler_per_layer, X_invented_per_layer: list[np.ndarray],
    invented_words_per_stimulus: list[str],
) -> dict[int, dict[str, Counter]]:
    """For each layer with a fitted probe, predict on all invented stimuli and
    bucket by invented word. Returns layer -> {word -> Counter of predictions}."""
    out: dict[int, dict[str, Counter]] = {}
    for layer, (clf, scaler) in enumerate(zip(clf_per_layer, scaler_per_layer)):
        if clf is None:
            continue
        preds = predict(clf, scaler, X_invented_per_layer[layer])
        per_word: dict[str, Counter] = {}
        for w in INVENTED_WORDS:
            per_word[w] = Counter()
        for word, pred in zip(invented_words_per_stimulus, preds.tolist()):
            per_word[word][pred] += 1
        out[layer] = per_word
    return out


def run_condition(
    name: str,
    canonical_stimuli_fn,
    invented_stimuli_fn,
    model, tok, device: str,
) -> dict:
    """Run a full condition: extract canonicals, train per-layer probes,
    extract inventeds at all layers, evaluate, return diagnostic results."""
    print(f"\n{'=' * 80}")
    print(f"Condition: {name}")
    print(f"{'=' * 80}")

    # Build and extract canonicals.
    print(f"\nBuilding canonical A ({N_PER_CLASS} per class x 5 classes)...")
    A: list[str] = []
    A_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(SEED + hash((name, op)) % 100000)
        A.extend(canonical_stimuli_fn(op, op_rng, N_PER_CLASS))
        A_labels.extend([op] * N_PER_CLASS)

    print("  Sample stimuli:")
    for op in CANONICALS:
        idx = A_labels.index(op)
        print(f"    {op:<14s}: {A[idx]!r}")

    print("\nExtracting canonical A activations across all layers...")
    t0 = time.time()
    X_A = extract_anchored_activations(model, tok, A, CANONICALS, device)
    n_layers = len(X_A)
    hidden_dim = X_A[0].shape[1]
    print(f"  extraction time: {time.time() - t0:.1f}s")
    print(f"  n_layers: {n_layers}, hidden_dim: {hidden_dim}")

    y_A = np.array(A_labels)

    # Build and extract inventeds (once, all layers).
    print("\nBuilding invented stimuli (50 per word x 5 words)...")
    invented_prompts: list[str] = []
    invented_word_per_stim: list[str] = []
    for w in INVENTED_WORDS:
        w_rng = random.Random(SEED + hash((name, w, "invent")) % 100000)
        stimuli = invented_stimuli_fn(w, w_rng, N_PER_CLASS)
        invented_prompts.extend(stimuli)
        invented_word_per_stim.extend([w] * len(stimuli))

    print("\nExtracting invented activations across all layers...")
    t0 = time.time()
    X_inv = extract_anchored_activations(
        model, tok, invented_prompts, INVENTED_WORDS, device
    )
    print(f"  extraction time: {time.time() - t0:.1f}s")

    # Per-layer probe training + CV accuracy + invented-prediction sweep.
    print(f"\nPer-layer probe sweep across all {n_layers} layers...")
    per_layer_cv: list[float] = []
    clf_per_layer: list = []
    scaler_per_layer: list = []

    sweep_t0 = time.time()
    for layer in range(n_layers):
        acc = cv_accuracy(X_A[layer], y_A, seed=SEED)
        per_layer_cv.append(acc)
        clf, scaler = fit_probe(X_A[layer], y_A)
        clf_per_layer.append(clf)
        scaler_per_layer.append(scaler)
    print(f"  sweep time: {time.time() - sweep_t0:.1f}s")

    print("\nPer-layer CV accuracy on canonical A:")
    print(f"  {'layer':>6s}  {'cv_acc':>7s}  {'unary_mass':>11s}  {'binary_repl_unary':>17s}")
    per_layer_pred = per_layer_invented_predictions(
        clf_per_layer, scaler_per_layer, X_inv, invented_word_per_stim
    )
    per_layer_unary_mass: list[float] = []
    per_layer_binary_repl_unary_mass: list[float] = []
    for layer in range(n_layers):
        unary = compute_unary_mass(per_layer_pred[layer])
        binary_repl = compute_binary_replacement_unary_mass(per_layer_pred[layer])
        per_layer_unary_mass.append(unary)
        per_layer_binary_repl_unary_mass.append(binary_repl)
        print(
            f"  {layer:>6d}  {per_layer_cv[layer]:>7.3f}  "
            f"{unary:>10.1f}%  {binary_repl:>16.1f}%"
        )

    # Locate three diagnostic layers.
    peak_cv_layer = int(np.argmax(per_layer_cv))
    peak_unary_layer = int(np.argmax(per_layer_unary_mass))
    reference_layer = min(FIXED_REFERENCE_LAYER, n_layers - 1)

    print(f"\nDiagnostic layers identified:")
    print(f"  Peak CV accuracy:        layer {peak_cv_layer:>3d} (cv={per_layer_cv[peak_cv_layer]:.3f}, unary={per_layer_unary_mass[peak_cv_layer]:.1f}%)")
    print(f"  Peak unary mass:         layer {peak_unary_layer:>3d} (cv={per_layer_cv[peak_unary_layer]:.3f}, unary={per_layer_unary_mass[peak_unary_layer]:.1f}%)")
    print(f"  Fixed reference (~20%):  layer {reference_layer:>3d} (cv={per_layer_cv[reference_layer]:.3f}, unary={per_layer_unary_mass[reference_layer]:.1f}%)")

    # Detailed reporting at each diagnostic layer.
    diagnostic_layers = sorted(set([peak_cv_layer, peak_unary_layer, reference_layer]))
    for layer in diagnostic_layers:
        labels = []
        if layer == peak_cv_layer:
            labels.append("peak-CV")
        if layer == peak_unary_layer:
            labels.append("peak-unary")
        if layer == reference_layer:
            labels.append("fixed-reference")
        label_str = ", ".join(labels)

        print_invented_distribution(
            INVENTED_WORDS, CANONICALS, per_layer_pred[layer],
            f"Invented predictions at layer {layer} ({label_str})"
        )

        # Arity-aware breakdown for diagnostic layers.
        print(f"\n  Arity-aware breakdown at layer {layer}:")
        binary_replacements = [w for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 2]
        unary_replacements = [w for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 1]

        for tag, ws in [("Binary-replacement", binary_replacements), ("Unary-replacement", unary_replacements)]:
            print(f"    {tag} words ({', '.join(ws)}):")
            for w in ws:
                c = per_layer_pred[layer][w]
                total = sum(c.values())
                unary_pct = 100 * (c.get("not", 0) + c.get("necessarily", 0)) / total
                binary_pct = 100 * (
                    c.get("and", 0) + c.get("or", 0) + c.get("implies", 0)
                ) / total
                print(f"      {w:<8s}: unary {unary_pct:5.1f}%, binary {binary_pct:5.1f}%")

    return {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "per_layer_cv": per_layer_cv,
        "per_layer_unary_mass": per_layer_unary_mass,
        "per_layer_binary_repl_unary_mass": per_layer_binary_repl_unary_mass,
        "peak_cv_layer": peak_cv_layer,
        "peak_unary_layer": peak_unary_layer,
        "reference_layer": reference_layer,
        "per_layer_pred": per_layer_pred,
    }


def main() -> None:
    device = (
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Model:  {MODEL_ID}")
    print(f"PyTorch: {torch.__version__}")

    # Gemma 2 is gated. Surface a helpful error if the user has not yet
    # accepted the license or logged in.
    print(f"\nLoading {MODEL_ID} (bfloat16; ~18 GB download on first run)...")
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
    except OSError as e:
        msg = str(e)
        if "gated" in msg.lower() or "authentication" in msg.lower() or "401" in msg:
            print()
            print("ERROR: Gemma 2 is a gated model. Required setup:")
            print("  1. Visit https://huggingface.co/google/gemma-2-9b and accept the license")
            print("  2. Run: huggingface-cli login")
            print("  3. Re-run this script")
            print(f"\nOriginal error: {e}")
            sys.exit(1)
        raise
    print(f"  load time: {time.time() - t0:.1f}s")

    # Quick sanity probe: a single forward pass to confirm no MPS fallbacks and
    # that bf16 outputs are finite (Gemma 2's soft-cap can NaN in fp16, hence
    # bf16; verify here before committing to extraction).
    with torch.no_grad():
        sample = tok("The function and(p, q) returns a boolean.", return_tensors="pt").to(device)
        out = model(**sample, output_hidden_states=True)
        if device == "mps":
            torch.mps.synchronize()
        last_h = out.hidden_states[-1]
        if not torch.isfinite(last_h).all():
            print("\nWARNING: non-finite values detected in hidden states. "
                  "Gemma 2 9B is sensitive to fp16 overflow via its soft-cap mechanism. "
                  "Verify that torch_dtype=torch.bfloat16 is set.")
            sys.exit(1)
        print(f"  sanity forward pass: {len(out.hidden_states)} hidden states, "
              f"hidden_dim={last_h.shape[-1]}, all-finite=True")

    print("\nValidating tokenization (with leading space, matching extraction):")
    print(f"  {'word':<14s}  {'n_subwords':>10s}  pieces")
    for c in CANONICALS:
        pieces = subwords(tok, c)
        print(f"  canonical {c:<6s}  {len(pieces):>10d}  {pieces}")
    for w in INVENTED_WORDS:
        pieces = subwords(tok, w)
        print(f"  invented  {w:<6s}  {len(pieces):>10d}  {pieces}")

    # =========================================================================
    # Condition 1: NEUTRAL metalinguistic templates (replicating script 15)
    # =========================================================================
    def cond1_canonical(op, rng, n):
        return make_neutral_stimuli(op, rng, n)

    def cond1_invented(w, rng, n):
        return make_neutral_stimuli(w, rng, n)

    cond1 = run_condition(
        name="1 (NEUTRAL — replicating script 15)",
        canonical_stimuli_fn=cond1_canonical,
        invented_stimuli_fn=cond1_invented,
        model=model, tok=tok, device=device,
    )

    # =========================================================================
    # Condition 2: FUNCTIONAL-PREFIX notation (replicating script 16 Cond 2)
    # =========================================================================
    def cond2_canonical(op, rng, n):
        return make_functional_stimuli(op, CANONICAL_ARITY[op], rng, n)

    def cond2_invented(w, rng, n):
        return make_functional_stimuli(w, CANONICAL_ARITY[W_TO_CANONICAL[w]], rng, n)

    cond2 = run_condition(
        name="2 (FUNCTIONAL-PREFIX — replicating script 16 Condition 2)",
        canonical_stimuli_fn=cond2_canonical,
        invented_stimuli_fn=cond2_invented,
        model=model, tok=tok, device=device,
    )

    # =========================================================================
    # Cross-model side-by-side summary
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("CROSS-MODEL SUMMARY: Gemma 2 9B vs OLMo 2 7B")
    print("=" * 80)
    print()
    print(f"  Gemma 2 9B layer count: {cond1['n_layers']} (vs OLMo 2 7B: 33 incl. embedding)")
    print(f"  Gemma 2 9B hidden dim:  {cond1['hidden_dim']} (vs OLMo 2 7B: 4096)")

    def summary_row(label, cond, key_g, ref_key, ref_dict):
        diag = cond["reference_layer"]
        per_word = cond["per_layer_pred"][diag]
        if key_g == "cv":
            g_val = cond["per_layer_cv"][diag]
            r_val = ref_dict[ref_key]
            print(f"  {label:<55s} {g_val:>9.3f}  {r_val:>9.3f}")
        elif key_g == "unary":
            g_val = compute_unary_mass(per_word)
            r_val = ref_dict[ref_key]
            print(f"  {label:<55s} {g_val:>8.1f}%  {r_val:>8.1f}%")
        elif key_g == "binary_repl_unary":
            g_val = compute_binary_replacement_unary_mass(per_word)
            r_val = ref_dict.get(ref_key)
            if r_val is None:
                print(f"  {label:<55s} {g_val:>8.1f}%  {'-':>8s}")
            else:
                print(f"  {label:<55s} {g_val:>8.1f}%  {r_val:>8.1f}%")

    for cond_label, cond, ref in [
        ("Condition 1 (NEUTRAL)", cond1, OLMO2_7B_REFERENCE["neutral"]),
        ("Condition 2 (FUNCTIONAL-PREFIX)", cond2, OLMO2_7B_REFERENCE["functional_prefix"]),
    ]:
        print()
        print(f"  {cond_label} — at fixed reference layer {cond['reference_layer']} (Gemma 2) "
              f"vs layer {ref['diagnostic_layer']} (OLMo 2 7B)")
        print(f"  {'metric':<55s} {'Gemma 2':>9s}  {'OLMo 2':>9s}")
        print("  " + "-" * 74)
        summary_row("Probe CV accuracy on canonicals", cond, "cv", "cv_accuracy", ref)
        summary_row("Mean unary-region mass on invented operators", cond, "unary", "mean_unary_mass", ref)
        if "binary_replacement_unary_mass" in ref:
            summary_row(
                "  ...for binary-replacement words only",
                cond, "binary_repl_unary", "binary_replacement_unary_mass", ref,
            )

    print("\n" + "=" * 80)
    print("VERDICT GUIDE")
    print("=" * 80)
    print()
    print("  CROSS-MODEL REPLICATION CONFIRMED if:")
    print("    - Probe CV accuracy on Gemma 2 canonicals is high (>= 0.9) in")
    print("      both conditions at the peak-CV layer.")
    print("    - Mean unary-region mass in Condition 2 (functional-prefix) is")
    print("      >= 80% on Gemma 2 (vs OLMo 2 7B's 100%).")
    print("    - The peak-CV layer and the peak-unary layer are similar (both")
    print("      indicating the same depth at which arity is encoded).")
    print()
    print("  CROSS-MODEL REPLICATION PARTIAL if:")
    print("    - Probe CV accuracy is high but Condition 2 unary mass is 50-80%,")
    print("      suggesting the attractor exists but is weaker / less rigid in")
    print("      Gemma 2.")
    print("    - Binary-replacement words show a non-trivial fraction going to")
    print("      arity-matched binary canonicals (and/or/implies), indicating")
    print("      Gemma 2 has some arity-respecting flexibility OLMo 2 lacks.")
    print()
    print("  REPLICATION FAILS if:")
    print("    - Mean unary mass in Condition 2 drops below 50% on Gemma 2.")
    print("    - The headline finding must be scoped to OLMo-family models and")
    print("      we need to investigate the OLMo-specific cause.")
    print()
    print("  Per-layer unary-mass trajectory (compare to OLMo 2 7B's monotonic")
    print("  rise from ~20% at layer 0 to plateau at layer 5+):")
    print()
    print("  Condition 2 per-layer unary-mass curve:")
    layers = list(range(cond2["n_layers"]))
    # Print as a compact ASCII chart with every layer.
    max_label_len = 6
    for layer in layers:
        unary = cond2["per_layer_unary_mass"][layer]
        bar_len = int(round(unary / 2))  # 50 columns for 100%
        bar = "#" * bar_len
        print(f"  layer {layer:>3d}: {unary:>5.1f}%  {bar}")


if __name__ == "__main__":
    main()
