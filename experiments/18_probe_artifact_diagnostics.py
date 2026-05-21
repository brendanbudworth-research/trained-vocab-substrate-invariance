"""Probe-artifact diagnostics for Gemma 2 9B vs OLMo 2 7B (script 17 follow-up).

Script 17 produced a clean replication of the arity-region attractor on Gemma 2 9B
in Condition 1 (NEUTRAL metalinguistic templates) but a problematic, non-monotonic
result in Condition 2 (FUNCTIONAL-PREFIX notation). Three observations made the
Condition 2 result hard to interpret:

  (1) Probe CV accuracy on canonicals was 1.000 at *every* layer from 1 to 42 in
      Gemma 2 Condition 2 — consistent with the probe reading the previous-token
      identity propagated through the residual stream rather than a structural
      representation.

  (2) The per-layer unary-mass curve for invented operators oscillated between
      0% and 100% with two distinct peaks (layers 2 and 16-17) and intervening
      zeros — qualitatively different from OLMo 2's monotonic rise to plateau.

  (3) Per-word landings at the fixed-reference layer 8 split cleanly by *last
      subword identity* (q/' dren' -> not; usp/lex/ph -> implies), not by the
      word's role-in-test arity. Strong surface-form signature in what should
      have been a structural measurement.

These three patterns are consistent with the probe at most layers being a
surface-form classifier rather than an arity-structure classifier. The
diagnostics in this script disambiguate that interpretation.

Four diagnostics, all run on both Gemma 2 9B and OLMo 2 7B for direct cross-
model comparison:

  Diagnostic A — Cross-condition probe transfer
    Train a probe on canonicals from one condition, evaluate it on inventeds
    from the *other* condition. If the arity-region attractor is a robust
    property of the operator-anchored representation, the NEUTRAL-trained probe
    should still place FUNCTIONAL-PREFIX invented words in the unary region
    (and vice versa). If transfer fails, the probe is reading template-specific
    surface features.

  Diagnostic B — Held-out canonical probe
    For each canonical X in {and, or, not, implies, necessarily}, train a probe
    on the other four and evaluate on X's stimuli. If the probe is structural,
    X should map to its arity-class partner(s) (e.g., `or` -> {and, implies}
    when held out). If the probe is overfitted to per-canonical surface tokens,
    held-out canonicals will be predicted at chance over the four trained
    classes.

  Diagnostic C — Direct geometric inspection (probe-free)
    For each diagnostic layer, compute the mean residual per canonical and per
    invented word. Report:
      - raw cosine similarity (invented x canonical) — does the layer-2 100%
        unary-mass peak correspond to a real geometric attractor?
      - cosine similarity to the unary-region centroid vs the binary-region
        centroid — direct geometric test of arity-region membership without a
        probe between the measurement and the geometry
      - the per-canonical and centroid separation magnitudes for context

  Diagnostic D — Last-subword baseline
    Test the layer-8 per-word split hypothesis explicitly. For each invented
    word, get the last-subword's embedding-layer vector and compute cosine
    similarity to each canonical's embedding-layer vector. If the layer-8
    probe predictions correlate with last-subword embedding similarity, the
    layer-8 result is a propagated-token-identity artifact.

Memory plan: load Gemma 2 9B, extract all condition x layer activations,
delete the model and free MPS memory, then load OLMo 2 7B and repeat. All
diagnostics consume the cached activation arrays; no further forward passes
required.

Run time on M4: estimated 6-10 minutes once both models are cached.
"""

from __future__ import annotations

import datetime as _dt
import gc
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Tee logging — every run also writes its full stdout to outputs/18_<ts>.log
# so long outputs don't get lost from the terminal scrollback buffer.
# ==============================================================================
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
            s.flush()
        return n

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _setup_logging() -> str | None:
    """Tee stdout (and stderr) to a timestamped log file under outputs/.
    Returns the log file path so we can report it at the end of the run.
    Set NO_LOG=1 to disable.
    """
    if os.environ.get("NO_LOG"):
        return None
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"18_{ts}.log")
    log_f = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    print(f"[logging] (set NO_LOG=1 to disable)")
    return log_path

# ==============================================================================
# Constants — identical to scripts 15/16/17 for direct comparability
# ==============================================================================
SEED = 17
N_PER_CLASS = 50

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
UNARY_CANONICALS = ["not", "necessarily"]
BINARY_CANONICALS = ["and", "or", "implies"]
CANONICAL_ARITY = {"and": 2, "or": 2, "implies": 2, "not": 1, "necessarily": 1}

INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL = dict(zip(INVENTED_WORDS, CANONICALS))


@dataclass
class ModelSpec:
    short_name: str
    model_id: str
    dtype: "torch.dtype"
    diagnostic_layers: list[int]  # condition-2 features of interest
    cond1_focus_layer: int        # peak-CV/peak-unary layer for cond 1
    cond2_focus_layers: list[int] # layers to detail for cond 2


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        short_name="Gemma 2 9B",
        model_id="google/gemma-2-9b",
        dtype=torch.bfloat16,
        diagnostic_layers=[2, 4, 8, 16, 17],
        cond1_focus_layer=4,
        cond2_focus_layers=[2, 8, 16],
    ),
    ModelSpec(
        short_name="OLMo 2 7B",
        model_id="allenai/OLMo-2-1124-7B",
        dtype=torch.float16,
        diagnostic_layers=[4, 7, 10, 16, 24],
        cond1_focus_layer=7,
        cond2_focus_layers=[7, 10, 16],
    ),
]


# ==============================================================================
# Stimulus generation — copied verbatim from script 17
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


def subwords(tok, word: str) -> list[str]:
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return [tok.decode([i]) for i in ids]


def find_operator_anchor(tok, prompt: str, operators: list[str]) -> int | None:
    """Find the position immediately after the operator's last subword.

    Replicates the helper used in scripts 15/16/17 verbatim. `joined`
    accumulates decoded subwords across the prompt; multi-subword operators
    (e.g., ' bli' + 'q' for bliq) match once the suffix equals " <op>".
    """
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
    return best_pos


def extract_anchored_activations(
    model, tok, prompts: list[str], operators: list[str], device: str
) -> list[np.ndarray]:
    """Return a list of (n_prompts, hidden_dim) arrays, one per hidden-state layer."""
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


# ==============================================================================
# Probe helpers
# ==============================================================================
def make_classifier() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")


def fit_probe(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = make_classifier()
    clf.fit(X_scaled, y)
    return clf, scaler


def predict(clf, scaler, X: np.ndarray) -> np.ndarray:
    return clf.predict(scaler.transform(X))


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


def unary_mass_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 100 * (counts.get("not", 0) + counts.get("necessarily", 0)) / total


def binary_mass_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 100 * (
        counts.get("and", 0) + counts.get("or", 0) + counts.get("implies", 0)
    ) / total


# ==============================================================================
# Activation cache build
# ==============================================================================
@dataclass
class ConditionActivations:
    canonical_X: list[np.ndarray]            # per-layer (n_canonical_stim, dim)
    canonical_labels: np.ndarray              # (n_canonical_stim,)
    invented_X: list[np.ndarray]              # per-layer (n_invented_stim, dim)
    invented_word_per_stim: list[str]         # (n_invented_stim,)


def build_condition(
    name: str,
    model, tok, device: str,
    canonical_stim_fn, invented_stim_fn,
) -> ConditionActivations:
    print(f"\n  Building condition: {name}")

    canon_prompts: list[str] = []
    canon_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(SEED + hash((name, op)) % 100000)
        canon_prompts.extend(canonical_stim_fn(op, op_rng, N_PER_CLASS))
        canon_labels.extend([op] * N_PER_CLASS)

    inv_prompts: list[str] = []
    inv_words: list[str] = []
    for w in INVENTED_WORDS:
        w_rng = random.Random(SEED + hash((name, w, "invent")) % 100000)
        stim = invented_stim_fn(w, w_rng, N_PER_CLASS)
        inv_prompts.extend(stim)
        inv_words.extend([w] * len(stim))

    print(f"    canonicals: {len(canon_prompts)} stimuli ({len(CANONICALS)} classes)")
    print(f"    invented:   {len(inv_prompts)} stimuli ({len(INVENTED_WORDS)} words)")

    print("    extracting canonical activations...")
    t0 = time.time()
    X_canon = extract_anchored_activations(model, tok, canon_prompts, CANONICALS, device)
    print(f"      {time.time() - t0:.1f}s, n_layers={len(X_canon)}, dim={X_canon[0].shape[1]}")

    print("    extracting invented activations...")
    t0 = time.time()
    X_inv = extract_anchored_activations(model, tok, inv_prompts, INVENTED_WORDS, device)
    print(f"      {time.time() - t0:.1f}s")

    return ConditionActivations(
        canonical_X=X_canon,
        canonical_labels=np.array(canon_labels),
        invented_X=X_inv,
        invented_word_per_stim=inv_words,
    )


# ==============================================================================
# DIAGNOSTIC A — Cross-condition probe transfer
# ==============================================================================
def diagnostic_a(
    model_name: str,
    cond1: ConditionActivations,
    cond2: ConditionActivations,
    cond1_layer: int,
    cond2_layers: list[int],
) -> dict:
    """For each pairing of (source-condition, target-condition, layer):
    train probe on source canonicals, predict on target invented.
    Report per-word unary mass."""

    print(f"\n{'=' * 80}")
    print(f"Diagnostic A: cross-condition probe transfer ({model_name})")
    print(f"{'=' * 80}")
    print()
    print("  Reading: a row of 'high unary mass' under the cross-condition column")
    print("  means the probe direction trained on one notation places invented")
    print("  operators in the unary region under a different notation - i.e.,")
    print("  the attractor is a robust property of the residual stream and not")
    print("  template-specific. A drop to near-zero under cross-condition")
    print("  evaluation means the probe is template-bound.")
    print()

    results: dict = {}

    pairings: list = []
    seen_keys: set = set()

    def add_pair(src_name, src, src_layer, tgt_name, tgt, tgt_layer):
        key = (src_name, src_layer, tgt_name, tgt_layer)
        if key in seen_keys:
            return
        seen_keys.add(key)
        pairings.append((src_name, src, src_layer, tgt_name, tgt, tgt_layer))

    add_pair("NEUTRAL", cond1, cond1_layer, "FUNC-PFX", cond2, cond1_layer)
    add_pair("NEUTRAL", cond1, cond1_layer, "FUNC-PFX", cond2, cond2_layers[0])
    for fl in cond2_layers:
        add_pair("FUNC-PFX", cond2, fl, "NEUTRAL", cond1, fl)
        add_pair("FUNC-PFX", cond2, fl, "NEUTRAL", cond1, cond1_layer)

    for src_name, src, src_layer, tgt_name, tgt, tgt_layer in pairings:
        if src_layer >= len(src.canonical_X) or tgt_layer >= len(tgt.invented_X):
            continue

        clf, scaler = fit_probe(src.canonical_X[src_layer], src.canonical_labels)
        cv = cv_accuracy(src.canonical_X[src_layer], src.canonical_labels, seed=SEED)

        # Within-source sanity: predict source's own invented words.
        preds_within = predict(clf, scaler, src.invented_X[src_layer])
        within_counts_per_word: dict[str, Counter] = {w: Counter() for w in INVENTED_WORDS}
        for w, p in zip(src.invented_word_per_stim, preds_within.tolist()):
            within_counts_per_word[w][p] += 1
        within_unary = float(np.mean([
            unary_mass_from_counts(within_counts_per_word[w]) for w in INVENTED_WORDS
        ]))

        # Cross-condition: predict target's invented words.
        preds_cross = predict(clf, scaler, tgt.invented_X[tgt_layer])
        cross_counts_per_word: dict[str, Counter] = {w: Counter() for w in INVENTED_WORDS}
        for w, p in zip(tgt.invented_word_per_stim, preds_cross.tolist()):
            cross_counts_per_word[w][p] += 1
        cross_unary = float(np.mean([
            unary_mass_from_counts(cross_counts_per_word[w]) for w in INVENTED_WORDS
        ]))

        # Cross-condition canonical generalisation: predict target's canonical
        # operators with source-trained probe (CV-style accuracy).
        preds_canon = predict(clf, scaler, tgt.canonical_X[tgt_layer])
        canon_acc = float(np.mean(preds_canon == tgt.canonical_labels))

        key = (src_name, src_layer, tgt_name, tgt_layer)
        results[key] = {
            "src_cv": cv,
            "within_unary": within_unary,
            "cross_canon_acc": canon_acc,
            "cross_unary": cross_unary,
            "within_counts": within_counts_per_word,
            "cross_counts": cross_counts_per_word,
        }

        print(
            f"  TRAIN: {src_name}@L{src_layer:<2d}  (CV={cv:.3f})  ->  "
            f"TEST: {tgt_name}@L{tgt_layer:<2d}"
        )
        print(
            f"    within-src invented unary mass: {within_unary:5.1f}%   |   "
            f"cross-tgt canonical acc: {canon_acc:.3f}   |   "
            f"cross-tgt invented unary mass: {cross_unary:5.1f}%"
        )
        # Per-word cross-condition breakdown.
        print(f"    cross-target per-word predictions:")
        print(
            f"      {'word':<8s} | "
            + " | ".join(f"{c:>13s}" for c in CANONICALS)
            + f" | {'unary %':>9s}"
        )
        for w in INVENTED_WORDS:
            c = cross_counts_per_word[w]
            total = sum(c.values())
            row = f"      {w:<8s} | " + " | ".join(
                f"{c.get(cl, 0):>5d} ({100*c.get(cl, 0)/total:>5.1f}%)"
                for cl in CANONICALS
            )
            row += f" | {unary_mass_from_counts(c):>8.1f}%"
            print(row)
        print()

    return results


# ==============================================================================
# DIAGNOSTIC B — Held-out canonical probe
# ==============================================================================
def diagnostic_b(
    model_name: str,
    cond: ConditionActivations,
    cond_label: str,
    layers: list[int],
) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Diagnostic B: held-out canonical probe ({model_name} / {cond_label})")
    print(f"{'=' * 80}")
    print()
    print("  Reading: when canonical X is held out at training, on which of the")
    print("  other four canonical classes does the probe place X's stimuli?")
    print("  STRUCTURAL: held-out class maps to its arity-class partner(s).")
    print("    or held out -> {and, implies}; not held out -> {necessarily}.")
    print("  SURFACE-OVERFIT: held-out class maps roughly uniformly over the")
    print("    four trained classes (probe has no arity prior to fall back on).")
    print()

    results: dict[int, dict] = {}

    for layer in layers:
        if layer >= len(cond.canonical_X):
            continue

        print(f"  Layer {layer}:")
        print(
            f"    {'held-out':<13s} | "
            + " | ".join(f"{c:>13s}" for c in CANONICALS)
            + f" | {'arity-cls':>9s}"
        )

        layer_result = {}
        for held_out in CANONICALS:
            train_mask = cond.canonical_labels != held_out
            test_mask = cond.canonical_labels == held_out

            X_train = cond.canonical_X[layer][train_mask]
            y_train = cond.canonical_labels[train_mask]
            X_test = cond.canonical_X[layer][test_mask]

            clf, scaler = fit_probe(X_train, y_train)
            preds = predict(clf, scaler, X_test)
            counts = Counter(preds.tolist())
            # arity-class fraction: when held-out is binary, fraction that go
            # to {and, or, implies} \ {held-out}; when held-out is unary,
            # fraction that go to {not, necessarily} \ {held-out}.
            arity_class = (
                BINARY_CANONICALS if CANONICAL_ARITY[held_out] == 2 else UNARY_CANONICALS
            )
            arity_class_in_train = [c for c in arity_class if c != held_out]
            arity_count = sum(counts.get(c, 0) for c in arity_class_in_train)
            total = sum(counts.values())
            arity_pct = 100 * arity_count / total if total else 0.0

            row = f"    {held_out:<13s} | " + " | ".join(
                f"{counts.get(c, 0):>5d} ({100*counts.get(c, 0)/total:>5.1f}%)"
                if c != held_out else f"{'(held)':>13s}"
                for c in CANONICALS
            )
            row += f" | {arity_pct:>8.1f}%"
            print(row)
            layer_result[held_out] = {"counts": counts, "arity_class_pct": arity_pct}

        results[layer] = layer_result
        print()

    # Summary across layers.
    print("  Summary (mean arity-class % across held-out canonicals per layer):")
    print(
        f"    {'layer':>6s}  {'binary held-out':>17s}  {'unary held-out':>16s}  {'all':>8s}"
    )
    for layer in layers:
        if layer not in results:
            continue
        bin_pcts = [results[layer][c]["arity_class_pct"] for c in BINARY_CANONICALS]
        un_pcts = [results[layer][c]["arity_class_pct"] for c in UNARY_CANONICALS]
        all_pcts = bin_pcts + un_pcts
        print(
            f"    {layer:>6d}  {np.mean(bin_pcts):>16.1f}%  "
            f"{np.mean(un_pcts):>15.1f}%  {np.mean(all_pcts):>7.1f}%"
        )
    print()
    print("  Chance baseline (4-class arity-class %):")
    print("    binary held-out: 2/4 = 50.0% (two remaining binaries of four trained)")
    print("    unary held-out:  1/4 = 25.0% (one remaining unary of four trained)")

    return results


# ==============================================================================
# DIAGNOSTIC C — Direct geometric inspection
# ==============================================================================
def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def diagnostic_c(
    model_name: str,
    cond: ConditionActivations,
    cond_label: str,
    layers: list[int],
) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Diagnostic C: direct geometric inspection ({model_name} / {cond_label})")
    print(f"{'=' * 80}")
    print()
    print("  Reading: bypass the probe and look at the raw geometry.")
    print("  - canonical-pair separation: mean cosine sim between canonicals")
    print("    (lower = more distinct; 1.0 = identical)")
    print("  - per-word cos sim to unary vs binary centroids: positive delta")
    print("    means the word's mean residual sits closer to the unary half-space.")
    print()

    results: dict[int, dict] = {}

    for layer in layers:
        if layer >= len(cond.canonical_X):
            continue

        # Build canonical centroids.
        canon_centroid: dict[str, np.ndarray] = {}
        for c in CANONICALS:
            mask = cond.canonical_labels == c
            canon_centroid[c] = cond.canonical_X[layer][mask].mean(axis=0)

        unary_centroid = np.mean(
            [canon_centroid[c] for c in UNARY_CANONICALS], axis=0
        )
        binary_centroid = np.mean(
            [canon_centroid[c] for c in BINARY_CANONICALS], axis=0
        )

        # Build invented-word centroids.
        inv_centroid: dict[str, np.ndarray] = {}
        for w in INVENTED_WORDS:
            mask = np.array([word == w for word in cond.invented_word_per_stim])
            inv_centroid[w] = cond.invented_X[layer][mask].mean(axis=0)

        # Canonical-pair separation (5x5 symmetric matrix, off-diagonals).
        print(f"  Layer {layer}: canonical-canonical cosine similarity")
        print(f"    {'':<14s}" + "".join(f"{c:>14s}" for c in CANONICALS))
        canon_pair_sims: list[float] = []
        for c1 in CANONICALS:
            row = f"    {c1:<14s}"
            for c2 in CANONICALS:
                s = cosine_np(canon_centroid[c1], canon_centroid[c2])
                if c1 != c2:
                    canon_pair_sims.append(s)
                row += f"{s:>14.4f}"
            print(row)
        mean_canon_pair = float(np.mean(canon_pair_sims))
        print(f"    mean off-diagonal canonical-canonical sim: {mean_canon_pair:.4f}")
        print(f"    (lower = more distinct canonicals; 1.0 = collapsed to a point)")

        # Invented x canonical sim.
        print()
        print(f"  Layer {layer}: invented x canonical cosine similarity")
        print(f"    {'':<10s}" + "".join(f"{c:>14s}" for c in CANONICALS) + f"   {'top can':>10s}")
        inv_top_canon: dict[str, str] = {}
        for w in INVENTED_WORDS:
            row = f"    {w:<10s}"
            sims = {c: cosine_np(inv_centroid[w], canon_centroid[c]) for c in CANONICALS}
            for c in CANONICALS:
                row += f"{sims[c]:>14.4f}"
            top = max(sims.items(), key=lambda kv: kv[1])
            inv_top_canon[w] = top[0]
            row += f"   {top[0]:>10s}"
            print(row)

        # Invented to unary-centroid vs binary-centroid.
        print()
        print(f"  Layer {layer}: invented x region-centroid cosine similarity")
        print(f"    {'word':<10s}  {'sim(unary)':>11s}  {'sim(binary)':>12s}  {'delta':>10s}  {'closer to':>11s}")
        region_results: dict[str, dict] = {}
        deltas: list[float] = []
        for w in INVENTED_WORDS:
            s_un = cosine_np(inv_centroid[w], unary_centroid)
            s_bn = cosine_np(inv_centroid[w], binary_centroid)
            delta = s_un - s_bn
            deltas.append(delta)
            closer = "UNARY" if delta > 0 else "BINARY"
            print(
                f"    {w:<10s}  {s_un:>+11.4f}  {s_bn:>+12.4f}  "
                f"{delta:>+10.4f}  {closer:>11s}"
            )
            region_results[w] = {
                "sim_unary": s_un, "sim_binary": s_bn, "delta": delta, "closer": closer,
                "top_canon": inv_top_canon[w],
            }

        n_to_unary = sum(1 for d in deltas if d > 0)
        print(f"\n    {n_to_unary}/{len(INVENTED_WORDS)} invented words closer to unary centroid")
        print(f"    mean delta (unary - binary): {np.mean(deltas):+.4f}")

        results[layer] = {
            "canon_pair_mean_sim": mean_canon_pair,
            "region_per_word": region_results,
            "n_to_unary": n_to_unary,
            "mean_delta_unary_minus_binary": float(np.mean(deltas)),
        }
        print()

    return results


# ==============================================================================
# DIAGNOSTIC D — Last-subword baseline
# ==============================================================================
def diagnostic_d(
    model_name: str,
    model, tok,
    device: str,
    inv_words_landings_per_layer_cond2: dict[int, dict[str, dict[str, int]]] | None,
) -> dict:
    """Test whether per-layer Cond-2 invented-word predictions track last-subword
    embedding closeness to canonicals (the "propagated token identity" hypothesis)."""

    print(f"\n{'=' * 80}")
    print(f"Diagnostic D: last-subword embedding baseline ({model_name})")
    print(f"{'=' * 80}")
    print()
    print("  For each invented word, take its last subword's layer-0 embedding")
    print("  and compute cosine similarity to each canonical's embedding.")
    print("  If a Condition-2 layer's invented-word landings track last-subword")
    print("  embedding similarity (i.e., the closest canonical at layer 0 ==")
    print("  the predicted canonical at that layer), the layer's result is a")
    print("  propagated-token-identity artifact rather than a structural finding.")
    print()

    embed = model.get_input_embeddings()
    results: dict[str, dict] = {}

    canon_emb: dict[str, np.ndarray] = {}
    for c in CANONICALS:
        ids = tok(" " + c, add_special_tokens=False, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            vecs = embed(ids.to(device)).float().detach().cpu().numpy()
        canon_emb[c] = vecs.mean(axis=0)

    print(
        f"  {'word':<10s}  {'last_subword':<14s}  "
        + "".join(f"{c:>14s}" for c in CANONICALS)
        + f"  {'top_can':>10s}"
    )
    for w in INVENTED_WORDS:
        ids = tok(" " + w, add_special_tokens=False, return_tensors="pt").input_ids[0]
        pieces = [tok.decode([i]) for i in ids.tolist()]
        with torch.no_grad():
            vecs = embed(ids.to(device)).float().detach().cpu().numpy()
        last_sub_emb = vecs[-1]
        sims = {c: cosine_np(last_sub_emb, canon_emb[c]) for c in CANONICALS}
        top = max(sims.items(), key=lambda kv: kv[1])
        results[w] = {
            "last_subword": pieces[-1],
            "sims": sims,
            "top": top[0],
        }
        row = (
            f"  {w:<10s}  {pieces[-1]:<14s}  "
            + "".join(f"{sims[c]:>14.4f}" for c in CANONICALS)
            + f"  {top[0]:>10s}"
        )
        print(row)

    if inv_words_landings_per_layer_cond2:
        for layer, landings in sorted(inv_words_landings_per_layer_cond2.items()):
            print()
            print(f"  Comparison: last-subword top canonical vs Cond-2 layer-{layer} top landing")
            print(
                f"    {'word':<10s}  {'last-sub closest':<20s}  "
                f"{f'L{layer} top landing':<18s}  {'match?':<8s}"
            )
            n_match = 0
            n_eval = 0
            for w in INVENTED_WORDS:
                w_landings = landings.get(w, {})
                if not w_landings:
                    continue
                n_eval += 1
                l_top = max(w_landings.items(), key=lambda kv: kv[1])[0]
                embed_top = results[w]["top"]
                match = embed_top == l_top
                n_match += int(match)
                print(
                    f"    {w:<10s}  {embed_top:<20s}  {l_top:<18s}  "
                    f"{'YES' if match else 'no':<8s}"
                )
            print(f"\n    {n_match}/{n_eval} match")
            if n_match == n_eval and n_eval == len(INVENTED_WORDS):
                print(f"    HIGH match => layer-{layer} Cond-2 dominated by last-subword identity (artifact)")
            elif n_match >= n_eval * 0.6:
                print(f"    MODERATE match => layer-{layer} Cond-2 result partially driven by last-subword identity")
            else:
                print(f"    LOW match => layer-{layer} Cond-2 reflects something built by layers 1-{layer}")

    return results


# ==============================================================================
# Per-model orchestration
# ==============================================================================
def run_model(spec: ModelSpec, device: str) -> dict:
    print()
    print("#" * 80)
    print(f"# Model: {spec.short_name} ({spec.model_id})")
    print("#" * 80)

    print(f"\nLoading {spec.model_id} ({spec.dtype})...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=spec.dtype,
            attn_implementation="eager",
        ).to(device).eval()
    except OSError as e:
        msg = str(e).lower()
        if "gated" in msg or "401" in msg or "authentication" in msg:
            print(f"\nERROR loading {spec.model_id}: this looks like a gated-model auth failure.")
            print("Steps:")
            print(f"  1. Accept the license at https://huggingface.co/{spec.model_id}")
            print(f"  2. Run: huggingface-cli login")
            sys.exit(1)
        raise
    print(f"  load time: {time.time() - t0:.1f}s")

    print("\nSubword tokenisation (with leading space):")
    print(f"  {'word':<14s}  {'n_subwords':>10s}  pieces")
    for c in CANONICALS:
        pieces = subwords(tok, c)
        print(f"  canonical {c:<6s}  {len(pieces):>10d}  {pieces}")
    for w in INVENTED_WORDS:
        pieces = subwords(tok, w)
        print(f"  invented  {w:<6s}  {len(pieces):>10d}  {pieces}")

    print("\nBuilding activation cache (NEUTRAL + FUNCTIONAL-PREFIX)...")

    cond1 = build_condition(
        name=f"{spec.short_name}/NEUTRAL",
        model=model, tok=tok, device=device,
        canonical_stim_fn=lambda op, rng, n: make_neutral_stimuli(op, rng, n),
        invented_stim_fn=lambda w, rng, n: make_neutral_stimuli(w, rng, n),
    )

    cond2 = build_condition(
        name=f"{spec.short_name}/FUNC-PFX",
        model=model, tok=tok, device=device,
        canonical_stim_fn=lambda op, rng, n: make_functional_stimuli(
            op, CANONICAL_ARITY[op], rng, n
        ),
        invented_stim_fn=lambda w, rng, n: make_functional_stimuli(
            w, CANONICAL_ARITY[W_TO_CANONICAL[w]], rng, n
        ),
    )

    # ---- Compute Cond-2 within-condition probe landings at all focus layers ----
    # Diagnostic D compares last-subword embedding closeness against the actual
    # within-Cond-2 probe landings at each focus layer to test the
    # "propagated last-subword identity" hypothesis. We compute the landings
    # before model deletion so Diagnostic D can use both.
    landings_per_layer_cond2: dict[int, dict[str, dict[str, int]]] = {}
    for layer in spec.cond2_focus_layers:
        if layer >= len(cond2.canonical_X):
            continue
        clf_l, scaler_l = fit_probe(
            cond2.canonical_X[layer], cond2.canonical_labels
        )
        preds_l = predict(clf_l, scaler_l, cond2.invented_X[layer])
        landings_counter: dict[str, Counter] = {w: Counter() for w in INVENTED_WORDS}
        for w, p in zip(cond2.invented_word_per_stim, preds_l.tolist()):
            landings_counter[w][p] += 1
        landings_per_layer_cond2[layer] = {
            w: dict(counts) for w, counts in landings_counter.items()
        }
    print(
        f"\n  (Diagnostic D will compare last-subword embedding closeness against "
        f"within-Cond-2 probe landings at layers {sorted(landings_per_layer_cond2.keys())})"
    )

    # ---- DIAGNOSTIC D needs the live model for embeddings ----
    diag_d = diagnostic_d(
        spec.short_name, model, tok, device,
        inv_words_landings_per_layer_cond2=landings_per_layer_cond2,
    )

    # ---- Free model before running cpu-bound diagnostics ----
    print(f"\nReleasing {spec.short_name} weights to free memory...")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ---- DIAGNOSTIC A ----
    diag_a = diagnostic_a(
        spec.short_name, cond1, cond2,
        cond1_layer=spec.cond1_focus_layer,
        cond2_layers=spec.cond2_focus_layers,
    )

    # ---- DIAGNOSTIC B (per condition) ----
    diag_b_cond1 = diagnostic_b(
        spec.short_name, cond1, "NEUTRAL", [0] + spec.diagnostic_layers
    )
    diag_b_cond2 = diagnostic_b(
        spec.short_name, cond2, "FUNC-PFX", [0] + spec.diagnostic_layers
    )

    # ---- DIAGNOSTIC C (per condition) ----
    diag_c_cond1 = diagnostic_c(
        spec.short_name, cond1, "NEUTRAL", spec.diagnostic_layers
    )
    diag_c_cond2 = diagnostic_c(
        spec.short_name, cond2, "FUNC-PFX", spec.diagnostic_layers
    )

    # Cleanup activations.
    del cond1, cond2
    gc.collect()

    return {
        "spec": spec,
        "diag_a": diag_a,
        "diag_b_cond1": diag_b_cond1,
        "diag_b_cond2": diag_b_cond2,
        "diag_c_cond1": diag_c_cond1,
        "diag_c_cond2": diag_c_cond2,
        "diag_d": diag_d,
    }


# ==============================================================================
# Cross-model summary
# ==============================================================================
def cross_model_summary(per_model: dict[str, dict]) -> None:
    print("\n\n" + "#" * 80)
    print("# CROSS-MODEL DIAGNOSTIC SUMMARY")
    print("#" * 80)

    print()
    print("Diagnostic A — cross-condition probe transfer")
    print("-" * 80)
    print("  Read: NEUTRAL-trained probe -> FUNC-PFX invented unary mass.")
    print("  If high (~80%+) on both models, attractor is robust across notations.")
    print("  If high on OLMo only, the attractor is OLMo-specific in functional notation.")
    print()
    print(f"  {'model':<14s}  {'NEUTRAL-trained probe -> FUNC-PFX invented unary mass':<55s}")
    for name, res in per_model.items():
        # Find the cross-condition entry that goes NEUTRAL -> FUNC-PFX at the
        # neutral-source layer evaluated at the neutral-source-layer in FUNC-PFX.
        ckey = None
        for k in res["diag_a"]:
            src_name, src_layer, tgt_name, tgt_layer = k
            if src_name == "NEUTRAL" and tgt_name == "FUNC-PFX" and src_layer == tgt_layer:
                ckey = k
                break
        if ckey is None:
            print(f"  {name:<14s}  (no matching transfer entry)")
            continue
        r = res["diag_a"][ckey]
        print(
            f"  {name:<14s}  CV={r['src_cv']:.3f}  within-NEUTRAL={r['within_unary']:5.1f}%  "
            f"cross-canon-acc={r['cross_canon_acc']:.3f}  cross-FUNC-PFX={r['cross_unary']:5.1f}%"
        )

    print()
    print("Diagnostic B — held-out canonical arity-class %")
    print("-" * 80)
    print("  Read: mean arity-class % across held-out canonicals at the focus layer.")
    print("  Structural: binary held-out ~50%+ to binaries, unary held-out ~25%+ to unaries.")
    print("  Surface-overfit: held-out class predicted at chance over four trained classes.")
    print()
    print(f"  {'model / cond':<22s}  {'layer':>6s}  {'binary held-out':>17s}  {'unary held-out':>16s}")
    for name, res in per_model.items():
        spec = res["spec"]
        for cond_label, diag_b in [("NEUTRAL", res["diag_b_cond1"]),
                                    ("FUNC-PFX", res["diag_b_cond2"])]:
            focus = spec.cond1_focus_layer if cond_label == "NEUTRAL" else spec.cond2_focus_layers[0]
            if focus not in diag_b:
                continue
            bin_pcts = [diag_b[focus][c]["arity_class_pct"] for c in BINARY_CANONICALS]
            un_pcts = [diag_b[focus][c]["arity_class_pct"] for c in UNARY_CANONICALS]
            print(
                f"  {name + '/' + cond_label:<22s}  {focus:>6d}  "
                f"{np.mean(bin_pcts):>16.1f}%  {np.mean(un_pcts):>15.1f}%"
            )

    print()
    print("Diagnostic C — geometric (probe-free) region delta")
    print("-" * 80)
    print("  Read: mean (sim_unary - sim_binary) across invented words at the focus layer.")
    print("  Positive => invented words geometrically closer to unary centroid (real attractor).")
    print("  Near zero or negative => no geometric attractor at this layer.")
    print()
    print(f"  {'model / cond':<22s}  {'layer':>6s}  {'mean delta':>11s}  {'n closer to unary':>22s}")
    for name, res in per_model.items():
        spec = res["spec"]
        for cond_label, diag_c in [("NEUTRAL", res["diag_c_cond1"]),
                                    ("FUNC-PFX", res["diag_c_cond2"])]:
            focus = spec.cond1_focus_layer if cond_label == "NEUTRAL" else spec.cond2_focus_layers[0]
            if focus not in diag_c:
                continue
            n_to_unary = diag_c[focus]["n_to_unary"]
            ratio_str = f"{n_to_unary} of {len(INVENTED_WORDS)}"
            print(
                f"  {name + '/' + cond_label:<22s}  {focus:>6d}  "
                f"{diag_c[focus]['mean_delta_unary_minus_binary']:>+10.4f}  "
                f"{ratio_str:>22s}"
            )

    print()
    print("Diagnostic D — last-subword top canonical (embedding layer)")
    print("-" * 80)
    print("  Read: per-invented-word: closest canonical by last-subword embedding.")
    print("  If layer-N landings track this list, layer-N result is a surface-token artifact.")
    print()
    for name, res in per_model.items():
        print(f"  {name}:")
        for w in INVENTED_WORDS:
            entry = res["diag_d"][w]
            print(f"    {w:<10s} (last subword '{entry['last_subword']}') -> {entry['top']}")
        print()


def main() -> None:
    log_path = _setup_logging()
    device = (
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    per_model: dict[str, dict] = {}
    for spec in MODEL_SPECS:
        per_model[spec.short_name] = run_model(spec, device)

    cross_model_summary(per_model)

    if log_path:
        print()
        print(f"[logging] full transcript written to: {log_path}")

    print()
    print("=" * 80)
    print("INTERPRETATION RUBRIC")
    print("=" * 80)
    print()
    print("Diagnostics jointly answer: is the Condition-2 Gemma 2 result a probe")
    print("artifact (most layers' apparent unary mass is template-bound surface")
    print("information leakage) or a real cross-model representational difference")
    print("(Gemma 2 genuinely does not build the arity attractor in functional-")
    print("prefix notation)?")
    print()
    print("Joint readings (per model x condition):")
    print()
    print("  PROBE ARTIFACT (recommended interpretation requires):")
    print("    A: NEUTRAL-trained probe -> FUNC-PFX invented unary mass is HIGH")
    print("       (i.e., the probe direction generalises across notations; the")
    print("        Condition-2 result was just the within-Cond-2 probe finding a")
    print("        different surface feature at each layer).")
    print("    B: Held-out canonical predictions in Cond-2 are at chance over the")
    print("       four trained classes (i.e., the Cond-2 probe is surface-overfit).")
    print("    C: Mean unary - binary centroid delta in Cond-2 at the focus layer")
    print("       is positive (geometric attractor exists, probe just didn't read it).")
    print("    D: Layer-8 per-word landings track last-subword embedding closeness")
    print("       (the layer-8 finding was propagated last-subword identity).")
    print()
    print("  REAL CROSS-MODEL DIFFERENCE requires:")
    print("    A: NEUTRAL-trained probe -> FUNC-PFX invented unary mass is LOW on")
    print("       Gemma 2 (probe direction does not survive notation change for")
    print("       Gemma 2 specifically) but HIGH on OLMo 2 (it survives for OLMo 2).")
    print("    C: Centroid delta in Cond-2 at the focus layer is near zero or")
    print("       negative for Gemma 2 (no geometric attractor) but positive for")
    print("       OLMo 2 (geometric attractor present).")
    print()
    print("  MIXED (genuine within-arity reshuffling but arity-region still encoded):")
    print("    A: cross-condition canonical accuracy is high but invented unary")
    print("       mass is intermediate (50-80%); attractor exists but is broader.")
    print("    C: positive but small delta; many but not all invented words closer")
    print("       to unary centroid.")
    print()
    print("Print this section to the writeup verbatim for any future reader who")
    print("needs to interpret an updated version of this experiment.")


if __name__ == "__main__":
    main()
