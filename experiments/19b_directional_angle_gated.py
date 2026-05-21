"""Script 19b - directional-angle analysis with canonical-transfer gate and
bootstrap confidence intervals (script 19 follow-up).

Reviewer follow-up to script 19. Adds three things on top of script 19's
per-layer centroid + probe angles:

  (i)  Canonical-transfer gate at each (train_cond, train_layer,
       test_cond, test_layer) pair. Trains a 5-class logistic-regression
       probe on source canonicals; evaluates accuracy on target canonicals.
       Accuracy >= 0.65 (~3x chance for 5-class) is required before the
       corresponding directional-angle or cross-condition transfer result
       is accepted as transportable. Threshold is empirically calibrated
       against the script-18 numbers (Gemma 2 FUNC-PFX@L16 -> NEUTRAL@L4
       cross-canonical = 0.200, which we want to fail; NEUTRAL@L4 ->
       FUNC-PFX@L4 = 1.000, which we want to pass; FUNC-PFX@L2 ->
       NEUTRAL@L2 = 0.756, which we want to pass).

  (ii) Bootstrap 95% confidence intervals on the cross-notation angles.
       100 within-class resamples (with replacement, n_per_class samples
       per class) for both the centroid direction and the probe direction.
       Reports 2.5 / 97.5 percentile bands.

  (iii) Cross-layer pairings beyond same-layer-only. Reproduces the
       script-18 critical pairings (NEUTRAL@L4 vs FUNC-PFX@L2 in Gemma 2,
       FUNC-PFX@L16 vs NEUTRAL@L4 in Gemma 2, etc.) with directional-
       angle measurements layered on top of the canonical-transfer gate
       accuracies.

Plus: disk-caches the activations to `experiments/outputs/cache/` so
repeat runs on the same model/condition/n_per_class skip the extraction
step entirely. The cache invalidates automatically if N_PER_CLASS or the
stimulus generation changes (filename includes N_PER_CLASS; per-stimulus
seeds are deterministic so as long as the script's RNG layout is stable,
cache hits are valid).

Run time on M4: ~12 min first-run (extraction dominated); ~3 min on
cache-hit re-run.
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Tee logging - same pattern as scripts 18 and 19.
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
    if os.environ.get("NO_LOG"):
        return None
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"19b_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    print(f"[logging] (set NO_LOG=1 to disable)")
    return log_path


# ==============================================================================
# Constants - identical to scripts 17/18/19 for direct comparability.
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
N_BOOTSTRAP = 100
GATE_THRESHOLD = 0.65

# Bump on any change that affects activations: stimulus templates, anchor
# logic, prompt generation seed mixing, preprocessing, etc.
STIMULUS_VERSION = "v2-stable-seeds"
ANCHOR_MODE = "operator-after"  # i + 1 after the last operator subword

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
UNARY_CANONICALS = ["not", "necessarily"]
BINARY_CANONICALS = ["and", "or", "implies"]
CANONICAL_ARITY = {"and": 2, "or": 2, "implies": 2, "not": 1, "necessarily": 1}

INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL = dict(zip(INVENTED_WORDS, CANONICALS))


# ==============================================================================
# Stable seeding + prompt-hash helpers (added after script-20 peer review).
# Python's built-in hash() is per-process salted unless PYTHONHASHSEED is
# fixed, so hash((name, op)) for seed derivation breaks stimulus
# reproducibility across runs. Replace with hashlib.blake2b for a stable,
# deterministic per-stimulus seed.
# ==============================================================================
def stable_seed(*parts, base: int = SEED, modulo: int = 100_000) -> int:
    s = "::".join(map(str, parts)).encode("utf-8")
    h = int(hashlib.blake2b(s, digest_size=8).hexdigest(), 16)
    return base + (h % modulo)


def prompts_checksum(prompts: list[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class ModelSpec:
    short_name: str
    model_id: str
    dtype: "torch.dtype"
    diagnostic_layers: list[int]
    cond1_focus_layer: int
    cond2_focus_layers: list[int]
    cross_layer_pairings: list[tuple]  # (src_cond, src_layer, tgt_cond, tgt_layer)


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        short_name="Gemma 2 9B",
        model_id="google/gemma-2-9b",
        dtype=torch.bfloat16,
        diagnostic_layers=[2, 4, 8, 16, 17],
        cond1_focus_layer=4,
        cond2_focus_layers=[2, 8, 16],
        # Critical pairings from script 18's table.
        cross_layer_pairings=[
            ("NEUTRAL", 4, "FUNC-PFX", 4),     # script-18 clean transfer
            ("NEUTRAL", 4, "FUNC-PFX", 2),     # the L2 sweet spot from script 19
            ("NEUTRAL", 2, "FUNC-PFX", 2),     # mirror at L2
            ("FUNC-PFX", 2, "NEUTRAL", 2),     # script-18 reverse, score 0.756
            ("FUNC-PFX", 2, "NEUTRAL", 4),     # script-18 reverse, score 0.672
            ("FUNC-PFX", 8, "NEUTRAL", 4),     # the L8 artifact layer
            ("FUNC-PFX", 16, "NEUTRAL", 16),   # script-18 "candidate L16", score 0.564
            ("FUNC-PFX", 16, "NEUTRAL", 4),    # script-18 cross-layer, score 0.200
        ],
    ),
    ModelSpec(
        short_name="OLMo 2 7B",
        model_id="allenai/OLMo-2-1124-7B",
        dtype=torch.float16,
        diagnostic_layers=[4, 7, 10, 16, 24],
        cond1_focus_layer=7,
        cond2_focus_layers=[7, 10, 16],
        cross_layer_pairings=[
            ("NEUTRAL", 7, "FUNC-PFX", 7),
            ("FUNC-PFX", 7, "NEUTRAL", 7),
            ("NEUTRAL", 7, "FUNC-PFX", 10),
            ("FUNC-PFX", 10, "NEUTRAL", 7),
            ("NEUTRAL", 4, "FUNC-PFX", 4),
            ("FUNC-PFX", 16, "NEUTRAL", 7),
        ],
    ),
]


# ==============================================================================
# Stimulus generation - verbatim from scripts 18/19.
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


def make_functional_canonical_stimuli(op: str, rng: random.Random, n: int) -> list[str]:
    arity = CANONICAL_ARITY[op]
    return _make_functional_stimuli_for_arity(op, arity, rng, n)


def make_functional_invented_stimuli(w: str, rng: random.Random, n: int) -> list[str]:
    arity = CANONICAL_ARITY[W_TO_CANONICAL[w]]
    return _make_functional_stimuli_for_arity(w, arity, rng, n)


def _make_functional_stimuli_for_arity(
    op: str, arity: int, rng: random.Random, n: int
) -> list[str]:
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


@dataclass
class ConditionActivations:
    canonical_X: list[np.ndarray]
    canonical_labels: np.ndarray
    invented_X: list[np.ndarray]
    invented_word_per_stim: list[str]


# ==============================================================================
# Disk cache for activations.
# ==============================================================================
def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    os.makedirs(base, exist_ok=True)
    slug = model_short_name.replace(" ", "_")
    # Filename includes STIMULUS_VERSION so changing stimulus generation
    # automatically invalidates the cache (no overwrite of valid older
    # caches; old files just become unused).
    return os.path.join(
        base,
        f"19b_{slug}_{condition_name}_npc{N_PER_CLASS}_{STIMULUS_VERSION}.npz",
    )


def _cache_save(
    path: str,
    cond: ConditionActivations,
    *,
    model_id: str,
    condition_name: str,
    canon_prompts_hash: str,
    inv_prompts_hash: str,
    dtype_before_cache: str,
) -> None:
    n_layers = len(cond.canonical_X)
    dim = cond.canonical_X[0].shape[1]
    canon_stack = np.stack(cond.canonical_X).astype(np.float16)
    inv_stack = np.stack(cond.invented_X).astype(np.float16)
    np.savez_compressed(
        path,
        canonical_X=canon_stack,
        canonical_labels=cond.canonical_labels,
        invented_X=inv_stack,
        invented_word_per_stim=np.array(cond.invented_word_per_stim),
        n_layers=np.array([n_layers]),
        dim=np.array([dim]),
        n_per_class=np.array([N_PER_CLASS]),
        meta_stimulus_version=np.array([STIMULUS_VERSION]),
        meta_anchor_mode=np.array([ANCHOR_MODE]),
        meta_model_id=np.array([model_id]),
        meta_condition=np.array([condition_name]),
        meta_canon_prompts_hash=np.array([canon_prompts_hash]),
        meta_inv_prompts_hash=np.array([inv_prompts_hash]),
        meta_dtype_before_cache=np.array([dtype_before_cache]),
    )
    print(f"    [cache] saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"    [cache]   meta: stimulus_version={STIMULUS_VERSION}, "
          f"anchor_mode={ANCHOR_MODE}, dtype={dtype_before_cache}")
    print(f"    [cache]   canon_prompts_hash={canon_prompts_hash[:16]}..., "
          f"inv_prompts_hash={inv_prompts_hash[:16]}...")


def _cache_load(
    path: str,
    *,
    expected_canon_prompts_hash: str,
    expected_inv_prompts_hash: str,
) -> ConditionActivations | None:
    """Returns the cached activations only if every metadata field
    matches the expectation derived from the current script. Any mismatch
    returns None (cache miss) so the caller falls through to re-extraction.
    Old caches without metadata fields are rejected on the first missing
    field."""
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)

        required_meta_keys = [
            "n_per_class", "meta_stimulus_version", "meta_anchor_mode",
            "meta_canon_prompts_hash", "meta_inv_prompts_hash",
        ]
        for k in required_meta_keys:
            if k not in z.files:
                print(f"    [cache] {os.path.basename(path)} missing key "
                      f"{k!r}; ignoring (likely a pre-v2 cache).")
                return None

        npc = int(z["n_per_class"][0])
        if npc != N_PER_CLASS:
            print(f"    [cache] n_per_class mismatch ({npc} != "
                  f"{N_PER_CLASS}); ignoring")
            return None

        sv = str(z["meta_stimulus_version"][0])
        if sv != STIMULUS_VERSION:
            print(f"    [cache] stimulus_version mismatch "
                  f"({sv} != {STIMULUS_VERSION}); ignoring")
            return None

        am = str(z["meta_anchor_mode"][0])
        if am != ANCHOR_MODE:
            print(f"    [cache] anchor_mode mismatch "
                  f"({am} != {ANCHOR_MODE}); ignoring")
            return None

        ch = str(z["meta_canon_prompts_hash"][0])
        if ch != expected_canon_prompts_hash:
            print(f"    [cache] canon_prompts_hash mismatch "
                  f"({ch[:16]}... != {expected_canon_prompts_hash[:16]}...); "
                  f"ignoring (stimulus generation drift)")
            return None

        ih = str(z["meta_inv_prompts_hash"][0])
        if ih != expected_inv_prompts_hash:
            print(f"    [cache] inv_prompts_hash mismatch "
                  f"({ih[:16]}... != {expected_inv_prompts_hash[:16]}...); "
                  f"ignoring (stimulus generation drift)")
            return None

        canon_stack = z["canonical_X"].astype(np.float32)
        inv_stack = z["invented_X"].astype(np.float32)
        cond = ConditionActivations(
            canonical_X=[canon_stack[i] for i in range(canon_stack.shape[0])],
            canonical_labels=z["canonical_labels"],
            invented_X=[inv_stack[i] for i in range(inv_stack.shape[0])],
            invented_word_per_stim=list(z["invented_word_per_stim"]),
        )
        print(f"    [cache] hit {os.path.basename(path)} "
              f"(n_layers={canon_stack.shape[0]}, dim={canon_stack.shape[2]})")
        print(f"    [cache]   verified: stimulus_version={sv}, "
              f"anchor_mode={am}")
        return cond
    except Exception as e:
        print(f"    [cache] failed to load {path}: {e}")
        return None


def _generate_prompts(
    name: str, canonical_stim_fn, invented_stim_fn,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Deterministically generate canonical and invented prompts using
    stable_seed. Returns (canon_prompts, canon_labels, inv_prompts,
    inv_words). Stable across runs (no PYTHONHASHSEED dependency)."""
    canon_prompts: list[str] = []
    canon_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(stable_seed(name, "canon", op))
        canon_prompts.extend(canonical_stim_fn(op, op_rng, N_PER_CLASS))
        canon_labels.extend([op] * N_PER_CLASS)

    inv_prompts: list[str] = []
    inv_words: list[str] = []
    for w in INVENTED_WORDS:
        w_rng = random.Random(stable_seed(name, "invent", w))
        stim = invented_stim_fn(w, w_rng, N_PER_CLASS)
        inv_prompts.extend(stim)
        inv_words.extend([w] * len(stim))

    return canon_prompts, canon_labels, inv_prompts, inv_words


def build_condition_cached(
    name: str,
    spec: ModelSpec,
    model, tok, device: str,
    canonical_stim_fn, invented_stim_fn,
) -> ConditionActivations:
    canon_prompts, canon_labels, inv_prompts, inv_words = _generate_prompts(
        name, canonical_stim_fn, invented_stim_fn
    )
    expected_canon_hash = prompts_checksum(canon_prompts)
    expected_inv_hash = prompts_checksum(inv_prompts)

    cache_p = _cache_path(spec.short_name, name)
    cached = _cache_load(
        cache_p,
        expected_canon_prompts_hash=expected_canon_hash,
        expected_inv_prompts_hash=expected_inv_hash,
    )
    if cached is not None:
        return cached

    assert model is not None and tok is not None, (
        "cache miss but model/tokenizer not loaded; "
        "delete stale partial files in outputs/cache/ and rerun"
    )

    print(f"\n  Building condition: {name} (no cache; extracting)")
    print(f"    canonicals: {len(canon_prompts)} stimuli ({len(CANONICALS)} classes)")
    print(f"    invented:   {len(inv_prompts)} stimuli ({len(INVENTED_WORDS)} words)")
    print(f"    canon_prompts_hash={expected_canon_hash[:16]}...")
    print(f"    inv_prompts_hash={expected_inv_hash[:16]}...")

    print("    extracting canonical activations...")
    t0 = time.time()
    X_canon = extract_anchored_activations(model, tok, canon_prompts, CANONICALS, device)
    print(f"      {time.time() - t0:.1f}s, n_layers={len(X_canon)}, dim={X_canon[0].shape[1]}")

    print("    extracting invented activations...")
    t0 = time.time()
    X_inv = extract_anchored_activations(model, tok, inv_prompts, INVENTED_WORDS, device)
    print(f"      {time.time() - t0:.1f}s")

    cond = ConditionActivations(
        canonical_X=X_canon,
        canonical_labels=np.array(canon_labels),
        invented_X=X_inv,
        invented_word_per_stim=inv_words,
    )
    _cache_save(
        cache_p, cond,
        model_id=spec.model_id,
        condition_name=name,
        canon_prompts_hash=expected_canon_hash,
        inv_prompts_hash=expected_inv_hash,
        dtype_before_cache=str(spec.dtype).replace("torch.", ""),
    )
    return cond


# ==============================================================================
# Direction primitives - copied from script 19.
# ==============================================================================
def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def cosine_angle_deg(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    cos = float(np.dot(u, v))
    cos = max(-1.0, min(1.0, cos))
    deg = float(np.degrees(np.arccos(cos)))
    return cos, deg


def centroid_unary_direction(
    canon_X_layer: np.ndarray, canon_labels: np.ndarray
) -> np.ndarray:
    unary_centroids = np.stack(
        [canon_X_layer[canon_labels == op].mean(axis=0) for op in UNARY_CANONICALS]
    )
    binary_centroids = np.stack(
        [canon_X_layer[canon_labels == op].mean(axis=0) for op in BINARY_CANONICALS]
    )
    return unit(unary_centroids.mean(axis=0) - binary_centroids.mean(axis=0))


def raw_binary_probe_direction(
    canon_X_layer: np.ndarray, canon_labels: np.ndarray
) -> np.ndarray:
    y = np.array([1 if op in UNARY_CANONICALS else 0 for op in canon_labels])
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(canon_X_layer, y)
    return unit(clf.coef_[0])


# ==============================================================================
# Canonical-transfer gate and within-condition CV.
# ==============================================================================
def within_cond_5class_cv(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores))


def canonical_transfer_accuracy(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> float:
    """Train 5-class probe on (X_train, y_train), evaluate on (X_test, y_test).
    Both X must have the same dim. Returns accuracy in [0, 1]."""
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def gate_verdict(acc: float) -> str:
    """Map cross-canonical accuracy to a 3-tier gate verdict."""
    if acc >= GATE_THRESHOLD:
        return "PASS"
    if acc >= 0.30:  # ~1.5x chance for 5-class
        return "AMBIG"
    return "FAIL"


# ==============================================================================
# Bootstrap CI on cross-notation angles.
# ==============================================================================
def _within_class_resample_indices(
    labels: np.ndarray, classes: list[str], rng: np.random.Generator,
) -> np.ndarray:
    idx_parts: list[np.ndarray] = []
    for c in classes:
        pool = np.where(labels == c)[0]
        idx_parts.append(rng.choice(pool, size=len(pool), replace=True))
    return np.concatenate(idx_parts)


def bootstrap_cross_notation_angles(
    Xa: np.ndarray, ya: np.ndarray,
    Xb: np.ndarray, yb: np.ndarray,
    n_boot: int, seed: int,
) -> dict:
    """For each bootstrap iter, resample within-class in both conditions,
    recompute centroid and probe directions, compute cross-notation angles.
    Returns summary statistics."""
    rng = np.random.default_rng(seed)
    centroid_degs: list[float] = []
    probe_degs: list[float] = []
    for _ in range(n_boot):
        ia = _within_class_resample_indices(ya, CANONICALS, rng)
        ib = _within_class_resample_indices(yb, CANONICALS, rng)
        Xa_b, ya_b = Xa[ia], ya[ia]
        Xb_b, yb_b = Xb[ib], yb[ib]

        d_a = centroid_unary_direction(Xa_b, ya_b)
        d_b = centroid_unary_direction(Xb_b, yb_b)
        _, deg_c = cosine_angle_deg(d_a, d_b)
        centroid_degs.append(deg_c)

        w_a = raw_binary_probe_direction(Xa_b, ya_b)
        w_b = raw_binary_probe_direction(Xb_b, yb_b)
        _, deg_p = cosine_angle_deg(w_a, w_b)
        probe_degs.append(deg_p)

    return {
        "n_boot": n_boot,
        "centroid_mean": float(np.mean(centroid_degs)),
        "centroid_ci_lo": float(np.percentile(centroid_degs, 2.5)),
        "centroid_ci_hi": float(np.percentile(centroid_degs, 97.5)),
        "probe_mean": float(np.mean(probe_degs)),
        "probe_ci_lo": float(np.percentile(probe_degs, 2.5)),
        "probe_ci_hi": float(np.percentile(probe_degs, 97.5)),
    }


# ==============================================================================
# Per-model analysis: per-layer gated table + cross-layer pairings.
# ==============================================================================
def analyse_model(
    spec: ModelSpec, cond1: ConditionActivations, cond2: ConditionActivations,
) -> dict:
    print(f"\n{'=' * 88}")
    print(f"DIRECTIONAL-ANGLE + GATE ANALYSIS - {spec.short_name}")
    print(f"{'=' * 88}")
    print(f"  Bootstrap: {N_BOOTSTRAP} within-class resamples per measurement.")
    print(f"  Gate threshold: cross-canonical 5-class accuracy >= {GATE_THRESHOLD:.2f}")
    print(f"  (= ~3x chance for 5-class; ambiguous between 0.30 and {GATE_THRESHOLD:.2f}; "
          f"FAIL below 0.30).")
    print()

    results: dict = {"model": spec.short_name,
                     "per_layer": [],
                     "cross_layer": []}

    layers = sorted(set(spec.diagnostic_layers
                        + [spec.cond1_focus_layer]
                        + list(spec.cond2_focus_layers)))

    # Same-layer per-L table
    print(f"  Same-layer cross-notation angles, with gate (NEUTRAL@L <-> FUNC-PFX@L):")
    print(f"    Each row: NEUTRAL CV / FUNC-PFX CV (within-cond 5-class), then")
    print(f"    cross-canonical accuracies in both directions, then centroid and")
    print(f"    probe angles with 95% bootstrap CIs, then the gate verdict.")
    print()
    header = ("    layer | NEUT CV | FUNC CV | gate N->F | gate F->N | "
              "centroid deg [95% CI] | probe deg [95% CI] | gate")
    print(header)
    print(f"    {'-' * (len(header) - 4)}")

    for L in layers:
        Xc_neut, y_neut = cond1.canonical_X[L], cond1.canonical_labels
        Xc_func, y_func = cond2.canonical_X[L], cond2.canonical_labels

        cv_neut = within_cond_5class_cv(Xc_neut, y_neut)
        cv_func = within_cond_5class_cv(Xc_func, y_func)
        gate_n2f = canonical_transfer_accuracy(Xc_neut, y_neut, Xc_func, y_func)
        gate_f2n = canonical_transfer_accuracy(Xc_func, y_func, Xc_neut, y_neut)

        boot = bootstrap_cross_notation_angles(
            Xc_neut, y_neut, Xc_func, y_func, N_BOOTSTRAP, SEED + L
        )

        # Gate verdict for the same-layer pairing: PASS only if BOTH directions
        # exceed the threshold (i.e., the per-layer arity direction is mutually
        # transportable in both notations).
        v_n2f = gate_verdict(gate_n2f)
        v_f2n = gate_verdict(gate_f2n)
        same_layer_gate = (
            "PASS" if (v_n2f == "PASS" and v_f2n == "PASS") else
            "FAIL" if (v_n2f == "FAIL" or v_f2n == "FAIL") else
            "AMBIG"
        )

        print(f"    {L:>5d} | {cv_neut:7.3f} | {cv_func:7.3f} | "
              f"{gate_n2f:>9.3f} | {gate_f2n:>9.3f} | "
              f"{boot['centroid_mean']:5.1f} [{boot['centroid_ci_lo']:.1f},"
              f"{boot['centroid_ci_hi']:.1f}]  | "
              f"{boot['probe_mean']:5.1f} [{boot['probe_ci_lo']:.1f},"
              f"{boot['probe_ci_hi']:.1f}]  | {same_layer_gate}")

        results["per_layer"].append({
            "layer": L,
            "neut_cv": cv_neut, "func_cv": cv_func,
            "gate_n2f": gate_n2f, "gate_f2n": gate_f2n,
            "v_n2f": v_n2f, "v_f2n": v_f2n,
            "same_layer_gate": same_layer_gate,
            **boot,
        })

    # Cross-layer pairings
    print()
    print(f"  Cross-layer critical pairings (script-18 table reproduced with")
    print(f"  directional-angle measurements layered on top):")
    print()
    cl_header = ("    src_cond  src_L  ->  tgt_cond  tgt_L | gate src->tgt | "
                 "centroid deg [95% CI] | probe deg [95% CI] | gate")
    print(cl_header)
    print(f"    {'-' * (len(cl_header) - 4)}")

    cond_map = {"NEUTRAL": cond1, "FUNC-PFX": cond2}

    for (src_cond_name, src_L, tgt_cond_name, tgt_L) in spec.cross_layer_pairings:
        src_cond = cond_map[src_cond_name]
        tgt_cond = cond_map[tgt_cond_name]
        Xs = src_cond.canonical_X[src_L]
        ys = src_cond.canonical_labels
        Xt = tgt_cond.canonical_X[tgt_L]
        yt = tgt_cond.canonical_labels

        gate_acc = canonical_transfer_accuracy(Xs, ys, Xt, yt)
        verdict = gate_verdict(gate_acc)

        boot = bootstrap_cross_notation_angles(
            Xs, ys, Xt, yt, N_BOOTSTRAP, SEED + 1000 + src_L * 50 + tgt_L
        )

        print(f"    {src_cond_name:<9} L{src_L:<4}  ->  {tgt_cond_name:<9} L{tgt_L:<4} | "
              f"{gate_acc:>13.3f} | "
              f"{boot['centroid_mean']:5.1f} [{boot['centroid_ci_lo']:.1f},"
              f"{boot['centroid_ci_hi']:.1f}]  | "
              f"{boot['probe_mean']:5.1f} [{boot['probe_ci_lo']:.1f},"
              f"{boot['probe_ci_hi']:.1f}]  | {verdict}")

        results["cross_layer"].append({
            "src_cond": src_cond_name, "src_layer": src_L,
            "tgt_cond": tgt_cond_name, "tgt_layer": tgt_L,
            "gate_acc": gate_acc,
            "verdict": verdict,
            **boot,
        })

    # Headline summary at the cond-1 focus layer
    focus_row = next(r for r in results["per_layer"]
                     if r["layer"] == spec.cond1_focus_layer)
    print()
    print(f"  Headline at cond-1 focus layer L={spec.cond1_focus_layer}:")
    print(f"    NEUTRAL CV: {focus_row['neut_cv']:.3f},  FUNC-PFX CV: {focus_row['func_cv']:.3f}")
    print(f"    canonical-transfer N->F: {focus_row['gate_n2f']:.3f} ({focus_row['v_n2f']})")
    print(f"    canonical-transfer F->N: {focus_row['gate_f2n']:.3f} ({focus_row['v_f2n']})")
    print(f"    centroid cross-notation: "
          f"{focus_row['centroid_mean']:.2f} deg "
          f"[{focus_row['centroid_ci_lo']:.2f}, {focus_row['centroid_ci_hi']:.2f}]")
    print(f"    probe    cross-notation: "
          f"{focus_row['probe_mean']:.2f} deg "
          f"[{focus_row['probe_ci_lo']:.2f}, {focus_row['probe_ci_hi']:.2f}]")
    print(f"    same-layer gate verdict: {focus_row['same_layer_gate']}")

    return results


# ==============================================================================
# Driver - mirrors script 19.
# ==============================================================================
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_model(spec: ModelSpec, device: str) -> dict:
    print()
    print(f"########  {spec.short_name}  ({spec.model_id})  ########")
    t_total = time.time()

    # Up-front cache check: regenerate prompts deterministically (fast),
    # compute expected hashes, then test the cache against those. Only
    # load the model if at least one condition is a cache miss.
    neut_canon, _, neut_inv, _ = _generate_prompts(
        "NEUTRAL", make_neutral_stimuli, make_neutral_stimuli
    )
    func_canon, _, func_inv, _ = _generate_prompts(
        "FUNC-PFX", make_functional_canonical_stimuli, make_functional_invented_stimuli
    )
    neut_cache = _cache_load(
        _cache_path(spec.short_name, "NEUTRAL"),
        expected_canon_prompts_hash=prompts_checksum(neut_canon),
        expected_inv_prompts_hash=prompts_checksum(neut_inv),
    )
    func_cache = _cache_load(
        _cache_path(spec.short_name, "FUNC-PFX"),
        expected_canon_prompts_hash=prompts_checksum(func_canon),
        expected_inv_prompts_hash=prompts_checksum(func_inv),
    )

    model = None
    tok = None
    if neut_cache is None or func_cache is None:
        print(f"  loading tokenizer + model ({spec.dtype}) on {device}...")
        tok = AutoTokenizer.from_pretrained(spec.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=spec.dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        print(f"  model loaded, n_layers={model.config.num_hidden_layers}, "
              f"hidden={model.config.hidden_size}")
    else:
        print(f"  both conditions cached; skipping model load")

    if neut_cache is None:
        print()
        print(f"  -- Building NEUTRAL condition --")
        cond1 = build_condition_cached(
            "NEUTRAL", spec, model, tok, device,
            canonical_stim_fn=make_neutral_stimuli,
            invented_stim_fn=make_neutral_stimuli,
        )
    else:
        cond1 = neut_cache

    if func_cache is None:
        print()
        print(f"  -- Building FUNCTIONAL-PREFIX condition --")
        cond2 = build_condition_cached(
            "FUNC-PFX", spec, model, tok, device,
            canonical_stim_fn=make_functional_canonical_stimuli,
            invented_stim_fn=make_functional_invented_stimuli,
        )
    else:
        cond2 = func_cache

    if model is not None:
        free_model(model)
    if tok is not None:
        del tok
    gc.collect()

    results = analyse_model(spec, cond1, cond2)
    print(f"\n  -- {spec.short_name} total time: {time.time() - t_total:.1f}s --")
    return results


def cross_model_summary(all_results: list[dict]) -> None:
    print()
    print("=" * 88)
    print("CROSS-MODEL SUMMARY (focus-layer same-layer pairing)")
    print("=" * 88)
    print()
    print("    model        | NEUT CV | FUNC CV | gate N->F | gate F->N | "
          "centroid deg (CI) | probe deg (CI) | gate")
    print("    " + "-" * 100)
    for r in all_results:
        spec = next(s for s in MODEL_SPECS if s.short_name == r["model"])
        row = next(rr for rr in r["per_layer"]
                   if rr["layer"] == spec.cond1_focus_layer)
        print(f"    {r['model']:<12} | "
              f"{row['neut_cv']:>7.3f} | {row['func_cv']:>7.3f} | "
              f"{row['gate_n2f']:>9.3f} | {row['gate_f2n']:>9.3f} | "
              f"{row['centroid_mean']:5.1f} [{row['centroid_ci_lo']:.1f},"
              f"{row['centroid_ci_hi']:.1f}] | "
              f"{row['probe_mean']:5.1f} [{row['probe_ci_lo']:.1f},"
              f"{row['probe_ci_hi']:.1f}] | {row['same_layer_gate']}")
    print()
    print("  Interpretation guide:")
    print("    PASS  cross-canonical >= 0.65 in both directions; the per-layer")
    print("          arity direction is mutually transportable across notations.")
    print("    AMBIG one direction passes, the other is borderline; treat the")
    print("          corresponding directional-angle as suggestive, not")
    print("          confirmed.")
    print("    FAIL  one direction has chance-level canonical-transfer; the")
    print("          directional-angle at this pairing reflects decision-")
    print("          boundary bias rather than a transferable arity direction.")


def main() -> None:
    log_path = _setup_logging()

    print(f"Script 19b - directional-angle + canonical-transfer gate + bootstrap CI")
    print(f"  HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '<unset>')}")
    print(f"  TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE', '<unset>')}")
    device = pick_device()
    print(f"  device: {device}")
    print(f"  seed:   {SEED}")
    print(f"  N_PER_CLASS: {N_PER_CLASS}")
    print(f"  N_BOOTSTRAP: {N_BOOTSTRAP}")
    print(f"  GATE_THRESHOLD: {GATE_THRESHOLD}")
    print(f"  models: {[s.short_name for s in MODEL_SPECS]}")
    print(f"  cache dir: experiments/outputs/cache/")

    all_results: list[dict] = []
    for spec in MODEL_SPECS:
        r = run_one_model(spec, device)
        all_results.append(r)

    cross_model_summary(all_results)

    if log_path:
        print()
        print(f"[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
