"""Script 25c -- corpus-frequency / lexical-identity control for Fact 1.

Purpose. The principal Fact-1 result (paper.md §4.1) is cross-notation
canonical-operator transfer at M2-canonical = 1.000, bootstrap CI
[1.000, 1.000] under 15-class readout at the N->F opera->opera L 4 cell
in all three model families. The reviewer (round 1 external) flagged a
potential alternative explanation: the probe may be finding a generic
substrate-independent lexical-identity hyperplane that would also
produce ceiling-level accuracy on any 15-word readout vocabulary, not
a specifically operator-class abstraction. This script tests that
alternative by replacing the 15 v6 canonicals with 15 heterogeneous
non-operator content words in syntactically identical NEUTRAL and
FUNC-PFX templates and running the principal Fact-1 cell.

Pre-registered adjudication.
  (a) If M2-canonical >= 0.65 at the principal Fact-1 cell on the
      content-word set, Fact 1 generalises to trained-vocabulary-set
      substrate-invariance broadly; the "operator-set-bound" framing
      in paper.md §1, abstract, §4.1, §5.1 reframes as
      "trained-vocabulary-set-bound" and the operator-class-specific
      claim is retracted.
  (b) If M2-canonical < 0.65 at the principal Fact-1 cell on the
      content-word set, Fact 1 is operator-class-specific and the
      headline framing stands. Report M2-canonical and bootstrap 95%
      CI on the content-word set as a supporting datum.

Content-word set design.
  - 15 heterogeneous content words spanning frequency tiers comparable
    to the v6 canonical set's 4-5 order-of-magnitude span.
  - 8 "binary"-tagged + 7 "unary"-tagged purely for template position
    matching: "binary" words go into the 2-argument FUNC-PFX template
    `op(p, q)`, "unary" words into the 1-argument `op(p)`. The arity
    tag carries no semantic meaning -- it is a syntactic position
    designator only.
  - High freq (5): house, water, music, light, paper
  - Mid  freq (5): pattern, theory, system, region, period
  - Low  freq (5): archipelago, mosaic, plinth, ledger, cassowary

Methodology.
  - Monkey-patch 19b's CANONICALS / CANONICAL_ARITY / UNARY_CANONICALS
    / BINARY_CANONICALS at module import to use the content-word set.
  - Reuse 21_multi_anchor_battery.py's extract_multi_anchor_activations
    unchanged for fresh content-word activation extraction.
  - Train logistic-regression probe on NEUTRAL operator-after content-
    word activations at L 4; test M2-canonical at FUNC-PFX
    operator-after L 4 against the same 15-class readout.
  - Bootstrap M2-canonical with B = 500 stimulus resamples.

Cache layout. `cache/25c_<model_short>.npz` with content-word
activations only (canonical_X / canonical_labels / FUNC-PFX same
structure as script 24's carryover cache; invented set not extracted
since this script does not test Fact 2).

Runtime estimate (M4 MPS):
  Gemma 2 9B  carryover extraction: ~20-25 min
  OLMo 2 7B   carryover extraction: ~15-20 min
  Pythia 6.9B carryover extraction: ~10-15 min
  Probe + bootstrap + adjudication: ~2-3 min per model
  Total first-run: ~50-70 min wall-clock
  Cache hit re-run: ~5-10 min total.

Tees all output to outputs/25c_<ts>.log.
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import importlib.machinery
import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Tee logging
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


def _setup_logging() -> Optional[str]:
    if os.environ.get("NO_LOG"):
        return None
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"25c_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Content-word set + design
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
N_BOOTSTRAP = 500

STIMULUS_VERSION = "25c-content-word-control"

# 8 "binary"-tagged content words (template position matching only).
CONTENT_BINARY_8 = [
    "house", "water", "music", "light", "paper",
    "pattern", "theory", "system",
]
# 7 "unary"-tagged content words.
CONTENT_UNARY_7 = [
    "region", "period",
    "archipelago", "mosaic", "plinth", "ledger", "cassowary",
]

CONTENT_WORDS = CONTENT_BINARY_8 + CONTENT_UNARY_7
assert len(CONTENT_WORDS) == 15
assert len(set(CONTENT_WORDS)) == 15

# "Arity" map: purely a syntactic-position designator for FUNC-PFX
# template structure (`op(p, q)` vs `op(p)`); carries no semantic meaning.
CONTENT_ARITY: dict[str, int] = {}
for w in CONTENT_BINARY_8:
    CONTENT_ARITY[w] = 2
for w in CONTENT_UNARY_7:
    CONTENT_ARITY[w] = 1

# Frequency tier annotation (for reporting only; not used in adjudication).
CONTENT_FREQ_TIER: dict[str, str] = {
    "house": "high", "water": "high", "music": "high",
    "light": "high", "paper": "high",
    "pattern": "mid", "theory": "mid", "system": "mid",
    "region": "mid", "period": "mid",
    "archipelago": "low", "mosaic": "low", "plinth": "low",
    "ledger": "low", "cassowary": "low",
}

# Pre-registered adjudication threshold.
M2C_REFRAME_THRESHOLD = 0.65


# ==============================================================================
# Anchors + cells (we only need the principal Fact-1 cell + a few sanity cells)
# ==============================================================================
ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]


# Principal Fact-1 cell + Phase 1/2 paper-cited cells. The L 4 cross-family
# cell is the load-bearing diagnostic; the others are reported for context.
@dataclass(frozen=True)
class TargetCell:
    direction: str      # "N->F" or "F->N"
    train_anchor: str
    test_anchor: str
    layer: int          # absolute layer index (model-specific)


def cells_for_model(focus_layers: list[int]) -> list[TargetCell]:
    # The principal Fact-1 cell across all three models: L 4 (Gemma), L 4 (OLMo, Pythia).
    layer_4_or_4 = 4 if 4 in focus_layers else focus_layers[0]
    out = [
        TargetCell("N->F", "operator-after", "operator-after", layer_4_or_4),
        TargetCell("N->F", "operator-after", "operator-after", focus_layers[2]),  # OLMo L 10
        TargetCell("F->N", "operator-after", "operator-after", layer_4_or_4),
    ]
    seen = set()
    uniq = []
    for c in out:
        key = (c.direction, c.train_anchor, c.test_anchor, c.layer)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


# ==============================================================================
# Model specs (mirror 24)
# ==============================================================================
@dataclass
class ModelSpec:
    short_name: str
    model_id: str
    dtype: "torch.dtype"
    focus_layers: list[int]


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        short_name="Gemma 2 9B",
        model_id="google/gemma-2-9b",
        dtype=torch.bfloat16,
        focus_layers=[2, 4, 8, 16, 17],
    ),
    ModelSpec(
        short_name="OLMo 2 7B",
        model_id="allenai/OLMo-2-1124-7B",
        dtype=torch.float16,
        focus_layers=[4, 7, 10, 16, 24],
    ),
    ModelSpec(
        short_name="Pythia 6.9B-deduped",
        model_id="EleutherAI/pythia-6.9b-deduped",
        dtype=torch.float16,
        focus_layers=[4, 7, 10, 16, 24],
    ),
]


def stable_seed(*parts, base: int = SEED, modulo: int = 100_000) -> int:
    s = "::".join(map(str, parts)).encode("utf-8")
    h = int(hashlib.blake2b(s, digest_size=8).hexdigest(), 16)
    return base + (h % modulo)


# ==============================================================================
# Load 19b module + monkey-patch CANONICALS -> CONTENT_WORDS
# ==============================================================================
def _load_module(filename: str, alias: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    loader = importlib.machinery.SourceFileLoader(alias, path)
    spec = importlib.util.spec_from_loader(alias, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    loader.exec_module(mod)
    return mod


_M19B = _load_module("19b_directional_angle_gated.py", "_m19b_25c")
_M19B.CANONICALS = CONTENT_WORDS
_M19B.CANONICAL_ARITY = CONTENT_ARITY
_M19B.UNARY_CANONICALS = CONTENT_UNARY_7
_M19B.BINARY_CANONICALS = CONTENT_BINARY_8
# We do not use the invented machinery in 25c, but the import surfaces it;
# leave it un-overridden so any accidental call would fail loudly.
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
assert _M19B.SEED == SEED
assert _M19B.N_PER_CLASS == N_PER_CLASS

_M21 = _load_module("21_multi_anchor_battery.py", "_m21_25c")
extract_multi_anchor_activations = _M21.extract_multi_anchor_activations


# ==============================================================================
# Tokenization audit
# ==============================================================================
def _tokenize_pieces(tok, word: str) -> list[str]:
    """Return subword pieces for the word as encountered in mid-sentence (with
    a leading space). Strips special tokens."""
    ids = tok.encode(" " + word, add_special_tokens=False)
    return [tok.convert_ids_to_tokens(i) for i in ids]


@dataclass
class TokenAudit:
    word: str
    n_pieces: int
    pieces: list[str]


def audit_content_word_tokens(tok, words: list[str]) -> list[TokenAudit]:
    out: list[TokenAudit] = []
    for w in words:
        pcs = _tokenize_pieces(tok, w)
        out.append(TokenAudit(word=w, n_pieces=len(pcs), pieces=pcs))
    return out


def print_token_audit(audits: list[TokenAudit], model_name: str) -> None:
    print(f"\n[audit] {model_name} content-word tokenization:")
    print(f"  {'word':<14} {'pcs':>4} pieces")
    for a in audits:
        print(f"  {a.word:<14} {a.n_pieces:>4}  {a.pieces}")
    multi = [a for a in audits if a.n_pieces > 1]
    print(f"  [summary] {len(multi)}/{len(audits)} multi-piece "
          f"(matching v6 canonical set: 1/15 multi-piece for `unprovably`)")


def diagnose_func_prompt_tokenization(
    tok, func_prompts: list[str], func_words: list[str], model_name: str,
    *, n_per_word: int = 1,
) -> None:
    """Print decoded tokenization of one FUNC-PFX prompt per content word.

    Surfaces why `first-arg` / `close-paren` anchors fall back: if `(` is
    merged into the operator subword (e.g., `▁house(`) or tokenized in a
    form whose `tok.decode([i])` doesn't contain the literal `(` character,
    `_find_token_after(decoded, "(", ...)` in 21_multi_anchor_battery.py
    returns None and the anchor is replaced with sentence-final.
    """
    print(f"\n[diagnose] {model_name} FUNC-PFX tokenization (1 sample per word):")
    seen: dict[str, int] = {}
    samples: list[tuple[str, str]] = []
    for prompt, word in zip(func_prompts, func_words):
        if seen.get(word, 0) >= n_per_word:
            continue
        seen[word] = seen.get(word, 0) + 1
        samples.append((word, prompt))
        if len(seen) == len(set(func_words)) and all(
            seen[w] >= n_per_word for w in seen
        ):
            break

    paren_open_findable = 0
    paren_close_findable = 0
    for word, prompt in samples:
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        decoded = [tok.decode([i]) for i in ids]
        has_open = any("(" in t for t in decoded)
        has_close = any(")" in t for t in decoded)
        paren_open_findable += int(has_open)
        paren_close_findable += int(has_close)
        marker = "OK" if (has_open and has_close) else "MISS"
        print(f"  [{marker}] {word:<12}  prompt={prompt!r}")
        print(f"           decoded={decoded}")
    n = len(samples)
    print(f"  [summary] {paren_open_findable}/{n} prompts contain a token "
          f"with literal '('  ;  {paren_close_findable}/{n} contain ')'")
    print("  (if either summary count is < n, _find_token_after will fall "
          "back to sentence-final for that anchor)")


# ==============================================================================
# Cache
# ==============================================================================
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(short: str) -> str:
    slug = short.lower().replace(" ", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"25c_{slug}.npz")


def _save_cache(path: str, *,
                neutral_X: np.ndarray, neutral_labels: np.ndarray,
                neutral_prompts: list[str],
                func_X: np.ndarray, func_labels: np.ndarray,
                func_prompts: list[str],
                content_words: list[str], anchors_neutral: list[str],
                anchors_func: list[str], n_layers: int) -> None:
    np.savez_compressed(
        path,
        neutral_X=neutral_X,
        neutral_labels=neutral_labels,
        neutral_prompts=np.array(neutral_prompts, dtype=object),
        func_X=func_X,
        func_labels=func_labels,
        func_prompts=np.array(func_prompts, dtype=object),
        content_words=np.array(content_words, dtype=object),
        anchors_neutral=np.array(anchors_neutral, dtype=object),
        anchors_func=np.array(anchors_func, dtype=object),
        n_layers=np.int64(n_layers),
        stimulus_version=STIMULUS_VERSION,
    )


def _load_cache(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as e:
        print(f"[cache] failed to load {path}: {e}")
        return None
    return {
        "neutral_X": z["neutral_X"],
        "neutral_labels": z["neutral_labels"],
        "neutral_prompts": list(z["neutral_prompts"]),
        "func_X": z["func_X"],
        "func_labels": z["func_labels"],
        "func_prompts": list(z["func_prompts"]),
        "content_words": list(z["content_words"]),
        "anchors_neutral": list(z["anchors_neutral"]),
        "anchors_func": list(z["anchors_func"]),
        "n_layers": int(z["n_layers"]),
    }


# ==============================================================================
# Stimulus generation
# ==============================================================================
def build_stimuli(words: list[str], rng_base: int) -> tuple[list[str], list[str], list[str], list[str]]:
    """Build NEUTRAL + FUNC-PFX stimuli for each content word.
    Returns (neutral_prompts, neutral_labels, func_prompts, func_labels)
    where labels[i] = content_words[i]'s row index, but we just store
    the word string for downstream consumption (label encoding handled later).
    """
    neutral_prompts: list[str] = []
    neutral_words: list[str] = []
    func_prompts: list[str] = []
    func_words: list[str] = []
    for w in words:
        rng_n = random.Random(stable_seed(rng_base, "neutral", w))
        rng_f = random.Random(stable_seed(rng_base, "func", w))
        ns = make_neutral_stimuli(w, rng_n, N_PER_CLASS)
        fs = make_functional_canonical_stimuli(w, rng_f, N_PER_CLASS)
        neutral_prompts.extend(ns)
        neutral_words.extend([w] * len(ns))
        func_prompts.extend(fs)
        func_words.extend([w] * len(fs))
    return neutral_prompts, neutral_words, func_prompts, func_words


# ==============================================================================
# Extraction (per model)
# ==============================================================================
def extract_for_model(spec: ModelSpec, device: str) -> dict:
    cache_path = _cache_path(spec.short_name)
    cached = _load_cache(cache_path)
    cache_hit = cached is not None and cached["content_words"] == CONTENT_WORDS

    if cache_hit:
        print(f"\n[cache] {spec.short_name} cache hit at {cache_path}")
        tok = AutoTokenizer.from_pretrained(spec.model_id)
        audits = audit_content_word_tokens(tok, CONTENT_WORDS)
        print_token_audit(audits, spec.short_name)
        diagnose_func_prompt_tokenization(
            tok, list(cached["func_prompts"]),
            list(cached["func_labels"]), spec.short_name,
        )
        del tok
        gc.collect()
        return cached

    if cached is not None:
        print(f"[cache] {spec.short_name} cache mismatch (content_words differ); re-extracting")

    print(f"\n[extract] {spec.short_name} -> {cache_path}")
    print(f"  loading model {spec.model_id} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=spec.dtype
    ).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s; dtype={spec.dtype}; device={device}")

    audits = audit_content_word_tokens(tok, CONTENT_WORDS)
    print_token_audit(audits, spec.short_name)

    neutral_prompts, neutral_words, func_prompts, func_words = build_stimuli(
        CONTENT_WORDS, rng_base=stable_seed(spec.short_name)
    )

    diagnose_func_prompt_tokenization(tok, func_prompts, func_words, spec.short_name)

    t1 = time.time()
    print(f"\n[extract] NEUTRAL n={len(neutral_prompts)} stimuli ...")
    neutral_X, _, n_layers_n = extract_multi_anchor_activations(
        model, tok, neutral_prompts, neutral_words,
        ANCHORS_NEUTRAL, condition="NEUTRAL", device=device,
    )
    print(f"  NEUTRAL shape={neutral_X.shape} in {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"\n[extract] FUNC-PFX n={len(func_prompts)} stimuli ...")
    func_X, _, n_layers_f = extract_multi_anchor_activations(
        model, tok, func_prompts, func_words,
        ANCHORS_FUNC_PFX, condition="FUNC-PFX", device=device,
    )
    print(f"  FUNC-PFX shape={func_X.shape} in {time.time() - t2:.1f}s")
    assert n_layers_n == n_layers_f, (n_layers_n, n_layers_f)

    neutral_labels = np.array(neutral_words, dtype=object)
    func_labels = np.array(func_words, dtype=object)

    _save_cache(
        cache_path,
        neutral_X=neutral_X, neutral_labels=neutral_labels, neutral_prompts=neutral_prompts,
        func_X=func_X, func_labels=func_labels, func_prompts=func_prompts,
        content_words=CONTENT_WORDS, anchors_neutral=ANCHORS_NEUTRAL,
        anchors_func=ANCHORS_FUNC_PFX, n_layers=n_layers_n,
    )
    print(f"[cache] saved to {cache_path}")

    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return {
        "neutral_X": neutral_X, "neutral_labels": neutral_labels,
        "neutral_prompts": neutral_prompts,
        "func_X": func_X, "func_labels": func_labels,
        "func_prompts": func_prompts,
        "content_words": CONTENT_WORDS,
        "anchors_neutral": ANCHORS_NEUTRAL,
        "anchors_func": ANCHORS_FUNC_PFX,
        "n_layers": n_layers_n,
    }


# ==============================================================================
# Probe + bootstrap
# ==============================================================================
def _slice_anchor_layer(X: np.ndarray, anchors: list[str], anchor: str,
                        layer: int) -> np.ndarray:
    """X shape: (n_anchors, n_stim, n_layers, dim). Return (n_stim, dim)."""
    a_idx = anchors.index(anchor)
    return X[a_idx, :, layer, :]


def m1_cv(X: np.ndarray, y: np.ndarray, *, seed: int) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs: list[float] = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    return float(np.mean(accs))


def m2_canonical(train_X: np.ndarray, train_y: np.ndarray,
                 test_X: np.ndarray, test_y: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
    clf.fit(train_X, train_y)
    return float(clf.score(test_X, test_y))


def bootstrap_m2c(train_X: np.ndarray, train_y: np.ndarray,
                  test_X: np.ndarray, test_y: np.ndarray, *,
                  n_boot: int, seed: int) -> tuple[float, float, float]:
    """Resample test stimuli with replacement (within-class stratified)
    and report mean / 2.5%-ile / 97.5%-ile of M2-canonical."""
    rng = np.random.default_rng(seed)
    clf = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
    clf.fit(train_X, train_y)
    pred_all = clf.predict(test_X)
    classes = np.unique(test_y)
    by_class = {c: np.where(test_y == c)[0] for c in classes}
    accs: list[float] = []
    for _ in range(n_boot):
        idx: list[int] = []
        for c in classes:
            ci = by_class[c]
            idx.extend(rng.choice(ci, size=len(ci), replace=True).tolist())
        idx_arr = np.array(idx)
        accs.append(float(np.mean(pred_all[idx_arr] == test_y[idx_arr])))
    return float(np.mean(accs)), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


# ==============================================================================
# Run per model
# ==============================================================================
@dataclass
class CellResult:
    cell: TargetCell
    m1_neutral: float
    m1_func: float
    m2c_point: float
    m2c_boot_mean: float
    m2c_boot_lo: float
    m2c_boot_hi: float


def run_for_model(spec: ModelSpec, cache: dict) -> list[CellResult]:
    print(f"\n[run] {spec.short_name}")
    results: list[CellResult] = []
    cells = cells_for_model(spec.focus_layers)
    for cell in cells:
        # Train condition data.
        if cell.direction == "N->F":
            train_X_all = cache["neutral_X"]
            train_anchors = cache["anchors_neutral"]
            train_labels = cache["neutral_labels"]
            test_X_all = cache["func_X"]
            test_anchors = cache["anchors_func"]
            test_labels = cache["func_labels"]
        else:
            train_X_all = cache["func_X"]
            train_anchors = cache["anchors_func"]
            train_labels = cache["func_labels"]
            test_X_all = cache["neutral_X"]
            test_anchors = cache["anchors_neutral"]
            test_labels = cache["neutral_labels"]

        train_X = _slice_anchor_layer(train_X_all, train_anchors,
                                       cell.train_anchor, cell.layer)
        test_X = _slice_anchor_layer(test_X_all, test_anchors,
                                      cell.test_anchor, cell.layer)

        seed = stable_seed(spec.short_name, cell.direction, cell.train_anchor,
                            cell.test_anchor, cell.layer)
        m1_n = m1_cv(train_X, train_labels.astype(str),
                     seed=seed)
        m1_f = m1_cv(test_X, test_labels.astype(str),
                     seed=seed + 1)
        m2c_point = m2_canonical(train_X, train_labels.astype(str),
                                  test_X, test_labels.astype(str))
        m2c_mean, m2c_lo, m2c_hi = bootstrap_m2c(
            train_X, train_labels.astype(str),
            test_X, test_labels.astype(str),
            n_boot=N_BOOTSTRAP, seed=seed + 2,
        )
        results.append(CellResult(
            cell=cell, m1_neutral=m1_n, m1_func=m1_f,
            m2c_point=m2c_point, m2c_boot_mean=m2c_mean,
            m2c_boot_lo=m2c_lo, m2c_boot_hi=m2c_hi,
        ))
        print(f"  {cell.direction} {cell.train_anchor}->{cell.test_anchor} L{cell.layer}: "
              f"M1n={m1_n:.3f} M1f={m1_f:.3f} "
              f"M2c={m2c_point:.3f} (mean {m2c_mean:.3f} CI [{m2c_lo:.3f}, {m2c_hi:.3f}])")
    return results


# ==============================================================================
# Top-level adjudication
# ==============================================================================
def adjudicate(per_model: dict[str, list[CellResult]]) -> None:
    print("\n" + "=" * 78)
    print("Pre-registered adjudication (M2-canonical at principal Fact-1 cell)")
    print("=" * 78)
    print(f"\nThreshold: M2c >= {M2C_REFRAME_THRESHOLD:.2f} on content words at the")
    print("           N->F opera->opera L 4 cell triggers a REFRAME of paper.md")
    print("           §1, abstract, §4.1, §5.1 from 'operator-set-bound' to")
    print("           'trained-vocabulary-set-bound'.")
    print()
    chance_15 = 1.0 / len(CONTENT_WORDS)
    print(f"           Chance baseline: 1/{len(CONTENT_WORDS)} = {chance_15:.3f}")
    print(f"           v6 canonical-set comparison: M2c = 1.000, CI [1.000, 1.000]")
    print(f"           per paper.md §4.1 / paper_notes §3.7.16.")
    print()
    print(f"{'Model':<22} {'Cell':<32} {'M2c':>6} {'CI':>18} {'Verdict':<24}")
    print("-" * 100)
    triggers_reframe = 0
    n_principal_cells = 0
    for model_name, results in per_model.items():
        for r in results:
            cell_key = f"{r.cell.direction} {r.cell.train_anchor}->{r.cell.test_anchor} L{r.cell.layer}"
            ci_str = f"[{r.m2c_boot_lo:.3f}, {r.m2c_boot_hi:.3f}]"
            if r.m2c_point >= M2C_REFRAME_THRESHOLD:
                verdict = "REFRAME-TRIGGER"
                if r.cell.direction == "N->F" and r.cell.train_anchor == "operator-after" \
                        and r.cell.test_anchor == "operator-after":
                    triggers_reframe += 1
                    n_principal_cells += 1
            else:
                verdict = "operator-class-specific"
                if r.cell.direction == "N->F" and r.cell.train_anchor == "operator-after" \
                        and r.cell.test_anchor == "operator-after":
                    n_principal_cells += 1
            print(f"{model_name:<22} {cell_key:<32} {r.m2c_point:>6.3f} {ci_str:>18} {verdict:<24}")
    print()
    if n_principal_cells == 0:
        print("[adjudication] no principal Fact-1 cells found in sweep")
        return
    if triggers_reframe == n_principal_cells:
        print("[adjudication] ALL principal Fact-1 cells trigger REFRAME.")
        print("    -> Fact 1 generalises to trained-vocabulary-set-bound.")
        print("    -> Update paper.md §1, abstract, §4.1, §5.1 accordingly.")
    elif triggers_reframe == 0:
        print("[adjudication] ZERO principal Fact-1 cells trigger REFRAME.")
        print("    -> Fact 1 is operator-class-specific; headline framing stands.")
        print("    -> Report content-word M2c + CI in paper.md §4.1.1 as supporting datum.")
    else:
        print(f"[adjudication] PARTIAL: {triggers_reframe}/{n_principal_cells} principal "
              "cells trigger REFRAME.")
        print("    -> Examine model-specific patterns; partial reframe may be needed.")


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    _setup_logging()
    print(f"[start] {_dt.datetime.now().isoformat()}")
    print(f"[stimulus_version] {STIMULUS_VERSION}")
    print(f"[content_words] {CONTENT_WORDS}")
    print(f"[N_PER_CLASS] {N_PER_CLASS}")
    print(f"[N_BOOTSTRAP] {N_BOOTSTRAP}")
    print(f"[M2C_REFRAME_THRESHOLD] {M2C_REFRAME_THRESHOLD}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[device] {device}")

    per_model: dict[str, list[CellResult]] = {}
    for spec in MODEL_SPECS:
        t0 = time.time()
        cache = extract_for_model(spec, device=device)
        per_model[spec.short_name] = run_for_model(spec, cache)
        print(f"[done] {spec.short_name} in {time.time() - t0:.1f}s")

    adjudicate(per_model)
    print(f"\n[end] {_dt.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
