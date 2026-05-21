"""Script 24 - v6 canonical-set expansion (15 canonicals, three-model
multi-scope sweep, pre-registered per experiments/preregistration_v6.md).

Purpose. Test the default-to-rarest-canonical mechanism identified in
§3.7.14 (OLMo + Gemma) / §3.7.15 (Pythia) by expanding the canonical
set with 5 new operators that disentangle two confounded dimensions:
  - Token frequency in training data
  - Subword-tokenization shape (single-piece vs multi-piece)

In v5, all three default-attraction targets (`xor`, `nand`, `negate`)
are simultaneously multi-subword and low-frequency. The new v6
canonicals span the {frequency x subword-count} cross-tab so the
mechanism can be adjudicated:

  - `nor`        (B, target 1pc, LF) -- single-pc LF (frequency would
                                         predict ATTRACTS; subword
                                         would predict NOT)
  - `iff`        (B, target 2-3pc, very LF) -- multi-pc LF (both
                                                predict ATTRACTS;
                                                control for v5
                                                attractors)
  - `unless`     (B, target 1pc, MF) -- single-pc MF (neither
                                         predicts attraction)
  - `definitely` (U, target 1pc, MF) -- single-pc MF unary control
  - `unprovably` (U, target multi-pc, very LF) -- multi-pc LF unary
                                                   (both predict
                                                   ATTRACTS)

Plus the v5 canonicals (10) for total 15 = 8 binary + 7 unary.

This script:
  (1) Audits the v6 canonical tokenization across all three model
      tokenizers and flags OUT-OF-DESIGN canonicals (where the actual
      tokenization profile does not match the pre-registered target).
  (2) Extracts a v6-expanded-canonical cache per model (Gemma 2 9B,
      OLMo 2 7B, Pythia 6.9B-deduped), reusing v5 caches' invented
      activations where possible (the invented set is unchanged from
      v4/v5) but re-extracting all canonical activations since the
      canonical set has changed.
  (3) Extracts a v6-heldout canonicals-only cache per model on a
      template set syntactically disjoint from the carryover templates,
      to validate that M1 within-condition CV does not collapse on
      held-out templates (template-leakage check).
  (4) Runs the full anchor x layer sweep at FOUR scopes (v3, v4, v5,
      v6) per model from the carryover cache.
  (5) Adjudicates the pre-registered predictions P_FREQ / P_SUBWORD /
      P_INTERACTION against the v6 aggregate per-word top-canonical
      distribution per model.
  (6) Bootstraps M4b on the top-M2c v6 cell per model and M2-canonical
      at the four cross-family Fact-1 anchor cells (the
      operator-set-bound paper's principal positive finding).

Runtime estimate (M4 MPS):
  Gemma 2 9B carryover + heldout extraction: ~25-30 min
  OLMo 2 7B  carryover + heldout extraction: ~15-20 min
  Pythia 6.9B-d carryover + heldout extraction: ~10-15 min
  Sweep + bootstrap + adjudication (cache-only): ~10-15 min per model
  Total first-run: ~75-100 min wall-clock
  Cache hit re-run: ~10-15 min total.

See experiments/preregistration_v6.md for the frozen analysis plan.
Tees all output to outputs/24_<ts>.log.
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
    log_path = os.path.join(log_dir, f"24_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Constants
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
N_BOOTSTRAP = 500
SKIP_HELDOUT = bool(os.environ.get("SKIP_HELDOUT"))
AUDIT_ONLY = bool(os.environ.get("AUDIT_ONLY"))

STIMULUS_VERSION = "v6-expanded-canonical"

# v5 carryover canonicals (10).
CANONICALS_V5 = [
    "and", "or", "implies", "xor", "nand",
    "not", "necessarily", "possibly", "always", "negate",
]

# v6 NEW canonicals (5) - the disentanglement set.
NEW_5_CANONICALS_V6 = ["nor", "iff", "unless", "definitely", "unprovably"]

# Full v6 canonical set (15).
CANONICALS = CANONICALS_V5 + NEW_5_CANONICALS_V6

# Arity map (v6).
CANONICAL_ARITY = {
    # binary (8)
    "and": 2, "or": 2, "implies": 2, "xor": 2, "nand": 2,
    "nor": 2, "iff": 2, "unless": 2,
    # unary (7)
    "not": 1, "necessarily": 1, "possibly": 1, "always": 1, "negate": 1,
    "definitely": 1, "unprovably": 1,
}

UNARY_CANONICALS_V6 = [c for c in CANONICALS if CANONICAL_ARITY[c] == 1]
BINARY_CANONICALS_V6 = [c for c in CANONICALS if CANONICAL_ARITY[c] == 2]

# Backward-compatible aliases (for v3/v4/v5 scopes).
ORIGINAL_5_CANONICALS = ["and", "or", "not", "implies", "necessarily"]
NEW_5_CANONICALS_V5 = ["xor", "nand", "possibly", "always", "negate"]
UNARY_CANONICALS_5 = ["not", "necessarily"]
BINARY_CANONICALS_5 = ["and", "or", "implies"]
UNARY_CANONICALS_10 = ["not", "necessarily", "possibly", "always", "negate"]
BINARY_CANONICALS_10 = ["and", "or", "implies", "xor", "nand"]


# ==============================================================================
# Pre-registered tokenization profile expectations (per
# experiments/preregistration_v6.md §1)
# ==============================================================================
@dataclass
class CanonicalProfile:
    target_subword_count: str  # "1pc", "2-3pc", "multi-pc", or "any"
    target_freq: str           # "high", "mid", "low", "very-low", or "any"


PROFILE: dict[str, CanonicalProfile] = {
    "and":         CanonicalProfile("1pc", "high"),
    "or":          CanonicalProfile("1pc", "high"),
    "implies":     CanonicalProfile("1pc", "mid"),
    "xor":         CanonicalProfile("2-3pc", "very-low"),
    "nand":        CanonicalProfile("2-3pc", "very-low"),
    "not":         CanonicalProfile("1pc", "high"),
    "necessarily": CanonicalProfile("1pc", "mid"),
    "possibly":    CanonicalProfile("1pc", "high"),
    "always":      CanonicalProfile("1pc", "high"),
    "negate":      CanonicalProfile("2-3pc", "low"),
    # v6 additions:
    "nor":         CanonicalProfile("1pc", "low"),
    "iff":         CanonicalProfile("2-3pc", "very-low"),
    "unless":      CanonicalProfile("1pc", "mid"),
    "definitely":  CanonicalProfile("1pc", "mid"),
    "unprovably":  CanonicalProfile("multi-pc", "very-low"),
}


def _subword_bucket(n_pieces: int) -> str:
    if n_pieces == 1:
        return "1pc"
    if n_pieces in (2, 3):
        return "2-3pc"
    return "multi-pc"


# ==============================================================================
# Invented set (unchanged from v4/v5)
# ==============================================================================
INVENTED_16 = [
    "bliq", "dren", "molex",
    "krev", "sond", "glin", "twiv", "fump",
    "vusp", "perph",
    "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",
]
INVENTED_5 = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL_16 = {
    "bliq": "and", "dren": "or", "molex": "implies",
    "krev": "and", "sond": "or", "glin": "implies",
    "twiv": "and", "fump": "or",
    "vusp": "not", "perph": "necessarily",
    "kelm": "not", "zorf": "not",
    "gleph": "necessarily", "drelth": "necessarily",
    "vrith": "not", "nilph": "necessarily",
}


# ==============================================================================
# Anchors + verdict thresholds (mirror 22b/23)
# ==============================================================================
ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]

GATE_CANONICAL_PASS = 0.65
GATE_ARITY_PASS = 0.65
M4B_PASS = 0.65
M4C_DISTRIBUTED = 0.70
LUCKY_DEFAULT_PWMIN = 0.95


# ==============================================================================
# Pre-registered falsification thresholds (per preregistration_v6.md §7)
# ==============================================================================
PFREQ_NEW_LF_SINGLE_THRESHOLD = 0.10    # P_FREQ.1-3: nor/unprovably each >= 10%
PFREQ_MF_CONTROL_THRESHOLD = 0.05       # P_FREQ.4-5: unless/definitely each <= 5%
PFREQ_AGGREGATE_THRESHOLD = 0.35        # P_FREQ.6: sum nor+iff+unprovably >= 35%
PSUBWORD_MULTI_THRESHOLD = 0.15         # P_SUBWORD.1-2: iff/unprovably each >= 15%
PSUBWORD_SINGLE_LF_THRESHOLD = 0.05     # P_SUBWORD.3: nor <= 5%
PSUBWORD_AGGREGATE_MULTI = 0.70         # P_SUBWORD.4: all multi-pc together >= 70%
PSUBWORD_AGGREGATE_SINGLE = 0.30        # P_SUBWORD.5: all single-pc together <= 30%
PINT_INTERMEDIATE_LO = 0.05             # P_INT.2: nor in [5, 15] range
PINT_INTERMEDIATE_HI = 0.15


# ==============================================================================
# Model specs (Gemma first because it's largest; clear memory in between)
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


# ==============================================================================
# Stable seeding (same protocol as 22d/23 v2)
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


# ==============================================================================
# Held-out templates (syntactically disjoint from script 19b's templates)
# ==============================================================================
NEUTRAL_TEMPLATES_HELDOUT = [
    "Why does the symbol {op} occur here?",
    "Is {op} a word you recognize?",
    "Whether {op} fits is unclear.",
    "{op} is a token in this passage.",
    "When {op} is read, its meaning emerges.",
    "How shall we interpret {op}?",
    "Among the symbols, {op} stands apart.",
    "The reader encounters {op} mid-sentence.",
    "Take note of {op} as it appears.",
    "Does {op} carry a special meaning?",
    "Asked about {op}, the student paused.",
    "From the page, {op} jumps to the eye.",
    "Once {op} is spotted, attention shifts.",
    "Encoded as {op}, the meaning persists.",
    "Whatever {op} denotes, it is recorded.",
    "{op} sits between two punctuation marks.",
    "After typing {op}, she paused thoughtfully.",
    "Before {op} appears, the context is set.",
    "Beside the figure, {op} is annotated.",
    "{op} was the answer to question seven.",
    "Just print {op} and continue.",
    "Across the document, {op} occurs eight times.",
    "Until {op} is defined, the script will not run.",
    "Hardly anyone reads {op} aloud.",
    "Indeed, {op} is the focal token.",
    "Whether intentional or not, {op} is present.",
    "If {op} appears, the parser logs it.",
    "Said the lecturer, consider {op}.",
    "Were {op} omitted, the sentence would change.",
    "Each occurrence of {op} is counted.",
    "{op}, a token of interest, is listed.",
    "Recently, {op} entered the lexicon.",
    "{op} (in italics) was emphasized.",
    "Several texts include {op}.",
    "{op} appears verbatim in line forty-two.",
    "By convention, {op} is enclosed in quotes.",
    "Surprisingly, {op} is recognized by the system.",
    "{op} the topic was discussed.",
    "Apparently, {op} is significant.",
    "{op} was emphasized through underlining.",
    "Pages cited contain {op} multiple times.",
    "Marked in yellow, {op} is hard to miss.",
    "{op} read aloud sounds strange.",
    "Until further notice, {op} stays in the document.",
    "Whenever {op} is parsed, an event fires.",
    "{op}, despite its brevity, conveys meaning.",
    "{op} has been marked for review.",
    "{op}, a curious term, is logged.",
    "Hence {op} is annotated below.",
    "{op} in bold starts the line.",
]
assert len(NEUTRAL_TEMPLATES_HELDOUT) >= N_PER_CLASS

FUNCTIONAL_TEMPLATE_FRAMES_HELDOUT = [
    "Within the codebase, {call} returns its result.",
    "Whenever {call} executes, the output is logged.",
    "Should {call} be called, the program proceeds.",
    "By invoking {call}, the developer obtains a value.",
    "Among the helpers, {call} is widely used.",
    "Once {call} returns, the caller continues.",
    "Through {call}, the system computes truth values.",
    "Until {call} completes, the thread blocks.",
    "Apparently, {call} is the principal predicate.",
    "Across modules, {call} is referenced extensively.",
]


# ==============================================================================
# Load 19b module and monkey-patch for v6
# ==============================================================================
def _load_module(filename: str, alias: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    loader = importlib.machinery.SourceFileLoader(alias, path)
    spec = importlib.util.spec_from_loader(alias, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    loader.exec_module(mod)
    return mod


_M19B = _load_module("19b_directional_angle_gated.py", "_m19b_24")
_M19B.CANONICALS = CANONICALS
_M19B.CANONICAL_ARITY = CANONICAL_ARITY
_M19B.UNARY_CANONICALS = UNARY_CANONICALS_V6
_M19B.BINARY_CANONICALS = BINARY_CANONICALS_V6
_M19B.INVENTED_WORDS = INVENTED_16
_M19B.W_TO_CANONICAL = W_TO_CANONICAL_16
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
make_functional_invented_stimuli = _M19B.make_functional_invented_stimuli
assert _M19B.SEED == SEED
assert _M19B.N_PER_CLASS == N_PER_CLASS

_M21 = _load_module("21_multi_anchor_battery.py", "_m21_24")
extract_multi_anchor_activations = _M21.extract_multi_anchor_activations
ConditionMultiAnchor = _M21.ConditionMultiAnchor


# ==============================================================================
# Held-out stimulus generators (parallel to 19b's generators)
# ==============================================================================
def make_neutral_stimuli_heldout(op: str, rng: random.Random, n: int) -> list[str]:
    templates = NEUTRAL_TEMPLATES_HELDOUT[:]
    rng.shuffle(templates)
    return [templates[i % len(templates)].format(op=op) for i in range(n)]


def _build_functional_stimulus(op: str, frame: str, p: str, q: Optional[str]) -> str:
    if q is None:
        call = f"{op}({p})"
    else:
        call = f"{op}({p}, {q})"
    return frame.format(call=call)


def make_functional_stimuli_heldout(op: str, rng: random.Random, n: int, *,
                                    arity: int) -> list[str]:
    vars_ = ["p", "q", "r", "s", "x", "y"]
    frames = FUNCTIONAL_TEMPLATE_FRAMES_HELDOUT[:]
    stimuli: list[str] = []
    for _ in range(n):
        frame = rng.choice(frames)
        if arity == 1:
            arg_p = rng.choice(vars_)
            stimuli.append(_build_functional_stimulus(op, frame, arg_p, None))
        else:
            arg_p, arg_q = rng.sample(vars_, 2)
            stimuli.append(_build_functional_stimulus(op, frame, arg_p, arg_q))
    return stimuli


# ==============================================================================
# Generate stimuli for carryover + heldout
# ==============================================================================
def _generate_prompts_carryover(
    condition: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Generate v6 canonical + v6 invented prompts (carryover templates)."""
    if condition == "NEUTRAL":
        canonical_fn = make_neutral_stimuli
        invented_fn = make_neutral_stimuli
    elif condition == "FUNC-PFX":
        canonical_fn = make_functional_canonical_stimuli
        invented_fn = make_functional_invented_stimuli
    else:
        raise ValueError(condition)

    canon_prompts: list[str] = []
    canon_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(stable_seed("v6", condition, "canon", op))
        canon_prompts.extend(canonical_fn(op, op_rng, N_PER_CLASS))
        canon_labels.extend([op] * N_PER_CLASS)

    inv_prompts: list[str] = []
    inv_words: list[str] = []
    for w in INVENTED_16:
        w_rng = random.Random(stable_seed("v6", condition, "inv", w))
        inv_prompts.extend(invented_fn(w, w_rng, N_PER_CLASS))
        inv_words.extend([w] * N_PER_CLASS)

    return canon_prompts, canon_labels, inv_prompts, inv_words


def _generate_prompts_heldout(
    condition: str,
) -> tuple[list[str], list[str]]:
    """Generate v6 canonical-only prompts on held-out templates (no
    invented words; M1heldout is canonical-only validation)."""
    canon_prompts: list[str] = []
    canon_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(stable_seed("v6", "heldout", condition, "canon", op))
        if condition == "NEUTRAL":
            canon_prompts.extend(make_neutral_stimuli_heldout(op, op_rng, N_PER_CLASS))
        elif condition == "FUNC-PFX":
            arity = CANONICAL_ARITY[op]
            canon_prompts.extend(
                make_functional_stimuli_heldout(op, op_rng, N_PER_CLASS, arity=arity)
            )
        else:
            raise ValueError(condition)
        canon_labels.extend([op] * N_PER_CLASS)
    return canon_prompts, canon_labels


# ==============================================================================
# Cache (carryover + heldout per (model, condition))
# ==============================================================================
def _cache_path(model_short_name: str, condition_name: str, kind: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    os.makedirs(base, exist_ok=True)
    slug = model_short_name.replace(" ", "_")
    suffix = "v6-expanded-canonical" if kind == "carryover" else "v6-heldout"
    return os.path.join(
        base,
        f"24_{slug}_{condition_name}_npc{N_PER_CLASS}_{suffix}.npz",
    )


def _cache_save_carryover(
    path: str, cond: ConditionMultiAnchor, *,
    model_id: str, condition_name: str,
    canon_prompts_hash: str, inv_prompts_hash: str,
    dtype_before_cache: str,
) -> None:
    np.savez_compressed(
        path,
        canonical_X=cond.canonical_X.astype(np.float16),
        canonical_labels=cond.canonical_labels,
        invented_X=cond.invented_X.astype(np.float16),
        invented_word_per_stim=cond.invented_word_per_stim,
        anchor_names=np.array(cond.anchor_names),
        canon_anchor_positions=np.array(cond.canon_anchor_positions),
        inv_anchor_positions=np.array(cond.inv_anchor_positions),
        meta_stimulus_version=np.array([STIMULUS_VERSION]),
        meta_model_id=np.array([model_id]),
        meta_condition=np.array([condition_name]),
        meta_canon_prompts_hash=np.array([canon_prompts_hash]),
        meta_inv_prompts_hash=np.array([inv_prompts_hash]),
        meta_dtype_before_cache=np.array([dtype_before_cache]),
        n_per_class=np.array([N_PER_CLASS]),
        canonical_list=np.array(CANONICALS),
        invented_word_list=np.array(INVENTED_16),
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"    [cache] saved {os.path.basename(path)} ({size_mb:.1f} MB)")


def _cache_save_heldout(
    path: str, *, canonical_X: np.ndarray, canonical_labels: np.ndarray,
    anchor_names: list[str], anchor_positions: list[list[int]],
    model_id: str, condition_name: str, canon_prompts_hash: str,
    dtype_before_cache: str,
) -> None:
    np.savez_compressed(
        path,
        canonical_X=canonical_X.astype(np.float16),
        canonical_labels=canonical_labels,
        anchor_names=np.array(anchor_names),
        canon_anchor_positions=np.array(anchor_positions),
        meta_stimulus_version=np.array(["v6-heldout"]),
        meta_model_id=np.array([model_id]),
        meta_condition=np.array([condition_name]),
        meta_canon_prompts_hash=np.array([canon_prompts_hash]),
        meta_dtype_before_cache=np.array([dtype_before_cache]),
        n_per_class=np.array([N_PER_CLASS]),
        canonical_list=np.array(CANONICALS),
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"    [cache] saved {os.path.basename(path)} ({size_mb:.1f} MB)")


def _cache_load_carryover(
    path: str, *,
    expected_canon_hash: str, expected_inv_hash: str,
    expected_anchors: list[str],
) -> Optional[ConditionMultiAnchor]:
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        required = [
            "n_per_class", "meta_stimulus_version", "anchor_names",
            "meta_canon_prompts_hash", "meta_inv_prompts_hash",
            "canonical_list", "invented_word_list",
        ]
        for k in required:
            if k not in z.files:
                return None
        if int(z["n_per_class"][0]) != N_PER_CLASS:
            return None
        if str(z["meta_stimulus_version"][0]) != STIMULUS_VERSION:
            return None
        if str(z["meta_canon_prompts_hash"][0]) != expected_canon_hash:
            return None
        if str(z["meta_inv_prompts_hash"][0]) != expected_inv_hash:
            return None
        if list(z["anchor_names"]) != expected_anchors:
            return None
        if list(z["canonical_list"]) != CANONICALS:
            return None
        if list(z["invented_word_list"]) != INVENTED_16:
            return None
        cond = ConditionMultiAnchor(
            canonical_X=z["canonical_X"].astype(np.float32),
            canonical_labels=z["canonical_labels"],
            invented_X=z["invented_X"].astype(np.float32),
            invented_word_per_stim=z["invented_word_per_stim"],
            anchor_names=list(z["anchor_names"]),
            canon_anchor_positions=z["canon_anchor_positions"].tolist(),
            inv_anchor_positions=z["inv_anchor_positions"].tolist(),
        )
        print(f"    [cache] hit {os.path.basename(path)} "
              f"canon={cond.canonical_X.shape}, inv={cond.invented_X.shape}")
        return cond
    except Exception as e:
        print(f"    [cache] failed to load {path}: {e}")
        return None


def _cache_load_heldout(
    path: str, *, expected_canon_hash: str, expected_anchors: list[str],
) -> Optional[tuple[np.ndarray, np.ndarray, list[str], list[list[int]]]]:
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["meta_stimulus_version"][0]) != "v6-heldout":
            return None
        if str(z["meta_canon_prompts_hash"][0]) != expected_canon_hash:
            return None
        if list(z["anchor_names"]) != expected_anchors:
            return None
        if list(z["canonical_list"]) != CANONICALS:
            return None
        print(f"    [cache] hit {os.path.basename(path)} "
              f"canon={z['canonical_X'].shape}")
        return (
            z["canonical_X"].astype(np.float32),
            z["canonical_labels"],
            list(z["anchor_names"]),
            z["canon_anchor_positions"].tolist(),
        )
    except Exception as e:
        print(f"    [cache] failed to load heldout {path}: {e}")
        return None


# ==============================================================================
# Tokenization audit -- gates which canonicals are in-design per model
# ==============================================================================
@dataclass
class TokAuditRow:
    word: str
    subwords: list[str]
    n_pieces: int
    expected_bucket: str
    actual_bucket: str
    in_design: bool


def audit_tokenization(spec: ModelSpec) -> list[TokAuditRow]:
    print()
    print("=" * 120)
    print(f"v6 TOKENIZATION AUDIT - {spec.short_name}")
    print("=" * 120)
    try:
        tok = AutoTokenizer.from_pretrained(spec.model_id)
    except Exception as e:
        print(f"  [tokenizer load failed: {e}]")
        return []
    rows: list[TokAuditRow] = []
    print(f"  {'word':<14} {'arity':<3} {'subwords':<54} {'n':>3}  "
          f"{'target':<10} {'actual':<10} {'flag':<16}")
    for c in CANONICALS:
        ids = tok.encode(" " + c, add_special_tokens=False)
        subs = [tok.decode([i]) for i in ids]
        actual_bucket = _subword_bucket(len(ids))
        prof = PROFILE[c]
        target_bucket = prof.target_subword_count
        if target_bucket == "any":
            in_design = True
        elif target_bucket == "multi-pc":
            in_design = len(ids) >= 2
        elif target_bucket == "2-3pc":
            in_design = 2 <= len(ids) <= 3
        elif target_bucket == "1pc":
            in_design = len(ids) == 1
        else:
            in_design = False
        flag = ""
        if c in NEW_5_CANONICALS_V6:
            flag = "NEW"
            if not in_design:
                flag = "NEW / OUT-OF-DESIGN"
        elif not in_design:
            flag = "carryover-OOD"
        arity_tag = "B" if CANONICAL_ARITY[c] == 2 else "U"
        print(f"  {c:<14} {arity_tag:<3} {str(subs):<54} {len(ids):>3}  "
              f"{target_bucket:<10} {actual_bucket:<10} {flag:<16}")
        rows.append(TokAuditRow(
            word=c, subwords=subs, n_pieces=len(ids),
            expected_bucket=target_bucket, actual_bucket=actual_bucket,
            in_design=in_design,
        ))
    print()
    print(f"  Invented words (target multi-pc):")
    print(f"  {'word':<14} {'subwords':<54} {'n':>3}")
    for w in INVENTED_16:
        ids = tok.encode(" " + w, add_special_tokens=False)
        subs = [tok.decode([i]) for i in ids]
        print(f"  {w:<14} {str(subs):<54} {len(ids):>3}")
    n_new_ood = sum(
        1 for r in rows
        if r.word in NEW_5_CANONICALS_V6 and not r.in_design
    )
    print()
    print(f"  NEW canonicals OUT-OF-DESIGN: {n_new_ood}/5")
    if n_new_ood > 2:
        print(f"  ** WARNING: more than 2 NEW canonicals failed their target "
              f"profile. Per pre-reg §1, results from this model are reported "
              f"as OUT-OF-DESIGN. Disentanglement analysis will be unreliable.**")
    return rows


# ==============================================================================
# Slice + subset helpers
# ==============================================================================
def slice_canonical(cond: ConditionMultiAnchor, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.canonical_X[a_idx, :, layer, :], np.asarray(cond.canonical_labels)


def slice_invented(cond: ConditionMultiAnchor, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.invented_X[a_idx, :, layer, :], np.asarray(cond.invented_word_per_stim)


def subset_canonical(
    X: np.ndarray, y: np.ndarray, keep: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, keep)
    return X[mask], y[mask]


def subset_invented(
    X: np.ndarray, w: np.ndarray, keep: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(w, keep)
    return X[mask], w[mask]


# ==============================================================================
# Metric primitives
# ==============================================================================
def m1_cv(X: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    for tr_idx, te_idx in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X[tr_idx], y[tr_idx])
        accs.append(clf.score(X[te_idx], y[te_idx]))
    return float(np.mean(accs))


def m1_heldout(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
) -> float:
    """Train probe on carryover templates; evaluate on held-out templates.
    Template-leakage diagnostic: a CV-1.000 within-condition probe should
    still classify above some defensible floor on held-out templates if it
    is reading structural information rather than template lexical bias."""
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)
    return float(clf.score(X_eval, y_eval))


def m2_metrics(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> tuple[float, float, np.ndarray, LogisticRegression]:
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)
    preds = clf.predict(X_test)
    m2_cano = float(np.mean(preds == y_test))
    n_ar = sum(
        CANONICAL_ARITY[str(t)] == CANONICAL_ARITY[str(p)]
        for t, p in zip(y_test, preds)
    )
    m2_arity = n_ar / len(y_test)
    return m2_cano, m2_arity, preds, clf


@dataclass
class M4Result:
    m4a: float
    m4b: float
    m4c: float
    breakdown_pct: dict[str, float]
    per_word_top: dict[str, str]
    per_word_top_pct: dict[str, float]
    per_word_unary_pct: dict[str, float]


def m4_breakdown(
    clf: LogisticRegression, X_inv: np.ndarray, words: np.ndarray,
    *, canonicals: list[str], unary_canonicals: list[str],
    invented_set: list[str], w_to_canonical: dict[str, str],
) -> M4Result:
    preds = clf.predict(X_inv)
    n_total = len(preds)
    breakdown_pct = {c: float(np.mean(preds == c)) for c in canonicals}
    m4a = sum(breakdown_pct[c] for c in unary_canonicals)
    m4c = float(sum(v ** 2 for v in breakdown_pct.values()))

    n_match = 0
    for w, p in zip(words, preds):
        intended_can = w_to_canonical[str(w)]
        if CANONICAL_ARITY[intended_can] == CANONICAL_ARITY[str(p)]:
            n_match += 1
    m4b = n_match / n_total

    per_word_top: dict[str, str] = {}
    per_word_top_pct: dict[str, float] = {}
    per_word_unary_pct: dict[str, float] = {}
    for w in invented_set:
        mask = (words == w)
        if not mask.any():
            continue
        word_preds = preds[mask]
        counts = {c: int(np.sum(word_preds == c)) for c in canonicals}
        top_c = max(counts, key=lambda c: counts[c])
        per_word_top[w] = top_c
        per_word_top_pct[w] = counts[top_c] / mask.sum()
        per_word_unary_pct[w] = sum(counts[c] for c in unary_canonicals) / mask.sum()

    return M4Result(
        m4a=m4a, m4b=m4b, m4c=m4c, breakdown_pct=breakdown_pct,
        per_word_top=per_word_top, per_word_top_pct=per_word_top_pct,
        per_word_unary_pct=per_word_unary_pct,
    )


def bootstrap_m4b(
    clf: LogisticRegression, X_inv: np.ndarray, words: np.ndarray,
    *, invented_set: list[str], w_to_canonical: dict[str, str],
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, float]:
    preds = clf.predict(X_inv)
    word_to_idx = {w: np.where(words == w)[0] for w in invented_set}
    m4b_samples = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_bootstrap):
        idx_list = []
        for w in invented_set:
            w_idx = word_to_idx[w]
            if len(w_idx) == 0:
                continue
            idx_list.append(rng.choice(w_idx, size=len(w_idx), replace=True))
        all_idx = np.concatenate(idx_list)
        sub_preds = preds[all_idx]
        sub_words = words[all_idx]
        n_match = sum(
            CANONICAL_ARITY[w_to_canonical[str(w)]] == CANONICAL_ARITY[str(p)]
            for w, p in zip(sub_words, sub_preds)
        )
        m4b_samples.append(n_match / len(sub_preds))
    arr = np.array(m4b_samples)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p_pass_065": float(np.mean(arr >= M4B_PASS)),
    }


def bootstrap_m2_canonical(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *, n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, float]:
    """Resample the TRAINING set (with replacement, stratified) and
    refit the probe each time. Same protocol as 22a."""
    samples: list[float] = []
    rng = np.random.default_rng(SEED)
    unique_y = np.unique(y_train)
    class_idx = {c: np.where(y_train == c)[0] for c in unique_y}
    for _ in range(n_bootstrap):
        idx_list: list[np.ndarray] = []
        for c in unique_y:
            ci = class_idx[c]
            if len(ci) == 0:
                continue
            idx_list.append(rng.choice(ci, size=len(ci), replace=True))
        all_idx = np.concatenate(idx_list)
        X_sub = X_train[all_idx]
        y_sub = y_train[all_idx]
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_sub, y_sub)
        samples.append(float(np.mean(clf.predict(X_test) == y_test)))
    arr = np.array(samples)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p_pass_065": float(np.mean(arr >= GATE_CANONICAL_PASS)),
    }


def detect_lucky_default(per_word_top_pct: dict[str, float]) -> bool:
    pcts = list(per_word_top_pct.values())
    if not pcts:
        return False
    return min(pcts) >= LUCKY_DEFAULT_PWMIN


# ==============================================================================
# SweepCell + Scope
# ==============================================================================
@dataclass
class SweepCell:
    scope: str
    direction: str
    train_cond: str
    train_anchor: str
    test_cond: str
    test_anchor: str
    layer: int

    M1_tr: float = 0.0
    M1_te: float = 0.0
    M2_cano: float = 0.0
    M2_arity: float = 0.0
    M4a: float = 0.0
    M4b: float = 0.0
    M4c: float = 0.0
    per_word_min_top_pct: float = 0.0
    per_word_max_top_pct: float = 0.0
    per_word_top: dict[str, str] = field(default_factory=dict)
    per_word_top_pct: dict[str, float] = field(default_factory=dict)
    breakdown_pct: dict[str, float] = field(default_factory=dict)
    lucky_default: bool = False

    @property
    def m2_gap(self) -> float:
        return self.M2_arity - self.M2_cano

    @property
    def verdict(self) -> str:
        if self.lucky_default:
            return "LUCKY-NEG"
        passes = (
            self.M2_arity >= GATE_ARITY_PASS
            and self.M4b >= M4B_PASS
            and self.M4c < M4C_DISTRIBUTED
            and 0.10 <= self.M4a <= 0.90
        )
        if passes:
            return "PASS-arity"
        if self.M2_arity >= GATE_ARITY_PASS and self.M4b >= M4B_PASS:
            return "ARITY-AXIS-ONLY"
        if self.M2_arity >= GATE_ARITY_PASS:
            return "M2A-ONLY"
        if self.M2_cano >= GATE_CANONICAL_PASS:
            return "M2C-ONLY"
        return "FAIL"


@dataclass
class Scope:
    name: str
    canonicals: list[str]
    unary_canonicals: list[str]
    binary_canonicals: list[str]
    invented_set: list[str]
    w_to_canonical: dict[str, str]


SCOPE_V3 = Scope(
    name="v3", canonicals=ORIGINAL_5_CANONICALS,
    unary_canonicals=UNARY_CANONICALS_5, binary_canonicals=BINARY_CANONICALS_5,
    invented_set=INVENTED_5,
    w_to_canonical={w: W_TO_CANONICAL_16[w] for w in INVENTED_5},
)
SCOPE_V4 = Scope(
    name="v4", canonicals=ORIGINAL_5_CANONICALS,
    unary_canonicals=UNARY_CANONICALS_5, binary_canonicals=BINARY_CANONICALS_5,
    invented_set=INVENTED_16,
    w_to_canonical=W_TO_CANONICAL_16,
)
SCOPE_V5 = Scope(
    name="v5", canonicals=CANONICALS_V5,
    unary_canonicals=UNARY_CANONICALS_10, binary_canonicals=BINARY_CANONICALS_10,
    invented_set=INVENTED_16,
    w_to_canonical=W_TO_CANONICAL_16,
)
SCOPE_V6 = Scope(
    name="v6", canonicals=CANONICALS,
    unary_canonicals=UNARY_CANONICALS_V6, binary_canonicals=BINARY_CANONICALS_V6,
    invented_set=INVENTED_16,
    w_to_canonical=W_TO_CANONICAL_16,
)
ALL_SCOPES = [SCOPE_V3, SCOPE_V4, SCOPE_V5, SCOPE_V6]


def enumerate_cells(scope: str, layers: list[int]) -> list[SweepCell]:
    cells: list[SweepCell] = []
    for L in layers:
        for tr_a in ANCHORS_NEUTRAL:
            for te_a in ANCHORS_FUNC_PFX:
                cells.append(SweepCell(
                    scope=scope, direction="N->F",
                    train_cond="NEUTRAL", train_anchor=tr_a,
                    test_cond="FUNC-PFX", test_anchor=te_a, layer=L,
                ))
        for tr_a in ANCHORS_FUNC_PFX:
            for te_a in ANCHORS_NEUTRAL:
                cells.append(SweepCell(
                    scope=scope, direction="F->N",
                    train_cond="FUNC-PFX", train_anchor=tr_a,
                    test_cond="NEUTRAL", test_anchor=te_a, layer=L,
                ))
    return cells


def run_cell(
    cell: SweepCell, cond_by_name: dict[str, ConditionMultiAnchor], scope: Scope,
) -> SweepCell:
    train_cond = cond_by_name[cell.train_cond]
    test_cond = cond_by_name[cell.test_cond]

    X_tr_full, y_tr_full = slice_canonical(train_cond, cell.train_anchor, cell.layer)
    X_te_full, y_te_full = slice_canonical(test_cond, cell.test_anchor, cell.layer)
    X_inv_full, w_inv_full = slice_invented(test_cond, cell.test_anchor, cell.layer)

    X_tr, y_tr = subset_canonical(X_tr_full, y_tr_full, scope.canonicals)
    X_te, y_te = subset_canonical(X_te_full, y_te_full, scope.canonicals)
    X_inv, w_inv = subset_invented(X_inv_full, w_inv_full, scope.invented_set)

    cell.M1_tr = m1_cv(X_tr, y_tr)
    cell.M1_te = m1_cv(X_te, y_te)
    cell.M2_cano, cell.M2_arity, _, clf = m2_metrics(X_tr, y_tr, X_te, y_te)

    m4 = m4_breakdown(
        clf, X_inv, w_inv,
        canonicals=scope.canonicals, unary_canonicals=scope.unary_canonicals,
        invented_set=scope.invented_set, w_to_canonical=scope.w_to_canonical,
    )
    cell.M4a = m4.m4a
    cell.M4b = m4.m4b
    cell.M4c = m4.m4c
    cell.per_word_top = m4.per_word_top
    cell.per_word_top_pct = m4.per_word_top_pct
    cell.breakdown_pct = m4.breakdown_pct
    pcts = list(m4.per_word_top_pct.values())
    cell.per_word_min_top_pct = float(min(pcts)) if pcts else 0.0
    cell.per_word_max_top_pct = float(max(pcts)) if pcts else 0.0
    cell.lucky_default = detect_lucky_default(m4.per_word_top_pct)
    return cell


# ==============================================================================
# Reporting
# ==============================================================================
def _cell_short(c: SweepCell) -> str:
    return (
        f"{c.scope} {c.direction} {c.train_anchor[:5]:>5}->{c.test_anchor[:5]:<5} "
        f"L{c.layer:>2}"
    )


def print_sweep_table(cells: list[SweepCell], model_short_name: str, scope_name: str) -> None:
    print()
    print("=" * 200)
    print(f"  {model_short_name} - {scope_name} scope - full sweep ({len(cells)} cells)")
    print("=" * 200)
    h = (
        f"  {'cell':<32} | {'M1tr':<5} | {'M1te':<5} | {'M2cano':<7} | "
        f"{'M2arty':<7} | {'gap':<6} | {'M4a':<6} | {'M4b':<6} | "
        f"{'M4c':<5} | {'pwmin':<5} | {'verdict':<14}"
    )
    print(h)
    print(f"  {'-' * (len(h) - 2)}")
    for c in cells:
        flag = " *L" if c.lucky_default else ""
        print(
            f"  {_cell_short(c):<32} | "
            f"{c.M1_tr:.2f}  | {c.M1_te:.2f}  | "
            f"{c.M2_cano:.3f}   | {c.M2_arity:.3f}   | "
            f"{c.m2_gap:+.3f} | "
            f"{c.M4a*100:>4.1f}% | {c.M4b*100:>4.1f}% | "
            f"{c.M4c:.2f}  | {c.per_word_min_top_pct:.2f}  | "
            f"{c.verdict + flag:<14}"
        )


def print_top_k(cells: list[SweepCell], scope_name: str, k: int = 8) -> list[SweepCell]:
    eligible = [c for c in cells if not c.lucky_default]
    sorted_cells = sorted(
        eligible,
        key=lambda c: (c.M2_arity, c.M4b, -c.M4c), reverse=True,
    )[:k]
    print()
    print(f"  TOP-{k} cells under {scope_name} (sort: M2-arity desc, M4b desc, M4c asc):")
    for i, c in enumerate(sorted_cells):
        print(
            f"    {i+1:>2}. {_cell_short(c):<32} | M2c={c.M2_cano:.3f} | "
            f"M2a={c.M2_arity:.3f} | M4b={c.M4b*100:>5.1f}% | "
            f"M4c={c.M4c:.2f} | M4a={c.M4a*100:>5.1f}% | verdict={c.verdict}"
        )
    return sorted_cells


def print_canonical_breakdown(
    cells: list[SweepCell], scope: Scope, *, top_cell_count: int = 3,
) -> None:
    sorted_by_m2c = sorted(cells, key=lambda c: -c.M2_cano)[:top_cell_count]
    print()
    print(f"  Canonical-readout breakdown of invented mass at the top-"
          f"{top_cell_count} M2-canonical cells under {scope.name}:")
    for c in sorted_by_m2c:
        print()
        print(f"    >> {_cell_short(c)} (M2c={c.M2_cano:.3f}, M4b={c.M4b:.3f}, "
              f"M4c={c.M4c:.2f})")
        for w in scope.invented_set:
            if w not in c.per_word_top:
                continue
            intended_can = scope.w_to_canonical[w]
            intended_arity_tag = "B" if CANONICAL_ARITY[intended_can] == 2 else "U"
            top_c = c.per_word_top[w]
            top_pct = c.per_word_top_pct[w]
            match = (CANONICAL_ARITY[top_c] == CANONICAL_ARITY[intended_can]
                     if top_c in CANONICAL_ARITY else False)
            print(f"       {w:<10} (int.{intended_arity_tag}: {intended_can:<11}) "
                  f"-> {top_c:<13} {top_pct*100:5.1f}%   "
                  f"{'arity-match' if match else 'arity-MISMATCH'}")


# ==============================================================================
# Aggregate per-word top distribution + adjudication
# ==============================================================================
@dataclass
class AggregateDistribution:
    counts: dict[str, int]
    total: int

    def pct(self, canonical: str) -> float:
        return (self.counts.get(canonical, 0) / self.total) if self.total else 0.0


def aggregate_per_word_tops(cells: list[SweepCell]) -> AggregateDistribution:
    counts: dict[str, int] = {c: 0 for c in CANONICALS}
    for cell in cells:
        for _w, top_c in cell.per_word_top.items():
            if top_c in counts:
                counts[top_c] += 1
    total = sum(counts.values())
    return AggregateDistribution(counts=counts, total=total)


def adjudicate_predictions(
    agg: AggregateDistribution, model_name: str, audit: list[TokAuditRow],
) -> dict[str, object]:
    """Test pre-registered P_FREQ, P_SUBWORD, P_INTERACTION."""
    nor_pct = agg.pct("nor")
    iff_pct = agg.pct("iff")
    unless_pct = agg.pct("unless")
    definitely_pct = agg.pct("definitely")
    unprovably_pct = agg.pct("unprovably")
    aggregate_new_lf_single = nor_pct  # only 1 candidate in this slot
    aggregate_new_lf_total = nor_pct + iff_pct + unprovably_pct
    multi_pc_canonicals = [r.word for r in audit if r.n_pieces >= 2]
    single_pc_canonicals = [r.word for r in audit if r.n_pieces == 1]
    multi_pc_total = sum(agg.pct(c) for c in multi_pc_canonicals)
    single_pc_total = sum(agg.pct(c) for c in single_pc_canonicals)

    pfreq_1 = nor_pct >= PFREQ_NEW_LF_SINGLE_THRESHOLD
    pfreq_2 = iff_pct >= PFREQ_NEW_LF_SINGLE_THRESHOLD
    pfreq_3 = unprovably_pct >= PFREQ_NEW_LF_SINGLE_THRESHOLD
    pfreq_4 = unless_pct <= PFREQ_MF_CONTROL_THRESHOLD
    pfreq_5 = definitely_pct <= PFREQ_MF_CONTROL_THRESHOLD
    pfreq_6 = aggregate_new_lf_total >= PFREQ_AGGREGATE_THRESHOLD
    pfreq_pass = all([pfreq_1, pfreq_2, pfreq_3, pfreq_4, pfreq_5, pfreq_6])

    psubword_1 = iff_pct >= PSUBWORD_MULTI_THRESHOLD
    psubword_2 = unprovably_pct >= PSUBWORD_MULTI_THRESHOLD
    psubword_3 = nor_pct <= PSUBWORD_SINGLE_LF_THRESHOLD
    psubword_4 = multi_pc_total >= PSUBWORD_AGGREGATE_MULTI
    psubword_5 = single_pc_total <= PSUBWORD_AGGREGATE_SINGLE
    psubword_pass = all([psubword_1, psubword_2, psubword_4, psubword_5])
    psubword_falsifies_freq = psubword_3 and not pfreq_1

    pint_1 = aggregate_new_lf_total > (unless_pct + definitely_pct)
    pint_2 = PINT_INTERMEDIATE_LO < nor_pct < PINT_INTERMEDIATE_HI
    pint_3 = unless_pct <= PFREQ_MF_CONTROL_THRESHOLD and definitely_pct <= PFREQ_MF_CONTROL_THRESHOLD
    pint_pass = all([pint_1, pint_2, pint_3])

    if pfreq_pass:
        verdict = "P_FREQ supported"
    elif psubword_pass and psubword_falsifies_freq:
        verdict = "P_SUBWORD supported (P_FREQ falsified)"
    elif pint_pass:
        verdict = "P_INTERACTION supported"
    elif unless_pct >= 0.10 or definitely_pct >= 0.10:
        verdict = "NONE -- MF controls destabilized"
    else:
        verdict = "NONE -- no prediction met cleanly"

    n_new_ood = sum(1 for r in audit if r.word in NEW_5_CANONICALS_V6 and not r.in_design)
    if n_new_ood > 2:
        verdict = f"OUT-OF-DESIGN ({n_new_ood} new canonicals failed audit) -- {verdict}"

    return {
        "model": model_name,
        "nor_pct": nor_pct,
        "iff_pct": iff_pct,
        "unless_pct": unless_pct,
        "definitely_pct": definitely_pct,
        "unprovably_pct": unprovably_pct,
        "aggregate_new_lf_total": aggregate_new_lf_total,
        "multi_pc_total": multi_pc_total,
        "single_pc_total": single_pc_total,
        "pfreq": {
            "P_FREQ.1 (nor>=10%)": pfreq_1, "P_FREQ.2 (iff>=10%)": pfreq_2,
            "P_FREQ.3 (unprovably>=10%)": pfreq_3,
            "P_FREQ.4 (unless<=5%)": pfreq_4, "P_FREQ.5 (definitely<=5%)": pfreq_5,
            "P_FREQ.6 (aggregate>=35%)": pfreq_6, "PASS": pfreq_pass,
        },
        "psubword": {
            "P_SUBWORD.1 (iff>=15%)": psubword_1,
            "P_SUBWORD.2 (unprovably>=15%)": psubword_2,
            "P_SUBWORD.3 (nor<=5%)": psubword_3,
            "P_SUBWORD.4 (multi-pc>=70%)": psubword_4,
            "P_SUBWORD.5 (single-pc<=30%)": psubword_5,
            "PASS": psubword_pass,
            "falsifies_pfreq": psubword_falsifies_freq,
        },
        "pint": {
            "P_INT.1 (new-LF > MF controls)": pint_1,
            "P_INT.2 (nor in [5,15]%)": pint_2,
            "P_INT.3 (MF controls hold)": pint_3,
            "PASS": pint_pass,
        },
        "verdict": verdict,
    }


def print_aggregate_distribution(agg: AggregateDistribution, audit: list[TokAuditRow]) -> None:
    bucket_lookup = {r.word: r.actual_bucket for r in audit}
    freq_lookup = {r.word: PROFILE[r.word].target_freq for r in audit}
    print(f"  Aggregate per-word top-canonical (v6, all 80 cells, 16 inv words = 1280 readouts):")
    print(f"  {'canonical':<14} {'arity':<3} {'tok':<8} {'freq':<10} "
          f"{'count':>6}  {'pct':>6}  {'flag':<6}")
    for c in CANONICALS:
        n = agg.counts.get(c, 0)
        pct = agg.pct(c) * 100
        is_new = "NEW" if c in NEW_5_CANONICALS_V6 else ""
        bucket = bucket_lookup.get(c, "?")
        freq = freq_lookup.get(c, "?")
        arity = "B" if CANONICAL_ARITY[c] == 2 else "U"
        print(f"  {c:<14} {arity:<3} {bucket:<8} {freq:<10} "
              f"{n:>6}  {pct:>5.1f}%  {is_new:<6}")
    print()


def print_adjudication(adj: dict[str, object]) -> None:
    print(f"  Adjudication for {adj['model']}:")
    print(f"    nor          ={adj['nor_pct']*100:5.1f}% (target P_FREQ.1 >= 10%)")
    print(f"    iff          ={adj['iff_pct']*100:5.1f}% (target P_FREQ.2 + P_SUBWORD.1)")
    print(f"    unprovably   ={adj['unprovably_pct']*100:5.1f}% (target P_FREQ.3 + P_SUBWORD.2)")
    print(f"    unless       ={adj['unless_pct']*100:5.1f}% (target P_FREQ.4 <= 5%)")
    print(f"    definitely   ={adj['definitely_pct']*100:5.1f}% (target P_FREQ.5 <= 5%)")
    print(f"    aggregate new-LF (nor+iff+unprovably) = {adj['aggregate_new_lf_total']*100:.1f}%")
    print(f"    aggregate multi-pc canonicals = {adj['multi_pc_total']*100:.1f}%")
    print(f"    aggregate single-pc canonicals = {adj['single_pc_total']*100:.1f}%")
    print()
    for block_name, block_key in (("P_FREQ", "pfreq"), ("P_SUBWORD", "psubword"), ("P_INTERACTION", "pint")):
        print(f"    {block_name}:")
        block = adj[block_key]
        for k, v in block.items():
            print(f"      [{'PASS' if v else 'fail'}] {k}")
    print()
    print(f"    VERDICT: {adj['verdict']}")


# ==============================================================================
# Cross-scope retraction chain
# ==============================================================================
def cross_scope_chain(
    cells_by_scope: dict[str, list[SweepCell]], model_name: str,
) -> None:
    """Trace v3-PASS-arity candidates through v4/v5/v6 to see whether
    each retracts under successive expansions. Mirrors §3.7.15 PHASE D
    + extends to v6."""
    print()
    print("=" * 120)
    print(f"  {model_name} - cross-scope retraction chain (v3 PASS-arity candidates)")
    print("=" * 120)
    v3_candidates = [c for c in cells_by_scope["v3"] if c.verdict == "PASS-arity"]
    if not v3_candidates:
        print(f"  No v3-scope PASS-arity cells. Chain is vacuous (this is itself")
        print(f"  informative: model does not produce the surface-level positive at v3).")
        return
    for v3c in v3_candidates:
        same_v4 = _same_cell(cells_by_scope.get("v4", []), v3c)
        same_v5 = _same_cell(cells_by_scope.get("v5", []), v3c)
        same_v6 = _same_cell(cells_by_scope.get("v6", []), v3c)
        print(f"  >> {_cell_short(v3c)}")
        for label, c in (("v3", v3c), ("v4", same_v4), ("v5", same_v5), ("v6", same_v6)):
            if c is None:
                continue
            print(f"     {label}: M2a={c.M2_arity:.3f} M4b={c.M4b:.3f} M4c={c.M4c:.2f} "
                  f"pwmin={c.per_word_min_top_pct:.2f} verdict={c.verdict}")
        survives_v6 = (same_v6 is not None and same_v6.verdict == "PASS-arity")
        survives_v5 = (same_v5 is not None and same_v5.verdict == "PASS-arity")
        survives_v4 = (same_v4 is not None and same_v4.verdict == "PASS-arity")
        if survives_v6:
            tag = "*** SURVIVES v6 *** (operator-set-bound finding RETRACTED for this model)"
        elif survives_v5:
            tag = "survives v5 but retracted by v6"
        elif survives_v4:
            tag = "survives v4 but retracted by v5"
        else:
            tag = "retracted by v4 (16-invented expansion)"
        print(f"     verdict: {tag}")


def _same_cell(cells: list[SweepCell], target: SweepCell) -> Optional[SweepCell]:
    for c in cells:
        if (c.direction == target.direction
                and c.train_anchor == target.train_anchor
                and c.test_anchor == target.test_anchor
                and c.layer == target.layer):
            return c
    return None


# ==============================================================================
# Build conditions (carryover + heldout)
# ==============================================================================
def build_condition_carryover(
    spec: ModelSpec, condition_name: str, anchor_names: list[str],
    model, tok, device: str,
) -> ConditionMultiAnchor:
    canon_prompts, canon_labels, inv_prompts, inv_words = _generate_prompts_carryover(condition_name)
    canon_hash = prompts_checksum(canon_prompts)
    inv_hash = prompts_checksum(inv_prompts)

    path = _cache_path(spec.short_name, condition_name, "carryover")
    cached = _cache_load_carryover(
        path, expected_canon_hash=canon_hash, expected_inv_hash=inv_hash,
        expected_anchors=anchor_names,
    )
    if cached is not None:
        return cached

    assert model is not None, "cache miss but model not loaded"
    print(f"\n  Building carryover {condition_name} for {spec.short_name} "
          f"(anchors={anchor_names}; extracting)")

    print(f"    extracting canonical activations ({len(canon_prompts)} stim)...")
    t0 = time.time()
    canon_X, canon_pos, _ = extract_multi_anchor_activations(
        model, tok, canon_prompts, CANONICALS, anchor_names, condition_name, device
    )
    print(f"      {time.time() - t0:.1f}s, shape={canon_X.shape}")

    print(f"    extracting invented activations ({len(inv_prompts)} stim)...")
    t0 = time.time()
    inv_X, inv_pos, _ = extract_multi_anchor_activations(
        model, tok, inv_prompts, INVENTED_16, anchor_names, condition_name, device
    )
    print(f"      {time.time() - t0:.1f}s, shape={inv_X.shape}")

    cond = ConditionMultiAnchor(
        canonical_X=canon_X,
        canonical_labels=np.array(canon_labels),
        invented_X=inv_X,
        invented_word_per_stim=np.array(inv_words),
        anchor_names=list(anchor_names),
        canon_anchor_positions=canon_pos,
        inv_anchor_positions=inv_pos,
    )
    _cache_save_carryover(
        path, cond, model_id=spec.model_id, condition_name=condition_name,
        canon_prompts_hash=canon_hash, inv_prompts_hash=inv_hash,
        dtype_before_cache=str(spec.dtype).replace("torch.", ""),
    )
    return cond


def build_condition_heldout(
    spec: ModelSpec, condition_name: str, anchor_names: list[str],
    model, tok, device: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[int]]]:
    canon_prompts, canon_labels = _generate_prompts_heldout(condition_name)
    canon_hash = prompts_checksum(canon_prompts)

    path = _cache_path(spec.short_name, condition_name, "heldout")
    cached = _cache_load_heldout(
        path, expected_canon_hash=canon_hash, expected_anchors=anchor_names,
    )
    if cached is not None:
        return cached

    assert model is not None, "heldout cache miss but model not loaded"
    print(f"\n  Building heldout {condition_name} for {spec.short_name} "
          f"(anchors={anchor_names}; extracting)")
    t0 = time.time()
    canon_X, canon_pos, _ = extract_multi_anchor_activations(
        model, tok, canon_prompts, CANONICALS, anchor_names, condition_name, device
    )
    print(f"      {time.time() - t0:.1f}s, shape={canon_X.shape}")

    _cache_save_heldout(
        path, canonical_X=canon_X, canonical_labels=np.array(canon_labels),
        anchor_names=list(anchor_names), anchor_positions=canon_pos,
        model_id=spec.model_id, condition_name=condition_name,
        canon_prompts_hash=canon_hash,
        dtype_before_cache=str(spec.dtype).replace("torch.", ""),
    )
    return canon_X, np.array(canon_labels), list(anchor_names), canon_pos


# ==============================================================================
# M1heldout report (template-leakage sanity check)
# ==============================================================================
def report_m1_heldout(
    spec: ModelSpec,
    carryover: dict[str, ConditionMultiAnchor],
    heldout: dict[str, tuple[np.ndarray, np.ndarray, list[str], list[list[int]]]],
) -> None:
    print()
    print("=" * 120)
    print(f"  {spec.short_name} - M1heldout sanity check (template-leakage diagnostic)")
    print(f"  Train probe on v6 carryover; evaluate on v6 heldout. v6 chance = 1/15 = 0.067.")
    print("=" * 120)
    print(f"  {'condition':<10} {'anchor':<16} {'layer':>5}  {'M1tr (carry)':<14} "
          f"{'M1heldout':<14}  {'gap':<8}")
    for cond_name in ("NEUTRAL", "FUNC-PFX"):
        anchors = ANCHORS_NEUTRAL if cond_name == "NEUTRAL" else ANCHORS_FUNC_PFX
        carry = carryover[cond_name]
        if cond_name not in heldout:
            continue
        ho_X, ho_y, ho_anchors, _ = heldout[cond_name]
        for anchor in anchors:
            a_idx = carry.anchor_names.index(anchor)
            ho_a_idx = ho_anchors.index(anchor)
            for L in spec.focus_layers:
                X_tr = carry.canonical_X[a_idx, :, L, :]
                y_tr = np.asarray(carry.canonical_labels)
                X_ho = ho_X[ho_a_idx, :, L, :]
                y_ho = ho_y
                m1_tr = m1_cv(X_tr, y_tr)
                m1_ho = m1_heldout(X_tr, y_tr, X_ho, y_ho)
                gap = m1_tr - m1_ho
                print(f"  {cond_name:<10} {anchor:<16} {L:>5}  "
                      f"{m1_tr:<14.3f} {m1_ho:<14.3f}  {gap:+.3f}")


# ==============================================================================
# Device + cleanup
# ==============================================================================
def get_device() -> tuple[str, str]:
    if torch.backends.mps.is_available():
        return "mps", "MPS (Apple Silicon)"
    if torch.cuda.is_available():
        return "cuda", "CUDA"
    return "cpu", "CPU"


def free_model(model) -> None:
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# Per-model run
# ==============================================================================
def run_one_model(spec: ModelSpec, device: str) -> dict:
    print()
    print()
    print("#" * 120)
    print(f"#  MODEL: {spec.short_name}  ({spec.model_id})")
    print(f"#  dtype={spec.dtype}  focus_layers={spec.focus_layers}")
    print("#" * 120)

    audit = audit_tokenization(spec)

    # Phase A: caches
    print()
    print("=" * 100)
    print(f"PHASE A: cache extraction ({spec.short_name})")
    print("=" * 100)
    t_phase_a = time.time()

    canon_neut, _, inv_neut, _ = _generate_prompts_carryover("NEUTRAL")
    canon_func, _, inv_func, _ = _generate_prompts_carryover("FUNC-PFX")

    neut_cached = _cache_load_carryover(
        _cache_path(spec.short_name, "NEUTRAL", "carryover"),
        expected_canon_hash=prompts_checksum(canon_neut),
        expected_inv_hash=prompts_checksum(inv_neut),
        expected_anchors=ANCHORS_NEUTRAL,
    )
    func_cached = _cache_load_carryover(
        _cache_path(spec.short_name, "FUNC-PFX", "carryover"),
        expected_canon_hash=prompts_checksum(canon_func),
        expected_inv_hash=prompts_checksum(inv_func),
        expected_anchors=ANCHORS_FUNC_PFX,
    )
    heldout_neut_cached = None
    heldout_func_cached = None
    if not SKIP_HELDOUT:
        ho_neut_prompts, _ = _generate_prompts_heldout("NEUTRAL")
        ho_func_prompts, _ = _generate_prompts_heldout("FUNC-PFX")
        heldout_neut_cached = _cache_load_heldout(
            _cache_path(spec.short_name, "NEUTRAL", "heldout"),
            expected_canon_hash=prompts_checksum(ho_neut_prompts),
            expected_anchors=ANCHORS_NEUTRAL,
        )
        heldout_func_cached = _cache_load_heldout(
            _cache_path(spec.short_name, "FUNC-PFX", "heldout"),
            expected_canon_hash=prompts_checksum(ho_func_prompts),
            expected_anchors=ANCHORS_FUNC_PFX,
        )

    need_model = (
        neut_cached is None or func_cached is None
        or (not SKIP_HELDOUT and (heldout_neut_cached is None or heldout_func_cached is None))
    )
    model = None
    tok = None
    if need_model:
        print(f"\n  Loading model: {spec.model_id}")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(spec.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=spec.dtype, low_cpu_mem_usage=True,
        ).to(device).eval()
        print(f"    loaded in {time.time() - t0:.1f}s "
              f"({model.config.num_hidden_layers} layers, "
              f"hidden_size={model.config.hidden_size})")

    if neut_cached is None:
        neut = build_condition_carryover(spec, "NEUTRAL", ANCHORS_NEUTRAL, model, tok, device)
    else:
        neut = neut_cached
    if func_cached is None:
        func = build_condition_carryover(spec, "FUNC-PFX", ANCHORS_FUNC_PFX, model, tok, device)
    else:
        func = func_cached
    carryover = {"NEUTRAL": neut, "FUNC-PFX": func}

    heldout: dict[str, tuple[np.ndarray, np.ndarray, list[str], list[list[int]]]] = {}
    if not SKIP_HELDOUT:
        if heldout_neut_cached is None:
            heldout["NEUTRAL"] = build_condition_heldout(
                spec, "NEUTRAL", ANCHORS_NEUTRAL, model, tok, device
            )
        else:
            heldout["NEUTRAL"] = heldout_neut_cached
        if heldout_func_cached is None:
            heldout["FUNC-PFX"] = build_condition_heldout(
                spec, "FUNC-PFX", ANCHORS_FUNC_PFX, model, tok, device
            )
        else:
            heldout["FUNC-PFX"] = heldout_func_cached

    if model is not None:
        free_model(model)
    if tok is not None:
        del tok
    gc.collect()
    print(f"\n  -- Phase A total time: {time.time() - t_phase_a:.1f}s --")

    # Phase B: 4-scope sweep
    print()
    print("=" * 100)
    print(f"PHASE B: 4-scope sweep ({spec.short_name})")
    print("=" * 100)
    cells_by_scope: dict[str, list[SweepCell]] = {}
    for scope in ALL_SCOPES:
        t0 = time.time()
        cells = enumerate_cells(scope.name, spec.focus_layers)
        for c in cells:
            run_cell(c, carryover, scope)
        cells_by_scope[scope.name] = cells
        print(f"  scope {scope.name}: {len(cells)} cells in {time.time() - t0:.1f}s")

    # Phase C: M1heldout sanity (only if held-out caches exist)
    if heldout:
        report_m1_heldout(spec, carryover, heldout)

    # Phase D: sweep tables
    print()
    print("=" * 100)
    print(f"PHASE D: per-scope sweep tables ({spec.short_name})")
    print("=" * 100)
    scope_lookup = {s.name: s for s in ALL_SCOPES}
    for scope_name in ("v3", "v4", "v5", "v6"):
        print_sweep_table(cells_by_scope[scope_name], spec.short_name, scope_name)
        print_top_k(cells_by_scope[scope_name], scope_name, k=8)
        print_canonical_breakdown(cells_by_scope[scope_name], scope_lookup[scope_name],
                                  top_cell_count=3)

    # Phase E: cross-scope retraction chain
    cross_scope_chain(cells_by_scope, spec.short_name)

    # Phase F: aggregate distribution + adjudication (v6 only)
    print()
    print("=" * 120)
    print(f"  {spec.short_name} - v6 aggregate per-word distribution + adjudication")
    print("=" * 120)
    agg = aggregate_per_word_tops(cells_by_scope["v6"])
    print_aggregate_distribution(agg, audit)
    adjudication = adjudicate_predictions(agg, spec.short_name, audit)
    print_adjudication(adjudication)

    # Phase G: bootstrap on top-M2c v6 cell + M2-canonical at top-M2c v6 cell
    print()
    print("=" * 120)
    print(f"  {spec.short_name} - bootstrap CIs (top-M2c v6 cell)")
    print("=" * 120)
    v6_cells = cells_by_scope["v6"]
    eligible = sorted(v6_cells, key=lambda c: -c.M2_cano)
    top_v6 = next((c for c in eligible if c.M2_cano >= GATE_CANONICAL_PASS), None)
    if top_v6 is None:
        print(f"  No v6 cell passes M2-canonical >= {GATE_CANONICAL_PASS}; bootstrap skipped.")
    else:
        print(f"  Top-M2c v6 cell: {_cell_short(top_v6)}")
        train_cond = carryover[top_v6.train_cond]
        test_cond = carryover[top_v6.test_cond]
        X_tr_full, y_tr_full = slice_canonical(train_cond, top_v6.train_anchor, top_v6.layer)
        X_te_full, y_te_full = slice_canonical(test_cond, top_v6.test_anchor, top_v6.layer)
        X_inv_full, w_inv_full = slice_invented(test_cond, top_v6.test_anchor, top_v6.layer)
        X_tr, y_tr = subset_canonical(X_tr_full, y_tr_full, CANONICALS)
        X_te, y_te = subset_canonical(X_te_full, y_te_full, CANONICALS)
        X_inv, w_inv = subset_invented(X_inv_full, w_inv_full, INVENTED_16)

        bs_m2c = bootstrap_m2_canonical(X_tr, y_tr, X_te, y_te)
        print(f"    M2-canonical bootstrap: mean={bs_m2c['mean']:.3f}, "
              f"std={bs_m2c['std']:.3f}, 95%CI [{bs_m2c['ci95_low']:.3f}, "
              f"{bs_m2c['ci95_high']:.3f}], P(>=0.65)={bs_m2c['p_pass_065']*100:.1f}%")

        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_tr, y_tr)
        bs_m4b = bootstrap_m4b(
            clf, X_inv, w_inv, invented_set=INVENTED_16, w_to_canonical=W_TO_CANONICAL_16,
        )
        print(f"    M4b bootstrap: point={top_v6.M4b:.3f}, mean={bs_m4b['mean']:.3f}, "
              f"std={bs_m4b['std']:.3f}, 95%CI [{bs_m4b['ci95_low']:.3f}, "
              f"{bs_m4b['ci95_high']:.3f}], P(>=0.65)={bs_m4b['p_pass_065']*100:.1f}%")

    return {
        "spec": spec,
        "audit": audit,
        "cells_by_scope": cells_by_scope,
        "aggregate": agg,
        "adjudication": adjudication,
    }


# ==============================================================================
# Cross-model headline + P_RETRACT check
# ==============================================================================
def print_cross_model_headline(results: list[dict]) -> None:
    print()
    print()
    print("=" * 200)
    print("CROSS-MODEL HEADLINE (v6)")
    print("=" * 200)
    print(f"  v6 chance baselines: M2-canonical = 1/15 = 0.067, M2-arity = 0.53, "
          f"M4b lucky-default = 0.50")
    print()
    print(f"  {'Model':<22} {'best v6 M2c':<13} {'best v6 M2a':<13} "
          f"{'v6 PASS-arity':<14} {'verdict':<60}")
    for r in results:
        cells = r["cells_by_scope"]["v6"]
        best_m2c = max(c.M2_cano for c in cells)
        best_m2a = max(c.M2_arity for c in cells)
        n_pass_arity = sum(1 for c in cells if c.verdict == "PASS-arity")
        verdict = r["adjudication"]["verdict"]
        print(f"  {r['spec'].short_name:<22} {best_m2c:<13.3f} {best_m2a:<13.3f} "
              f"{n_pass_arity:<14} {verdict:<60}")

    print()
    print(f"  Pre-registered P_RETRACT (zero PASS-arity at v6 in any model):")
    total_pass_arity = sum(
        sum(1 for c in r["cells_by_scope"]["v6"] if c.verdict == "PASS-arity")
        for r in results
    )
    if total_pass_arity == 0:
        print(f"    [PASS] All three models have zero PASS-arity cells at v6. "
              f"operator-set-bound finding survives v6.")
    else:
        print(f"    [FAIL] {total_pass_arity} PASS-arity cell(s) across all models at v6. "
              f"P_RETRACT falsified -- operator-set-bound finding partially retracted; "
              f"see retraction-chain tables for details.")

    print()
    print(f"  Adjudication summary (P_FREQ vs P_SUBWORD vs P_INTERACTION) per model:")
    for r in results:
        adj = r["adjudication"]
        print(f"    {r['spec'].short_name:<22}: {adj['verdict']}")


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    log_path = _setup_logging()
    print("Script 24 - v6 canonical-set expansion (three-model multi-scope sweep)")
    print(f"STIMULUS_VERSION={STIMULUS_VERSION}  SEED={SEED}  N_PER_CLASS={N_PER_CLASS}")
    print(f"SKIP_HELDOUT={SKIP_HELDOUT}  N_BOOTSTRAP={N_BOOTSTRAP}")
    print()
    print(f"v6 canonical set (15): {CANONICALS}")
    print(f"  binary (8):  {BINARY_CANONICALS_V6}")
    print(f"  unary  (7):  {UNARY_CANONICALS_V6}")
    print(f"  NEW (5):     {NEW_5_CANONICALS_V6}")
    print(f"Invented set (16, unchanged from v4/v5): {INVENTED_16}")
    print()
    print(f"Scopes under test:")
    for s in ALL_SCOPES:
        print(f"  {s.name}: {len(s.canonicals)} canonicals + {len(s.invented_set)} invented")
    print()
    print(f"Models under test:")
    for m in MODEL_SPECS:
        print(f"  {m.short_name:<22}  {m.model_id}")
    print()

    device, device_name = get_device()
    print(f"Device: {device_name}")
    print()

    if AUDIT_ONLY:
        print("=" * 100)
        print("AUDIT_ONLY=1: running tokenization audit for all models, then exiting.")
        print("=" * 100)
        for spec in MODEL_SPECS:
            audit_tokenization(spec)
        print("\nAudit complete. Run again without AUDIT_ONLY to extract caches + sweep.")
        if log_path:
            print(f"\n[logging] full transcript written to: {log_path}")
        return

    t_total = time.time()
    results: list[dict] = []
    for spec in MODEL_SPECS:
        result = run_one_model(spec, device)
        results.append(result)

    print_cross_model_headline(results)

    print()
    print(f"Total wall-clock: {time.time() - t_total:.1f}s")
    if log_path:
        print(f"\n[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
