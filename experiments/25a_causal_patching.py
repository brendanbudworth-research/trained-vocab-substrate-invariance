"""Script 25a - causal patching at Gemma L2 v6 emergent PASS-arity cells +
OLMo Fact-1 anchor cell.

Purpose. The §3.7.16 v6 expansion produced two empirical situations that
linear probes alone cannot adjudicate:

  (Q1) Fact 1 (operator-set-bound canonical-operator transfer; M2-canonical
       = 1.000 with bootstrap CI [1.000, 1.000] at three model families) is
       a *correlational* finding from linear probes. The §5 limitation
       "linear probes only" applies.

  (Q2) Gemma 2 9B has two v6 emergent PASS-arity cells at L2 close-paren
       (opera->close L2 with M4b = 82.2%; sente->close L2 with M4b =
       66.2%) whose M4b jumped from chance at v3-v5 to PASS at v6 without
       any change to the underlying activations. The §3.7.16 reading is
       that this is a methodological caveat on M4b's threshold-sensitivity
       to readout granularity rather than a substantive retraction of
       operator-set-bound; activation patching is the principled
       adjudicator.

This script answers both with the same instrument. For each target cell,
we patch the FUNC-PFX residual stream at the target anchor & layer with a
NEUTRAL canonical-c activation, then measure (i) whether the
v6-carryover-trained M2 probe reads the patched canonical (sanity check),
and (ii) whether the model's *behavioural* downstream output (next-token
distribution at the sentence-final position) shifts toward canonical c's
reference distribution.

Q1 reading: if patching NEUTRAL-c into FUNC-PFX-w at the cell shifts the
behavioural distribution toward FUNC-PFX-c's reference (i.e., delta-KL >
0), the cross-notation canonical transfer is causally load-bearing at
that cell, not merely probe-readable.

Q2 reading: at Gemma L2 close-paren, patch each intended-unary invented
word with `and` (canonical binary) and each intended-binary invented word
with `not` (canonical unary). If patching with the wrong arity flips the
behavioural distribution to that wrong arity's reference, the L2
close-paren representation is causally arity-respecting and the v6
emergent PASS-arity at this cell is a real Gemma exception to
operator-set-bound. If it does not flip, M4b at this cell was probe-only
and the §3.7.16 methodological-caveat reading holds.

Target cells (3 original + 2 reviewer-round-1 follow-ups = 5 total):
  * Gemma 2 9B  L2  opera->close  (N->F)                              [original]
  * Gemma 2 9B  L2  sente->close  (N->F)                              [original]
  * OLMo  2 7B  L10 sente->close  (N->F, the project-flagship Fact-1) [original]
  * OLMo  2 7B  L10 opera->close  (N->F, same target as #3 with       [extra]
                                   opera source: within-target
                                   source-anchor flip; tests whether
                                   #3's inertness is sentence-final-
                                   source-specific)
  * Gemma 2 9B  L4  opera->opera  (N->F, the principal Fact-1 anchor; [extra]
                                   tests whether the project-flagship
                                   ceiling-transfer cell is causally
                                   load-bearing)

The two extra cells address paper.md §4.5's source-anchor direction-
specificity hedge: the original 3-cell sweep tested 1 operator-after-
sourced patch (PASS) and 2 sentence-final-sourced (FAIL); we cannot
tell from that sample whether source-anchor is deterministic in this
direction or whether the pattern is joint-source-target idiosyncrasy.
The extra cells add an operator-after-sourced patch at a different
target (cell #4) and at the principal geometric Fact-1 anchor (cell
#5). Run with `CELL_FILTER=extra` to run only the two new cells (~5-10
min compute; reuses script-24 caches).

Patch conditions (4):
  * BASELINE     - no patch (per invented word + per canonical reference)
  * PATCH_NOT    - replace target anchor activation with mean NEUTRAL-not
                   source activation at (source_anchor, layer)
  * PATCH_AND    - replace with mean NEUTRAL-and source activation
  * RANDOM_NORM  - replace with a random unit vector x norm matched to the
                   mean canonical activation norm (control: severe
                   disruption expected)

Runtime estimate (M4 MPS), all 5 cells:
  Gemma 2 9B forward passes: 3 cells x 16 inv x 4 conds x 10 stim + refs
                             ~= 1980 passes @ ~0.4 s = ~13 min
  OLMo  2 7B forward passes: 2 cells x 16 inv x 4 conds x 10 stim + refs
                             ~= 1320 passes @ ~0.25 s = ~6 min
  Model loads + cache loads:  ~2-3 min
  Total: ~22-30 min wall-clock for all 5 cells.

Runtime estimate (M4 MPS), extra cells only (CELL_FILTER=extra):
  Gemma 2 9B + OLMo 2 7B extras: ~10-12 min including model loads.

Caches: reuses script 24's v6-expanded-canonical carryover NPZ caches
(both NEUTRAL and FUNC-PFX, for both models). No new caches written.

Tees output to outputs/25a_<ts>.log. Env flags:
  CELL_FILTER=<substring>  -- only run cells whose label matches
                              (e.g., CELL_FILTER=opera)
  N_PATCH_STIM=<int>       -- override per-word patched-stimulus count
                              (default 10)
  SKIP_OLMO=1              -- skip OLMo entirely (faster Gemma-only debug)
  SKIP_GEMMA=1             -- skip Gemma entirely (faster OLMo-only debug)

See paper_notes.md §3.7.16 / §4.1.8 / §6 (scripts 25a/25b priority block)
for context.
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import importlib.machinery
import importlib.util
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Tee logging (mirror 24)
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
    log_path = os.path.join(log_dir, f"25a_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Constants
# ==============================================================================
SEED = 17
N_PER_CLASS = 50  # must match script 24's caches
N_PATCH_STIM = int(os.environ.get("N_PATCH_STIM", "10"))
CELL_FILTER = os.environ.get("CELL_FILTER", "").strip().lower()
SKIP_GEMMA = bool(os.environ.get("SKIP_GEMMA"))
SKIP_OLMO = bool(os.environ.get("SKIP_OLMO"))

STIMULUS_VERSION = "v6-expanded-canonical"  # must match script 24

# v6 canonical set + invented set (mirror script 24 exactly).
CANONICALS = [
    "and", "or", "implies", "xor", "nand",
    "not", "necessarily", "possibly", "always", "negate",
    "nor", "iff", "unless", "definitely", "unprovably",
]
CANONICAL_ARITY = {
    "and": 2, "or": 2, "implies": 2, "xor": 2, "nand": 2,
    "nor": 2, "iff": 2, "unless": 2,
    "not": 1, "necessarily": 1, "possibly": 1, "always": 1, "negate": 1,
    "definitely": 1, "unprovably": 1,
}
INVENTED_16 = [
    "bliq", "dren", "molex", "krev", "sond", "glin", "twiv", "fump",
    "vusp", "perph", "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",
]
W_TO_CANONICAL_16 = {
    "bliq": "and", "dren": "or", "molex": "implies",
    "krev": "and", "sond": "or", "glin": "implies",
    "twiv": "and", "fump": "or",
    "vusp": "not", "perph": "necessarily",
    "kelm": "not", "zorf": "not",
    "gleph": "necessarily", "drelth": "necessarily",
    "vrith": "not", "nilph": "necessarily",
}

ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]

# Patch source/target canonicals.
PATCH_CANONICALS = ["not", "and"]  # one unary + one binary
REF_CANONICALS = ["not", "and"]    # canonicals to use as behavioural references


# ==============================================================================
# Model specs (Pythia deferred per §6)
# ==============================================================================
@dataclass
class ModelSpec:
    short_name: str
    model_id: str
    dtype: "torch.dtype"


MODEL_SPECS_ALL: list[ModelSpec] = [
    ModelSpec("Gemma 2 9B", "google/gemma-2-9b", torch.bfloat16),
    ModelSpec("OLMo 2 7B", "allenai/OLMo-2-1124-7B", torch.float16),
]


# ==============================================================================
# Patch cells (3 total)
# ==============================================================================
@dataclass(frozen=True)
class PatchCell:
    model_short_name: str
    layer: int
    source_condition: str  # "NEUTRAL"
    source_anchor: str     # "operator-after" or "sentence-final"
    target_condition: str  # "FUNC-PFX"
    target_anchor: str     # "close-paren"
    label: str             # short label

    def matches_filter(self) -> bool:
        if not CELL_FILTER:
            return True
        key = f"{self.model_short_name} {self.label}".lower()
        return CELL_FILTER in key


PATCH_CELLS: list[PatchCell] = [
    PatchCell(
        "Gemma 2 9B", 2, "NEUTRAL", "operator-after",
        "FUNC-PFX", "close-paren",
        "opera->close L 2 (v6 PASS-arity, M4b=0.822)",
    ),
    PatchCell(
        "Gemma 2 9B", 2, "NEUTRAL", "sentence-final",
        "FUNC-PFX", "close-paren",
        "sente->close L 2 (v6 PASS-arity, M4b=0.662)",
    ),
    PatchCell(
        "OLMo 2 7B", 10, "NEUTRAL", "sentence-final",
        "FUNC-PFX", "close-paren",
        "sente->close L 10 (Fact-1 anchor, M2-arity=1.000)",
    ),
    # ----------------------------------------------------------------------
    # Reviewer-round-1 follow-up cells (paper.md §4.5 source-anchor
    # direction-specificity hedge). Original 3-cell sweep tested two source
    # anchors (operator-after at 1 cell, sentence-final at 2 cells) at two
    # target cells; all 3 sentence-final-sourced patches failed, the 1
    # operator-after-sourced passed. The two cells below disambiguate the
    # source-anchor direction-specificity claim from joint-source-target
    # idiosyncrasy:
    #   (4) OLMo 2 7B opera->close L 10 -- same target as cell (3) above
    #       (the inert Fact-1 flagship cell), but with operator-after source
    #       instead of sentence-final. Within-target source-anchor flip:
    #       if PASS, OLMo's flagship inertness is sentence-final-source-
    #       specific; if FAIL, the target itself is causally inert
    #       regardless of source.
    #   (5) Gemma 2 9B opera->opera L 4 -- the principal Fact-1 anchor
    #       (M2c=1.000, paper.md §4.1) with operator-after source/target,
    #       testing whether geometric Fact 1's project-flagship ceiling-
    #       transfer cell is causally load-bearing. Run with CELL_FILTER
    #       =extra to run only these two.
    # ----------------------------------------------------------------------
    PatchCell(
        "OLMo 2 7B", 10, "NEUTRAL", "operator-after",
        "FUNC-PFX", "close-paren",
        "extra: opera->close L 10 (Fact-1 anchor, source-anchor flip)",
    ),
    PatchCell(
        "Gemma 2 9B", 4, "NEUTRAL", "operator-after",
        "FUNC-PFX", "operator-after",
        "extra: opera->opera L 4 (principal Fact-1 anchor causal test)",
    ),
]


# ==============================================================================
# Stable seeding (mirror script 24)
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
# Import script 19b stimulus generators + script 21 anchor utilities
# (same monkey-patch pattern as script 24).
# ==============================================================================
def _load_module(filename: str, alias: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    loader = importlib.machinery.SourceFileLoader(alias, path)
    spec = importlib.util.spec_from_loader(alias, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    loader.exec_module(mod)
    return mod


_M19B = _load_module("19b_directional_angle_gated.py", "_m19b_25a")
_M19B.CANONICALS = CANONICALS
_M19B.CANONICAL_ARITY = CANONICAL_ARITY
_M19B.UNARY_CANONICALS = [c for c in CANONICALS if CANONICAL_ARITY[c] == 1]
_M19B.BINARY_CANONICALS = [c for c in CANONICALS if CANONICAL_ARITY[c] == 2]
_M19B.INVENTED_WORDS = INVENTED_16
_M19B.W_TO_CANONICAL = W_TO_CANONICAL_16
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
make_functional_invented_stimuli = _M19B.make_functional_invented_stimuli
assert _M19B.SEED == SEED
assert _M19B.N_PER_CLASS == N_PER_CLASS

_M21 = _load_module("21_multi_anchor_battery.py", "_m21_25a")
compute_anchor_positions = _M21.compute_anchor_positions
ConditionMultiAnchor = _M21.ConditionMultiAnchor


# ==============================================================================
# Stimulus regeneration (must match script 24's stable_seed for cache parity)
# ==============================================================================
def _gen_funcpfx_canonical_prompts(op: str) -> list[str]:
    """Regenerate FUNC-PFX canonical prompts (full N_PER_CLASS = 50)."""
    rng = random.Random(stable_seed("v6", "FUNC-PFX", "canon", op))
    return make_functional_canonical_stimuli(op, rng, N_PER_CLASS)


def _gen_funcpfx_invented_prompts(w: str) -> list[str]:
    """Regenerate FUNC-PFX invented prompts (full N_PER_CLASS = 50)."""
    rng = random.Random(stable_seed("v6", "FUNC-PFX", "inv", w))
    return make_functional_invented_stimuli(w, rng, N_PER_CLASS)


def _gen_neutral_canonical_prompts(op: str) -> list[str]:
    rng = random.Random(stable_seed("v6", "NEUTRAL", "canon", op))
    return make_neutral_stimuli(op, rng, N_PER_CLASS)


def _gen_neutral_invented_prompts(w: str) -> list[str]:
    rng = random.Random(stable_seed("v6", "NEUTRAL", "inv", w))
    return make_neutral_stimuli(w, rng, N_PER_CLASS)


# ==============================================================================
# Cache loading (read-only, mirrors script 24's load path exactly)
# ==============================================================================
def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    slug = model_short_name.replace(" ", "_")
    return os.path.join(
        base,
        f"24_{slug}_{condition_name}_npc{N_PER_CLASS}_v6-expanded-canonical.npz",
    )


def _load_carryover_cache(model_short_name: str, condition_name: str) -> Optional[ConditionMultiAnchor]:
    path = _cache_path(model_short_name, condition_name)
    if not os.path.exists(path):
        print(f"  [cache] MISS: {path}")
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["meta_stimulus_version"][0]) != STIMULUS_VERSION:
            print(f"  [cache] stimulus_version mismatch in {path}")
            return None
        if list(z["canonical_list"]) != CANONICALS:
            print(f"  [cache] canonical_list mismatch in {path}")
            return None
        if list(z["invented_word_list"]) != INVENTED_16:
            print(f"  [cache] invented_word_list mismatch in {path}")
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
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [cache] HIT  {os.path.basename(path)} "
              f"canon={cond.canonical_X.shape} inv={cond.invented_X.shape} "
              f"({size_mb:.1f} MB)")
        return cond
    except Exception as e:
        print(f"  [cache] failed to load {path}: {e}")
        return None


# ==============================================================================
# Source-activation extraction from NEUTRAL cache
# ==============================================================================
def extract_source_activations(
    neut: ConditionMultiAnchor, source_anchor: str, layer: int,
    canonicals: list[str],
) -> tuple[dict[str, np.ndarray], float]:
    """Returns (source[c] = mean activation, mean_norm).

    source[c] is a (dim,) float32 vector = mean over the 50 NEUTRAL-c
    stimuli of the residual stream at (source_anchor, layer).
    mean_norm is the average L2 norm of canonical mean activations, used
    to scale the RANDOM_NORM control.
    """
    a_idx = neut.anchor_names.index(source_anchor)
    labels = np.asarray(neut.canonical_labels)
    X = neut.canonical_X[a_idx, :, layer, :]  # (n_stim, dim)
    out: dict[str, np.ndarray] = {}
    norms = []
    for c in canonicals:
        mask = (labels == c)
        if not mask.any():
            raise RuntimeError(f"no NEUTRAL stimuli for canonical {c!r}")
        v = X[mask].mean(axis=0).astype(np.float32)
        out[c] = v
        norms.append(float(np.linalg.norm(v)))
    return out, float(np.mean(norms))


# ==============================================================================
# Probe training (M2 N->F: train on NEUTRAL canonical, test on FUNC-PFX)
# ==============================================================================
def train_m2_probe(
    neut: ConditionMultiAnchor, func: ConditionMultiAnchor,
    source_anchor: str, target_anchor: str, layer: int,
) -> tuple[LogisticRegression, float, float]:
    """Train M2 probe on NEUTRAL canonical at source_anchor; return probe
    + (M2-canonical accuracy on FUNC-PFX canonical at target_anchor,
       M2-arity accuracy)."""
    sa_idx = neut.anchor_names.index(source_anchor)
    ta_idx = func.anchor_names.index(target_anchor)
    X_tr = neut.canonical_X[sa_idx, :, layer, :]
    y_tr = np.asarray(neut.canonical_labels)
    X_te = func.canonical_X[ta_idx, :, layer, :]
    y_te = np.asarray(func.canonical_labels)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    m2_cano = float(np.mean(preds == y_te))
    n_ar = sum(
        CANONICAL_ARITY[str(t)] == CANONICAL_ARITY[str(p)]
        for t, p in zip(y_te, preds)
    )
    m2_arity = n_ar / len(y_te)
    return clf, m2_cano, m2_arity


# ==============================================================================
# Patched forward pass.
#
# Bug-fix v2: the v1 implementation read `out.hidden_states[layer]` for
# the probe input. Empirically (script 25a first run, 25a_20260520_205228.log)
# the patch DID propagate to downstream computation (next-token logits
# shifted) but `out.hidden_states[layer]` did NOT reflect the patch on MPS
# (probe predictions identical across BASELINE / PATCH_not / PATCH_and /
# RANDOM_NORM in all three cells, 0.0% probe-causality everywhere). The
# fix is to capture the residual directly inside the forward hook so we
# have a guaranteed-post-patch value, and not rely on out.hidden_states
# for the probe input.
# ==============================================================================
def _layer_module_for_patch(model, layer: int):
    """Return the nn.Module whose forward-hook output corresponds to
    hidden_states[layer] in the post-pass output (i.e., the L-th block's
    output). The convention: model.model.layers[L-1] for L >= 1."""
    return model.model.layers[layer - 1]


def _make_patch_capture_hook(
    source_vec: Optional[torch.Tensor], position: int,
    capture_sink: list,
):
    """Forward hook that:
      (i)  optionally overwrites output[0][:, position, :] with source_vec
           (broadcasted over the batch dimension);
      (ii) captures the (post-patch) value at output[0][0, position, :]
           into capture_sink as a float32 numpy array.

    If source_vec is None, the hook only captures (does not modify the
    layer's output). This guarantees the probe sees what subsequent layers
    will see at this position.
    """
    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if source_vec is not None:
            hs_new = hs.clone()
            hs_new[:, position, :] = source_vec.to(hs_new.dtype).to(hs_new.device)
            captured = hs_new[0, position, :].detach().to(torch.float32).cpu().numpy()
            capture_sink.append(captured)
            if is_tuple:
                return (hs_new,) + output[1:]
            return hs_new
        else:
            captured = hs[0, position, :].detach().to(torch.float32).cpu().numpy()
            capture_sink.append(captured)
            return None  # do not modify
    return hook


@dataclass
class PassRecord:
    final_logits: np.ndarray         # (vocab,) float32
    target_residual: np.ndarray      # (dim,) float32 - residual at L_target [target_pos] AFTER patch
    target_pos: int


def run_one_pass(
    model, tok, device: str, prompt: str, layer: int,
    target_anchor: str, source_canon_for_anchor: list[str],
    *, patch_source: Optional[torch.Tensor] = None,
) -> PassRecord:
    """Run a single forward pass. If patch_source is provided, replace
    the residual stream at (layer, target_anchor) with patch_source.
    Returns the final-position logits and the residual at L_target
    [target_anchor] post-patch (or unpatched if patch_source is None).
    """
    enc = tok(prompt, return_tensors="pt").to(device)
    seq_len = enc.input_ids.shape[1]
    anchors = compute_anchor_positions(tok, prompt, source_canon_for_anchor,
                                       "FUNC-PFX")
    pos = anchors.get(target_anchor, None)
    if pos is None or pos >= seq_len:
        pos = seq_len - 1

    layer_mod = _layer_module_for_patch(model, layer)
    capture_sink: list = []
    handle = layer_mod.register_forward_hook(
        _make_patch_capture_hook(patch_source, pos, capture_sink)
    )

    try:
        with torch.no_grad():
            out = model(**enc)  # don't need output_hidden_states; hook captures it
    finally:
        handle.remove()

    if device == "mps":
        torch.mps.synchronize()

    final_logits = out.logits[0, -1, :].to(torch.float32).cpu().numpy()
    if not capture_sink:
        raise RuntimeError(f"forward hook did not fire at layer {layer}; "
                           f"layer module is {type(layer_mod).__name__}")
    target_residual = capture_sink[0]

    return PassRecord(
        final_logits=final_logits,
        target_residual=target_residual,
        target_pos=pos,
    )


# ==============================================================================
# KL divergence + behavioural-shift metrics
# ==============================================================================
def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) over the vocab dimension. p, q are probability vectors."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def topk_overlap(p: np.ndarray, q: np.ndarray, k: int = 20) -> float:
    """Top-K jaccard overlap between the top-K vocab tokens of p and q."""
    pk = set(np.argsort(p)[-k:].tolist())
    qk = set(np.argsort(q)[-k:].tolist())
    return len(pk & qk) / max(1, len(pk | qk))


# ==============================================================================
# Per-cell experiment
# ==============================================================================
@dataclass
class CellResult:
    cell: PatchCell
    m2_cano: float
    m2_arity: float
    # Per (invented word, patch condition):
    probe_pred_per: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    next_logits_per: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    # Per canonical c (no-patch FUNC-PFX-c reference):
    ref_logits_per: dict[str, np.ndarray] = field(default_factory=dict)
    # Norm for random-norm control:
    canon_mean_norm: float = 0.0


def run_cell_patching(
    cell: PatchCell, neut: ConditionMultiAnchor, func: ConditionMultiAnchor,
    model, tok, device: str,
) -> CellResult:
    print()
    print("=" * 120)
    print(f"  CELL: {cell.model_short_name} {cell.label}")
    print(f"        source = {cell.source_condition} {cell.source_anchor} L{cell.layer}")
    print(f"        target = {cell.target_condition} {cell.target_anchor} L{cell.layer}")
    print("=" * 120)

    t_cell0 = time.time()

    # 1. Train M2 probe (NEUTRAL source -> FUNC-PFX target).
    probe, m2_cano, m2_arity = train_m2_probe(
        neut, func, cell.source_anchor, cell.target_anchor, cell.layer,
    )
    print(f"  M2 probe: M2-canonical={m2_cano:.3f}, M2-arity={m2_arity:.3f}")

    # 2. Extract source activations from NEUTRAL cache.
    source_vecs, mean_norm = extract_source_activations(
        neut, cell.source_anchor, cell.layer, CANONICALS,
    )
    print(f"  source vec dim={source_vecs['not'].shape[0]}, mean canonical norm={mean_norm:.3f}")
    for c in PATCH_CANONICALS:
        print(f"    src[{c}] L2-norm = {np.linalg.norm(source_vecs[c]):.3f}")

    res = CellResult(cell=cell, m2_cano=m2_cano, m2_arity=m2_arity,
                     canon_mean_norm=mean_norm)

    # 3. Reference FUNC-PFX-c next-token logits (no patch).
    print(f"  --- Reference: FUNC-PFX-c canonical no-patch ---")
    for c in REF_CANONICALS:
        prompts = _gen_funcpfx_canonical_prompts(c)[:N_PATCH_STIM]
        logits_acc = []
        t0 = time.time()
        for p in prompts:
            rec = run_one_pass(model, tok, device, p, cell.layer,
                               cell.target_anchor, [c])
            logits_acc.append(softmax_np(rec.final_logits))
        ref_probs = np.mean(np.stack(logits_acc, axis=0), axis=0)
        res.ref_logits_per[c] = ref_probs
        print(f"    ref[{c:<12}] {N_PATCH_STIM} stim in {time.time() - t0:.1f}s")

    # 4. For each invented word and patch condition: forward passes.
    print(f"  --- Patched FUNC-PFX-w invented forward passes ---")
    rng = np.random.default_rng(SEED)
    random_unit = rng.standard_normal(source_vecs["not"].shape[0]).astype(np.float32)
    random_unit /= np.linalg.norm(random_unit)
    random_vec = random_unit * mean_norm

    conditions = ["BASELINE", "PATCH_not", "PATCH_and", "RANDOM_NORM"]
    cond_to_source: dict[str, Optional[torch.Tensor]] = {
        "BASELINE":    None,
        "PATCH_not":   torch.from_numpy(source_vecs["not"]),
        "PATCH_and":   torch.from_numpy(source_vecs["and"]),
        "RANDOM_NORM": torch.from_numpy(random_vec),
    }

    n_words = len(INVENTED_16)
    n_passes = n_words * len(conditions) * N_PATCH_STIM
    t_pass_total0 = time.time()
    pass_count = 0
    for w_idx, w in enumerate(INVENTED_16):
        prompts = _gen_funcpfx_invented_prompts(w)[:N_PATCH_STIM]
        for cond_name in conditions:
            src_t = cond_to_source[cond_name]
            probe_pred_counts: dict[str, int] = {c: 0 for c in CANONICALS}
            arity_match_count = 0
            logits_acc = []
            residuals_acc = []
            for p in prompts:
                rec = run_one_pass(
                    model, tok, device, p, cell.layer,
                    cell.target_anchor, [w],
                    patch_source=src_t,
                )
                logits_acc.append(softmax_np(rec.final_logits))
                residuals_acc.append(rec.target_residual)
                pass_count += 1
            # Aggregate probe prediction on the patched residuals.
            R = np.stack(residuals_acc, axis=0)
            preds = probe.predict(R)
            for pp in preds:
                probe_pred_counts[str(pp)] += 1
            # Arity match: did the probe predict the patched-canonical's arity?
            intended_canon = W_TO_CANONICAL_16[w]
            for pp in preds:
                if CANONICAL_ARITY[str(pp)] == CANONICAL_ARITY[intended_canon]:
                    arity_match_count += 1
            res.probe_pred_per[(w, cond_name)] = {
                **probe_pred_counts,
                "_arity_match": arity_match_count,
                "_n": N_PATCH_STIM,
            }
            res.next_logits_per[(w, cond_name)] = np.mean(
                np.stack(logits_acc, axis=0), axis=0
            )
        if (w_idx + 1) % 4 == 0 or w_idx == n_words - 1:
            elapsed = time.time() - t_pass_total0
            rate = pass_count / max(elapsed, 1e-6)
            print(f"    [{w_idx + 1}/{n_words}] {w:<8}  "
                  f"passes={pass_count}/{n_passes}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f} pass/s")

    print(f"  cell complete in {time.time() - t_cell0:.0f}s "
          f"({pass_count} forward passes)")
    return res


# ==============================================================================
# Per-cell reporting
# ==============================================================================
def _arity_str(c: str) -> str:
    return "B" if CANONICAL_ARITY.get(c, 0) == 2 else "U"


def report_cell(res: CellResult) -> None:
    cell = res.cell
    print()
    print("=" * 120)
    print(f"  RESULTS: {cell.model_short_name} {cell.label}")
    print(f"           M2-canonical={res.m2_cano:.3f}, M2-arity={res.m2_arity:.3f}")
    print("=" * 120)

    print()
    print(f"  --- (A) Probe-readout per (invented word, patch condition) ---")
    print(f"  For each row, columns show what the v6 M2-probe predicts when applied to the")
    print(f"  patched residual at L{cell.layer} {cell.target_anchor}. If patching is")
    print(f"  effective, the probe should read the patched canonical (not -> 'not';")
    print(f"  and -> 'and'). BASELINE shows what the probe reads with no patch (the")
    print(f"  v6 reading on this cell's invented activations).")
    print()
    header = (f"  {'word':<8} {'arity':<6} "
              f"| {'BASELINE top (%)':<28} "
              f"| {'PATCH_not top (%)':<28} "
              f"| {'PATCH_and top (%)':<28} "
              f"| {'RANDOM top (%)':<24}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for w in INVENTED_16:
        intended = W_TO_CANONICAL_16[w]
        arity = _arity_str(intended)
        row_parts: list[str] = []
        for cond in ("BASELINE", "PATCH_not", "PATCH_and", "RANDOM_NORM"):
            cnt = res.probe_pred_per.get((w, cond), {})
            n = cnt.get("_n", N_PATCH_STIM)
            counts = {c: cnt.get(c, 0) for c in CANONICALS}
            if all(v == 0 for v in counts.values()):
                row_parts.append("(no data)")
                continue
            top_c = max(counts, key=lambda c: counts[c])
            pct = 100.0 * counts[top_c] / max(1, n)
            row_parts.append(f"{top_c:<14}{_arity_str(top_c)} {pct:>5.1f}")
        print(f"  {w:<8} {arity:<6} "
              f"| {row_parts[0]:<28} "
              f"| {row_parts[1]:<28} "
              f"| {row_parts[2]:<28} "
              f"| {row_parts[3]:<24}")

    # Aggregate: how often does the probe predict the patched canonical?
    print()
    print(f"  --- (B) Probe causality summary (across all 16 inv x {N_PATCH_STIM} stim = {16 * N_PATCH_STIM} predictions) ---")
    for cond, expected_c in (("PATCH_not", "not"), ("PATCH_and", "and"),
                             ("RANDOM_NORM", None), ("BASELINE", None)):
        total = 0
        n_predicting_expected = 0
        per_canon: dict[str, int] = {c: 0 for c in CANONICALS}
        for w in INVENTED_16:
            cnt = res.probe_pred_per.get((w, cond), {})
            n = cnt.get("_n", N_PATCH_STIM)
            total += n
            for c in CANONICALS:
                per_canon[c] += cnt.get(c, 0)
            if expected_c is not None:
                n_predicting_expected += cnt.get(expected_c, 0)
        top3 = sorted(per_canon.items(), key=lambda kv: -kv[1])[:3]
        top3_str = ", ".join(f"{c}={100*v/total:.1f}%" for c, v in top3 if v > 0)
        if expected_c is not None:
            print(f"    {cond:<14} probe -> {expected_c!r}: "
                  f"{100 * n_predicting_expected / total:5.1f}% "
                  f"(top3: {top3_str})")
        else:
            print(f"    {cond:<14}: "
                  f"(top3: {top3_str})")

    print()
    print(f"  --- (C) Behavioural shift (next-token KL divergence) ---")
    print(f"  For each invented word w and each patch condition C, we compute the KL")
    print(f"  divergence between the patched next-token distribution and the no-patch")
    print(f"  reference canonical-c distribution at the same anchor. Lower KL means")
    print(f"  the model's behaviour is closer to canonical-c. Delta = KL(BASELINE || ref)")
    print(f"  - KL(PATCH_c || ref). Delta > 0 = patch pulled behavior toward c.")
    print()
    print(f"  {'word':<8} {'arity':<5} "
          f"| {'KL(BASE||ref_not)':<18} {'KL(PATCH_not||ref_not)':<22} {'Δ(not)':<10} "
          f"| {'KL(BASE||ref_and)':<18} {'KL(PATCH_and||ref_and)':<22} {'Δ(and)':<10}")
    print(f"  {'-' * 138}")
    delta_summary = {"PATCH_not": [], "PATCH_and": [], "RANDOM_NORM_not": [],
                     "RANDOM_NORM_and": []}
    arity_flip = {"unary_word_patched_and": [], "binary_word_patched_not": []}
    for w in INVENTED_16:
        intended = W_TO_CANONICAL_16[w]
        arity = _arity_str(intended)
        base_logits = res.next_logits_per.get((w, "BASELINE"))
        p_not_logits = res.next_logits_per.get((w, "PATCH_not"))
        p_and_logits = res.next_logits_per.get((w, "PATCH_and"))
        p_rnd_logits = res.next_logits_per.get((w, "RANDOM_NORM"))
        ref_not = res.ref_logits_per.get("not")
        ref_and = res.ref_logits_per.get("and")
        if base_logits is None or p_not_logits is None or p_and_logits is None:
            continue
        kl_base_not = kl_divergence(base_logits, ref_not)
        kl_pnot_not = kl_divergence(p_not_logits, ref_not)
        kl_base_and = kl_divergence(base_logits, ref_and)
        kl_pand_and = kl_divergence(p_and_logits, ref_and)
        d_not = kl_base_not - kl_pnot_not
        d_and = kl_base_and - kl_pand_and
        print(f"  {w:<8} {arity:<5} "
              f"| {kl_base_not:>14.3f}     {kl_pnot_not:>16.3f}       {d_not:>+7.3f}   "
              f"| {kl_base_and:>14.3f}     {kl_pand_and:>16.3f}       {d_and:>+7.3f}")
        delta_summary["PATCH_not"].append(d_not)
        delta_summary["PATCH_and"].append(d_and)
        if p_rnd_logits is not None:
            kl_prnd_not = kl_divergence(p_rnd_logits, ref_not)
            kl_prnd_and = kl_divergence(p_rnd_logits, ref_and)
            delta_summary["RANDOM_NORM_not"].append(kl_base_not - kl_prnd_not)
            delta_summary["RANDOM_NORM_and"].append(kl_base_and - kl_prnd_and)
        # Arity-flip: intended-unary word patched with `and` -> should flip
        # to binary behaviour if cell is causally arity-respecting.
        if CANONICAL_ARITY[intended] == 1:
            arity_flip["unary_word_patched_and"].append(d_and)
        else:
            arity_flip["binary_word_patched_not"].append(d_not)

    print()
    print(f"  --- (D) Aggregate ΔKL summary ---")
    for key, vals in delta_summary.items():
        if not vals:
            continue
        arr = np.array(vals)
        n_pos = int((arr > 0).sum())
        print(f"    Δ {key:<22}  mean = {arr.mean():+.3f}   "
              f"median = {np.median(arr):+.3f}   "
              f"{n_pos}/{len(arr)} positive   "
              f"(positive = patch pulled behavior toward the reference canonical)")

    print()
    print(f"  --- (E) Arity-flip test (for Gemma v6 emergent PASS-arity question) ---")
    print(f"    Q: if cell L{cell.layer} {cell.target_anchor} is causally")
    print(f"       arity-respecting, then patching with the WRONG-ARITY canonical")
    print(f"       should shift behavior strongly toward that wrong-arity reference.")
    for key, vals in arity_flip.items():
        if not vals:
            continue
        arr = np.array(vals)
        n_pos = int((arr > 0).sum())
        n_neg = int((arr < 0).sum())
        print(f"    {key:<32}  mean Δ = {arr.mean():+.3f}   "
              f"+/- = {n_pos}/{n_neg}/{len(arr)}   "
              f"(strong + with high probe-causality = arity flip is real)")

    print()


# ==============================================================================
# Cross-cell synthesis
# ==============================================================================
def report_synthesis(results: list[CellResult]) -> None:
    print()
    print("=" * 120)
    print(f"  CROSS-CELL SYNTHESIS")
    print("=" * 120)
    print(f"  Headline columns:")
    print(f"    P(probe -> not | PATCH_not)  = how often the patched residual reads as 'not'")
    print(f"    P(probe -> and | PATCH_and)  = how often the patched residual reads as 'and'")
    print(f"    Δ KL(not)   = mean (KL(BASE||ref_not) - KL(PATCH_not||ref_not)) over invented words")
    print(f"    Δ KL(and)   = mean (KL(BASE||ref_and) - KL(PATCH_and||ref_and)) over invented words")
    print(f"    Δ KL(rnd_not)= same with RANDOM_NORM (expected negative or near-zero)")
    print(f"    Δ KL(rnd_and)= same with RANDOM_NORM")
    print()
    print(f"  {'cell':<54} {'M2-cano':>8} {'M2-arity':>9} "
          f"{'P->not|p_not':>13} {'P->and|p_and':>13} "
          f"{'ΔKL(not)':>10} {'ΔKL(and)':>10} "
          f"{'ΔKL(rnd_not)':>13} {'ΔKL(rnd_and)':>13}")
    print(f"  {'-' * 154}")
    for res in results:
        cell = res.cell
        label = f"{cell.model_short_name} {cell.label[:38]}"
        # Probe causality
        n_pn = sum(res.probe_pred_per.get((w, "PATCH_not"), {}).get("not", 0)
                   for w in INVENTED_16)
        n_pa = sum(res.probe_pred_per.get((w, "PATCH_and"), {}).get("and", 0)
                   for w in INVENTED_16)
        total = 16 * N_PATCH_STIM
        # Behavioural ΔKL
        dk_not = []
        dk_and = []
        dk_rnd_not = []
        dk_rnd_and = []
        for w in INVENTED_16:
            base = res.next_logits_per.get((w, "BASELINE"))
            p_n = res.next_logits_per.get((w, "PATCH_not"))
            p_a = res.next_logits_per.get((w, "PATCH_and"))
            p_r = res.next_logits_per.get((w, "RANDOM_NORM"))
            rn = res.ref_logits_per.get("not")
            ra = res.ref_logits_per.get("and")
            if base is None or p_n is None or p_a is None or rn is None or ra is None:
                continue
            dk_not.append(kl_divergence(base, rn) - kl_divergence(p_n, rn))
            dk_and.append(kl_divergence(base, ra) - kl_divergence(p_a, ra))
            if p_r is not None:
                dk_rnd_not.append(kl_divergence(base, rn) - kl_divergence(p_r, rn))
                dk_rnd_and.append(kl_divergence(base, ra) - kl_divergence(p_r, ra))
        print(f"  {label:<54} "
              f"{res.m2_cano:>8.3f} {res.m2_arity:>9.3f} "
              f"{100*n_pn/total:>12.1f}% {100*n_pa/total:>12.1f}% "
              f"{np.mean(dk_not):>+10.3f} {np.mean(dk_and):>+10.3f} "
              f"{np.mean(dk_rnd_not):>+13.3f} {np.mean(dk_rnd_and):>+13.3f}")

    print()
    print(f"  Interpretation guide:")
    print(f"    * Q1 (Fact 1 causal grounding): if P->not|p_not and P->and|p_and are")
    print(f"      both high (~> 0.70) AND ΔKL(not), ΔKL(and) are both clearly positive,")
    print(f"      the cross-notation canonical transfer at this cell is causally")
    print(f"      load-bearing (the residual at the target anchor controls downstream")
    print(f"      behaviour). RANDOM_NORM ΔKLs should be near-zero or negative; if they")
    print(f"      are also positive, the test is non-discriminating.")
    print()
    print(f"    * Q2 (Gemma v6 emergent PASS-arity adjudication): at the Gemma L2")
    print(f"      cells, check the arity-flip block (E) per cell. If patching an")
    print(f"      intended-unary word with `and` shifts ΔKL(and) strongly positive")
    print(f"      AND the probe reads `and`, the L2 close-paren position is causally")
    print(f"      arity-respecting -> Gemma v6 emergence is real and operator-set-bound")
    print(f"      has a model-specific exception. If patching does not flip behaviour")
    print(f"      (small ΔKL(and) for unary words), M4b's 82% reading was probe-only")
    print(f"      and the §3.7.16 methodological-caveat reading holds.")
    print()


# ==============================================================================
# Per-model runner
# ==============================================================================
def free_model(model) -> None:
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def run_one_model(spec: ModelSpec, device: str) -> list[CellResult]:
    print()
    print("#" * 120)
    print(f"# MODEL: {spec.short_name}  ({spec.model_id})")
    print("#" * 120)

    cells = [c for c in PATCH_CELLS if c.model_short_name == spec.short_name
             and c.matches_filter()]
    if not cells:
        print(f"  No cells for this model; skipping.")
        return []

    print(f"  Cells in scope: {len(cells)}")
    for c in cells:
        print(f"    - {c.label}")

    print()
    print(f"  Loading caches:")
    neut = _load_carryover_cache(spec.short_name, "NEUTRAL")
    func = _load_carryover_cache(spec.short_name, "FUNC-PFX")
    if neut is None or func is None:
        print(f"  ** v6 carryover caches missing for {spec.short_name}. **")
        print(f"  ** Run script 24 first to populate the caches. **")
        return []

    print()
    print(f"  Loading model: {spec.model_id}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=spec.dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    n_layers_model = model.config.num_hidden_layers
    print(f"    loaded in {time.time() - t0:.1f}s "
          f"({n_layers_model} layers, hidden_size={model.config.hidden_size})")

    results: list[CellResult] = []
    try:
        for cell in cells:
            if cell.layer < 1 or cell.layer > n_layers_model:
                print(f"  ** Skipping {cell.label}: layer {cell.layer} out of range "
                      f"(model has {n_layers_model} layers) **")
                continue
            res = run_cell_patching(cell, neut, func, model, tok, device)
            report_cell(res)
            results.append(res)
    finally:
        free_model(model)
        del tok
        gc.collect()
        print(f"  freed {spec.short_name}")
    return results


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    log_path = _setup_logging()
    print()
    print("=" * 120)
    print(" script 25a - causal patching at v6 emergent PASS-arity cells "
          "(Gemma L2) + Fact-1 anchor (OLMo L10)")
    print("=" * 120)
    print(f"  N_PATCH_STIM      = {N_PATCH_STIM} stimuli per (cell, word, condition)")
    print(f"  SEED              = {SEED}")
    print(f"  CELL_FILTER       = {CELL_FILTER!r}")
    print(f"  patch conditions  = BASELINE, PATCH_not, PATCH_and, RANDOM_NORM")
    print(f"  patch canonicals  = {PATCH_CANONICALS}")
    print(f"  reference canon.  = {REF_CANONICALS}")
    print(f"  patch cells       = {len(PATCH_CELLS)} ({[c.label[:38] for c in PATCH_CELLS]})")
    print()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"  device            = {device}")

    specs = []
    for spec in MODEL_SPECS_ALL:
        if spec.short_name.startswith("Gemma") and SKIP_GEMMA:
            print(f"  skipping {spec.short_name} (SKIP_GEMMA=1)")
            continue
        if spec.short_name.startswith("OLMo") and SKIP_OLMO:
            print(f"  skipping {spec.short_name} (SKIP_OLMO=1)")
            continue
        specs.append(spec)

    all_results: list[CellResult] = []
    t_total0 = time.time()
    for spec in specs:
        results = run_one_model(spec, device)
        all_results.extend(results)

    report_synthesis(all_results)

    print()
    print(f"  total wall-clock: {time.time() - t_total0:.0f}s")
    if log_path:
        print(f"\n  [logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
