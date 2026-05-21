"""Directional-angle analysis between NEUTRAL and FUNCTIONAL-PREFIX arity
directions, per model (script 18 follow-up).

Script 18 established that:
  - Both models have a geometric arity-region attractor in both notations
    (Diagnostic C: 5/5 invented words closer to the unary centroid than to
    the binary centroid in every model x condition cell).
  - The NEUTRAL probe direction transfers to FUNC-PFX invented activations
    at 100% unary mass for Gemma 2, at 0% (all `and`) for OLMo 2 (Diag A).
  - The reverse transfer is symmetric in failure mode (FUNC-PFX NEUTRAL: 0%
    for OLMo 2 to `implies`; Gemma 2 86-100% across cross-transfer-validated
    layers).

The interpretation is that Gemma 2 has a globally-aligned arity direction
across notations, while OLMo 2 has notation-local arity directions. This
script measures that alignment quantitatively, by computing the cosine
angle between the NEUTRAL and FUNC-PFX arity directions in each model.

Two operationalisations of "the arity direction" at a given layer:

  (A) Centroid-based:  d_C = mean(unary canonical centroids)
                             - mean(binary canonical centroids)
      Probe-free, geometrically interpretable, no scaling required.

  (B) Probe-based:     d_P = unit-normalised weight vector of a binary
                             logistic-regression probe predicting
                             arity (unary vs binary) from canonical
                             activations.
      The probe is trained on RAW (un-standardised) activations so the
      weight vector lives directly in residual stream coordinates and
      the cosine angle between two condition-specific probe weights is
      a well-defined geometric measurement of the directional
      alignment between the two conditions' arity classifiers. (A
      standardised-space probe weight can be rotated relative to the
      raw-space arity direction when per-dimension variance differs
      across conditions; we want to compare directions in residual
      stream space, not in a standardised feature space whose axes
      vary between conditions.)

For each model at each diagnostic layer, we report:
  - angle(d_C^NEUT,  d_C^FUNC)  - centroid-based cross-notation angle
  - angle(d_P^NEUT,  d_P^FUNC)  - probe-based cross-notation angle
  - angle(d_C^cond,  d_P^cond)  - within-notation centroid-vs-probe angle
                                   (sanity check: should be small if both
                                   capture the same arity structure)
  - mean projection of invented activations onto d_C^cond, d_C^other-cond
    (sanity check: in-notation projection should be large positive in
    the unary direction; cross-notation projection follows the cross-
    condition transfer signature from script 18)

Plus baselines:
  - Random-unit-vector baseline: angle between a centroid direction and
    100 Gaussian-sampled unit vectors of the same dimension. Mean angle
    should be ~ 90 deg with tight CI. Establishes the "what does an
    unrelated direction look like" prior.

Predicted (qualitative) result, based on the script-18 transfer asymmetry:
  - Gemma 2 9B:  centroid + probe angles ~ small  (well below 90 deg)
  - OLMo 2 7B:   centroid + probe angles ~ near orthogonal (near 90 deg)

Memory plan: identical to script 18 - load Gemma 2 9B, extract, delete,
load OLMo 2 7B, repeat.

Run time on M4: estimated 8-12 minutes once both models are cached
(slightly less than script 18 because no probe CV sweeps).
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
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Tee logging - same pattern as script 18.
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
    log_path = os.path.join(log_dir, f"19_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    print(f"[logging] (set NO_LOG=1 to disable)")
    return log_path


# ==============================================================================
# Constants - identical to scripts 17/18 for direct comparability.
# ==============================================================================
SEED = 17
N_PER_CLASS = 50


def _stable_seed(*parts, base: int = SEED, modulo: int = 100_000) -> int:
    """Stable replacement for `SEED + hash((...)) % modulo`. Python's
    built-in hash() is per-process salted unless PYTHONHASHSEED is fixed,
    breaking reproducibility of stimulus generation across runs."""
    s = "::".join(map(str, parts)).encode("utf-8")
    h = int(hashlib.blake2b(s, digest_size=8).hexdigest(), 16)
    return base + (h % modulo)


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
    diagnostic_layers: list[int]
    cond1_focus_layer: int
    cond2_focus_layers: list[int]


# Same layer set as script 18 so the two scripts' tables line up.
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
# Stimulus generation - verbatim from script 18.
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
    """Match script 18: arity inherited from the canonical the invented word
    replaces (W_TO_CANONICAL)."""
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
    """Position immediately after the operator's last subword. Same helper as
    scripts 15/16/17/18."""
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


def build_condition(
    name: str,
    model, tok, device: str,
    canonical_stim_fn, invented_stim_fn,
) -> ConditionActivations:
    print(f"\n  Building condition: {name}")

    canon_prompts: list[str] = []
    canon_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(_stable_seed(name, "canon", op))
        canon_prompts.extend(canonical_stim_fn(op, op_rng, N_PER_CLASS))
        canon_labels.extend([op] * N_PER_CLASS)

    inv_prompts: list[str] = []
    inv_words: list[str] = []
    for w in INVENTED_WORDS:
        w_rng = random.Random(_stable_seed(name, "invent", w))
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
# Direction primitives - the new content in this script.
# ==============================================================================
def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def cosine_angle_deg(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Cosine similarity and angle in degrees between two unit vectors."""
    cos = float(np.dot(u, v))
    cos = max(-1.0, min(1.0, cos))
    deg = float(np.degrees(np.arccos(cos)))
    return cos, deg


def centroid_unary_direction(
    canon_X_layer: np.ndarray, canon_labels: np.ndarray
) -> np.ndarray:
    """mean(unary centroids) - mean(binary centroids), unit-normalised in the
    raw activation space."""
    unary_centroids = np.stack(
        [canon_X_layer[canon_labels == op].mean(axis=0) for op in UNARY_CANONICALS]
    )
    binary_centroids = np.stack(
        [canon_X_layer[canon_labels == op].mean(axis=0) for op in BINARY_CANONICALS]
    )
    return unit(unary_centroids.mean(axis=0) - binary_centroids.mean(axis=0))


def raw_binary_probe_directions(
    canon_X_A: np.ndarray, labels_A: np.ndarray,
    canon_X_B: np.ndarray, labels_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train two binary (unary vs binary) logistic-regression probes on RAW
    activations, one per condition. Return the two unit-normalised weight
    vectors in residual stream coordinates so the cosine angle between them
    is a direct geometric measurement of how aligned the two conditions'
    arity-classification directions are.

    Rationale: a StandardScaler-based variant would equalise per-dim variance
    across conditions, which can rotate the probe weight relative to the raw-
    space arity direction when per-dim variance differs across conditions
    (e.g., the NEUTRAL probe could lean on a high-variance noise dimension
    that's actually irrelevant to arity but happens to discriminate the small
    canonical sample). Training in raw space removes that artifact at the
    cost of a slightly less well-conditioned optimisation - which is fine
    for LM residual stream activations (post-LayerNorm features have
    approximately bounded magnitudes per layer).
    """
    y_A = np.array([1 if op in UNARY_CANONICALS else 0 for op in labels_A])
    y_B = np.array([1 if op in UNARY_CANONICALS else 0 for op in labels_B])

    clf_A = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(canon_X_A, y_A)
    clf_B = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(canon_X_B, y_B)

    return unit(clf_A.coef_[0]), unit(clf_B.coef_[0])


def random_unit_vectors(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n random unit vectors of dimension dim (rows). Used for the
    'random direction baseline' - what does cosine between an arity direction
    and an unrelated direction look like, given the embedding dimension."""
    v = rng.standard_normal((n, dim))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def random_baseline_summary(d: np.ndarray, n: int, rng: np.random.Generator) -> dict:
    """Return mean / std / quantiles of |cos| between d and n random unit
    vectors of the same dim. The expected magnitude is 1/sqrt(D) for
    isotropic Gaussian samples; we use 100 samples to bound the variance."""
    rs = random_unit_vectors(d.shape[0], n, rng)
    cos = rs @ d
    deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return {
        "n_samples": n,
        "mean_abs_cos": float(np.mean(np.abs(cos))),
        "mean_deg": float(np.mean(deg)),
        "std_deg": float(np.std(deg)),
        "q025_deg": float(np.quantile(deg, 0.025)),
        "q975_deg": float(np.quantile(deg, 0.975)),
    }


# ==============================================================================
# Per-model directional-angle analysis.
# ==============================================================================
def analyse_model(
    spec: ModelSpec, cond1: ConditionActivations, cond2: ConditionActivations,
) -> dict:
    print(f"\n{'=' * 80}")
    print(f"DIRECTIONAL-ANGLE ANALYSIS - {spec.short_name}")
    print(f"{'=' * 80}")
    print()
    print(f"  Reading: small cross-notation angle => the arity direction is")
    print(f"  approximately the same vector in residual stream space across")
    print(f"  notations (consistent with script 18 cross-condition transfer")
    print(f"  succeeding). Near-90-degree angle => the two arity directions")
    print(f"  are approximately unrelated despite both being structural within")
    print(f"  their own notation (consistent with cross-condition transfer")
    print(f"  failing). The random-unit-vector baseline establishes what an")
    print(f"  unrelated direction looks like in this dim - the cross-notation")
    print(f"  angle should be interpreted relative to that baseline.")
    print()

    results: dict = {"model": spec.short_name, "per_layer": []}
    rng = np.random.default_rng(SEED)

    # Establish random-direction baseline once at the focus layer of cond 1
    # (the chosen layer is geometrically arbitrary for this baseline; we only
    # care about the per-dim isotropic statistic).
    focus_layer = spec.cond1_focus_layer
    d_C_focus = centroid_unary_direction(
        cond1.canonical_X[focus_layer], cond1.canonical_labels
    )
    baseline = random_baseline_summary(d_C_focus, n=200, rng=rng)
    print(f"  Random-direction baseline (dim={d_C_focus.shape[0]}, n={baseline['n_samples']}):")
    print(f"    mean angle to random unit vector: {baseline['mean_deg']:.2f} deg "
          f"(std {baseline['std_deg']:.2f} deg, 95% CI [{baseline['q025_deg']:.2f}, {baseline['q975_deg']:.2f}])")
    print(f"    mean |cos|: {baseline['mean_abs_cos']:.4f}  (= 1/sqrt(D)~{1.0/np.sqrt(d_C_focus.shape[0]):.4f})")
    print()
    results["baseline"] = baseline

    # Build a union of diagnostic layers covering both conditions' interest.
    layers = sorted(set(spec.diagnostic_layers
                        + [spec.cond1_focus_layer]
                        + list(spec.cond2_focus_layers)))

    # Header
    print(f"  Per-layer cross-notation angles (NEUTRAL <-> FUNC-PFX):")
    print(f"    layer | centroid (cos, deg) | probe (cos, deg) | within-NEUT C<->P (deg) | within-FUNC C<->P (deg)")
    print(f"    ------+--------------------+------------------+-------------------------+------------------------")

    for L in layers:
        # Canonical activations at this layer
        Xc_neut = cond1.canonical_X[L]
        Xc_func = cond2.canonical_X[L]
        # Centroid directions in raw activation space
        d_C_neut = centroid_unary_direction(Xc_neut, cond1.canonical_labels)
        d_C_func = centroid_unary_direction(Xc_func, cond2.canonical_labels)
        cos_C, deg_C = cosine_angle_deg(d_C_neut, d_C_func)

        # Probe directions in raw residual stream space
        d_P_neut, d_P_func = raw_binary_probe_directions(
            Xc_neut, cond1.canonical_labels, Xc_func, cond2.canonical_labels
        )
        cos_P, deg_P = cosine_angle_deg(d_P_neut, d_P_func)

        # Within-notation centroid-vs-probe sanity check (raw space, both
        # directions live in residual stream coordinates).
        _, deg_within_neut = cosine_angle_deg(d_C_neut, d_P_neut)
        _, deg_within_func = cosine_angle_deg(d_C_func, d_P_func)

        # Invented projections onto centroid directions (raw space).
        # Per condition, project the invented activations onto its in-notation
        # centroid direction and the other-notation centroid direction.
        # Centering is done relative to mean(binary canonical centroid) so the
        # numbers are interpretable as "displacement along the unary axis".
        inv_neut = cond1.invented_X[L]
        inv_func = cond2.invented_X[L]
        bin_center_neut = np.stack(
            [Xc_neut[cond1.canonical_labels == op].mean(axis=0)
             for op in BINARY_CANONICALS]
        ).mean(axis=0)
        bin_center_func = np.stack(
            [Xc_func[cond2.canonical_labels == op].mean(axis=0)
             for op in BINARY_CANONICALS]
        ).mean(axis=0)
        proj_inv_neut_on_neut = (inv_neut - bin_center_neut) @ d_C_neut
        proj_inv_func_on_func = (inv_func - bin_center_func) @ d_C_func
        proj_inv_neut_on_func = (inv_neut - bin_center_neut) @ d_C_func
        proj_inv_func_on_neut = (inv_func - bin_center_func) @ d_C_neut

        print(f"    {L:>5d} | {cos_C:+.4f}, {deg_C:6.2f} | {cos_P:+.4f}, {deg_P:6.2f} | {deg_within_neut:7.2f}  | {deg_within_func:7.2f}")

        results["per_layer"].append({
            "layer": L,
            "centroid_cos": cos_C,
            "centroid_deg": deg_C,
            "probe_cos": cos_P,
            "probe_deg": deg_P,
            "within_neut_centroid_probe_deg": deg_within_neut,
            "within_func_centroid_probe_deg": deg_within_func,
            "proj_inv_neut_on_neut_mean": float(np.mean(proj_inv_neut_on_neut)),
            "proj_inv_func_on_func_mean": float(np.mean(proj_inv_func_on_func)),
            "proj_inv_neut_on_func_mean": float(np.mean(proj_inv_neut_on_func)),
            "proj_inv_func_on_neut_mean": float(np.mean(proj_inv_func_on_neut)),
        })

    # Detailed invented-projection table at focus layers.
    print()
    print(f"  Invented projections onto centroid directions (raw space).")
    print(f"  Positive => mean invented activation lies on the unary side of")
    print(f"  the binary centroid, measured along the named direction.")
    print()
    print(f"    layer | inv_NEUT . d_C^NEUT | inv_NEUT . d_C^FUNC | inv_FUNC . d_C^FUNC | inv_FUNC . d_C^NEUT")
    print(f"    ------+--------------------+--------------------+--------------------+--------------------")
    for row in results["per_layer"]:
        print(f"    {row['layer']:>5d} | "
              f"{row['proj_inv_neut_on_neut_mean']:+.4f}             | "
              f"{row['proj_inv_neut_on_func_mean']:+.4f}             | "
              f"{row['proj_inv_func_on_func_mean']:+.4f}             | "
              f"{row['proj_inv_func_on_neut_mean']:+.4f}")

    # Headline summary at the cond-1 focus layer.
    focus = next(r for r in results["per_layer"] if r["layer"] == spec.cond1_focus_layer)
    print()
    print(f"  Headline at cond-1 focus layer L={spec.cond1_focus_layer}:")
    print(f"    centroid cross-notation angle: {focus['centroid_deg']:.2f} deg "
          f"(cos {focus['centroid_cos']:+.4f})")
    print(f"    probe    cross-notation angle: {focus['probe_deg']:.2f} deg "
          f"(cos {focus['probe_cos']:+.4f})")
    print(f"    within-NEUT centroid-vs-probe: {focus['within_neut_centroid_probe_deg']:.2f} deg")
    print(f"    within-FUNC centroid-vs-probe: {focus['within_func_centroid_probe_deg']:.2f} deg")
    print(f"  (random-direction baseline for this dim: "
          f"{baseline['mean_deg']:.2f} +/- {baseline['std_deg']:.2f} deg)")

    return results


# ==============================================================================
# Driver - mirrors script 18 driver structure.
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

    print(f"  loading tokenizer + model ({spec.dtype}) on {device}...")
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=spec.dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"  model loaded, n_layers={model.config.num_hidden_layers}, "
          f"hidden={model.config.hidden_size}")

    print()
    print(f"  -- Building NEUTRAL condition --")
    cond1 = build_condition(
        "NEUTRAL", model, tok, device,
        canonical_stim_fn=make_neutral_stimuli,
        invented_stim_fn=make_neutral_stimuli,
    )

    print()
    print(f"  -- Building FUNCTIONAL-PREFIX condition --")
    cond2 = build_condition(
        "FUNC-PFX", model, tok, device,
        canonical_stim_fn=make_functional_canonical_stimuli,
        invented_stim_fn=make_functional_invented_stimuli,
    )

    # Done with the model
    free_model(model)
    del tok
    gc.collect()

    results = analyse_model(spec, cond1, cond2)
    print(f"\n  -- {spec.short_name} total time: {time.time() - t_total:.1f}s --")
    return results


def cross_model_summary(all_results: list[dict]) -> None:
    print()
    print("=" * 80)
    print("CROSS-MODEL SUMMARY")
    print("=" * 80)
    print()
    print("  Headline cross-notation angles at each model's cond-1 focus layer:")
    print()
    print("    model        | centroid (deg) | probe (deg) | within-NEUT C<->P | within-FUNC C<->P | baseline (deg)")
    print("    -------------+----------------+-------------+-------------------+-------------------+----------------")
    for r in all_results:
        spec = next(s for s in MODEL_SPECS if s.short_name == r["model"])
        focus = next(row for row in r["per_layer"] if row["layer"] == spec.cond1_focus_layer)
        bl = r["baseline"]
        print(f"    {r['model']:<12} | "
              f"{focus['centroid_deg']:7.2f}        | "
              f"{focus['probe_deg']:7.2f}     | "
              f"{focus['within_neut_centroid_probe_deg']:7.2f}           | "
              f"{focus['within_func_centroid_probe_deg']:7.2f}           | "
              f"{bl['mean_deg']:6.2f} +/- {bl['std_deg']:.2f}")
    print()
    print("  Interpretation (per script 18 transfer asymmetry):")
    print("    centroid + probe cross-notation angle WELL below the random")
    print("    baseline (~90 deg) => globally-aligned arity direction across")
    print("    notations. Angle near baseline => notation-local arity directions.")
    print()
    print("  Within-notation centroid-vs-probe should be small in both models")
    print("  if the centroid direction and the probe weight are both reading")
    print("  the same arity structure. A large within-notation angle would")
    print("  indicate the probe is fitting something other than the centroid-")
    print("  geometry arity attractor.")


def main() -> None:
    log_path = _setup_logging()

    print(f"Script 19 - directional-angle analysis")
    print(f"  HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '<unset>')}")
    print(f"  TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE', '<unset>')}")
    device = pick_device()
    print(f"  device: {device}")
    print(f"  seed:   {SEED}")
    print(f"  N_PER_CLASS: {N_PER_CLASS}")
    print(f"  models: {[s.short_name for s in MODEL_SPECS]}")

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
