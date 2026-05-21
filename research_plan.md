# Thesis Research Plan: Probing the Morphospace of Latent Logical Structure in Open LLMs

## 1. One-line framing

Use the geometric and mechanistic tooling of representation analysis to test whether large language models converge on substrate-independent ("Platonic") logical structures, and to map the conditions under which their internal representations either accommodate or fail to accommodate genuinely novel logical primitives.

This is a hybrid of the two paths originally considered:

- **Path B (epistemological)** provides the question — are there boundaries to the morphospace of representable logical structure, and can we find them?
- **Path A (mechanistic)** provides the method — the claims live in measurable representational geometry, not in interpretations of model output text.

The thesis is defensible only if every high-level claim rests on a lower-level geometric or causal measurement.

---

## 2. Research questions

In order of decreasing scope:

1. **The Platonic question.** Do LLMs represent logical/algebraic structures in a substrate-independent way, such that the same structure expressed in radically different surface forms aligns in representation space?
2. **The novelty question.** When presented with structural primitives that are absent from the training distribution, does the model (a) form genuinely new representational directions, (b) project the novelty onto existing structures, or (c) collapse into noise?
3. **The scaling question.** Do whatever boundaries exist move predictably with model scale, training compute, and architectural choice — or are they idiosyncratic per model?

Question 1 is the safest publishable result. Question 2 is the headline. Question 3 is the validation that the headline isn't a single-model artifact.

---

## 3. Model selection

### Primary: OLMo 2 (AI2)

**OLMo 2 7B for local laptop work, OLMo 2 13B for cluster experiments.** Same model family, identical infrastructure, scales the rigor as compute allows.

Rationale:

- **Fully open training data, code, and intermediate checkpoints.** This is the only sub-14B family where we can credibly answer "was this structure in training?" by actually searching the corpus. Every other choice forces us to wave hands about "novel relative to training" without verification.
- **Competitive capability.** Comparable to Llama 3.1 8B on logic/reasoning benchmarks; sufficient for our probes.
- **Released November 2024**, actively maintained by AI2.
- **Fits the M4 budget** — 7B in fp16 is ~14GB, leaving room for activations and tooling. 13B in int8 is ~13GB, still local-runnable for inference, though cluster is preferred for serious extraction.

### Secondary models (cluster-only)

- **Gemma 2 9B + Gemma Scope SAEs** — validation target for cross-model replication. Gemma Scope provides pre-trained, labeled sparse autoencoders across all layers, eliminating ~6 months of SAE training work. Use when we need to *name* which features carry the structural signature.
- **Pythia 1.4B and 6.9B** — used solely for the scaling/training-dynamics analyses. Pythia's 154 intermediate checkpoints per model are unique and irreplaceable. Weak models, but the question they answer (how does the representation emerge during training?) doesn't require frontier capability.

### Explicitly rejected

- **Llama 3.1/3.2** — closed training data, no training transparency.
- **Qwen 2.5** — capable but training data not auditable.
- **Mistral / Mixtral** — same issue.
- **Anything >14B** — unnecessary for the research question, and the marginal capability is not worth the compute cost or the loss of laptop iteration.

---

## 4. Local development on M4 (48GB unified memory)

The MacBook Pro M4 with 48GB unified RAM is a development workstation for this project, but it is **not** a guaranteed activation-extraction target. The PyTorch MPS backend has known silent-fallback behavior on the exact operations that interpretability libraries depend on (custom hooks, complex tensor slicing, modified hidden-state writes, einops-heavy ops). OLMo 2 in particular uses SwiGLU activations and RoPE; if a native MPS kernel for any hook-targeted op is missing, PyTorch will either silently fall back to CPU (with catastrophic throughput) or crash with `NotImplementedError`.

**Treat local extraction as conditional on a Week 1 smoke test (see Section 9).** Until that test passes, plan as if all real extraction will happen on a remote cluster (nnsight's NDIF service, university cluster, or rented GPU). The M4 is unambiguously useful for stimulus generation, pipeline scaffolding, geometric/TDA code development on dummy matrices, and analysis of *already-extracted* activations cached as tensors.

### What fits locally (if MPS smoke test passes)

| Model | Precision | Approx RAM | Use case |
|---|---|---|---|
| Pythia 1.4B | fp16 | ~3 GB | Pipeline development, fastest iteration |
| Pythia 6.9B | fp16 | ~14 GB | Small-scale probes |
| OLMo 2 7B | fp16 | ~14 GB | **Primary local target** |
| OLMo 2 7B | int8 | ~7 GB | Long-context activation extraction |
| OLMo 2 13B | int8 | ~13 GB | Local capability validation |
| OLMo 2 13B | 4-bit | ~7 GB | Quick local sanity checks |
| Gemma 2 9B | int8 | ~9 GB | Local Gemma Scope exploration |

Budget guideline: keep the model resident under ~25 GB to leave room for activations (which can be large for long sequences and wide layers), the OS, and your normal workflow tools. The 7B in fp16 is the sweet spot.

### M4 viability — confirmed by Phase 0 smoke test and 7B scale-up

The Week 1 MPS smoke test (`experiments/03_mps_smoke_test.py`) returned **GREEN**: OLMo 2 1B in fp16 achieves **527 tok/s** in batched forward passes on the M4 with sustained GPU activity, no MPS fallbacks, and 4.2 GB RSS. Hidden-state extraction across all 16 transformer layers works cleanly via `output_hidden_states=True`.

**Local-first development is confirmed viable for OLMo 2 7B as well.** Scripts 08, 09, 10 all run successfully against `allenai/OLMo-2-1124-7B` in fp16 on the M4: ~70s load time once weights are cached, ~14 GB RSS during forward passes, ~91s extraction time for 200 prompts (or ~141s for 1200 prompts in script 10's 4-condition pass), with no MPS CPU-fallbacks. The cluster-first fallback below remains documented for future operations (nnsight-specific hooks, custom kernels, very long contexts, SAE training, persistent homology on full-rank activations) that may exceed the M4's working set or exercise MPS ops not yet validated.

### Cluster-first fallback plan

If the MPS smoke test fails (or any subsequent op in the pipeline triggers a CPU fallback that throttles extraction below ~5 tok/s), the workflow shifts to:

- **nnsight remote execution via NDIF** for OLMo 2 7B/13B and Gemma 2 9B activation extraction. The M4 then runs the *driver* code; activations come back as tensors over the network.
- **University cluster / rented A100/H100** for SAE training, persistent-homology computations on full-rank activations, and any fine-tuning beyond LoRA.
- **Local M4** retains stimulus generation, geometric analysis on cached/projected activations, plotting, writing, and small Pythia runs (which use simpler architectures with better MPS coverage).

This is the more robust default; the local-first plan is only adopted if Week 1 evidence supports it.

### Software stack on macOS

Two parallel toolchains are worth maintaining:

**MLX (Apple's native framework)** — fastest inference and lowest memory on Apple Silicon. Use for:

- Bulk inference / stimulus generation
- Quick activation dumps with `mlx-lm`
- LoRA fine-tuning experiments

**PyTorch + MPS backend** — broader ecosystem compatibility. Required for:

- `TransformerLens` (some MPS support; some ops fall back to CPU)
- `nnsight` (works with MPS; preferred for OLMo 2)
- `SAELens`
- Anything depending on a HuggingFace `transformers` model

Recommended setup:

```bash
# Conda env, Python 3.11
conda create -n thesis python=3.11
conda activate thesis

# Core stack
pip install torch torchvision  # ships with MPS support
pip install transformers accelerate bitsandbytes
pip install transformer-lens nnsight sae-lens
pip install mlx mlx-lm  # Apple-native, used in parallel

# Geometric / topological analysis
pip install giotto-tda ripser scikit-tda umap-learn
pip install scikit-learn scipy einops

# Information theory
pip install nats-toolkit  # or roll our own MI estimators
```

### What does NOT fit locally

These need cluster time:

- Training SAEs from scratch on OLMo 2 (use Gemma Scope on Gemma instead, or budget cluster time)
- Full fine-tuning of 13B (LoRA only on laptop)
- Persistent homology on very large activation matrices (memory blow-up at scale)
- The full Pythia scaling sweep
- Anything in Phase 3 at headline-paper scale

---

## 5. Toolset

### Mechanistic interpretability
- **TransformerLens** — activation extraction, patching, on Pythia and smaller OLMo runs
- **nnsight** — preferred for OLMo 2 13B and Gemma 2 9B (handles modern architectures better than TransformerLens)
- **SAELens** — SAE training and loading
- **Gemma Scope** — pre-trained SAEs for the Gemma 2 validation target

### Geometric and topological
- **giotto-tda** or **ripser** — persistent homology on activation manifolds
- **scikit-tda** — supporting topological tools
- Custom **CKA** and **Procrustes** implementations — representational similarity
- **UMAP / PaCMAP** — visualization only, never as the basis for a claim

**Mandatory pre-TDA reduction step.** Persistent-homology computation scales catastrophically with both point count and ambient dimension. A raw activation cloud of 10,000 vectors from a 4096-dim OLMo layer will exhaust 48 GB of unified memory before \(H_1\) completes, and will not finish on a cluster either without bespoke optimization. Every TDA pipeline in this thesis must include a deterministic, reproducible reduction step before filtration:

1. **Johnson-Lindenstrauss random projections to 100–300 dimensions** as the default. JL projections preserve pairwise distances up to a multiplicative \(1 \pm \varepsilon\) factor with probability ≥ \(1 - \delta\), with \(k \geq O(\varepsilon^{-2} \log n / \delta)\). This in turn preserves persistent-homology births and deaths up to the same scaling factor. *Critically, JL projections do not preferentially preserve high-variance directions.* (External-review consequence, May 2026: an earlier version of this plan used PCA-to-30-50-dim. PCA preserves variance, which in LLM residual streams is dominated by token-frequency, positional, and gross-grammatical features. Operator-identity differences — exactly what we want to measure — likely live in lower-variance subspaces that PCA discards. JL projections are the correct default.)
2. **Landmark / witness complex** (e.g., maxmin sampling) for point-count reduction when the cloud has more than ~2,000 vectors. Applies orthogonally to the JL step.
3. **SAE-based feature filtering** (e.g., Gemma Scope SAEs) as a model-specific alternative when an SAE is available. Filters activations to the sparse subset of features active for the stimulus class, often a more interpretable basis than projection. *For OLMo 2 specifically, no public SAEs exist as of Phase 0 wrap-up; JL is the default.*
4. **PCA** only as a sensitivity-analysis baseline, never as the primary reduction. UMAP / PaCMAP for visualisation only, never as the basis for a claim, and never without sensitivity analysis on `n_neighbors` and `min_dist`.

The reduction parameters must be locked in before Phase 0 → Phase 1 and reported alongside every persistent-homology figure. This is the difference between a topological *measurement* and a topological *illustration*.

### Probing and causal
- **scikit-learn** — linear probes
- **PyTorch** — nonlinear probes when needed
- **Activation patching** via TransformerLens — causal claims
- **Causal scrubbing** — formal validation framework for any circuit-level claim

### Information theory
- **MINE / InfoNCE** — mutual information estimation between representations and structural variables
- **Compression-based Kolmogorov complexity proxies** (gzip, BWT, zstd) — for bounding stimulus complexity
- **nats-toolkit** or hand-rolled estimators — entropy / MI on discrete distributions

---

## 6. The taxonomy of edges

The single most important conceptual commitment of the thesis. "Morphospace boundary" is a metaphor; it must be operationalized into specific, measurable kinds of edge. Five candidates, three chosen:

| # | Edge type | What it tests | Status |
|---|---|---|---|
| 1 | **Compositional** | Operators seen separately, never composed | Rejected — too well-trodden |
| 2 | **Operator-novelty** | Structural primitives absent from training | **Chosen** — the headline claim |
| 3 | **Substrate-invariance** | Same structure, alien surface form | **Chosen** — the cleanest Platonic test |
| 4 | **Complexity** | Stimuli with controlled algorithmic complexity | Rejected — separate dissertation |
| 5 | **Distributional** | Quantitatively far from training, controlling for structure | **Chosen** — mandatory baseline |

The thesis commits to **(2), (3), and (5)**. Edge type (5) is the null-hypothesis class against which the other two are always measured.

### Operationalizations

**Substrate-invariance (edge type 3):**
Take a structure the model demonstrably knows — e.g., the symmetric group \(S_4\), propositional logic, or simple type theory. Re-present it through a progression of surface transformations:

1. Standard textbook notation
2. Renamed symbols (consistent but novel names)
3. Invented alphabet (Unicode private-use area)
4. Adversarial encodings (rare-tokenization sequences)

Measure representation alignment (CKA, Procrustes) at each step. The Platonic claim predicts high alignment across steps 1–3 with a possible break at step 4. The null predicts alignment that decays monotonically with surface distance.

**Operator-novelty (edge type 2):**
Construct a small formal language with primitives absent from the training distribution:

- Ternary logical connectives (no Boolean reduction)
- Non-classical quantifiers (e.g., "most," "exactly half" as primitives)
- Novel algebraic operations satisfying specific axioms

Train a linear probe to predict structural properties of expressions. Measure whether the probe finds a new, well-conditioned direction or projects onto existing logical-operator directions. Causal intervention: ablate the candidate direction and check whether structural-property prediction degrades selectively.

**Distributional baseline (edge type 5):**
For every stimulus in (2) and (3), construct matched controls:

- Random-noise stimuli at equivalent token-distance from training
- Structurally-scrambled versions of the same stimuli (same tokens, broken structure)
- Common natural-language paraphrases (the "easy" baseline)

Quantify "distance from training" using both embedding-space distance to nearest-k OLMo training samples and n-gram coverage against the OLMo corpus. Pre-register the threshold for "out-of-distribution" before running.

**Perplexity-matched controls — mandatory hidden confounder check.** A novel synthetic stimulus will warp the activation manifold for at least two structurally distinct reasons:

1. The model is encoding a genuinely new logical structure (the morphospace claim we want to defend).
2. The model is simply experiencing a large information-theoretic surprise spike (perplexity / cross-entropy) and the manifold is responding to *that*, not to structure.

These produce visually similar geometric and topological signatures. The control we owe ourselves: for every novel-structure stimulus, we need a "fluent but structureless" comparison that controls for surprise.

**Phase 0 result (`experiments/05_perplexity_matching.py`):** uniform-random token sampling produces sequences with perplexity in the **2M–11M** range against any meaningful target perplexity (1–100). Naive rejection sampling cannot bridge this gap — a known result in the literature that confirms uniform-random is the wrong null. The realistic control ladder is:

- **Token-shuffling of natural prompts.** Same token bag, broken structure. Perplexity typically within 1–2 orders of magnitude of the original. **Primary control** — already implemented in the toy substrate-invariance experiment.
- **High-temperature sampling from the base model itself.** Produces locally-plausible but globally-incoherent text. Perplexity is naturally mid-range.
- **Markov-chain / n-gram sampling from the training corpus.** Locally fluent, globally structureless, with very low local perplexity.
- **Uniform random tokens.** Retained only as a "perplexity ceiling" reference, not as the matched control.

The methodology generalizes from "find a random sequence at the same perplexity" to "show that the structured-stimulus signature is distinct from *every point* on this ladder of structure-stripped controls." This is actually more rigorous than the original framing — it tests whether the structural signature is qualitatively different from the entire family of surprise-only responses, not just one matched point.

**Tokenization handling for alien surface forms (substrate-invariance probe support).**
The substrate-invariance probe runs into a hard floor at the tokenizer: invented or rare Unicode symbols typically fall back to byte-level BPE encoding, fragmenting each "alien glyph" into 3–4 opaque byte tokens with no semantic embedding signal. This both destroys the per-symbol comparison structure and conflates "novel surface form" with "long input." The probe handles this through a three-tier ladder, run in sequence:

- **Tier 1 — single-token rare symbols.** Pre-screen the OLMo 2 tokenizer's existing vocabulary for tokens that are (a) single-token glyphs, (b) extremely low-frequency in the Dolma corpus, and (c) semantically inert in normal contexts.
- **Tier 2 — designed multi-token glyphs.** Multi-token sequences that *consistently* tokenize the same way across uses, so they can be treated as compound symbols with a known token span.
- **Tier 3 — Unicode private-use-area glyphs.** Genuinely invented characters. These will fragment under byte-fallback BPE; tokenization length per symbol is recorded as an explicit controlled variable, and representations at the symbol level are constructed by aggregating across the symbol's token span.

**Tier 1 inventory measured against the actual OLMo 2 tokenizer (Phase 0 result, see `experiments/02_tokenization_screening.py`):**

| Use | Available Tier 1 glyphs | Count |
|---|---|---|
| Variables | Greek lowercase: α β γ δ ε η θ ι κ λ μ; Cyrillic Э Я | ~13 |
| Operators | ¬ → (everything else, including ∧ ∨ ⊕ ↔ ∀ ∃ ∈, falls back to bytes) | 2 |

The "exotic glyph reservoir" hypothesis is **refuted** for OLMo 2 — Linear B, Tifinagh, Math Script, Fraktur, Alchemical, and PUA glyphs all byte-fragment. This has two consequences:

1. **Operator renaming cannot use Tier 1 single glyphs.** The substrate-invariance probe for operator renaming must instead use *invented operator words* (multi-letter strings that tokenize consistently as Tier 2), e.g., `XQUI`, `ZUVI`. Less visually alien but methodologically cleaner.
2. **Tier 3 (true byte-fallback) is the rule, not the exception, for any genuinely novel glyph.** This means the anchor-aggregation methodology for cross-tokenization-budget comparison is mandatory infrastructure, not a stress test.

A Week 1 pipeline utility takes a candidate symbol set and reports per-symbol: tokenization length, token IDs, Dolma-corpus frequency of each token, and embedding-layer cosine similarity to common symbols.

**Representation alignment across tokenization budgets.** When comparing representations across surface forms with different token counts (Tier 3 vs. canonical), we cannot align by sequence position. Instead we define structural anchors ("operator X appears here," "operand 1 appears here") and compare *anchor-aggregated* representations rather than raw position-indexed ones. The aggregation method (mean / attention-weighted / last-token) is reported alongside every CKA or Procrustes figure.

A subtle but important conceptual point: tokenization is itself the first morphospace boundary in a layered stack — tokenizer range → embedding-layer distinctions → deeper structural abstractions. The substrate-invariance probe in particular is testing the transition between the second and third of these. This should be made explicit in any paper that comes out of Phase 1.

### The role-bound asymmetry (Phase 0 finding)

A finding emerged from Phase 0 that materially refines the substrate-invariance edge: variables and operators do *not* behave the same under renaming. The thesis must from now on treat substrate-invariance as a per-role property, not a global one.

**The asymmetry, in one paragraph.** Under operator-anchored extraction at OLMo 2 7B layer 7, variable renaming (Greek-lowercase Tier-1 substitution for English letters in a propositional-logic template) preserves perfect operator-identity probe accuracy (1.000) — variables are essentially interchangeable. Operator renaming (Tier-2 invented words for `and`, `or`, `not`, `implies`) drops probe accuracy to 0.290, with 186 of 200 invented operators classified as `not` regardless of which canonical they replaced. The asymmetry replicates at 1B (probe acc 0.165, *below chance*) and at 7B (0.290, slightly above chance), and is *robust to a 4× variation in invented-operator BPE subword count* (peak gaps 0.71–0.75 across L ∈ {1, 2, 3, 4}).

**The H1 / H2 / H3 / H4 framework (finalized by scripts 14, 15, and the script-16 external-review-driven stress test).** The pre-registered hypotheses for the operator-renaming failure are now operationalized as:

- **H1 (default to the arity-region — formerly "unary-class region"):** the model has an *arity-region attractor* at the operator-anchored position. **Confirmed at 100% mass** in the cleanest single measurement (script 16, functional-prefix notation × functional-prefix-trained probe), at 99.6% mass in script 15 (neutral-train × neutral-test), at 94.6% in the rich-template factorial (script 13), and at 91–93% across the subword-length sweep (script 10). The attractor is **constructed by attention/MLP layers 1–7**, not inherited from layer-0 embedding geometry (script 14: all five invented words have peak layer-0 similarity to `and`/`or`, yet 0% of layer-7 probe predictions go to any binary canonical). It is **independent of syntactic position** (script 16: binary-replacement invented words in prefix-binary functional-call slots are still classified as unary 100% of the time, refuting the prefix-vs-infix syntactic-position confound). The within-arity identity (specifically which unary canonical an invented word maps to) is **fragile and probe-instrument-dependent** — the same word can split 100:0 in one notation and 0:98 in another. The Phase 1 claim is therefore *arity-encoded, within-arity-identity-not-encoded* for invented operators.
- **H2 (tokenization-position effect):** **Rejected** by script 10.
- **H3 (word-specific embedding-similarity bias):** **Confirmed in two regimes** but **largely NOT a layer-0 phenomenon** (script 14): cross-class escape (`bar` → `or` ~74% regardless of slot; script 11) and within-unary modulation (e.g., `molex` → 98% necessarily even in neutral templates, despite having the lowest layer-0 sim(necc) of the five invented words; script 15 Test 1). Layer-0 cosine similarity predicts script-13 top landing for only 1 of 5 words; the H3 mechanism is constructed by intermediate layers.
- **H4 (template-lexical-context bias):** **Reframed as probe-instrument artifact** by script 15. The script-13 finding of an H4 channel pulling toward the template's "owned" canonical does not survive a probe-instrument change. A probe trained on neutral templates sees the same B'_rich activations and routes them to `implies` (driven by "If... then..." scaffolding in the rich templates), not to the template's owned canonical. The H4 numbers from script 13 are valid probe-internal measurements but do not reflect a deep template-context channel in the residual stream.

The combined picture, post-scripts-14-and-15: a unary-region attractor (H1) catches 99–100% of invented-operator probability mass when probe-train and probe-test are matched-template-family; the within-unary not↔necessarily split is roughly uniform in the cleanest measurement (with one word-specific anomaly, molex → necessarily); cross-class escapes (H3 cross-class) occur for a small number of invented words whose mid-layer trajectories happen to land outside the unary region. The unary-region attractor is the principal Phase 0 finding; it survives all probe-instrument changes and is independent of token-embedding geometry.

**Phase 1 entry refinement (scripts 17–18, with peer-review-driven recalibration).** Cross-model replication on Gemma 2 9B (`google/gemma-2-9b`; different lab, training data, architecture, tokenizer) reproduces the within-notation arity-region attractor in both NEUTRAL-metalinguistic and FUNCTIONAL-PREFIX notations (peak unary mass 97.6% NEUTRAL @ L4; 100% FUNC-PFX @ L2 and L16-17 within-condition). A four-diagnostic probe-artifact battery (cross-condition probe transfer, held-out canonical, probe-free centroid geometry, last-subword embedding baseline) on both Gemma 2 9B and OLMo 2 7B reveals a substantive model-level difference: **the cross-context stability of the arity direction is model-specific**. After applying a canonical-transfer gate (cross-canonical 5-class accuracy ≥ 0.65 at the same train/test pairing), Gemma 2 has cross-condition-transfer-compatible arity directions at validated early layers (NEUTRAL@L4 ↔ FUNC-PFX@L4 with canonical-transfer accuracy 1.000; FUNC-PFX@L2 → NEUTRAL with 0.756). OLMo 2's tested arity directions at L7 do not transfer across notations: cross-canonical 5-class accuracy collapses with all 250 invented stimuli classified as `and` (NEUTRAL → FUNC-PFX direction) or `implies` (reverse). The Gemma 2 L16 "candidate late re-emergence" pairing has high invented-unary mass (99.6%) but cross-canonical accuracy 0.564 within-layer and 0.200 cross-layer to NEUTRAL@L4 — at or near chance, so it is reported as a candidate requiring stronger canonical-transfer validation rather than a confirmed finding. Both models have within-notation geometric arity attractors at the operator-anchored position; what differs is whether the arity direction is approximately the same direction across notations (Gemma 2, at validated early layers) or notation-local (OLMo 2, at all tested layers). The Phase 1 entry headline is therefore *cross-model arity-region substrate-invariance with model-specific cross-context stability of the arity direction* — a richer claim than "the finding replicates", and (after the recalibration) a defensible one: characterised quantitatively as a directional cosine angle (script 19), with the canonical-transfer gate as the methodological backbone, and ready for testing across additional models (Pythia next).

**Four methodological consequences for the rest of the thesis.**

1. **Probe A reports must factor variables vs operator-arity separately, with within-arity reports nested.** A single "substrate-invariance" number averaging variable and operator is misleading. Variables are at ceiling; operator-arity (unary vs binary) is at ceiling; operator identity within arity is at floor for binary and content-dependent for unary. The report structure: (variable invariance) | (operator-arity invariance) | (within-arity operator identity invariance, per arity).
2. **Every stimulus that uses invented surface forms must report each invented form's embedding distance to each canonical it could be confused with.** Useful for documentation purposes, but with one major caveat from script 14: layer-0 embedding similarity does *not* reliably predict layer-7 probe behaviour. Pre-screening of invented operators by embedding-similarity-to-canonical is necessary but not sufficient to identify which invented words will have anomalous probe behaviour (e.g., the molex → necessarily effect is invisible to a layer-0 screen).
3. **Probe results must be reported across multiple training-template regimes.** Single-instrument reporting is unsafe. Minimum: (a) probe trained on canonical-neutral templates, (b) probe trained on the same template family as the test data. Mismatched train/test produces qualitatively different prediction distributions on the same activations (script 15: 94.6% unary mass with matched probe vs 21.2% with mismatched).
4. **The cleanest H1 measurement is in canonical-neutral templates** (`experiments/15_template_neutral_probe.py`). Phase 1 stimulus design for any substrate-invariance probe should default to neutral-template form unless there is a specific reason to use lexically-rich templates.
5. **Cross-condition probe transfer is the gold-standard substrate-invariance instrument across notations and models** (Phase 1 entry, script 18 Diagnostic A). Within-condition probe CV accuracy alone is no longer sufficient — a probe can hit CV=1.000 at every layer in functional-prefix notation by reading propagated previous-token identity rather than structural arity (Gemma 2 FUNC-PFX, script 17; partial in OLMo 2 FUNC-PFX, script 16). Every Phase 1 / Phase 2 substrate-invariance claim should report (a) within-condition probe CV, (b) cross-condition probe transfer in both directions, and (c) the held-out canonical generalisation test. The cross-condition transfer measurement is what makes the model-specific cross-context stability finding visible at all.

---

## 7. Experimental scaffold

Four phases over four years. Decision points at the end of Phase 0 and Phase 1.

### Phase 0 — Infrastructure and de-risking (months 1–6)

**Deliverable:** working pipeline + go/no-go decision on the research program.

**Week 1 — Infrastructure smoke tests (gate everything else on these):**

- **MPS native-execution check.** Run nnsight against OLMo 2 7B on a 10-token prompt with residual-stream hooks at multiple layers. Use macOS Activity Monitor and PyTorch's profiler to verify: no CPU fallbacks, no swap usage, throughput acceptable. If this fails, switch immediately to the cluster-first plan in Section 4 before writing any stimulus code.
- **Tokenization screening utility.** Build the per-symbol report tool described in Section 6 against OLMo 2's tokenizer. Use this to pre-curate Tier 1 / Tier 2 / Tier 3 candidate symbol sets before the substrate-invariance probe goes near a model.
- **Dolma corpus mirror.** Set up the local or university-cluster mirror of the OLMo 2 training data (or its searchable index). Verify n-gram queries return correct frequency counts for both common and rare strings. Version-pin the dataset snapshot we use.

Pipeline components (Weeks 2–8):

1. Stimulus generation framework (parameterized, reproducible, version-controlled)
2. Tokenization sanity checks integrated as a pre-commit hook on the stimulus generator
3. Perplexity scorer + perplexity-matched random-stimulus generator (mandatory control infrastructure)
4. Activation extraction (nnsight, via the path determined by the Week 1 MPS test)
5. Geometric analysis (CKA, Procrustes, linear probes)
6. TDA pipeline with the mandatory PCA/landmark reduction step
7. Reporting / plotting

De-risking probes (each ~6 weeks):

- **Probe A: Substrate-invariance (revised after Phase 0).** Present a known structure (e.g., \(S_4\) or propositional logic) under multiple surface forms and report invariance metrics *separately for variable-role and operator-role substitutions*. Success criterion (pre-registered, revised three times during Phase 0):
  - **Per-role reporting mandate.** Report variable substrate-invariance and operator substrate-invariance as separate metrics. A single averaged number is not admissible. Phase 0 established that the two roles behave at the extreme ends of the invariance spectrum (variables at ceiling, operators at floor), so averaging hides both effects.
  - **Multi-pooling mandate.** Report at minimum three pooling/extraction strategies side-by-side: (a) last-token residual stream, (b) mean-pool over all tokens, (c) anchor-position extraction (extracted at the token position immediately following the renamed element). No claim is admissible from a single strategy alone — at least two of the three must show a consistent signal.
  - **Linear-probe gap as primary metric for operator invariance.** Train an operator-identity logistic-regression probe on canonical activations; report held-out accuracy on each substitution condition (variable-renamed, operator-renamed, both-renamed). The probe gap (canonical CV accuracy minus condition held-out accuracy) is the principal single-number metric for operator substrate-invariance, replacing the earlier CKA-based threshold. Phase 0 found the probe gap is the cleanest measurement; CKA is reported as a supporting curve but not as a success-criterion-bearing metric.
  - **Embedding-distance reporting for invented surface forms.** Before any substrate-invariance claim, report embedding-layer cosine similarity between each invented operator's leading-space tokenization and each canonical operator's tokenization. The `bar` → `or` finding (script 11) established that the H3 embedding-similarity channel produces false-positive substrate-invariance recoveries. Any invented surface form whose embedding distance to a non-target canonical is materially smaller than the within-canonical-set embedding distance distribution must be reported as a flagged confound.
  - **Confusion-matrix reporting at the peak-gap layer.** Probe accuracy alone is insufficient — the confusion-matrix structure tells us *which* canonical class invented operators get mapped to. Phase 0 found a uniform default to `not` (200/200 for B'' at 7B layer 7) that would be invisible in scalar-accuracy reporting.
  - **Direction consistency.** The B-vs-B' gap (variable-renaming CKA minus operator-renaming CKA at the same position) is reported across all three pooling strategies and used as a sanity check: artifacts manifest as inconsistency across the three.
  - Replicated across at least 3 seeds, at least 2 base structures, and at least 2 model scales (now established at 1B and 7B for OLMo 2; Phase 1 adds Gemma 2 9B at minimum).
  - Earlier criteria are retired: (1) "CKA > 0.7 absolute" (Phase 0 found mean-pool produces ~0.95 CKA even on token-bag overlap, making absolute CKA uninformative); (2) "use last-token, gap > 0.3" (Phase 0 found last-token has a severe positional bias that produces apparent strong-Platonic results on operator renaming which collapse under mean-pool and anchor-position extraction); (3) "single averaged substrate-invariance metric" (Phase 0 found variable-vs-operator asymmetry too large to average over).
- **Probe B: Novel operator.** Define a single ternary connective with a clear truth table; test whether a linear probe finds it as a new direction. Success criterion: probe accuracy > 80% with the direction being approximately orthogonal to learned binary-connective directions. Operator must be a Tier 2 invented word (not a Tier 1 alien glyph — see operator-tokenization result in Section 6). Probe is trained and evaluated at the operator-anchored position (not last-token), based on the Phase 0 finding that operator-position activations are where operator semantics are most clearly represented.
  - **Caveat from Phase 0:** the H1 default-to-`not` finding means a Probe B "new direction" claim must explicitly distinguish a *new direction* from a *collapse into the default-attractor direction* (which Phase 0 has now characterized at the operator-anchored position). Phase 1 Probe B must include a control showing the probe direction is geometrically distinct from the `not`-attractor direction identified in Phase 0.
- **Probe C: Topological characterization.** Compute persistent homology (H0, H1, H2) on activation manifolds for 10 stimulus families after the mandatory Johnson-Lindenstrauss reduction step (§5; replaces the earlier PCA-based reduction). Success criterion: at least one qualitative topology change between in-distribution and OOD families that survives perturbation testing.

**Decision point at month 6:**
- 2 of 3 probes show signal → proceed to Phase 1.
- 1 of 3 → narrow the thesis to the surviving probe; reconsider whether the program is still ambitious enough.
- 0 of 3 → pivot. This is not a failure; it's a four-month investment that saved three years.

### Phase 1 — Establish one rigorous primitive (months 7–18)

**Deliverable:** first publication, e.g., "Substrate-invariant representations of algebraic structure in OLMo 2."

Take the strongest signal from Phase 0 and make it bulletproof:

- Full counterfactual baseline matrix
- Statistical power analysis (sample sizes, multiple-comparison corrections)
- Replication across seeds and OLMo 2 intermediate checkpoints
- At least one causal intervention experiment (activation patching, not just correlational geometry)
- Sensitivity analysis on every methodological choice

The deliverable is a paper where every claim is mechanically traceable to a measurement, with no interpretive gaps. This phase is where the thesis earns its credibility.

### Phase 2 — Cross-model and scaling (months 19–30)

**Deliverable:** scaling-laws-style paper on how the morphospace boundary moves with size and training compute.

- Replicate Phase 1 on Gemma 2 9B using Gemma Scope to *name* the features involved. This is where we stop saying "the representation" and start saying "features F_2317 and F_8842 in layer 14 carry the structural signature."
- Replicate across the Pythia scale ladder (1.4B → 6.9B → 12B) using intermediate checkpoints to characterize emergence dynamics.
- If feasible, include Qwen 2.5 7B as an architectural-variant control.

### Phase 3 — The morphospace claim (months 31–42)

**Deliverable:** the headline paper the thesis is named for.

Now we have the right to attempt the big claim. Design stimulus families that demonstrably live outside OLMo 2's training distribution (verified by corpus search and embedding distance) and test the operator-novelty edge at full rigor:

- Persistent homology on activation manifolds as the centerpiece — looking for new homology classes appearing under novel-operator stimuli, not present under matched controls
- Sparse-autoencoder analysis on OLMo 2 (now budget-justified given Phase 1–2 results) to identify whether novel features are emerging or existing features are being recombined
- Cross-model replication of the headline result

### Phase 4 — Synthesis and theory (months 43–48)

**Deliverable:** the dissertation.

Includes a theoretical chapter that ties the empirical findings to:

- The Platonic Representation Hypothesis (Huh, Isola et al. 2024) and surrounding literature
- Algorithmic information theory framings of representational novelty
- Category-theoretic / structural realist accounts of what "structure" means in this context
- Cognitive science and philosophy of mind implications

This is the chapter that earns the philosophical scope. It is the last thing written, not the first.

---

## 8. Risks and explicit decision criteria

A few things to flag now rather than discover at month 18:

- **Phase 0 → Phase 1 transition is where this kind of thesis usually dies.** The success criteria above are pre-registered intentionally; do not weaken them retroactively. If the probes don't hit, the honest move is to pivot, not to redefine success. *Phase 0 status: Probe A has produced a publishable result — **arity-region substrate-invariance at the operator-anchored position in OLMo 2 1B and 7B**. The arity-attractor finding is confirmed at 99.6–100% mass across four independent probe instruments (multi-op-rich, single-op-rich, neutral-metalinguistic, functional-prefix), is independent of layer-0 token-embedding geometry (script 14), and survives a prefix/infix syntactic-position dissociation (script 16, external-review-driven stress test). Within-arity identity is **not** robustly encoded for invented operators (within-unary not-vs-necessarily varies 0:98 to 100:0 across notations); this is reported as a within-arity-fragility caveat rather than a positive claim. Probe B (operator-novelty) and Probe C (topological) are not yet started. Decision-point check is satisfied by Probe A alone; Phase 1 begins.*
- **MPS / Apple-Silicon execution risk.** Silent CPU fallbacks on hook-based activation extraction are a real and well-documented failure mode. The Week 1 smoke test exists specifically to catch this before we invest weeks in stimulus design that depends on a local extraction workflow.
- **Perplexity confound.** Without perplexity-matched controls, a manifold warp caused by sheer model surprise will be indistinguishable from a structural attractor in the activation geometry. Treating perplexity-matched controls as optional in any experiment is a thesis-killer.
- **Tokenization fragmentation.** Naive use of invented Unicode glyphs produces byte-fallback fragmentation that destroys per-symbol comparison structure. The Tier 1/2/3 ladder and the Week 1 screening utility exist to prevent this; never skip them for any substrate-invariance experiment.
- **TDA computational cliff.** Persistent homology on raw activation clouds will exhaust any reasonable memory budget. Every TDA computation in the thesis must go through the mandatory dimensionality-reduction step.
- **Gemma Scope dependency.** If DeepMind deprecates Gemma Scope or it turns out to have systematic issues for our stimulus types, Phase 2 needs a fallback (training SAEs on OLMo 2, which is itself a year of cluster work).
- **OLMo training-data search is the unique value proposition.** If access ever changes, the rigor argument weakens. Mirror the corpus locally early (resolved — see Section 10) and version-pin the dataset release we use.
- **"Novel relative to training" is a sliding scale, not binary.** Commit to a quantitative operationalization (probably embedding distance to nearest-k training samples plus n-gram coverage) in Phase 1 so it cannot be gerrymandered later.
- **The interpretive trap.** The single biggest risk is producing geometric measurements and over-interpreting them as morphospace claims. The discipline of always reporting the null-hypothesis baselines (distributional *and* perplexity-matched) is the structural safeguard.
- **The single-pooling trap (Phase 0 lesson).** Script 07 produced an artifact (apparent strong-Platonic result for operator renaming) under last-token pooling alone; script 08's three-pooling triangulation corrected it to a 0.259-point gap in the operator-anchored measurement. Any Phase 1 or Phase 2 representational-similarity claim must report at least two pooling strategies; single-pooling pre-registration is now formally retired.
- **The H3 channel (Phase 0 lesson).** The `bar` → `or` recovery in script 10 was an embedding-similarity accident, not a structural finding. Any Phase 1 substrate-invariance claim that uses invented surface forms must include a per-stimulus embedding-distance report against canonical, or it risks publishing an embedding accident as a structural result.
- **The within-condition-probe trap (Phase 1 entry lesson; scripts 17 and 18).** A linear probe can hit CV=1.000 at every single layer in functional-prefix notation by reading propagated previous-token identity at the operator-anchored position, with no structural arity content (Gemma 2 FUNC-PFX in script 17; partial in OLMo 2 FUNC-PFX in script 16). Within-condition probe accuracy is therefore not by itself a substrate-invariance signal in functional-prefix notation. Cross-condition probe transfer in both directions is now a hard requirement for any Phase 1 or Phase 2 substrate-invariance claim — without it, the publication risk is reporting a propagated-token-identity classifier as a substrate-invariance finding.
- **The single-model trap.** The cross-condition transfer asymmetry between Gemma 2 9B and OLMo 2 7B is from a sample of n=2 models. Either model alone would have produced a different (and possibly publishable) Phase 1 paper, but the model-specific cross-context-stability finding would have been invisible. Phase 1 / Phase 2 work must keep adding models (Pythia next, then Qwen and/or Mistral) before any model-level property is claimed as a general pattern.

---

## 9. Immediate next steps

The original 6-week plan below describes what was intended at thesis-proposal time. Phase 0 has executed steps 1–9 in a compressed timeline using OLMo 2 1B and 7B, before the formal "month 1" begins, via the M4 local-extraction pipeline. The status of each step is annotated; the un-annotated items remain to be done.

1. **Week 1, Day 1–2: MPS smoke test.** ✓ Done — `experiments/03_mps_smoke_test.py` returned GREEN on OLMo 2 1B (527 tok/s, no CPU fallback). 7B follow-up runs of scripts 08–12 confirm viability at 7B in fp16 with ~14 GB RSS, ~91s per 200-prompt extraction.
2. **Week 1, Day 3–5: Tokenization screening utility.** ✓ Done — `experiments/02_tokenization_screening.py`. Tier-1 inventory is sparse (Greek lowercase variables + `¬ →` operators); Linear B, Tifinagh, Math Script, Fraktur, Alchemical, PUA glyphs all byte-fragment. Tier-2 invented words (`bliq`, `dren`, `vusp`, `molex`) confirmed as the workable surface form for operator-renaming probes.
3. **Week 1–2: Dolma corpus mirror.** Not yet done. Now lower priority than originally because the Phase 0 results are surface-form-based (so corpus-search isn't on the critical path until Phase 1's operator-novelty probe). Should still be set up before Phase 1 begins.
4. **Weeks 2–3: Stimulus generation module.** ✓ Done at Phase 0 scale (template-based generation in scripts 06–12). Needs hardening for Phase 1: per-stimulus embedding-distance reporting, multi-structure support beyond propositional logic, formal pre-registration of stimulus seeds.
5. **Weeks 3–4: Perplexity scorer + matched-control generator.** Partially done — `experiments/05_perplexity_matching.py` established that uniform-random tokens are the wrong null (5-6 orders of magnitude off canonical perplexity). The realistic control ladder (token-shuffle, high-temp-sample, n-gram-sample, uniform-random) is specified but not yet built as a callable utility.
6. **Weeks 4–5: Activation extraction pipeline.** ✓ Done — `extract_anchored_activations` in scripts 09–12 handles per-prompt operator-anchored extraction across all transformer layers for both OLMo 2 1B and 7B.
7. **Weeks 5–6: Geometric utilities.** ✓ Linear probes done (scripts 09–12). CKA done (script 08). Procrustes not yet done; TDA pipeline not yet done.
8. **End of Week 6: Sanity-check experiment.** ✓ Done — script 06's variable-substrate-invariance result (probe accuracy 1.000 from layer 2; CKA 0.93–0.99) is the cleanest possible three-way separation between canonical, variable-renamed, and structurally-scrambled. Operator-substrate-invariance scaled this from "does the model represent structure" to "does the model represent structure *uniformly across syntactic roles*"; the answer is no, with characterized failure modes.
9. **Weeks 7+: Probe A (substrate-invariance).** ✓ Done at Phase 0 scale for OLMo 2 1B and 7B. Per-role asymmetry characterized; uniform-default-to-`not` failure mode identified; H1/H2/H3 framework specified with H1 supported, H2 rejected, H3 emergent. See `paper_notes.md` for the full Phase 0 writeup.

**Phase 0 → Phase 1 transition checklist (final):**

- **✓ Done.** Script 11 (`11_l1_bar_anomaly_probe.py`) — H3 cross-class regime confirmed.
- **✓ Done.** Script 12 (`12_second_unary_probe.py`) — H1a vs H1b adjudicated provisionally; subsequently refined by script 15.
- **✓ Done.** Script 13 (`13_template_context_quantification.py`) — factorial H4 measurement; established the unary-region framing.
- **✓ Done.** Script 14 (`14_embedding_similarity_audit.py`) — H1 not inherited from layer-0 embeddings; mostly NOT a layer-0 phenomenon.
- **✓ Done.** Script 15 (`15_template_neutral_probe.py`) — H1 confirmed at 99.6% mass in template-neutral measurement; H4 reframed as probe artifact.
- **✓ Done.** Script 16 (`16_syntactic_confound_stress_test.py`) — external-review-driven; prefix/infix syntactic-position confound decisively refuted. Arity-attractor at 100% mass in functional-prefix notation; within-arity identity revealed to be even more fragile than scripts 9-15 indicated.
- **✓ Done.** Script 17 (`17_gemma2_cross_model_replication.py`) — Phase 1 entry test on Gemma 2 9B. NEUTRAL replicates (peak unary 97.6% at L4). FUNCTIONAL-PREFIX is non-monotonic (peaks 100% at L2 and L16-17, trough 0% L6-L12). Three observations made the FUNC-PFX result hard to interpret from script 17 alone (probe CV=1.000 at every layer, surface-feature last-subword split at fixed-reference L8, non-monotonic per-layer trajectory).
- **✓ Done.** Script 18 (`18_probe_artifact_diagnostics.py`) — four-diagnostic battery on both Gemma 2 9B and OLMo 2 7B. **Principal new finding: cross-condition probe transfer asymmetry.** Gemma 2's NEUTRAL probe transfers to FUNC-PFX invented at 100% unary mass; OLMo 2's NEUTRAL probe transfers at 0% unary mass (all 250 stimuli to `and`). Reverse direction also diverges: Gemma 2 86-100% at L2 and L16; OLMo 2 0% (all to `implies`). Both models pass the structural-probe sanity checks (held-out canonical at 4× chance NEUTRAL in both; centroid arity delta positive 5/5 in every model × condition). Phase 1 entry verdict: **cross-model arity-region attractor with model-specific cross-context representational stability**.
- Set up the Dolma corpus mirror (step 3 above) before Phase 1's operator-novelty probe.
- Build the perplexity-matched control-ladder utility (step 5 above) as a callable Python module so every Phase 1 experiment can invoke it.
- Build the embedding-distance-reporting utility (per the §6 methodological requirement) as a pre-experiment artifact for any invented-surface-form probe. Important caveat from script 14: layer-0 similarity does not reliably predict layer-7 probe behaviour, so this utility documents but does not screen-out potential H3 anomalies.
- Implement Procrustes alignment and the TDA pipeline (Johnson-Lindenstrauss-reduced; see §5 for the corrected default) as reusable utilities — neither is on the Phase 0 critical path but both are needed for Phase 1.
- **Directional-angle analysis (script 19, in progress).** Scale-invariant follow-up to the script-18 cross-condition transfer asymmetry. Computes cosine angle between (a) centroid-based unary direction (NEUTRAL vs FUNC-PFX) and (b) binary probe weight direction (NEUTRAL vs FUNC-PFX) per model. Predicted result: small angle for Gemma 2, large for OLMo 2. Closes the "raw centroid delta magnitudes are not comparable across models" gap in the script-18 report.
- **Cross-model replication on Pythia (scale ladder: 1.4B / 6.9B / 12B).** Next cross-model target after Gemma 2. Pythia trained on the Pile (a third training distribution distinct from Dolma and Gemma 2's mix) and Pythia's scale ladder lets us measure whether cross-context stability correlates with parameter count or is a property of architecture / training data alone. Compute: 12B in bf16 fits on M4 with ≈ 28 GB RSS; if MPS bf16 has issues, fp32-on-CPU at ≈ 5 min per script run is also acceptable.
- **Per-layer cross-condition probe transfer sweep, both models.** Generalises script 18's single-layer cross-transfer report to a full per-layer trajectory. Identifies whether Gemma 2 is globally aligned at every layer or only at the attractor-construction layers.
- **Per-layer mechanism trace of H1 construction.** Probe the residual stream at each layer on B'_neutral to identify the layer range over which the unary attractor is built. The natural follow-up to script 14's "H1 is not layer-0" finding, partially started by script 17's per-layer probe sweep.

**Phase 0 verdict: GO.** The arity-attractor finding is real, robust across four probe instruments and multiple syntactic notations, embedding-layer-independent, and survives the external-review-driven syntactic-confound stress test. Decision-point criterion (≥ 2 of 3 probes producing publishable findings) is satisfied by Probe A alone. Phase 1 begins. *The within-arity-identity claim is explicitly downgraded to a limitation; the headline is arity-only.*

**Phase 1 entry verdict (after scripts 17–18): the arity attractor replicates cross-model with model-specific representational stability.** Both Gemma 2 9B and OLMo 2 7B encode operator arity at the operator-anchored position with positive geometric centroid attractors in both NEUTRAL and FUNCTIONAL-PREFIX notations. The cross-condition transfer asymmetry (Gemma 2 globally aligned; OLMo 2 notation-local) is a model-level property that is invisible to within-condition probe accuracy alone and surfaces only under the cross-condition transfer instrument. The principal Phase 1 paper will therefore likely be structured around the cross-model substrate-invariance dimension — specifically the directional-angle (script 19) and per-layer cross-transfer sweep generalisations of script 18.

The pipeline matters more than any single early experiment. The probes are designed to test the *research program*; the pipeline is what makes every subsequent experiment in the thesis cheap to run. Phase 0's lesson: the pipeline-development discipline paid off — a probe-iteration cycle (write stimulus → extract → train probe → confusion matrix) now takes ~10 minutes on the M4 against OLMo 2 7B, which is what allowed us to converge on a publishable finding before formal month-1 of the thesis.

---

## 10. Open questions and resolved decisions

### Resolved (working defaults — confirm with supervisor but don't litigate)

- **Data infrastructure: mirror, not API.** Do not depend on a live AI2 API for four years. Mirror the tokenized Dolma index locally or on the university cluster in Week 1. We only need the metadata/tokenized shards sufficient to check n-gram frequencies for our logical primitives. Version-control the snapshot. Treat any live AI2 service as a convenience, not a dependency.
- **Pre-registration: private OSF + timestamped git.** Use a private OSF project with public timestamps, backed by a private GitHub/GitLab repository with signed commits. Preserves academic priority on the Question 2 headline results while keeping exact stimuli confidential until Phase 1 publication. Public-OSF-only is too exposed for the headline work.
- **Collaboration: independent through end of Phase 1.** Mechanistic interpretability moves fast and multi-institution entanglement before there is a stable, reproducible metric will slow academic autonomy. Once a paper proves substrate-invariance on OLMo 2 with hard CKA/Procrustes data (end of Phase 1), use that result as the asset to approach EleutherAI / Apollo Research / DeepMind for compute partnerships or SAE access in Phase 2.

### Still open (genuine supervisor / committee discussion)

- **Cluster budget for Phase 2–3.** Realistic envelope for SAE training on OLMo 2 (if Gemma Scope's fallback becomes necessary), persistent homology at scale, and cross-model replication. Need a rough number to plan against by end of Phase 0.
- **Compute partnerships.** Which Phase 2 partnership (if any) is worth pursuing first, and on what terms — co-authorship, compute-only, full collaboration?
- **Theory chapter scope.** How much philosophical / category-theoretic apparatus the committee expects in Phase 4 — i.e., whether the dissertation is positioned primarily in CS/ML or has genuine cognitive-science/philosophy weight.
- **Phase 1 headline framing (refined after scripts 17–18; cross-model dimension now available).** The cleanest single result is no longer single-model. Three of the four framings below are viable post-script-18; preference is now option 4.
  1. *Hierarchical-arity substrate-invariance in OLMo 2* — emphasises the H1 unary-region attractor confirmed across multiple probe instruments and independent of layer-0 embeddings. Mechanistically tightest, narrowest scope. The cross-model dimension would be relegated to "future work". Probably no longer the best Phase 1 paper given that the cross-model data already exists in scripts 17–18.
  2. *Operator representations in OLMo 2: an arity-shaped attractor at the operator-anchored position* — same finding, slightly broader framing that foregrounds the architectural-locus claim (the attractor is at a specific position in the residual stream) and motivates the Phase 2 per-layer-mechanism trace.
  3. *Surface-form perturbation produces compositional-hierarchy-graded failure modes in language model logical representations* — the broadest framing, positioned within the Platonic Representation Hypothesis discourse. Requires Phase 1 to establish the same hierarchical-arity pattern across (a) a second model (Gemma 2 9B — done) and (b) at least one non-propositional-logic domain (not done).
  4. **Cross-model arity-region substrate-invariance with model-specific cross-context stability of the arity direction** (new after scripts 17–18, peer-review-recalibrated). Principal finding: across OLMo 2 7B and Gemma 2 9B, invented operator forms are absorbed into unary-canonical regions within both metalinguistic and functional-prefix notations. The models differ in whether this arity readout transfers across notations. Gemma 2 shows clean cross-notation transfer at validated early layers (especially NEUTRAL@L4 ↔ FUNC-PFX@L4, and FUNC-PFX@L2 → NEUTRAL with the canonical-transfer gate passed), while OLMo 2's tested arity readouts at L7 collapse under the same cross-context transfer tests. Gemma 2's FUNC-PFX layer trajectory is non-monotonic: early layers (L2) show transferable arity structure, middle layers (L6-L12) show non-transportable surface readouts (the L8 result reframed as the cleanest single illustration of "within-condition probe success ≠ structural arity"), and later layers (L16-L17) show a candidate re-emergence requiring further validation. The cross-condition probe transfer instrument with a canonical-transfer gate is itself a methodological contribution. Mechanistically tightest framing that uses the full Phase 1 entry data; positions the thesis as both a substrate-invariance finding and a cross-model representational-geometry measurement. Pythia replication is a hard prerequisite for the final version of this paper; the cross-context-stability claim is materially stronger with a third model and especially with scale-ladder data.

  Pre-supervisor preference: framing 4, conditional on (a) Pythia replication confirming that the Gemma 2 vs OLMo 2 split is not a two-model artifact, (b) script 19 directional-angle results confirming the geometric interpretation, and (c) stronger canonical-transfer validation of the Gemma 2 L16 candidate stage. Fallback framing: framing 2, foregrounded by the Phase 0 OLMo 2 results, with cross-model dimension as a section rather than the headline. To be discussed.
- **The unary-attractor's training-corpus grounding.** Script 13's finding of zero `and` predictions across 1250 invented-operator inputs already partially answers this: if the H1 default were driven by raw corpus frequency, `and` (the most frequent canonical operator in English text) would dominate. It doesn't. The default is specifically *unary*. The remaining question is whether the unary-class attractor reflects some training-distributional property of how the model encounters unary operators (e.g., `not` appears in different syntactic contexts than `and`/`or`) or a more abstract structural prior. This is testable once the Dolma mirror is set up.
- **The molex anomaly.** Molex shows 98% necessarily even in the cleanest neutral-template measurement, against zero embedding-similarity evidence at layer 0. No current Phase 0 hypothesis explains this. Phase 1 should run a focused mini-experiment (per-subword patching, attention-pattern analysis at the layer the bias emerges) to identify the mechanism. If it's a generalizable channel, it deserves a name (H5?); if it's a one-off OLMo 2 7B idiosyncrasy, it's a footnote.
