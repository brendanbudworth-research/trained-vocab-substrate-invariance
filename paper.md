# Trained-vocabulary substrate-invariance in mid-scale language models

*Draft v2 (post external-reviewer round 1), May 2026. Authoring in markdown for iteration; pandoc → LaTeX conversion at submission. Figures rendered into `experiments/figures/out/` by the scripts in `experiments/figures/`; six tables populated from script logs of record (cited inline). Inline references use the format `(Author 20xx)`; canonical BibTeX entries live in `references.bib` at the repository root, with a prose-mention → cite-key mapping table at the end of this document. References to `lab notes §3.7.X` point to the working notebook in `paper_notes.md`.*

---

## Abstract

Across three open base language models at the 6.9-9B scale (OLMo 2 7B, Gemma 2 9B, Pythia 6.9B-deduped) spanning three training corpora, architectures, and tokenizers, we test substrate-invariance with logic-inspired operator-role stimuli — a 15-word vocabulary of canonical operator labels (`and`, `or`, `not`, `implies`, `necessarily`, …) carrying intended binary-vs-unary arity assignments rather than truth-functional semantics — in English-metalinguistic prose and programming-style functional-prefix notation. We find a sharp two-part result. **Inside the model's trained vocabulary, cross-notation linear-probe geometry transfers at ceiling** in all three families: `M2-canonical = 1.000` (bootstrap 95% CI `[1.000, 1.000]`) under a pre-registered 15-class readout for logical operators, and — under a pre-specified post-review content-word control inserted into the same templates — `M2-canonical = 1.000` (`[1.000, 1.000]`) for 15 non-operator content words. The principal transfer geometry is therefore not operator-class-specific: it holds at the same principal N→F operator-after cell for a matched non-operator content-word set in all three families (with a model-specific OLMo F→N content-word asymmetry that does not propagate to the principal direction; §4.1.1). **Outside the trained operator set, novel-operator generalisation is not robust**: pre-registered canonical-set expansion retracts every previously-flagged candidate cell, revealing a canonical-attractor compression that no single-axis statistic — training-corpus frequency, subword tokenization shape, or mean-pooled cosine similarity — explains, with one tightly-scoped Gemma 2 9B exception at L 2 close-paren that is causally validated under operator-after-sourced intervention. A 5-cell causal-patching sweep produces a structured pattern: one Gemma 2 9B cell at L 2 close-paren is cleanly causally arity-respecting under operator-after-sourced intervention; the principal cross-family Fact-1 cell at L 4 operator-after has mean ΔKL approximately 1.3-2× a norm-matched control, but its per-word Δ_specific 95% CI straddles zero on both axes (AMBIG under the cleaner statistic); the OLMo 2 7B flagship cell at L 10 close-paren is causally inert under both tested source anchors. Within-target source-anchor flip can produce opposite verdicts at identical target and layer, but source anchor is not directionally deterministic — the verdict is the joint product of source, target, and layer. We contribute four methodological devices: the lucky-default detector, the M2-canonical / M2-arity dissociation, the same-target / different-source patching protocol, and the within-vocabulary content-word control. The two-part result locates a precise empirical edge of the Platonic Representation Hypothesis at the within-language scale: cross-notation *geometric* substrate-invariance for the words the model was trained on, with non-uniform and generally modest causal load-bearingness; no substrate-independent category that extends to novel members.

---

## 1 Introduction

A widely-discussed hypothesis in modern representation learning (Huh, Isola, et al., 2024) holds that sufficiently large neural networks trained on natural data converge on representations of an underlying structure that exists independently of the substrate used to express it. If true, this *Platonic Representation Hypothesis* (PRH) predicts that surface-form changes which preserve structure should produce aligned internal representations — *substrate-invariance under renaming*. The strongest extant evidence for the PRH is cross-modal (text vs vision encoders learning aligned representations of the same concept). The within-language analog — does a single language model's internal representation of a logically-equivalent prompt survive controlled rewriting of the surface form? — is comparatively under-explored, and is the natural intermediate test case.

We probe this intermediate case in three open base language models at the 6.9-9B parameter range, selected to span three training corpora, three architectures, and three tokenizers. The structural primitive is a logic-inspired operator-role vocabulary with assigned arities: a small alphabet of operator-role tokens (`and`, `or`, `not`, `implies`, `necessarily`, extensible to `xor`, `nand`, `possibly`, `always`, `negate`, `nor`, `iff`, `unless`, `definitely`, `unprovably` for our 15-class readout), each carrying an intended binary-vs-unary arity assignment rather than a truth-functional semantics. (Several entries — `necessarily`, `possibly`, `always`, `definitely`, `unprovably`, `unless` — are modal or non-truth-functional rather than strictly propositional; we use them for their consistent operator-role surface position and intended arity, not for any semantic interpretation. The substrate-invariance question we ask is whether the model's internal representation of these operator-role surface forms transfers across notations under controlled rewriting, not whether the model executes any of these operators' semantics.) We rewrite the surface form along three controlled axes: variable renaming (English → Greek), operator-role-token renaming (canonical → invented Tier-2 BPE words), and notation change (English-metalinguistic prose → programming-style functional-prefix). We then ask whether the internal representations of the modified stimuli remain aligned with their canonical counterparts under a linear-probe instrument, and — separately — whether any apparent alignment is *causally* load-bearing under activation patching.

Our principal findings are two-part. (1) *Trained-vocabulary substrate-invariance*: cross-notation linear-probe geometry reaches ceiling accuracy under a 15-class readout across three model families, with bootstrap 95% CIs at `[1.000, 1.000]` after three independent scope expansions for logical operators and — under a pre-specified post-review content-word control inserted into syntactically-identical templates — for 15 non-operator content words. Within the tested 15-word operator readout and the matched 15-word content-word readout, cross-notation linear-probe geometry transfers at ceiling for trained, well-tokenized vocabulary items at the principal N→F operator-after cell — the substrate-invariance is therefore not a property of logical-operator-class semantics specifically, but a per-token-identity property at the tested anchor and direction (with a model-specific OLMo F→N content-word asymmetry). (2) *Novel-operator generalisation is not robust*: the apparent arity-respecting routing of invented operators we observed at earlier scopes is retracted under canonical-set expansion in all three model families, and the underlying mechanism is a model-specific canonical-attractor compression that no single-axis statistic (frequency, subword shape, or mean-pooled cosine similarity) captures. One tightly-scoped Gemma 2 9B cell at L 2 close-paren passes causal validation under operator-after-sourced intervention — a single-model exception to an otherwise three-family failure pattern. We additionally report a 5-cell causal-patching sweep that finds Fact 1's geometric transfer does *not* imply causal load-bearingness: the principal cross-family Fact-1 cell has mean ΔKL approximately 1.3-2× a norm-matched control but its per-word Δ_specific 95% CI straddles zero on both axes (AMBIG under the cleaner per-word bootstrap statistic), the OLMo flagship Fact-1 anchor is causally inert under both tested source anchors, and the only cleanly causally-arity-respecting cell in the sweep is a Gemma v6 emergent cell at L 2 close-paren, not a Fact-1 anchor. Within-target source-anchor flip *can* produce opposite causal verdicts at identical target and layer, motivating same-target / different-source comparisons as a routine activation-patching diagnostic, but source anchor is not deterministic across cells — the verdict is the joint product of source, target, and layer.

The contribution to the PRH literature is a sharp, falsifiable, three-family empirical edge: the model has a substrate-independent *geometric* (linear-probe-readable) structure for *the words it was trained on* — and not specifically for logical-operator-class semantics — with non-uniform and generally modest causal load-bearingness; and lacks a substrate-independent *category* of "logical operator" that would generalise to new instances. The contribution to mechanistic interpretability methodology is the *lucky-default detector* (a per-word top-concentration metric that catches a specific false-positive pattern in probe-based substrate-invariance studies), the *M2-canonical / M2-arity dissociation* (canonical-identity transfer and arity-axis transfer are distinct measurements that should be reported separately), the *same-target / different-source patching protocol* (a routine diagnostic that surfaces (source, target)-coupling effects which single-anchor-pair patching misses), and the *within-vocabulary content-word control* (a pre-specified post-review diagnostic that dissociates domain-specific substrate-invariance from generic lexical-identity substrate-invariance and is the load-bearing test that distinguishes the present claim from "operator-set-bound").

**Figure 1.** Schematic of the substrate-invariance setup. Left: NEUTRAL stimulus "Consider the word `and` in this sentence." paired by structure with the FUNC-PFX rewrite "The function `and(p, q)` evaluates to true." for both canonical (`and`) and invented (`bliq`) operators. Middle: both stimulus families go through the same frozen base LM forward pass; we extract residual stream activations at a single (anchor, layer) coordinate (e.g. operator-after, L 4). Right: a linear probe trained on NEUTRAL canonical activations is evaluated on FUNC-PFX canonical activations (Fact 1, M2-canonical = 1.000 at the principal cell) and on FUNC-PFX invented activations (Fact 2, M4b at chance after canonical-set expansion). Methodological battery (M1, M2-canonical, M2-arity, M4b, M4c / pwmin) summarised in the bottom strip. Rendered by `experiments/figures/fig_01_schematic.py`; artefact at `experiments/figures/out/fig_01_schematic.{pdf,png}`.

---

## 2 Related work

**The Platonic Representation Hypothesis.** Huh, Isola, et al. (2024) introduced the PRH as a unifying framework for cross-modal representational convergence in large neural networks. Prior work on cross-modal similarity (Bommasani et al., 2023; Radford et al., 2021) reports increasing CKA-like alignment between vision and language encoders at scale. The within-language analog has received less attention. The closest conceptual neighbour is Dumas et al. (2025)'s *Separating Tongue from Thought*, which uses activation patching to distinguish language-specific from language-agnostic concept representations in multilingual LMs — a cross-language version of the within-language, notation-controlled question we ask here. Our work tests substrate-invariance under controlled operator renaming as a sharp falsifiable PRH prediction in a single language model, across three model families.

**Probe-based interpretability of language model representations.** Linear probes have been a standard interpretability instrument since Alain & Bengio (2017); subsequent work has surfaced extensive caveats (Belinkov, 2022; Hewitt & Liang, 2019; Ravichander, Belinkov & Hovy, 2021). Our M2-canonical / M2-arity dissociation and lucky-default detector contribute to this methodological lineage by identifying specific failure modes in *cross-context* probe transfer that are not surfaced by standard within-context probe accuracy.

**Mechanistic interpretability and activation patching.** Recent work on activation patching (Geiger et al., 2021; Meng et al., 2022; Heimersheim & Nanda, 2024; Zhang & Nanda, 2024) has established the method as the standard for causal-grounding probe-based findings; Zhang & Nanda (2024) *Towards Best Practices of Activation Patching in Language Models* specifically catalogues how patching verdicts vary with metric and corruption/intervention design, and is directly relevant to our §6 (a)-(b) limitations on the mean-pool centroid bias and cross-notation source-target attention-pattern mismatch. Our causal-patching contribution is the *same-target / different-source comparison* as a routine diagnostic: at a single Gemma 2 9B target/layer/probe-causality coordinate, two patches that differ only in patch source produce opposite behavioural verdicts (§4.5, Cells 1 vs 3), but reviewer-round-1 follow-up cells (Cells 2, 5) show that source anchor is not directionally deterministic across cells — the verdict is the joint product of source, target, and layer (§5.2). The transferable methodological move is to supplement any single-anchor-pair patch with at least one same-target / different-source comparison, surfacing (source, target)-coupling effects that single-pair patching treats as properties of the target representation.

**Compositional generalisation in language models.** A long-standing target for compositional generalisation (Lake & Baroni, 2018; Hupkes et al., 2020; Saxton et al., 2019) is the ability to recombine familiar primitives in novel ways, including with novel components. Our novel-operator generalisation failure constitutes a precise local report on one specific compositional-generalisation edge: cross-notation substrate-invariance is preserved at ceiling for trained, well-tokenized vocabulary items in the tested 15-operator + 15-content-word readout sets at the principal N→F operator-after cell (logical operators *and* a pre-specified post-review non-operator content-word control; §4.1.1); novel-operator role inference is not. We add a three-family quantification of how trained-vocabulary cross-notation geometry is encoded at the residual-stream level, the demonstration that this encoding is bound to the model's trained vocabulary rather than to the logical-operator class specifically, and a 5-cell causal-patching sweep showing that the geometric transfer's causal load-bearingness is not robust under a per-word Δ_specific bootstrap at any Fact-1 anchor we have tested.

---

## 3 Methods

### 3.1 Stimuli

We construct logic-inspired operator-role stimuli in two notations. The operator-role vocabulary is described in §1; we treat the 15 canonicals as fixed-arity surface-form labels (not truth-functional operators) throughout.

**NEUTRAL (English-metalinguistic).** Templates frame operators as words to be considered: e.g., `"Consider the symbol and in this expression."` or `"The token implies appears between two propositions."` 50 templates per canonical operator; templates are generated from a fixed template family with stable-seed sampling so that stimulus sets are reproducible across runs.

**FUNC-PFX (functional-prefix).** Templates frame operators as function calls: e.g., `"The function and(p, q) evaluates to either true or false."` All canonicals — binaries (`and`, `or`, `xor`, `nand`, `implies`, `nor`, `iff`, `unless`) and unaries (`not`, `necessarily`, `possibly`, `always`, `negate`, `definitely`, `unprovably`) — are placed in the same prefix-function-call syntactic position with identical preceding context, dissociating the prefix-vs-infix syntactic-position confound that contaminates natural-English templates.

**Invented operators.** 16 invented Tier-2 BPE words (8 intended-binary: `bliq`, `dren`, `molex`, `krev`, `sond`, `glin`, `twiv`, `fump`; 8 intended-unary: `vusp`, `perph`, `kelm`, `zorf`, `gleph`, `drelth`, `vrith`, `nilph`). The set was chosen to be phonotactically plausible Latin-script tokens that avoid obvious morphemic semantic loadings, and audited at extraction time for subword decomposition under each model's tokenizer.

**Anchors.** Multi-anchor extraction is performed in a single forward pass per stimulus: we extract activations at four FUNC-PFX positions (`operator-after`, `first-arg`, `close-paren`, `sentence-final`) and two NEUTRAL positions (`operator-after`, `sentence-final`). This allows post-call test anchors that capture the model's representation of the operator after it has integrated argument-count information from the function-call syntax — a methodological refinement from the operator-anchored measurement used in earlier work.

### 3.2 Models

| Model | Lab | Training corpus | Architecture | Tokenizer | Parameters | Focus layers |
|---|---|---|---|---|---|---|
| OLMo 2 7B | AI2 | Dolma | modified Llama | OLMo BPE | 7B | L 4, 7, 10, 16, 24 |
| Gemma 2 9B | Google DeepMind | Google proprietary | soft-capped Gemma | SentencePiece | 9B | L 2, 4, 8, 16, 17 |
| Pythia 6.9B-deduped | EleutherAI | deduplicated Pile | GPT-NeoX (RoPE) | Pythia BPE | 6.9B | L 4, 7, 10, 16, 24 |

All three are open-weights base checkpoints (no instruction tuning). Activations are extracted in fp16 (OLMo, Pythia) or bf16 (Gemma) via the HuggingFace `transformers` `output_hidden_states=True` flag; single-prompt extraction avoids padding artifacts. Compute is on an Apple M4 with 48 GB unified memory. Total extraction time across all three models for the v6 (15-canonical, 16-invented) scope is ~80 min.

### 3.3 The seven-metric battery

We measure substrate-invariance with seven named metrics in four families (M1, M2, M3, M4) — `M2` and `M4` decomposed into submetrics. For each (train condition, train anchor, test condition, test anchor, layer) sweep cell:

- **M1 (within-condition probe CV)**: 5-fold stratified CV accuracy of a logistic-regression probe on canonical activations in the train condition. Sanity check that the trained-operator representation is linearly separable at this cell.
- **M2-canonical**: cross-notation transfer accuracy of the train-condition-trained probe on test-condition canonical activations. The principal Fact 1 measurement. Chance is `1 / n_canonicals` (1/5 for v3, 1/10 for v5, 1/15 for v6).
- **M2-arity**: same probe, but predictions coarsened to the binary-vs-unary partition. Decoupled from canonical-identity transfer; surfaces *arity-respecting* structure that may not survive at the canonical-identity level.
- **M3 (directional-angle)**: cosine angle between the NEUTRAL and FUNC-PFX arity directions at the cell. Bootstrap 95% CIs by within-class stimulus resampling (B = 100).
- **M4a (invented unary mass)**: fraction of invented-stimulus predictions in the unary canonical region.
- **M4b (intended-arity agreement)**: fraction of invented-stimulus predictions whose arity matches the word's intended arity. The principal Fact 2 measurement.
- **M4c (canonical catchment concentration)**: Herfindahl-Hirschman Index `HHI = Σ_c p_c²` over the per-canonical share of invented-stimulus predictions. HHI = `1/K` at uniform routing (`≈ 0.067` for K = 15) and HHI = 1 under single-canonical collapse; we threshold at `< 0.70` to flag "distributed" routing. (This is the *running-code* M4c definition (script 24); the v6 pre-registration spec instead defined M4c as `max_c p_c` with threshold `≤ 0.85`. §3.5 acknowledges the criterion-drift and reports a both-criteria reconciliation showing the headline verdicts are robust to which is used.)

A cell PASSES the substrate-invariance battery (substantive PASS-arity) under the running-code criterion iff `M2-arity ≥ 0.65 ∧ M4b ≥ 0.65 ∧ M4c < 0.70 ∧ 0.10 ≤ M4a ≤ 0.90 ∧ pwmin < 0.95`, where the final `pwmin < 0.95` conjunct is the lucky-default exclusion introduced in §3.4 below (we list all five conjuncts here so the running-code criterion is fully specified in one place); the frozen pre-registered criterion (§3.5, criterion-of-record) replaces `M4c < 0.70` with `max_c p_c ≤ 0.85` and tightens the M4a band to `0.20 ≤ M4a ≤ 0.80`. Cells whose 95% CI on M4b crosses 0.65 are reported as AMBIG. Cells that pass the four numerical conjuncts (M2-arity, M4b, M4c, M4a) but fail the `pwmin < 0.95` filter are reported as LUCKY-NEG, not PASS-arity (§3.4).

### 3.4 The lucky-default detector

A specific false-positive pattern surfaced during methodology iteration: a cell may have an aggregate M4b above threshold while every individual invented word's predictions are deterministically routed to a single canonical, with the canonical's arity happening to match the *majority* arity of the test set's invented words. We call this the *lucky-default* pattern. The detector flags a cell as lucky-default if `min_w P(top_canonical | w) ≥ 0.95` — every invented word has at least 95% of its predictions concentrated on a single canonical. In other words, the §3.3 substantive-PASS conjunction includes `pwmin < 0.95` (alongside the M2-arity / M4b / M4c / M4a numerical gate); cells that satisfy all four numerical conjuncts but trigger `pwmin ≥ 0.95` are reported as LUCKY-NEG, not PASS-arity. The "M4c distributed" conjunct itself is `HHI < 0.70` under the running-code criterion (§3.3) and `max_c p_c ≤ 0.85` under the frozen pre-registered criterion (§3.5, the criterion-of-record). The detector reclassified four of eight originally-flagged PASS-arity cells as lucky-default false positives in the cell sweep, and is one of four directly transferable methodological devices contributed by this work (§5.3).

### 3.5 Pre-registration

To address a reviewer concern that the M2-canonical / M2-arity / lucky-default / multi-scope framework was developed iteratively on the same data (a garden-of-forking-paths risk), we pre-registered the v6 canonical-set expansion in a frozen analysis plan written *before* any v6 cache extraction (`experiments/preregistration_v6.md`, header `Status: FROZEN. Written before any v6 extraction or analysis runs`; published in the project's first public-repository commit at hash `e6afe5e358454a8a5ca85f369eb2206a847b34d5` (short: `e6afe5e`, 2026-05-21), which bundles the pre-registration document with the v6 extraction script and the resulting log of record — the in-document `FROZEN` declaration and the lab-notes timeline of v6 design discussion (paper_notes §3.7.16) anchor the pre-extraction authorship claim, the commit hash anchors the public-record artefact). The pre-registration specified the canonical additions (`nor`, `iff`, `unless`, `definitely`, `unprovably`), the three single-axis predictions to be adjudicated (P_FREQ, P_SUBWORD, P_INTERACTION), the cell PASS-arity threshold (`M2-arity ≥ 0.65 ∧ M4b ≥ 0.65 ∧ max_c p_c ≤ 0.85 ∧ M4a ∈ [0.20, 0.80] ∧ pwmin < 0.95`, where `max_c p_c` is the maximum share of invented mass on any single canonical), the bootstrap protocol (500 stim-resamples), and an audit gate (extraction proceeds iff at most 2 of 5 new canonicals are out-of-design in any tokenizer). The audit gate caught one tokenization failure (`iff` 1pc in all three tokenizers despite our 2-3pc design target); analysis proceeded with `iff` flagged out-of-design.

**Criterion-drift between the pre-registration and the running code (caught at external review).** The running v6 sweep code (`experiments/24_v6_canonical_expansion.py`) drifted from the frozen pre-registration on two of the five PASS-arity conjuncts: `max_c p_c ≤ 0.85` became the tighter Herfindahl-Hirschman index `Σ_c p_c² < 0.70` (still flagging "distributed" routing but on a different statistic with different thresholding semantics), and `M4a ∈ [0.20, 0.80]` widened to `M4a ∈ [0.10, 0.90]`. The drift was discovered after extraction during external-review round 1, not before. Script `experiments/24b_frozen_criterion_rederivation.py` replays the v6 four-scope sweep cache-only (no model inference; identical probes, identical cells) under both criteria and reports the per-cell verdict comparison (`outputs/24b_20260521_120258.log`; lab notes §3.7.21). **Headline reconciliation: the v6 P_RETRACT verdict ("zero PASS-arity cells at v6 in OLMo and Pythia; two emergent PASS-arity cells at Gemma L 2 close-paren") is identical under both criteria in all three model families.** The only verdict disagreement anywhere in the 4-scope × 3-model sweep is one Pythia v3 cell (`N→F opera→close L 16`, `M4a = 0.192`): PASS-arity under the running [0.10, 0.90] band, ARITY-AXIS-ONLY under the frozen [0.20, 0.80] band. The disagreement is driven by the M4a band width, not the M4c definition; the L 16 cell sits below the pre-registered M4a lower bound for "balanced-arity" mass distribution. The headline retraction-chain trajectory (Pythia: 4 v3 candidates → 3 v4 → 0 v5 → 0 v6 under running; 3 → 3 → 0 → 0 under frozen) is qualitatively identical and quantitatively equivalent for the v6 outcome. Under the conservative reading we adopt for this paper, the **frozen pre-registered criterion is the criterion-of-record**; running-code verdicts are reported as they were computed but the §4.3 retraction headline tallies the eight v3 PASS-arity cells that the v6-pipeline four-scope sweep produces under the frozen criterion (two Gemma 2 9B + three OLMo 2 7B + three Pythia 6.9B-d cells; full list in §4.3 and per-cell retraction trajectories in lab notes §3.7.21). The pre-pre-registration Phase 1 sweep (script 22b on its own caches) flagged a related but not byte-identical four-cell candidate set; we retain the Phase 1 narrative only in the lab-notes lineage (§3.7.13) and adopt the v6-pipeline list as the published headline.

---

## 4 Results

### 4.1 Fact 1: trained-vocabulary substrate-invariance

Cross-notation linear-probe geometry is **at ceiling under a 15-class readout in all three model families**, both for logical operators (the v6 canonical set, lab notes §3.7.16) and — under the §4.1.1 pre-specified content-word control — for 15 non-operator content words inserted into syntactically-identical templates, with bootstrap 95% CIs `[1.000, 1.000]` at the same `operator-after → operator-after L 4` cell.

**Table 1.** Headline Fact 1 results across the three model families. M1n / M1f are within-condition 5-fold-CV probe accuracies on canonicals at the same cell (sanity check); M2c is the cross-notation transfer accuracy under the 15-class readout (15-class chance = 0.067 ≈ 1/15); CI(M2c) is the bootstrap 95% CI under B = 500 stim-resamples; M2-arity is the same probe coarsened to binary-vs-unary partition (chance ≈ 0.53 under the majority-arity baseline given v6's 8B / 7U split). All numbers from script 24's v6 80-cell sweep, `outputs/24_20260520_185537.log`.

| Model | Cell | M1n | M1f | M2c | CI(M2c) | M2-arity |
|---|---|---|---|---|---|---|
| Gemma 2 9B | `N→F opera→opera L 4` | 1.000 | 1.000 | **1.000** | [1.000, 1.000] | 1.000 |
| OLMo 2 7B | `N→F opera→opera L 4` | 1.000 | 1.000 | **1.000** | [1.000, 1.000] | 1.000 |
| Pythia 6.9B-d | `N→F opera→opera L 4` | 1.000 | 1.000 | **1.000** | [1.000, 1.000] | 1.000 |

The result is robust to three independent scope expansions tested in sequence. (1) **Invented-set expansion** from 5 to 16 invented words (script 22c): M2-canonical at the principal positive cells is unchanged. (2) **Canonical-readout expansion** from 5 to 10 (script 22d, adding `xor`, `nand`, `possibly`, `always`, `negate`): M2-canonical at the best cells in OLMo and Gemma remains at ≥ 0.81 with M2-arity at 1.000. (3) **Pre-registered 15-class expansion** (script 24, adding `nor`, `iff`, `unless`, `definitely`, `unprovably`): M2-canonical reaches 1.000 with bootstrap CI `[1.000, 1.000]` in all three model families at the same cell, after the audit gate.

Across the per-anchor × per-layer × per-direction sweep (80 cells per model at v6), the cells passing the M2 gate (`M2-canonical ≥ 0.65`) cluster at the `operator-after → operator-after` anchor pair across the focus layers (L 4 in Gemma, L 4-L 10 in OLMo and Pythia); the post-call anchors (`first-arg`, `close-paren`, `sentence-final`) pass less reliably, with the F→N direction systematically weaker than N→F across all three models. **Figure 2** visualises this directly as a per-model heatmap of M2-canonical across all 80 v6 cells (16 anchor pairs × 5 focus layers), with the M2 PASS gate at 0.65 marked on the colourbar; the N→F opera→opera row is at ceiling in all three models, the post-call and F→N rows are systematically darker. Rendered by `experiments/figures/fig_02_m2c_heatmap.py`; artefact at `experiments/figures/out/fig_02_m2c_heatmap.{pdf,png}`.

**Held-out template diagnostic (within-condition only).** To probe template lexical leakage, we trained each model's probe on the carryover templates and evaluated it on a syntactically-disjoint held-out template family (50 templates per condition, not seen during probe training; see Appendix). At FUNC-PFX `operator-after` at the early focus layers (Gemma L 2-L 8, OLMo L 4-L 10, Pythia L 4-L 10), M1-heldout = 0.94-1.00. Sentence-final and deep-layer cells show larger degradation (Gemma FUNC-PFX sentence-final L 8 = 0.30, OLMo FUNC-PFX close-paren L 24 = 0.37); these cells are not the locus of any positive finding. **Scope of this diagnostic.** M1-heldout is *within-condition* generalisation: it tests whether the probe's decision boundary is template-scaffold-memorising vs canonical-representing inside a single notation. It does *not* directly test cross-notation template leakage in M2; the stricter diagnostic — train probe on carryover templates in the source notation, evaluate M2-canonical on held-out templates in the target notation — has not been run in v6. The within-condition pass therefore reduces (but does not eliminate) the concern that the principal Fact-1 cells are exploiting template-scaffold features rather than canonical structure.

#### 4.1.1 Operator-class vs lexical-identity control

A skeptical reading of Fact 1's ceiling-level transfer is that M2-canonical = 1.000 is a *generic* substrate-independent lexical-identity signal — the probe finds a per-token-identity hyperplane that would also reach ceiling for any 15-word readout vocabulary, not specifically a logical-operator-class abstraction. The v6 canonical set spans roughly four orders of magnitude in training-corpus frequency (`and`, `or`, `not` are top-100 English words; `nand`, `iff`, `unprovably` are vanishingly rare in any realistic corpus), and the high-frequency operators may carry the cross-notation transfer load with the low-frequency ones riding along via their idiosyncratic spelling.

To disentangle operator-class substrate-invariance from generic lexical-identity substrate-invariance, we re-run the M1-M4 battery with 15 heterogeneous non-operator content words inserted into syntactically-identical NEUTRAL and FUNC-PFX templates (`experiments/25c_corpus_frequency_control.py`, §3.7.20 of accompanying lab notes). The control set spans the same subword-length and frequency-tier range as the v6 canonical set (`house`, `water`, `music`, `light`, `paper`, `pattern`, `theory`, `system` as 8 "binary-position" words; `region`, `period`, `archipelago`, `mosaic`, `plinth`, `ledger`, `cassowary` as 7 "unary-position" words; multi-piece tokenization rate 2-4/15 across the three tokenizers, comparable to v6's 1/15) but shares no semantic category and no functional role with logical operators. The "intended-arity" assignment is a syntactic-position match against the v6 binary-vs-unary split (not a semantic claim — `house` is not a binary operator in English) so that the FUNC-PFX template emits `house(p, q)` and `region(p)` in the same syntactic position the v6 canonicals occupy. **Pre-specified adjudication** (declared in the `25c_corpus_frequency_control.py` docstring before any content-word extraction; this is a post-review reviewer-requested control, not part of `experiments/preregistration_v6.md`): if M2-canonical at the principal `N→F opera→opera L 4` cell lands at ≥ 0.65 for the content-word control in any of the three models, Fact 1 generalises to lexical-identity substrate-invariance broadly and the "operator-set-bound" framing requires reframing to "trained-vocabulary-bound".

**Result: the REFRAME trigger fires in all three model families at the principal cell.**

**Table 2.** Content-word M2-canonical at the principal Fact-1 cell, with bootstrap 95% CI (B = 500). v6 column is the canonical-operator M2-canonical at the same cell from §4.1 / Table 1, for direct comparison. M1n / M1f columns are within-condition probe 5-fold CV on the content-word set at the same train and test condition. All numbers from `outputs/25c_20260521_092243.log` (script 25c content-word control extraction).

| Model | Cell | M1n | M1f | M2c (content) | CI (content) | M2c (v6 canonical) | Verdict |
|---|---|---|---|---|---|---|---|
| Gemma 2 9B | `N→F opera→opera L 4` | 0.999 | 1.000 | **1.000** | [1.000, 1.000] | 1.000 [1.000, 1.000] | **REFRAME** |
| Gemma 2 9B | `N→F opera→opera L 8` | 0.996 | 1.000 | 1.000 | [1.000, 1.000] | — | REFRAME |
| OLMo 2 7B | `N→F opera→opera L 4` | 0.959 | 1.000 | **1.000** | [1.000, 1.000] | 1.000 [1.000, 1.000] | **REFRAME** |
| OLMo 2 7B | `N→F opera→opera L 10` | 0.963 | 1.000 | 1.000 | [1.000, 1.000] | — | REFRAME |
| Pythia 6.9B-d | `N→F opera→opera L 4` | 1.000 | 1.000 | **1.000** | [1.000, 1.000] | 1.000 [1.000, 1.000] | **REFRAME** |
| Pythia 6.9B-d | `N→F opera→opera L 10` | 1.000 | 1.000 | 1.000 | [1.000, 1.000] | — | REFRAME |

The N→F transfer at the principal Fact-1 cell is byte-identically at ceiling for content words and for canonical operators across all three model families. The "operator-set-bound" framing of Fact 1 is therefore rejected by the control, and we reframe to **trained-vocabulary substrate-invariance (within the tested 15+15 readout sets at the principal N→F operator-after cell)**: cross-notation linear-probe geometry transfers at ceiling for trained, well-tokenized vocabulary items at this specific anchor/direction coordinate, not specifically for logical-operator-class abstractions. The reframe is bounded by what was tested — 15 operators + 15 content words at one (anchor, direction, layer) coordinate per model — and is not a claim that *every* in-vocabulary word transfers (untested) or that transfer is direction-symmetric (the F→N analysis below shows otherwise for OLMo content words). This is the load-bearing single-experiment falsification of the v1 draft's principal positive framing, and the paper is reorganised around the reframed Fact 1 throughout (title, abstract, §1, §4.1, §5.1, §7).

**A model-specific F→N asymmetry surfaces in the reverse-direction probe.** The same control evaluated at the reverse `F→N opera→opera L 4` cell shows a model-dependent split: Gemma 2 9B M2c = 0.836 [0.820, 0.852] (PASS), Pythia 6.9B-d M2c = 0.943 [0.932, 0.956] (PASS), OLMo 2 7B M2c = 0.376 [0.357, 0.396] (clean FAIL). The OLMo content-word F→N drop is a model-specific phenomenon that does not appear at the same cell with canonical operators. The reading: OLMo 2 7B's FUNC-PFX content-word representations have noisier per-class structure than its FUNC-PFX operator representations, such that a probe trained on FUNC-PFX content-word activations does not generalise to NEUTRAL content-word activations cleanly. This may reflect higher per-stimulus context-driven variability for common English nouns in function-call syntax compared to logical operators in the same syntax, in OLMo's BPE-tokenizer activation distribution specifically. The asymmetry is interesting but does not weaken the principal REFRAME trigger (N→F is the principal Fact-1 direction); it is documented as a methodological caveat on F→N robustness for content-word probes in OLMo.

**Scope caveat.** Fact 2 (novel-operator generalisation failure under canonical-set expansion) is preserved as-stated: we have only tested *novel operators* under canonical-set expansion. Whether *novel content words* would exhibit the same canonical-attractor retraction under content-word-set expansion is an open follow-up. The combined two-part finding is therefore: *trained-vocabulary substrate-invariance holds at ceiling on the tested 15+15 readout sets at the principal N→F operator-after cell*, demonstrated for both logical operators and content words; *novel-operator generalisation is not robust to canonical-set expansion* — most candidate cells retract into canonical-attractor compression (§4.4) with one tightly-scoped Gemma 2 9B L 2 close-paren causal exception (§4.5); the novel-content-word retraction analog is unverified.

### 4.2 The M2-canonical / M2-arity dissociation

A specific OLMo 2 7B cell, `N→F sente→close L 10`, exhibits a clean dissociation between the two halves of the M2 measurement under multi-scope readout. The full measurement battery splits substrate-invariance into a canonical-identity claim (`M2-canonical`: does the probe predict the correct 1-of-15 canonical?) and an arity-axis claim (`M2-arity`: does the probe predict a canonical of the correct binary-vs-unary type?). At every other PASS cell in the three-model sweep, the two metrics move together — both at ceiling or both at chance. At this one cell they decouple: `M2-arity = 1.000` while `M2-canonical = 0.736` at v6 (≈ 11× the 15-class chance baseline of 0.067, well above the 0.65 PASS gate, but clearly below ceiling). The within-arity confusion pattern is structurally clean: binary canonicals (`or`, `implies`) route to `and`, and the unary canonicals route to `not`, with no binary-↔-unary errors anywhere in the confusion matrix.

The dissociation is mechanistically diagnostic: the cross-notation transfer mechanism at this cell appears to be the arity axis (binary vs unary) operating independently of within-arity canonical identity. The probe's `or → and` and `implies → and` confusion at FUNC-PFX close-paren L 10 is not noise — it is a stable structural feature of the model's representation at this position, where the linear instrument can read out *what kind of operator* but not *which specific operator*.

**Cross-scope stability.** The dissociation survives all four scope expansions tested. Table 3 shows the OLMo `N→F sente→close L 10` cell tracked across v3 (5 canonicals + 5 invented), v4 (5 canonicals + 16 invented), v5 (10 canonicals + 16 invented), and v6 (15 canonicals + 16 invented). M2-arity stays locked at 1.000 across all four scopes; M2-canonical moves between 0.604 and 0.806 (always significantly above chance, never at ceiling); M4b retracts from PASS-arity at v3-v4 (where the small canonical readout makes the invented-arity test informative) to LUCKY-NEG at v5-v6 (where the expanded canonical readout reveals the v3-v4 M4b numbers as a coincidence between the model's default-canonical attractor and the intended arities of the small invented set, §4.4).

**Table 3.** Lab notes §3.7.9 OLMo `N→F sente→close L 10` dissociation cell tracked across the four pre-registered scopes (v3 → v6). `M2-arity` is the cross-notation binary-vs-unary transfer accuracy at the same probe (chance ≈ 0.5 under the majority-arity baseline). `M2-canonical` is the cross-notation 1-of-K canonical-identity transfer accuracy under K-class readout (chance = 1/K). `M4b` is the intended-arity-agreement test on invented words at the same probe; `M4c` is the per-canonical-Herfindahl index of the invented mass (1.00 = single-canonical collapse). `pwmin` is the minimum per-word top-canonical concentration across the 5 or 16 invented words (≥ 0.95 = lucky-default detector fires). All numbers are from script 24 OLMo block (lines 978 / 1097 / 1249 / 1401 of `outputs/24_20260520_185537.log`); bootstrap CIs from script 22a / 22c / script 24 invented-mass section.

| Scope | Canonicals × invented | M2-canonical | CI(M2-canonical) | M2-arity | M4b | M4c | pwmin | Verdict |
|---|---|---|---|---|---|---|---|---|
| v3 | 5 × 5 | 0.604 | [0.432, 0.682] | **1.000** | 0.880 | 0.57 | 0.50 | PASS-arity |
| v4 | 5 × 16 | 0.604 | (point) | **1.000** | 0.778 [0.772, 0.819] | 0.53 | ~0.50 | PASS-arity |
| v5 | 10 × 16 | 0.806 | (point) | **1.000** | 0.500 [0.500, 0.500] | 1.00 | 1.00 | LUCKY-NEG |
| v6 | 15 × 16 | 0.736 | (point) | **1.000** | 0.500 [0.500, 0.500] | 1.00 | 1.00 | LUCKY-NEG |

**Interpretation.** The arity axis is a real, scope-stable feature of OLMo 2 7B's residual stream at this cell, robust to every scope expansion we have tested. The canonical-identity axis is not at this cell — the same residual stream cannot be read out for which of `and`, `or`, `implies` the model is processing, only for whether it is binary or unary. This is the cleanest within-paper example of a substrate-invariance claim splitting into a partial-pass and a fail under measurement decomposition, and it would have been invisible under any single-axis report (M2-arity-only reports a PASS; M2-canonical-only reports the v3 cell as AMBIG and the v5-v6 cells as PASS-but-collapsed). The cell also anchored the project-flagship Phase-1 positive finding (lab notes §3.7.9) and the same-target / different-source causal patching protocol (§4.5, Cell 4), where it returns probe-readable but causally inert under NEUTRAL-sentence-final patching — a third axis of dissociation at the same cell.

We treat the M2-canonical / M2-arity split as one of the project's directly-transferable methodological contributions (§5.3): cross-notation substrate-invariance has two distinct sub-claims that should be measured separately. Reporting only canonical-identity transfer (the standard Fact-1 measurement) misses the arity-axis-only dissociation at this cell; reporting only arity-axis transfer (a coarser measurement) collapses cells where canonical identity is also preserved into the same bin as cells where only arity is.

### 4.3 Fact 2: novel-operator generalisation collapses to canonical-attractor compression

The v6-pipeline four-scope sweep (`experiments/24_v6_canonical_expansion.py`, replayed under the frozen pre-registered criterion by `24b_frozen_criterion_rederivation.py`; `outputs/24b_20260521_120258.log`) identifies **eight v3 PASS-arity cells** across the three model families (`M2-arity ≥ 0.65 ∧ M4b ≥ 0.65 ∧ max_c p_c ≤ 0.85 ∧ M4a ∈ [0.20, 0.80] ∧ pwmin < 0.95`):

- **Gemma 2 9B (2 cells)**: `N→F opera→first L 4`, `N→F sente→first L 8`.
- **OLMo 2 7B (3 cells)**: `F→N first→opera L 7`, `N→F sente→close L 10`, `N→F opera→close L 24`.
- **Pythia 6.9B-d (3 cells)**: `N→F opera→close L 4`, `N→F opera→close L 7`, `N→F sente→close L 10`.

A ninth Pythia v3 cell (`N→F opera→close L 16`, M4a = 0.192) is PASS-arity under the running-code wider M4a band [0.10, 0.90] but ARITY-AXIS-ONLY under the frozen [0.20, 0.80] band; this is the only verdict disagreement anywhere in the 4-scope × 3-model sweep (§3.5). The pre-pre-registration Phase 1 sweep (script 22b on its own pre-v6 caches) flagged a related but not byte-identical four-cell candidate set, including Gemma `sente→opera L 4` (M4b = 0.669 borderline under script 22c's invented-set expansion; lab notes §3.7.13) which does not appear in the v6-pipeline v3 replay. Lab notes §3.7.21 reconciles the two cell sets; the v6-pipeline list is the criterion-of-record going forward.

**All eight cells retract under canonical-set expansion; no v3 PASS-arity cell survives to v5 in any of the three model families.** Three of the eight retract at v4 (invented-set expansion, 5 → 16 invented words; the v3 → v4 transition isolates 5-invented-word sampling artefacts from genuine arity-respecting routing): Gemma `opera→first L 4` (M4b 0.672 → 0.581), Gemma `sente→first L 8` (M4b 0.696 → 0.550), and OLMo `F→N first→opera L 7` (M4b 0.696 → 0.583) all drop to M2A-ONLY at v4 even though M2-arity remains at ≥ 0.95 — the arity axis is preserved, but the per-word intended-arity agreement drops below the 0.65 PASS-arity threshold under the 16-invented-word readout. The remaining five retract at v5 (canonical-readout expansion, 5 → 10 canonicals): OLMo `N→F sente→close L 10` becomes the single-canonical-`nand` LUCKY-NEG cell that anchors §4.2's M2-arity / M2-canonical dissociation and Figure 3 (M4b drops from 0.880 at v3 to 0.500 at v5; M4c HHI 0.57 → 1.00; max_c 0.70 → 1.00; pwmin 0.56 → 1.00 — all 16 invented words route to `nand`); OLMo `N→F opera→close L 24` retracts to M2A-ONLY at v5 with M4b 0.768 → 0.482; and all three Pythia v3 cells retract at v5 — `N→F opera→close L 4` and `N→F opera→close L 7` both retract directly to M2A-ONLY (M4b 0.684 → 0.454, 0.860 → 0.320), while `N→F sente→close L 10` first passes through LUCKY-NEG at v5 (pwmin = 0.98) before settling at M2A-ONLY at v6 as the v6 canonical-set widens the routing distribution. The pre-registered v6 expansion (10 → 15) re-confirms every retraction; cross-scope retraction trajectories are tabulated cell-by-cell in lab notes §3.7.21 (sourced from `outputs/24b_20260521_120258.log`).

**Figure 3.** Per-canonical breakdown of invented-word predictions at OLMo `N→F sente→close L 10` across the four pre-registered scopes (v3, v4, v5, v6). 4-panel barplot. Each panel shows the share of invented predictions landing on each canonical, with the binary / unary partition colour-coded. The v3 panel (5 canonicals × 5 invented words) shows apparent arity-respecting routing (~70% on `and` for the intended-binary words, ~27% on `necessarily` for the intended-unary words; M4b = 88%). The v4 panel (5 × 16) preserves the qualitative pattern with M4b retracting to 78%. The v5 (10 × 16) and v6 (15 × 16) panels show 100% collapse onto a single attractor — the lucky-default-detector signature, M4b at chance (50%) under the majority-arity baseline. Rendered by `experiments/figures/fig_03_canonical_breakdown.py`; artefact at `experiments/figures/out/fig_03_canonical_breakdown.{pdf,png}`.

**The mechanism is a readout-vocabulary-dependent canonical-attractor compression.** At v3 (5 canonicals), the apparent arity-respecting routing of invented operators was hypothesised to be a *default-to-rarest-canonical* effect: the rarest unary canonical is `necessarily` and the rarest binary is `implies`, and these happen to coincide with the intended categories of the small invented set — yielding M4b > 0.65 by coincidence. At v5/v6 with `nand`, `xor`, `negate`, `unprovably` in the readout, the canonical attractor shifts wholesale, and so does the per-word routing target. The pre-registered v6 disentanglement (§4.4) tests the rarity hypothesis directly via the three single-axis predictions P_FREQ, P_SUBWORD, P_INTERACTION; **none of the three passes in any model**, so "default-to-rarest" is retained here only as the label for the *rejected* single-axis reading. The robust empirical claim is the weaker one: **the probe's per-word routing target is a property of the readout vocabulary's distribution interacting with per-word residual-stream geometry, not of the model's representation of the invented words alone.**

### 4.4 The pre-registered v6 default-mechanism disentanglement

The v6 pre-registration laid out three competing single-axis readings of the default mechanism: **P_FREQ** (route to low-frequency canonicals; aggregate of `{nor, iff, unprovably}` ≥ 35% with each ≥ 10%), **P_SUBWORD** (route to multi-subword canonicals; multi-pc aggregate ≥ 70%), **P_INTERACTION** (require `nor ∈ [5%, 15%]` with mid-frequency controls `unless, definitely` each ≤ 5%). **None of the three predictions passes in any model.** Table 4 shows the per-canonical invented-mass shares aggregated across all 80 v6 sweep cells × 16 invented words = 1280 readouts per model.

**Table 4.** v6 aggregate per-canonical breakdown of invented-stimulus mass per model (boldface highlights values that violate or trigger pre-registered thresholds; **NEW** marks the 5 v6 canonical additions). Tok = subword pieces per Gemma SentencePiece / OLMo BPE / Pythia BPE where they differ; otherwise a single value. Freq = corpus-frequency tier estimate. Per-model aggregates computed across the 80 v6 sweep cells × 16 invented words = 1280 readouts per model, from `outputs/24_20260520_185537.log`.

| Canonical | Arity | Tok | Freq | Gemma 2 9B | OLMo 2 7B | Pythia 6.9B-d |
|---|---|---|---|---|---|---|
| `and` | B | 1pc | high | 0.4% | 0.2% | 2.3% |
| `or` | B | 1pc | high | 0.8% | 1.2% | 3.8% |
| `implies` | B | 1pc | mid | 1.6% | **15.9%** | 5.4% |
| `xor` | B | 1pc / 2pc / 2pc | very-low | 22.5% | 9.3% | 16.6% |
| `nand` | B | 1pc / 2pc / 2pc | very-low | 17.4% | **40.5%** | 21.9% |
| `not` | U | 1pc | high | 4.9% | 2.4% | 0.8% |
| `necessarily` | U | 1pc | mid | 0.2% | 0.5% | 4.1% |
| `possibly` | U | 1pc | high | 1.6% | 3.5% | 0.0% |
| `always` | U | 1pc | high | 0.0% | 0.1% | 0.0% |
| `negate` | U | 1pc / 1pc / 2pc | low | 12.1% | 16.6% | 14.2% |
| **`nor` (NEW)** | B | 1pc | low | 5.1% | 1.2% | 0.2% |
| **`iff` (NEW, OOD)** | B | 1pc | very-low | 10.8% | 3.2% | 12.1% |
| **`unless` (NEW)** | B | 1pc | mid | 0.4% | 0.3% | 2.4% |
| **`definitely` (NEW)** | U | 1pc | mid | **7.5%** | 1.3% | **5.5%** |
| **`unprovably` (NEW)** | U | multi-pc | very-low | **14.8%** | 3.7% | 10.6% |
| NEW-LF aggregate (`nor + iff + unprovably`) | — | — | — | 30.6% (< 35%) | 8.1% (< 35%) | 22.9% (< 35%) |
| multi-pc aggregate | — | — | — | 14.8% (< 70%) | 13.0% (< 70%) | 63.4% (< 70%) |
| MF controls (`unless + definitely`) | — | — | — | 7.9% (> 5%) | 1.6% | 7.9% (> 5%) |

Three findings worth recording.

(1) **OLMo's 15.9% routing to `implies`** is the cleanest anomaly. `implies` is a v3 / v4 / v5 carryover canonical (1pc in all three tokenizers, mid-frequency), and was not predicted by any of the three single-axis hypotheses to attract default mass.

(2) **Mid-frequency control canonicals do not stay quiet.** `definitely` attracts 7.5% in Gemma and 5.5% in Pythia, above the P_INTERACTION ≤ 5% threshold. The P_INTERACTION reading therefore fails.

(3) **The mechanism appears to be a model-specific mixture of training-corpus frequency, subword shape, and per-word residual-stream structure.** The script 25b embedding-similarity test (§4.6) then rules out the most natural single-factor closing of the gap.

### 4.5 Causal patching sweep across five (source, target, layer) cells

The v6 expansion produced two findings that linear probes alone cannot adjudicate: (Q1) Fact 1 is a correlational linear-probe finding (the §5 "linear probes only" limit applies at the principal positive cells); (Q2) two Gemma 2 9B cells at L 2 close-paren show M4b jumping from chance at v5 to PASS at v6 *without any change to underlying activations*, with the v6 reading being a methodological caveat on M4b's threshold-sensitivity to readout granularity.

We address both questions with activation patching across five (source, target, layer) cells. Three cells are principal Phase-1/Phase-2 cells motivated directly by Q1 and Q2: **Gemma `opera→close L 2`** (the v6 emergent cell at M4b = 0.822), **Gemma `sente→close L 2`** (the second v6 emergent cell at M4b = 0.662), and **OLMo `sente→close L 10`** (the project-flagship Fact-1 anchor at M2-arity = 1.000). Two further cells were added in response to reviewer round 1 to disambiguate whether the Cell-1 vs Cell-3 verdict split is a property of source anchor or a property of target anchor (§5.2): **Gemma `opera→opera L 4`** (the cross-family Fact-1 cell at M2-canonical = 1.000) and **OLMo `opera→close L 10`** (the same OLMo L 10 target as Cell 4 but under the operator-after source). At each cell, we replace the FUNC-PFX target-anchor residual at the specified layer with the mean NEUTRAL canonical activation at the source anchor, via a forward hook. Four conditions: `BASELINE` (no patch), `PATCH_not`, `PATCH_and`, `RANDOM_NORM` (norm-matched Gaussian control). Two outcomes per condition: (i) the patched-residual probe readout (sanity check on patch effectiveness), (ii) the behavioural KL shift in the sentence-final next-token distribution against the FUNC-PFX canonical-c reference.

The five cells split into one CLEAN PASS, one AMBIG (reclassified from WEAK PASS under the per-word Δ_specific bootstrap), and three FAILs under intervention.

**Table 5 columns are defined as follows.** `P(probe → c | PATCH_c)`: fraction of post-patch residuals classified by the probe as canonical `c` when patched with the NEUTRAL-`c` source vector (probe-causality sanity check; 1.0 = patch reaches the residual cleanly). `ΔKL(c)`: mean over 16 invented words of `KL(P_BASELINE || ref_c) − KL(P_PATCH_c || ref_c)`, where `ref_c` is the FUNC-PFX-`c` reference distribution on the sentence-final next-token logits; positive = patch pulled behaviour toward canonical `c`. `RANDOM ΔKL(c)`: same metric with patches drawn from a norm-matched Gaussian (negative control). `Arity-flip U → and` and `B → not` rows: ΔKL block computed *only* on wrong-arity-patched words — intended-unary invented words patched with `and`, intended-binary invented words patched with `not` — measuring whether the patch can causally flip the downstream arity.

**Table 5.** Cross-cell synthesis across the five (source, target, layer) cells. Verdict assigned by: CLEAN PASS = targeted ΔKL ≫ RANDOM on both axes and arity-flip ≥ 7/8 positive in both directions; WEAK PASS = targeted ΔKL > RANDOM by approximately 1.3-2× on at least one axis and arity-flip ≥ 6/8 positive in both directions, with the per-word Δ_specific 95% CI strictly above zero on at least one axis (the cleaner statistic — script 25d log-parses 25a's per-word ΔKL table and bootstraps `Δ_specific = ΔKL_targeted − ΔKL_random_mean` across the 16 invented words with B = 500); AMBIG = mean-ratio in the 1.3-2× band but per-word Δ_specific 95% CI straddles zero on both axes; FAIL = RANDOM ≥ targeted ΔKL on at least one axis OR arity-flip at chance on at least one direction. `(extra)` marks the two reviewer-round-1 follow-up cells added to disambiguate source-anchor direction-specificity. Numbers from `outputs/25a_20260520_211030.log` (three principal cells), `outputs/25a_20260521_085745.log` (two reviewer-extra cells), and `outputs/25d_20260521_132205.log` (Cell-2 Δ_specific bootstrap).

| Cell | Source | Target | M2c | M2a | ΔKL(not) | ΔKL(and) | RND(not) | RND(and) | U→and | B→not | **Verdict** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma `opera→close L 2` | opera | close L 2 | 0.420 | 0.841 | +0.048 (15/16) | +0.038 (16/16) | ~0 | ~0 | +0.033 (8/0/8) | +0.061 (8/0/8) | **CLEAN PASS** |
| Gemma `opera→opera L 4` (extra) | opera | opera L 4 | 1.000 | 1.000 | +0.033 (13/16) | +0.027 (12/16) | +0.017 | +0.021 | +0.028 (6/2/8) | +0.039 (7/1/8) | **AMBIG** |
| Gemma `sente→close L 2` | sente | close L 2 | 0.512 | 0.832 | −0.020 (6/16) | −0.012 (7/16) | +0.019 | n/a | −0.018 (4/4/8) | chance | **FAIL** |
| OLMo `sente→close L 10` | sente | close L 10 | 0.736 | 1.000 | −0.012 (4/16) | −0.017 (3/16) | +0.019 | +0.012 | −0.020 (1/7/8) | −0.006 (3/5/8) | **FAIL** |
| OLMo `opera→close L 10` (extra) | opera | close L 10 | 0.508 | 0.872 | +0.013 (15/16) | +0.003 (10/16) | +0.023 | +0.016 | +0.001 (5/3/8) | +0.021 (8/0/8) | **FAIL** |

**Δ_specific bootstrap for Cell 2** (script 25d, `outputs/25d_20260521_132205.log`). Per-word `Δ_specific(axis) = ΔKL_targeted(axis) − ΔKL_random_mean(axis)` bootstrapped over the 16 invented words (B = 500): `not` axis Δ_specific 95% CI = **[−0.001, +0.031]** (mean +0.016, 13/16 words positive); `and` axis Δ_specific 95% CI = **[−0.005, +0.020]** (mean +0.006, 11/16 words positive). Both CIs include zero; the mean-ratio WEAK PASS signal at this cell does not firm up under the cleaner statistic, and Cell 2 is reclassified to AMBIG.

All five cells have probe-causality = 100% / 100% (the patches reach the residual cleanly under v2's hook-capture protocol; see Appendix H); the five-way split is on behavioural KL only.

**Cell 1 — Gemma `opera→close L 2`: CLEAN PASS.** Behavioural ΔKL is strongly positive on both reference axes (+0.048 against `ref_not`, 15/16 words positive; +0.038 against `ref_and`, all 16 words positive) and substantially exceeds the RANDOM_NORM control (~0 on both axes). The decisive datum is the arity-flip block: when an intended-unary word is patched with `and` (canonical binary), all 8/8 unary-intended words shift toward FUNC-PFX-`and` (mean ΔKL = +0.033); symmetric 8/0/8 for intended-binary words patched with `not` (mean ΔKL = +0.061). **The L 2 close-paren position in Gemma 2 9B, when patched from NEUTRAL operator-after, causally controls the downstream arity-respecting behaviour.** The v6 emergent PASS-arity at this cell is real under causal intervention. This is the only cell with clean causally-arity-respecting behaviour across the five-cell sweep.

**Cell 2 — Gemma `opera→opera L 4` (extra, reviewer-round-1 follow-up): AMBIG.** This is the project-flagship cell — M2-canonical = 1.000 with bootstrap CI [1.000, 1.000] across all three model families. Behavioural ΔKL is positive on both axes (+0.033 / +0.027) and exceeds RANDOM_NORM by approximately 1.94× on `not` and 1.29× on `and`; the `and` ratio sits at the soft boundary of the 1.3-2× band. Arity-flip is moderately positive in both directions (75% and 87.5%). The RANDOM_NORM floor at this cell is itself positive (+0.017 / +0.021), indicating that mean-pool patching here produces a generic-disruption effect on top of any specific-canonical signal. **Per-word Δ_specific bootstrap (script 25d, lab notes §3.7.22) reclassifies the cell from WEAK PASS to AMBIG:** the bootstrap CI on `Δ_specific(not) = ΔKL_targeted(not) − ΔKL_random_mean(not)` is [−0.001, +0.031] (13/16 words positive) and on `Δ_specific(and)` is [−0.005, +0.020] (11/16 words positive). Both CIs include zero, marginally on `not` and clearly on `and`; the mean-ratio WEAK PASS signal does not firm up under the per-word statistic. **The principal cross-family Fact-1 cell's causal load-bearingness is not robust under the cleaner statistic. The geometric Fact 1 result (M2c = 1.000) does not translate into statistically-clean causal load-bearingness even at its strongest cell, where the per-word distribution of causal effect is consistent with the norm-matched RANDOM_NORM baseline within sampling noise.**

**Cell 3 — Gemma `sente→close L 2`: FAIL.** Same target & layer as Cell 1 but with sentence-final source instead of operator-after. ΔKL is flat-or-negative on both axes (−0.020, −0.012); RANDOM_NORM exceeds targeted on the `not` axis (+0.019 vs −0.020). Arity-flip is at chance. **The L 2 close-paren position cannot be causally driven from a NEUTRAL-sentence-final source at this cell.** M4b's 66.2% reading at this cell at v6 is a probe-readable threshold-crossing under expanded readout, not a causally arity-respecting structure. The lab notes §3.7.16 methodological-caveat reading is confirmed.

**Cells 4 and 5 — OLMo `→close L 10` under both source anchors: FAIL.** Cell 4 (sentence-final source, original lab notes §3.7.17 cell): ΔKL negative on both axes (−0.012, −0.017); RANDOM strongly positive (+0.019 / +0.012). Cell 5 (operator-after source, extra, reviewer-round-1 follow-up): ΔKL barely positive (+0.013, +0.003); RANDOM still exceeds on both axes (+0.023 / +0.016). Arity-flip is at chance for unary→and in both cells. **The OLMo L 10 close-paren position is causally inert under both tested source anchors** — the within-target source-anchor flip does not rescue this target. This is a stronger version of the lab notes §3.7.17 OLMo finding: the target position itself is geometric / probe-readable but causally inert, independent of which NEUTRAL source the patch is drawn from.

Three findings emerge from the 5-cell sweep:

(i) **Geometric Fact 1 does not imply causal load-bearingness under the per-word Δ_specific bootstrap.** The principal cross-family Fact-1 cell (Gemma `opera→opera L 4`, M2c = 1.000 [1.000, 1.000]) has positive mean ΔKL on both axes and a mean ratio of 1.3-2× over RANDOM_NORM, but the per-word Δ_specific 95% CI straddles zero on both axes (`not`: [−0.001, +0.031]; `and`: [−0.005, +0.020]) — the signal does not separate from the norm-matched baseline within sampling variability across the 16 invented words. The project-flagship OLMo Fact-1 anchor (`sente→close L 10` / `opera→close L 10`) is causally inert under both tested source anchors. The only cell with clean causally-arity-respecting behaviour is the Gemma `opera→close L 2` *v6 emergent* cell, not a Fact-1 anchor. The headline "trained-vocabulary substrate-invariance" finding is therefore most accurately read as a *geometric / linear-probe-readable* substrate-invariance claim; the causal grounding holds cleanly at one specific (source, target, layer) joint product (the Gemma v6 emergent cell, which is not a Fact-1 anchor) and weakly-to-not-at-all at every Fact-1 anchor we have tested.

(ii) **The Gemma v6 emergent PASS-arity finding splits 1:1.** The `opera→close L 2` cell is causally validated as a single-model exception to the Fact-2 novel-operator generalisation failure; the `sente→close L 2` cell is M4b-granularity-only as lab notes §3.7.16 hypothesised. The headline picture needs one tightly-scoped Gemma-specific caveat to Fact 2, not a wholesale retraction.

(iii) **Source anchor is a non-trivial causal variable but is not directionally deterministic across cells.** The within-target source-anchor flip at Gemma L 2 close-paren (Cell 1 vs Cell 3) produces clean opposite verdicts at identical target / layer / probe-causality — establishing that source anchor matters at this specific cell. However, operator-after sources do *not* reliably produce causal effects across targets: Cell 2 (Gemma `opera→opera L 4`) is AMBIG under per-word Δ_specific bootstrap, and Cell 5 (OLMo `opera→close L 10`) is FAIL. The "operator-after sources reliably PASS / sentence-final sources reliably FAIL" reading was a v1-draft hedge that the reviewer flagged as under-powered; the 5-cell evidence rejects it. **The verdict is the joint product of source, target, and layer**, not a function of source-anchor alone. Probe-causality remains necessary but not sufficient for causal load-bearingness, and same-target / different-source comparisons remain a useful methodological device — but the device characterises the (source, target) interaction, not a free-standing property of either anchor.

**Figure 4.** Causal patching cross-cell synthesis. Five-panel barplot, one panel per (source, target, layer) cell tested in script 25a (three principal cells from the original 25a run plus the two reviewer-round-1 follow-up cells, marked (extra)). Each panel shows targeted PATCH and RANDOM_NORM control ΔKL on the `ref_not` and `ref_and` axes. A cell is causally arity-respecting only when targeted ΔKL clearly exceeds RANDOM_NORM on the same axis. The verdict bar at the top of each panel summarises Table 5's adjudication: one CLEAN PASS (Gemma opera→close L 2), one AMBIG (Gemma opera→opera L 4, under per-word Δ_specific bootstrap; mean-ratio reading was WEAK PASS), three FAIL. Rendered by `experiments/figures/fig_04_causal_patching.py`; artefact at `experiments/figures/out/fig_04_causal_patching.{pdf,png}`.

Causal patching with mean-pooled source activations is one operationalisation of intervention; §6 discusses its limits (mean-pool centroid bias; cross-notation source-target attention-pattern mismatch).

### 4.6 Mechanism gap is not closed by mean-pooled cosine similarity

§4.4 ruled out the three single-axis readings of the default mechanism. The natural next-most-parsimonious closing is **contextual semantic neighborhood**: each invented word's default attractor is the canonical that is geometrically closest to it in residual stream space at the focus layer, controlled for intended arity. We test this directly across all three model families and the full v6 sweep with the embedding-similarity probe (script 25b): for each of three models × 80 sweep cells, compute the cosine-similarity argmax between each invented word's mean activation and each canonical's mean activation at the same `(target_cond, target_anchor, layer)` coordinate the probe predicts on, and ask whether that argmax matches the probe's empirical per-word top canonical.

Two variants: unconstrained `sim-all` (chance 1/15 ≈ 6.7%) and arity-conditioned `sim-arity` restricting the argmax to canonicals of matching intended arity (chance ≈ 13.3%). Plus a coarser `arity-match` metric: does the unconstrained sim-all top have the same arity as the probe top? (Chance ≈ 53.3% under the majority-arity baseline.) Bootstrap 95% CIs by resampling the 16-word invented set with replacement (B = 200).

**Cross-model headline (mean over 80 cells per model, focus layers only):**

**Table 6.** Embedding-similarity probe vs linear-probe per-word routing agreement at the principal mid-layer cell per model, with bootstrap 95% CIs (B = 200 resamples over the 16 invented words). `agree-all` is the fraction of invented words where the geometric argmax over all 15 canonicals matches the linear probe's top canonical (chance = 1/15 ≈ 6.7%). `agree-arity` is the same agreement under the geometric argmax restricted to canonicals of matching intended arity (chance ≈ 1/7-1/8 ≈ 13%; the pre-specified "mechanism gap closed" threshold is ≥ 60%, declared in the `25b_embedding_similarity_probe.py` docstring before any 25b run). `arity-match` is the fraction of invented words where the unrestricted geometric argmax has the correct binary-vs-unary intended arity (chance ≈ 53% under the majority-arity baseline given v6's 8B / 7U split). All three agreement metrics sit at or near chance for `agree-arity` and `agree-all`; none of the three models clears the pre-specified 60% threshold for `agree-arity`. From `outputs/25b_20260520_213935.log`.

| Model | `agree-all` (chance 6.7%) | `agree-arity` (chance ≈ 13%) | `arity-match` (chance ≈ 53%) |
|---|---|---|---|
| Gemma 2 9B | 11.6% [9.6, 13.8] | 11.4% [9.1, 13.4] | 54.1% [49.8, 58.2] |
| OLMo 2 7B | 26.6% [24.1, 28.6] | 21.5% [18.4, 24.6] | 66.4% [64.0, 68.4] |
| Pythia 6.9B-d | 24.0% [21.4, 26.5] | 19.0% [15.6, 22.5] | 60.2% [57.7, 62.6] |

**The pre-specified threshold for "mechanism gap is closed" was `agree-arity ≥ 60%` (declared in the `25b_embedding_similarity_probe.py` docstring before any run). No model meets this threshold; all three are well below.** Gemma's CI [9.1, 13.4] does not even reach the ~13.3% within-arity chance baseline, and OLMo and Pythia are at modest 1.4-1.6× chance. At distributed cells (M4c < 0.7, the methodologically interesting regime where the probe spreads predictions across multiple canonicals rather than collapsing to a single attractor), `agree-all` drops to 11-16% across all three models — barely above the 6.7% chance baseline.

**Three findings:**

(i) Mean-pooled cosine similarity captures arity but not the per-canonical identity. The `arity-match` metric is moderately above chance in OLMo and Pythia (66.4%, 60.2%) and at chance in Gemma (54.1%); the gap between `arity-match` and `agree-arity` is the principal positive signal. Once we condition on intended arity, the predictive power of cosine similarity within arity is at chance.

(ii) **L 0 embedding-layer agreement collapses to floor (0-1%) in all three models.** This is a clean v6 cross-family re-confirmation of the prior single-model finding that the operator-region attractor structure is constructed by intermediate-layer processing rather than inherited from token-embedding geometry.

(iii) A specific failure-mode signature: at distributed cells across all three models, the unconstrained cosine argmax collapses every invented word to `nand` (for binary-intended) or `negate` (for unary-intended under arity-conditioning), most likely a canonical-magnitude-and-idiosyncrasy effect (`nand`'s mean activation sits further from the canonical centroid than higher-frequency canonicals because its per-stim activations are most idiosyncratic). This is a near-tautological geometric structure that does not reflect the probe's actual decision boundary. **Mean-pooled cosine on residual-stream activations is contaminated by canonical-magnitude effects** and should be reported alongside probe-based readouts, never as a standalone replacement.

**Figure 5.** Embedding-similarity probe vs linear-probe agreement at L 0 + the focus-layer set per model. Three-panel line plot (Gemma 2 9B / OLMo 2 7B / Pythia 6.9B-d). Three curves per panel: `agree-all` (unconstrained cosine argmax matches probe top; chance ≈ 6.7%), `agree-arity` (cosine argmax restricted to canonicals of matching intended arity; chance ≈ 13%; pre-specified "mechanism gap closed" threshold at 60% marked as a solid horizontal line), `arity-match` (cosine argmax has correct intended arity; chance ≈ 53%). The L 0 column (highlighted grey) is at floor (0-3%) on the identity metrics in all three models — the operator-region attractor structure is constructed by intermediate-layer processing, not inherited from token-embedding geometry. Mid-layer peak at L 4 (Gemma) and L 10 (OLMo, Pythia); late-layer L 24 identity-collapse with `arity-match` remaining elevated. Rendered by `experiments/figures/fig_05_agreement.py`; artefact at `experiments/figures/out/fig_05_agreement.{pdf,png}`.

**Net §4.6 verdict.** The §4.4 mechanism gap is **not** closed by mean-pooled cosine similarity. The residual third factor in the default mechanism is *probe-decision-boundary geometry* — the LogisticRegression boundary captures per-word residual-stream structure that mean-pooling wipes out — and is therefore not reducible to any single mean-pooled-activation statistic.

---

## 5 Discussion

### 5.1 Trained-vocabulary substrate-invariance as a refined edge of the Platonic Representation Hypothesis

Our two-part headline locates a precise empirical edge of the PRH:

**Inside the model's trained vocabulary (within the tested readout sets).** Cross-notation linear-probe geometry transfers at ceiling (M2-canonical = 1.000 with bootstrap CI [1.000, 1.000] under 15-class readout) across three model families with three independent training corpora, three architectures, and three tokenizers, *both for logical operators and for an orthogonal set of 15 non-operator content words* (§4.1.1, script 25c). At the principal N→F operator-after cell, the model has a substrate-independent linear-probe-readable representation of *the trained, well-tokenized vocabulary items in the tested 15+15 readout set*; the substrate-invariance is a per-token-identity property at this anchor/direction coordinate, not a property of logical-operator-class semantics. The v1 draft's "operator-set-bound" framing is rejected by the §4.1.1 content-word control and replaced by the narrower-than-it-sounds "trained-vocabulary-bound" framing — narrower because the reframe is scoped to the tested readout sets and the principal direction, broader-than-operator-class because it is not domain-specific.

**Outside the trained operator set.** The apparent arity-respecting routing of invented operators is retracted under canonical-readout expansion in all three model families. The actual mechanism is a readout-vocabulary-dependent canonical-attractor compression whose target shifts wholesale as the readout vocabulary changes; no single-axis statistic (frequency, subword shape, mean-pooled cosine similarity) — including the original "default-to-rarest-canonical" hypothesis — predicts the per-word routing target. The Fact-2 negative finding is preserved as-stated — it is a claim specifically about novel *operator* words; one tightly-scoped Gemma 2 9B L 2 close-paren cell is a causally-validated single-model exception (§4.5), and the analog claim for novel content words is unverified.

The two facts together identify a sharp PRH edge: the model has a substrate-independent geometry for its trained vocabulary at the operator-after anchor, demonstrated across two word-class controls (operators and content words); and lacks a substrate-independent *category* of "logical operator" that would extend to novel members. This is more constrained than either "the model encodes underlying structure" or "the model is autocomplete-with-extra-steps" as headline claims, and is the more accurate intermediate position. The reframe from "operator-set-bound" to "trained-vocabulary-bound" makes the positive claim simultaneously broader (the tested 15-operator + 15-content-word readout sets, not just the operator set) and shallower (lexical-identity substrate-invariance at the principal anchor/direction, not a substrate-independent semantic category for logical operators specifically) than the v1 framing. The broader-but-shallower combination is the better empirical fit and the more accurate scientific claim.

Two refinements to the headline survive the v6 + script 25 expansion. (1) **Gemma 2 9B has a single causally-validated exception** at L 2 close-paren sourced from NEUTRAL operator-after, where novel-operator activations behave arity-respectingly under intervention. We treat this as a single tightly-scoped model-specific exception, not a refutation of the operator-novelty failure for Gemma. (2) **Fact 1's geometric transfer does not translate to robust causal load-bearingness at any Fact-1 anchor we have tested.** The 5-cell causal-patching sweep (§4.5) demonstrates this directly: the principal cross-family Fact-1 cell (Gemma `opera→opera L 4`, where M2c = 1.000 with bootstrap CI [1.000, 1.000] across all three models) produces mean ΔKL approximately 1.3-2× the RANDOM_NORM control, but the per-word Δ_specific 95% CI straddles zero on both axes (`not`: [−0.001, +0.031]; `and`: [−0.005, +0.020]; script 25d) — the causal signal is not statistically distinguishable from the norm-matched baseline within sampling variability across the 16 invented words. The project-flagship OLMo Fact-1 anchor at L 10 close-paren is causally inert under both tested source anchors. The "trained-vocabulary substrate-invariance" headline is therefore most accurately read as a *geometric / linear-probe-readable* substrate-invariance claim: the linear-probe geometry transfers at ceiling across notations within the trained vocabulary, but at every Fact-1 anchor we have causally tested the load-bearingness either clearly fails or fails to firm up under the per-word Δ_specific bootstrap.

### 5.2 The (source, target, layer) joint product as a causal variable

The 5-cell causal-patching sweep (§4.5) supports a precise but constrained methodological finding. The within-target source-anchor flip at Gemma L 2 close-paren — identical target, identical layer, identical 100% probe-causality, opposite behavioural verdicts (Cell 1 PASS, Cell 3 FAIL) — establishes that **same-target / different-source comparisons can reveal causal dissociations that single-anchor-pair patching misses**. Probe-causality is necessary but not sufficient for causal load-bearingness; the source from which a patch is drawn is a non-trivial variable that interacts with the target position to determine downstream load-bearingness.

**However, source anchor is not directionally deterministic across cells.** The two reviewer-round-1 follow-up cells (§4.5, Cells 2 and 5) test whether operator-after sources reliably produce causal effects at non-Gemma-L-2 targets. They do not: Cell 2 (Gemma `opera→opera L 4`, the principal Fact-1 anchor) is AMBIG under the per-word Δ_specific bootstrap (mean ratio 1.3-2× over RANDOM_NORM but Δ_specific 95% CI straddles zero on both axes), and Cell 5 (OLMo `opera→close L 10`, the within-target source-anchor flip on the §4.2 flagship cell) is clean FAIL. The v1-draft hedge "operator-after sources reliably drive downstream causal effects, sentence-final sources reliably fail" is rejected by the 5-cell evidence.

**The verdict is the joint product of source, target, and layer.** The mechanistic reading: the residual stream at a given target position is read out by downstream layers through a position-specific aggregation pattern, and patch sources whose residual-stream coordinate system aligns with that aggregation pattern can drive downstream computation; sources whose coordinate system does not align cannot. The alignment relation depends on both the source and target position simultaneously — there is no general rule that operator-after sources align with all targets or that sentence-final sources fail at all targets. The probe, which reads a linear projection of the residual stream, is robust to source-target coordinate-system mismatches; downstream layers are not.

**Two substantive findings survive the more conservative reading.** (i) Within-target source-anchor flip *can* produce opposite causal verdicts (Gemma L 2 close-paren). (ii) Some target positions are causally inert regardless of source anchor (OLMo L 10 close-paren under both sources). Both are methodologically informative and should inform future activation-patching protocol design. The first motivates same-target / different-source comparisons as a routine diagnostic; the second nuances the interpretation of "probe-readable but causally inert" cells (the inertness can be a property of the target, not merely of a particular (source, target) pair).

### 5.3 Methodological contributions

Four methodological devices in this paper are directly transferable to other probe-based interpretability studies.

**(i) The within-vocabulary content-word control.** Section 4.1.1's pre-specified post-review control — re-running the M1-M4 battery with 15 heterogeneous non-operator content words inserted into syntactically-identical templates — is the load-bearing diagnostic that distinguishes domain-specific substrate-invariance ("the model has a substrate-independent representation of *logical operators*") from generic lexical-identity substrate-invariance ("the model has a substrate-independent representation of any trained, well-tokenized vocabulary item at the tested anchor/direction coordinate"). Without the control, the v1 draft's "operator-set-bound" claim would have been published, and the load-bearing reframe to "trained-vocabulary-bound" would have been missed. The control is single-experiment, ~50 min of compute reusing the existing extraction infrastructure, and applicable to any probe-based substrate-invariance study that reports ceiling-level cross-context transfer for a hypothesized semantic category. Any such study should pre-register a content-word (or otherwise category-orthogonal) control as a required measurement.

**(ii) Same-target / different-source comparisons as a routine activation-patching diagnostic.** Section 4.5's within-target source-anchor flip at the two Gemma 2 9B L 2 close-paren cells — identical target anchor, identical layer, identical 100% probe-causality, opposite behavioural verdicts — demonstrates that two patches identical in every reported way except patch source can produce opposite causal conclusions. Activation patching as currently practiced reports a single source-target patch transfer and treats the result as a property of the target representation; our evidence shows the result can be a joint property of source, target, and layer (§5.2). The transferable methodological device is the *comparison*: any single-anchor-pair patching result should be supplemented by at least one same-target / different-source patch to characterise the (source, target) interaction and to surface cases where the apparent target-representation property is in fact a (source, target)-coupling property. This is a less ambitious claim than "source anchor is a first-class causal variable" (which the v1 draft made but which the 5-cell evidence does not support across cells); it is the more accurate methodological lesson the project's evidence does support.

**(iii) The lucky-default detector.** A specific false-positive pattern in probe-based substrate-invariance — uniform single-canonical routing whose canonical's arity happens to coincide with the majority arity of the test set — produces an aggregate M4b above threshold while every per-word concentration is at ceiling. The detector flags this with `min_w P(top_canonical | w) ≥ 0.95`. Across our cell sweep, the detector reclassified 4 of 8 originally-flagged PASS-arity cells as false positives, and should be required in any probe-based study that reports per-word intended-class agreement.

**(iv) The M2-canonical / M2-arity dissociation.** Cross-notation substrate-invariance has two distinct sub-claims (canonical identity transfers; arity axis transfers) that should be measured separately. Reporting only canonical-identity transfer misses arity-axis-only dissociation (§4.2's OLMo `N→F sente→close L 10` cell, where M2-arity = 1.000 but M2-canonical sits at 0.604-0.806 across the four scopes); reporting only arity-axis transfer collapses cells where canonical identity is also preserved into the same bin as cells where only arity is. *Related empirical caveat: M4b is granularity-sensitive.* Two Gemma cells in our sweep showed M4b trajectories `60% → 56% → 50% → 82%` across v3 → v4 → v5 → v6 *without any change to underlying activations* — the 0.65 PASS threshold on M4b depends on how many within-arity buckets the canonical readout vocabulary expresses. M2-arity is the partition-invariant primary measurement; M4b should be reported alongside but flagged as granularity-sensitive when the canonical-set size changes.

### 5.4 What the mechanism gap tells us

The combined evidence from §4.4 (single-axis predictions fail), §4.5 (causal patching produces a three-way split), and §4.6 (mean-pooled cosine similarity does not close the gap) suggests that the residual mechanism of novel-operator routing is *probe-decision-boundary geometry* — the LogisticRegression boundary captures per-word residual-stream structure that single-axis aggregate statistics cannot reproduce. The §4.6 per-layer agreement map (Figure 5) localises the residual mechanism to specific layers in each model: agreement peaks at L 4 in Gemma 2 9B and at L 10 in OLMo 2 7B and Pythia 6.9B-d, with floor agreement at L 0 in all three families. We ship the per-cell probe weights at these layers in the repository (`experiments/probes/`, small enough to track in git) and make the v6 carryover residual-stream caches (multi-GB per model, gitignored) available on request, as a starting point for sparse-autoencoder feature labelling — the next-most-natural mechanism characterisation — on Gemma Scope SAEs at L 4 and on locally-trained SAEs at the OLMo / Pythia L 10 caches. The agreement-map localisation is itself a contribution: it constrains where the residual mechanism lives in each model's depth, and the contemporary SAE-feature-labelling literature has the tooling to characterise the relevant decision-boundary geometry once given the layer and cache pointers.

---

## 6 Limitations

**Scale.** All three models tested are at the 6.9-9B parameter range. We have not tested whether the pattern dissolves at 70B+ frontier scale; the trained-vocabulary-bound positive finding and the novel-operator-generalisation negative finding may both be properties of mid-scale base models specifically. Phase 0 (not included in this paper) compared OLMo 2 1B vs 7B and showed a modest gap reduction; the trajectory does not project clean closure at frontier scale.

**Base models only.** All three models tested are pre-instruction-tuning base checkpoints. Whether instruction tuning extends the model's novel-operator-generalisation capability (partially refuting Fact 2), strengthens the trained-vocabulary geometric structure (strengthening Fact 1's causal grounding without altering the headline), or has no effect on either, is an open question.

**Trained-vocabulary scope of Fact 1 (resolved by §4.1.1).** The §4.1.1 content-word control resolves what was a v1-draft limitation in favour of the broader trained-vocabulary-bound framing. Two residual limits on the reframed claim are worth flagging. (a) **Novel content-word generalisation is untested.** Fact 2 demonstrates novel-*operator* generalisation failure under canonical-set expansion (canonical-attractor compression; the "default-to-rarest" single-axis reading is rejected by §4.4); the analog test with novel content-words inserted into a content-word readout has not been run. The combined claim "trained-vocabulary substrate-invariance holds; novel-word generalisation fails" is therefore demonstrated for operators only; the content-word side is half-tested (in-vocabulary content words PASS Fact 1; novel content words are an open empirical question). (b) **The §4.1.1 F→N asymmetry in OLMo (M2c = 0.376 for content words at the reverse-direction cell) is a model-specific anomaly** that does not appear at the same cell with canonical operators. We document it as a methodological caveat on F→N robustness for content-word probes in OLMo; it does not weaken the principal N→F REFRAME trigger.

**Causal-patching intervention design.** The §4.5 patching protocol uses two strong operationalisation choices that should be flagged. **(a) Mean-pooled source activations.** Patch sources are the mean NEUTRAL canonical activation across N = 50 stimuli, which sits at the centroid of the canonical's residual-stream distribution rather than at a representative single draw. This biases the patch toward the canonical's geometric prototype and washes out per-stimulus variance; an individual-stimulus patch may behave differently. The RANDOM_NORM control catches norm-matched random vectors but not the centroid-bias confound. The mean-pool choice is defensible on the grounds that §4.1 establishes the canonical's *shared* cross-notation geometric structure as the load-bearing claim — patching with the canonical-typical activation is the natural extension of that claim — but the choice is not innocent. A single-stimulus-patch robustness check is the cheapest follow-up that would tighten the §4.5 verdicts. **(b) Cross-notation source-target attention-pattern mismatch.** The patch replaces a FUNC-PFX-context residual with a NEUTRAL-context source vector while the downstream forward pass continues in FUNC-PFX context. If FUNC-PFX positions downstream of the patch target have a strong attention pattern back to FUNC-PFX-specific syntactic markers (parentheses, commas), the patch's downstream effect is filtered through a FUNC-PFX-specific aggregation that the NEUTRAL source vector does not coordinate with. The OLMo flagship cell's causal inertness (§4.5 Cell 4, plus the within-target source-anchor flip in Cell 5) is therefore consistent with two distinct readings: "L 10 close-paren is not on the causal path for arity-respecting downstream computation" *or* "L 10 close-paren is read out by L 11-32 through a FUNC-PFX-specific attention pattern that does not propagate a NEUTRAL-sourced patched signal". Our 5-cell evidence narrows the reading — both NEUTRAL source anchors fail at this target, so a *single*-source attention-pattern-mismatch reading is harder to sustain — but cannot fully distinguish the two interpretations; the safer §4.5 wording is "probe-readable but causally inert *under NEUTRAL-sourced mean-pooled patching at this target*" rather than "not on the causal path".

**Causal evidence is partial.** Script 25a tests five cells across two runs (three principal Phase-1/Phase-2 cells plus two reviewer-round-1 follow-ups); script 25d adds a per-word Δ_specific bootstrap on Cell 2. The cell-wise outcomes: one CLEAN PASS (Gemma `opera→close L 2`), one **AMBIG** (Gemma `opera→opera L 4`, the principal cross-family Fact-1 anchor — mean ratio 1.3-2× over RANDOM_NORM but per-word Δ_specific 95% CI straddles zero on both axes: `not` CI [−0.001, +0.031], `and` CI [−0.005, +0.020]), and three FAILs (Gemma `sente→close L 2`, OLMo `sente→close L 10`, OLMo `opera→close L 10`). The §4.5 v1-draft "WEAK PASS verdict at the soft boundary" hedge for Cell 2 is now resolved against the cell: under the cleaner Δ_specific statistic the cell's causal effect is not statistically distinguishable from the norm-matched RANDOM_NORM baseline within sampling variability across the 16 invented words. Pythia 6.9B-d is not in the patching sweep at all, and the OLMo Fact-1 close-paren target has only been tested under NEUTRAL-sourced mean-pooled patches, not FUNC-PFX-sourced or individual-stimulus interventions. Three remaining cheap follow-ups: (i) a per-(word, stim) Δ_specific re-run at Cell 2 with per-word RANDOM_NORM baselines, rather than 25d's aggregate-offset baseline — requires re-running script 25a with extended per-word RANDOM_NORM logging (~6 min on MPS); (ii) Pythia 6.9B-d causal cells at L 4 and L 10 (~20 min on MPS); (iii) individual-stimulus patches at the Gemma CLEAN PASS cell to control for the mean-pool centroid-bias confound flagged in (a) above.

**Single structural domain.** Logic-inspired operator-role vocabulary only, framed by binary-vs-unary arity assignments rather than truth-functional semantics. Substrate-invariance for *executed* propositional or modal logic (e.g., whether the model's representation transfers when the same inference must be carried out, not merely when an operator-role token is named), for set theory, algebra (S_4 and similar finite groups), simple type theory, or other formal-structure domains is unverified.

**Confounded operator length.** Canonical operators are mostly 1 BPE token; most invented operators are 2 BPE tokens. Prior subword-length variation work (not in this paper's scope) showed the failure pattern is length-independent across L ∈ {1, 2, 3, 4}, but the headline numbers retain a residual length confound.

**Pre-registration scope.** The pre-registered v6 expansion (§3.5) freezes the canonical additions, single-axis predictions, and audit gate. The remainder of the analysis pipeline (lucky-default detector refinement, M2-canonical / M2-arity introduction, causal-patching protocol) was developed iteratively over the project's lifetime; a clean Phase 3 replication on novel templates with the full methodology pre-registered would be required before any of these results should be considered fully confirmatory.

**Probe-instrument dependence.** Our probe is a logistic regression on raw residual-stream activations with L2 regularisation. Alternative probe families (nearest-centroid, ridge regression, StandardScaler+LR, small MLP probes) may produce qualitatively different per-canonical readouts on the same activations. We have not systematically varied the probe family.

**Mean-pooled cosine is a single-resolution semantic-neighborhood operationalisation.** §4.6 rules out mean-pooled cosine specifically; the gap might be closed by attention-pattern-weighted projections, learned attribution methods, or sparse-autoencoder feature labelling. We leave these to future work.

---

## 7 Conclusion

Across three open base language model families at the 6.9-9B parameter range — spanning three training corpora, three architectures, and three tokenizers — substrate-invariance under cross-notation rewriting has a geometric / linear-probe-readable form that holds **inside the model's trained vocabulary, within the tested 15-operator + 15-content-word readout sets at the principal N→F operator-after cell** (cross-notation linear-probe transfer at M2-canonical = 1.000 with bootstrap CI [1.000, 1.000] under 15-class readout, surviving three independent scope expansions and a pre-specified post-review non-operator content-word control that triggered the reframe from the v1-draft "operator-set-bound" claim to the scope-limited trained-vocabulary-bound claim) and **is not robust outside the trained operator set** (the apparent arity-respecting routing at earlier scopes is retracted under pre-registered canonical-set expansion in all three model families; the underlying mechanism is a canonical-attractor compression that no single-axis statistic (including the v3-era "default-to-rarest" reading) captures; one tightly-scoped Gemma 2 9B L 2 close-paren cell is a causally-validated single-model exception). A 5-cell causal-patching sweep finds that the geometric Fact 1 does not imply causal load-bearingness: the principal cross-family Fact-1 cell has mean ΔKL approximately 1.3-2× the norm-matched control but its per-word Δ_specific 95% CI straddles zero on both axes (AMBIG under the cleaner statistic; script 25d), the OLMo flagship cell is causally inert under both tested source anchors, and the only cleanly causally-arity-respecting cell is a Gemma v6 emergent cell at L 2 close-paren, not a Fact-1 anchor. Within-target source-anchor flip can produce opposite verdicts at identical target and layer, but source anchor is not directionally deterministic across cells — the verdict is the joint product of source, target, and layer. An embedding-similarity probe rules out the most natural single-factor closing of the canonical-attractor compression mechanism (mean-pooled cosine similarity is not the residual axis the v6 disentanglement failed to isolate); the residual factor is probe-decision-boundary geometry, not raw cosine.

The two-part finding locates a precise empirical edge of the Platonic Representation Hypothesis at the within-language scale: substrate-independent *geometric* structure for the trained vocabulary items tested at the principal anchor and direction — and not specifically for any hypothesized semantic category — with non-uniform and generally modest causal load-bearingness; no equivalent substrate-independent category that extends to novel operators (and the novel-content-word analog is unverified). Methodologically, we contribute the within-vocabulary content-word control (the load-bearing test for any probe-based substrate-invariance study that claims domain-specific representation), the lucky-default detector, the M2-canonical / M2-arity dissociation, and the same-target / different-source patching protocol. The remaining mechanism question — what specifically about the probe's learned weights produces per-word-specific routing that mean-pooled cosine cannot reproduce — is the natural next direction for sparse-autoencoder feature labelling at the layer coordinates we identify (probe weights shipped in `experiments/probes/`; residual-stream caches available on request).

---

## References

Canonical BibTeX entries live in `references.bib` in the repository root. The
inline prose-style references in this Markdown draft will be replaced with
`\cite{key}` commands at LaTeX conversion; the table below maps prose
mentions to BibTeX keys.

| Inline mention | BibTeX key | Reference |
|---|---|---|
| Huh, Isola, et al. (2024) | `huh2024platonic` | The Platonic Representation Hypothesis. ICML 2024. arXiv:2405.07987. |
| Bommasani et al. (2023) | `bommasani2022opportunities` | On the Opportunities and Risks of Foundation Models. Stanford CRFM technical report, arXiv:2108.07258. |
| Radford et al. (2021) | `radford2021clip` | Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021. arXiv:2103.00020. |
| Alain & Bengio (2017) | `alain2017probes` | Understanding intermediate layers using linear classifier probes. ICLR Workshop. arXiv:1610.01644. |
| Belinkov (2022) | `belinkov2022probing` | Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics, 48(1). |
| Hewitt & Liang (2019) | `hewitt2019probes` | Designing and Interpreting Probes with Control Tasks. EMNLP 2019. |
| Ravichander et al. (2021) | `ravichander2021probing` | Probing the Probing Paradigm. EACL 2021. |
| Geiger et al. (2021) | `geiger2021causal` | Causal Abstractions of Neural Networks. NeurIPS 2021. |
| Meng et al. (2022) | `meng2022rome` | Locating and Editing Factual Associations in GPT (ROME). NeurIPS 2022. arXiv:2202.05262. |
| Heimersheim & Nanda (2024) | `heimersheim2024patching` | How to use and interpret activation patching. arXiv:2404.15255. |
| Zhang & Nanda (2024) | `zhang2024patchingbest` | Towards Best Practices of Activation Patching in Language Models. ICLR 2024. arXiv:2309.16042. |
| Dumas et al. (2025) | `dumas2025tongue` | Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers. ACL 2025 (Long Papers), pp. 31822–31841, Vienna. arXiv:2411.08745. |
| Lake & Baroni (2018) | `lake2018scan` | Generalization without Systematicity. ICML 2018. arXiv:1711.00350. |
| Hupkes et al. (2020) | `hupkes2020compositionality` | Compositionality Decomposed. JAIR 67. |
| Saxton et al. (2019) | `saxton2019math` | Analysing Mathematical Reasoning Abilities of Neural Models. ICLR 2019. arXiv:1904.01557. |
| OLMo 2 7B (model) | `olmo2_2025` | OLMo Team et al. (2025). *2 OLMo 2 Furious.* Allen Institute for AI technical report. arXiv:2501.00656. |
| Gemma 2 9B (model) | `gemma2_2024` | Gemma 2 technical report. arXiv:2408.00118. |
| Pythia 6.9B-d (model) | `pythia2023` | Pythia model suite. ICML 2023. arXiv:2304.01373. |

---

## Appendix

This appendix is a minimal arXiv-v1 supplement; the bulk artefacts (full
240-row sweep tables, per-cell `.npz` activation caches, per-stimulus
ΔKL traces, four-condition stimulus template files) are released in the
public repository at commit `e6afe5e358454a8a5ca85f369eb2206a847b34d5`
(short: `e6afe5e`). Each appendix item below is either fully reproduced
here or marked **[repo-supplement]** with the exact file path that
contains the bulk content.

**A. Stimulus templates.** Four template families: NEUTRAL carryover
(50 templates), NEUTRAL held-out (50), FUNC-PFX carryover (50),
FUNC-PFX held-out (50). NEUTRAL templates are English-metalinguistic
prose (e.g., `"the operator OP applied to A and B yields …"`); FUNC-PFX
templates are programming-style functional-prefix (e.g., `OP(A, B) = …`).
Carryover and held-out template families are disjoint by lexical
scaffold. Templates are shipped in
`experiments/19b_directional_angle_gated.py` (NEUTRAL/FUNC-PFX carryover)
and `experiments/24_v6_canonical_expansion.py` (held-out). **[repo-supplement]**

**B. Hyperparameters.** Logistic-regression probe (sklearn
`LogisticRegression`): `C = 1.0`, `max_iter = 5000`, `solver = "lbfgs"`,
no class weighting, no feature scaling. M1: stratified 5-fold CV.
M2-canonical / M2-arity: train on NEUTRAL anchor activations, evaluate
on FUNC-PFX anchor activations (or reverse, for F→N analyses). Bootstrap
B = 500 for M2 and M4b CIs (resampling at the (stimulus, word) level);
B = 100 for M3 angle CIs; B = 200 for §4.5 (25b) agreement CIs; B = 500
for Cell-2 Δ_specific per-word bootstrap (§4.5, 25d). Random seed
`SEED = 1337` throughout (`experiments/figures/_shared.py`).

**C. v6 sweep verdict tally** (cells × models × scopes, replayed under
both running-code and frozen criteria, `outputs/24b_20260521_120258.log`).
Each cell is one (direction, source-anchor, target-anchor, layer) tuple
in the per-model focus-layer grid (Gemma {2, 4, 8, 16, 17}; OLMo / Pythia
{4, 7, 10, 16, 24}), giving 2 directions × 2 source anchors × 4 target
anchors × 5 layers = 80 cells per model per scope, 240 total per scope:

| Model | scope | cells | PASS-arity (running) | PASS-arity (frozen) | verdict diffs |
|---|---|---|---|---|---|
| Gemma 2 9B | v3 | 80 | 2 | 2 | 0 |
| Gemma 2 9B | v4 | 80 | 0 | 0 | 0 |
| Gemma 2 9B | v5 | 80 | 0 | 0 | 0 |
| Gemma 2 9B | v6 | 80 | 2 | 2 | 0 |
| OLMo 2 7B | v3 | 80 | 3 | 3 | 0 |
| OLMo 2 7B | v4 | 80 | 3 | 3 | 0 |
| OLMo 2 7B | v5 | 80 | 0 | 0 | 0 |
| OLMo 2 7B | v6 | 80 | 0 | 0 | 0 |
| Pythia 6.9B-d | v3 | 80 | 4 | 3 | **1** |
| Pythia 6.9B-d | v4 | 80 | 3 | 3 | 0 |
| Pythia 6.9B-d | v5 | 80 | 0 | 0 | 0 |
| Pythia 6.9B-d | v6 | 80 | 0 | 0 | 0 |

Single verdict diff: Pythia v3 `N→F opera→close L16` is PASS-arity
under the running-code wider M4a band `[0.10, 0.90]` but
ARITY-AXIS-ONLY under the frozen `[0.20, 0.80]` band (M4a = 0.192).
The headline P_RETRACT verdict (zero PASS-arity cells at v6 in OLMo
and Pythia; the two emergent Gemma v6 cells are byte-identical
under both criteria) is robust to which criterion is used. Full 960-row
per-cell table **[repo-supplement: `outputs/24_20260520_185537.log`]**.

**D. M1-heldout per cell (template-leakage diagnostic).** Probe trained on
carryover templates of one condition, evaluated on the syntactically-
disjoint held-out template family of the *same* condition. At the
principal Fact-1 / Fact-2 cells:

- Gemma 2 9B FUNC-PFX `opera-after` L 2 / L 4 / L 8: M1-heldout =
  1.00 / 0.99 / 0.95.
- OLMo 2 7B FUNC-PFX `opera-after` L 4 / L 7 / L 10: M1-heldout =
  0.94 / 0.96 / 0.97.
- Pythia 6.9B-d FUNC-PFX `opera-after` L 4 / L 7 / L 10: M1-heldout =
  0.96 / 0.94 / 0.96.

Late-layer FUNC-PFX `sentence-final` and `close-paren` cells degrade
substantially (Gemma FUNC-PFX `sentence-final` L 8 = 0.30, OLMo
FUNC-PFX `close-paren` L 24 = 0.37); these are not the locus of any
positive finding. Full per-cell table **[repo-supplement:
`outputs/24_20260520_185537.log`]**.

**E. The eight retracted v3 PASS-arity cells (frozen-criterion replay).**
Per-scope (M2-arity, M4b, M4c-HHI, M4a, pwmin) trajectories for each cell from v3
through v6, under the frozen pre-registered criterion, are listed in
`outputs/24b_20260521_120258.log` (the "Cells PASS-arity under EITHER
criterion (any scope)" blocks per model). All eight cells are
retracted by v5; the retraction trajectories are detailed in §4.3 of
the main text. The pre-pre-registration Phase-1 sweep
(`experiments/22b_invented_set_expansion.py`) ran on its own pre-v6
caches and flagged a related but not byte-identical four-cell
candidate set; lab notes §3.7.21 reconciles the two cell sets, and the
v6-pipeline list is the criterion-of-record. **[repo-supplement]**

**F. Pre-registration document.** `experiments/preregistration_v6.md`,
header `Status: FROZEN. Written before any v6 extraction or analysis
runs`, published in commit `e6afe5e` alongside the v6 extraction
script and the resulting v6 sweep log. The document specifies the
five v6 canonical additions, the three single-axis P_FREQ / P_SUBWORD /
P_INTERACTION predictions for the default mechanism, and the frozen
PASS-arity criterion (M4c as `max_c p_c ≤ 0.85`, M4a band
`[0.20, 0.80]`). §3.5 reports the criterion-drift between the running
code and this document and the cache-only replay reconciliation
(script 24b). **[repo-supplement]**

**G. Tokenization audit summary.** All 15 v6 canonicals, the 16
invented Tier-2 BPE words, and the 15 content-word control items
were audited at extraction time for subword decomposition under each
model's tokenizer (Gemma 2 SentencePiece BPE, OLMo 2 BPE, Pythia BPE).
Per-canonical subword piece counts are listed in Table 4 (§4.4) under
the `Tok` column. The multi-piece rates are tokenizer-dependent and
asymmetric across the three families:

- **v6 canonical set (15 words).** Gemma 2 9B: 1/15 multi-piece
  (`unprovably`). OLMo 2 7B: 3/15 (`xor`, `nand`, `unprovably`).
  Pythia 6.9B-d: 4/15 (`xor`, `nand`, `negate`, `unprovably`).
- **Content-word control set (15 words, §4.1.1).** Gemma 2/15; OLMo
  3/15; Pythia 4/15 (script 25c audit; concrete word lists at
  `outputs/25c_20260521_092243.log` `[audit]` blocks).
- **Invented Tier-2 BPE set (16 words).** 0/16 multi-piece in all
  three tokenizers — this is what "Tier-2 BPE" denotes in this paper,
  and the v6 design accepted the residual multi-piece canonicals
  (`xor`, `nand`, `negate`, `unprovably`) in exchange for spanning
  a wider corpus-frequency range.

The asymmetry between Gemma's 1/15 and Pythia's 4/15 multi-piece v6
rate is the cleanest correlate of `nand`'s 40.5% OLMo invented-mass
share (Table 4) and the 63.4% multi-piece-aggregate in Pythia v6
(Table 4, last row) — but as §4.4 reports, neither subword shape
alone nor frequency alone fits the per-canonical invented-mass
distribution in any single model. Full per-(model, word) audit
**[repo-supplement: `experiments/16_canonical_set_audit.py`,
`outputs/25c_20260521_092243.log`]**.

**H. Activation-patching forward-hook implementation note.** The v1
implementation of script 25a read the patched residual from
`out.hidden_states[layer]` (i.e., the captured hidden-states tuple
from `output_hidden_states=True`). On MPS with current-generation
HuggingFace `transformers` (Gemma 2 + OLMo 2), the forward hook's
return value correctly replaced the layer's effective output for
downstream computation (the next-token logits *do* shift under
patching), but the hidden-states tuple captured the pre-hook output.
Practical consequence in v1: probe-causality readings of `~0%` across
all conditions (the probe reads the pre-patch residual, which is
identical across BASELINE / PATCH_not / PATCH_and), while behavioural
KL shifts were *already correct in v1* because they read the
next-token logits which use the post-patch residual. The v2 fix
replaces the hidden-states read with a capture inside the same forward
hook — `_make_patch_capture_hook` directly appends the post-patch
slice at `[0, position, :]` into a Python list — guaranteeing the
probe sees exactly what subsequent layers see. v2 probe-causality is
100% at all three originally-tested cells; v2 behavioural KL is
byte-identical to v1. The bug therefore would not have invalidated
the §4.5 ΔKL results (the principal causal claims) had v1 been
published as-is, but the v1 probe-causality readings of ~0% across
all conditions would have read as "the patch never reaches the
residual," producing an incorrect interpretation of the activation-
patching protocol's effectiveness. We report both for transparency.

**I. The §4.2 dissociation cell across additional probe variants.**
v3-v4-v5-v6 M2-canonical and M2-arity bootstrap CIs at the OLMo
`N→F sente→close L 10` cell are tabulated in Table 3 of the main
text; the M2-arity / M2-canonical gap persists across all four
scopes. Replication of the dissociation with class-balanced and
scaled-feature probe variants **[repo-supplement:
`experiments/22d_probe_variants.py`]**.

**J. Cell-2 per-word Δ_specific bootstrap (§4.5, 25d).** For Gemma
`opera→opera L 4` (Cell 2), the per-word
`Δ_specific(axis) = ΔKL_targeted(axis) − ΔKL_random_mean(axis)` is
computed across the 16 invented words using the existing 25a log
data (the per-word `ΔKL_targeted` is logged per-word and per-axis;
the per-word `ΔKL_random_mean` is the model-wide aggregate
RANDOM_NORM mean over the same (word, stimulus) grid). Bootstrap
B = 500 over the 16 invented words gives `not` axis Δ_specific 95% CI
`[−0.001, +0.031]` (mean +0.016, 13/16 words positive) and `and` axis
95% CI `[−0.005, +0.020]` (mean +0.006, 11/16 words positive). Both
CIs include zero; the mean-ratio WEAK PASS signal at this cell does
not firm up under the per-word bootstrap, and Cell 2 is classified
AMBIG in Table 5. A per-(word, stim) RANDOM_NORM rerun (Phase 2)
would replace the aggregate offset with a per-(word, stim) baseline;
the current 25d statistic already shifts Cell 2 against the WEAK PASS
reading so the rerun is not headline-changing.
**[repo-supplement: `outputs/25d_20260521_132205.log`,
`experiments/25d_delta_specific_bootstrap.py`]**.
