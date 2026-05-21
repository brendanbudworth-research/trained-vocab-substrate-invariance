# Asymmetric Substrate-Invariance in OLMo 2: Variable and Operator Renaming Probe Different Morphospace Boundaries

**Status:** Phase 1 entry. Phase 0 complete, including an external-review-driven syntactic-confound stress test (script 16). The principal Phase 0 finding — *arity-region substrate-invariance at the operator-anchored position in OLMo 2* — is confirmed across two model scales (1B, 7B), four probe instruments (multi-op-rich-trained, single-op-rich-trained, neutral-metalinguistic-trained, functional-prefix-trained), and survives a prefix/infix syntactic-position dissociation. **Phase 1 entry adds a second model (Gemma 2 9B, scripts 17-20) and four new measurements (cross-condition probe transfer in script 18; cross-notation directional angle in script 19; canonical-transfer gating with bootstrap CIs in script 19b; gated invented-mass re-test in script 20). The refined headline: both models have within-notation arity-region attractors in both notations, but cross-notation behaviour differs sharply and the cross-notation finding is NOT strict arity-respecting transfer. Gemma 2 has cross-condition-transfer-compatible unary/modifier catchment basins at validated early layers (NEUTRAL L4 and FUNC-PFX L2), dominated by "necessarily" (a generic modal/adverbial modifier as well as a logical-unary canonical) and growing monotonically with depth in the N→F direction; these do not yet show *arity-respecting* assignment of invented operators. No tested OLMo 2 pairing at the operator-anchored position shows above-baseline bidirectional arity-respecting invented transfer. The negative-result headline was *softened* after the script 21 post-call anchor re-test, **partially upgraded** after script 22a bootstrap M2 + M2-arity introduction (the §3.7.9 cell upgrades to "demonstrated under M2-arity but not M2-canonical"), and **further extended** after the script 22b full anchor × layer sweep identifies three additional PASS-arity cells: Gemma 2 N→F sente→first L8 (M4b = 73.6%), Gemma 2 N→F sente→opera L4 (M4b = 72.8%, reinstating the Gemma transfer claim from scripts 17-18), and OLMo 2 F→N first→opera L7 (M4b = 65.6%, with both M2-canonical and M2-arity PASS — the first observed cell with simultaneous canonical-identity and arity-axis cross-notation transfer). The §3.7.9 cell remains unique-strongest by composite criterion. All four cells share a structural commonality: training at a post-call anchor (NEUT `sentence-final` or FUNC `first-arg`), never the operator-after position which produces lucky-default catchment basins (40 of the 160 cells in the sweep). Validation queued: expanded invented + canonical sets.** The Phase 1 claim is therefore *cross-model within-condition arity-region attractors are robust; cross-notation arity-respecting transfer is demonstrated at four cells across both models (one decisive single-cell finding in OLMo 2 N→F, plus three supporting cells with noisier per-word patterns), with the training anchor being a first-class methodological variable; the strict 5-class M2-canonical gate is satisfied at only one of the four cells (OLMo 2 F→N L7), making canonical-identity transfer a much rarer phenomenon than arity-axis transfer; replication with expanded stimuli is the principal remaining empirical task*.

## Abstract (draft)

The Platonic Representation Hypothesis (Huh et al., 2024) predicts that large neural networks converge on substrate-independent representations of structure. We test a sharp version of this claim — *substrate-invariance under renaming* — in OLMo 2 1B and 7B by measuring whether structurally-equivalent propositional-logic prompts produce aligned internal representations when their surface form is altered along two distinct axes: variable renaming (Greek-lowercase substitution for English letters) and operator renaming (invented Tier-2 BPE word substitution for `and`, `or`, `not`, `implies`, and `necessarily`). Using linear CKA across three principled pooling strategies and a linear operator-identity probe, we find a sharp and scale-robust asymmetry: variable substrate-invariance is essentially complete (probe accuracy 1.000 from layer 2 onward at both scales), while operator substrate-invariance is materially absent (probe accuracy 0.165 in 1B and 0.290 in 7B, against a 4-class chance baseline of 0.250). The asymmetry survives a 4× parameter scale-up and is therefore *role-bound*, not capacity-bound.

The operator-side failure has an **arity-region attractor mechanistic structure** that is robust across multiple independent probe instruments and syntactic notations, and is not derivable from layer-0 token-embedding geometry. The cleanest single measurement is a syntactic-confound stress test (script 16) where canonicals are placed in functional-prefix notation (`op(p, q)` for binaries, `op(p)` for unaries) with identical preceding context — all five canonicals in the same prefix-function-call syntactic role. The probe reaches 1.000 CV accuracy on canonicals; on invented Tier-2 BPE words substituted into the same notation, **all 250 invented-operator stimuli are classified as `not` (100% unary, 0% to any binary canonical, 0% even to the other unary `necessarily`)**. The result holds for binary-replacement invented words (bliq, dren, molex) placed in prefix-binary slots — the precise position where binary canonicals naturally appear in functional notation — decisively ruling out a prefix-vs-infix syntactic-position confound that the original infix-binary / prefix-unary template design could not distinguish from arity.

The unary-region attractor is **not inherited from layer-0 embeddings**. A direct embedding audit (script 14) shows that all five invented words have their highest layer-0 cosine similarity to `and` or `or` (the high-frequency canonical operators), with most invented words 4–8× closer to `and`/`or` than to `necessarily`. Mean-pool embedding similarity predicts the script-13 top-landing canonical for only 1 of 5 words; within-unary not-vs-necessarily sign-agreement is 2 of 5. The network actively *moves* invented-operator representations from the `and`/`or` region of embedding space into the unary `not`/`necessarily` region by layer 7 — the unary attractor is a genuine computational structure constructed during forward pass.

The mechanism has a four-channel decomposition: (H1) a structural default to the **unary-class region** at the operator-anchored position, with `not` and `necessarily` as the two probe-recoverable anchor points; confirmed at ~100% mass in the cleanest (neutral-train × neutral-test) measurement. (H2) no measurable tokenization-position effect — varying invented-operator subword count by 4× changes peak gaps by ≤ 4 percentage points (script 10). (H3) a word-specific bias operating in two regimes: cross-class escape (`bar` → `or` ~74% regardless of slot; script 11) when an invented word's embedding lies independently close to a non-unary canonical, and within-unary modulation (the molex → necessarily anomaly; script 15) which is constructed by intermediate layers rather than inherited from layer-0 embeddings. (H4) a *probe-instrument-dependent* template-context channel: a probe trained on lexically-rich templates can be biased toward the canonical its training templates "own", but the same residual-stream activations evaluated by a neutral-trained probe produce predictions dominated by the template's syntactic scaffolding (e.g., "If... then..." structure pulls toward `implies`) rather than the H4 effect of script 13. The H4 channel as originally characterised in script 13 is therefore reframed as a probe-training artifact rather than a deep property of the residual stream.

We argue that the morphospace boundary for substrate-invariance in OLMo 2 is structured *by operator arity at the operator-anchored position*: variable substrate-invariance is complete (probe accuracy 1.000); operator-arity substrate-invariance is complete and robust to probe instrument and syntactic notation (100% mass to unary in the cleanest measurement, 99.6% in the next, 94.6% in script 13's rich-natural-English condition); operator identity *within* arity is fragile and probe-instrument-dependent (the within-unary `not`-vs-`necessarily` split for the same invented word ranges from 0:98 to 100:0 across notations). The arity-attractor finding has direct implications for the Platonic Representation Hypothesis at the within-modality scale: the model encodes a compositional structural property (arity) robustly, while specific operator-identity within that compositional class is not abstracted away from canonical surface forms. We also report methodological findings on the failure modes of single-pooling pre-registration in representational similarity analysis, the importance of cross-probe-instrument reporting (single-op-rich vs multi-op-rich vs neutral-metalinguistic vs functional-prefix probes produce qualitatively different per-canonical prediction distributions on the same activations), the embedding-vs-residual-stream divergence (layer-0 embedding geometry does not predict layer-7 probe behaviour for invented words), and the prefix/infix syntactic-position confound that contaminates any operator-renaming probe based on natural-English propositional-logic templates alone.

**Phase 1 entry (cross-model, scripts 17–20).** Replication of the cleanest two Phase 0 probe instruments (script 15 NEUTRAL-metalinguistic and script 16 functional-prefix Condition 2) on Gemma 2 9B (`google/gemma-2-9b`, a deliberately different model: different lab, different training data, different architecture, different tokenizer, 42 layers vs OLMo 2 7B's 32, bf16-required) reproduces the within-notation arity-region attractor in both notations (Gemma 2 NEUTRAL peak unary mass 97.6% at layer 4; FUNC-PFX peak unary mass 100% at layers 2 and 16-17). A four-diagnostic probe-artifact battery (cross-condition probe transfer, held-out canonical generalisation, probe-free centroid geometry, last-subword embedding baseline) confirms the within-notation arity attractor is real in both models. **The principal new finding is the cross-condition probe transfer asymmetry.** The NEUTRAL-trained probe direction in Gemma 2 (validated structural by held-out canonical at 4× chance) places functional-prefix invented operators in the unary region at 100% mass; the same probe direction in OLMo 2 places functional-prefix invented operators at 0% unary mass (all 250 stimuli classified as `and`). The reverse-direction transfer also fails in OLMo 2 (0% unary, all to `implies`); in Gemma 2 the reverse direction is mixed: FUNC-PFX@L2 → NEUTRAL transfers at 86.8% unary mass with cross-canonical 5-class accuracy 0.756, which is a clean transfer; FUNC-PFX@L16 → NEUTRAL at 99.6% unary mass but with cross-canonical accuracy 0.564 (and 0.200 when cross-layer to NEUTRAL@L4), which is at or near chance for 5-class classification and downgrades the L16 result to a candidate late re-emergence requiring stronger validation. The reviewer-recommended canonical-transfer gate (cross-canonical 5-class accuracy ≥ 0.65 at the same train/test pairing) cleanly accepts NEUTRAL@L4 ↔ FUNC-PFX@L4 and FUNC-PFX@L2 → NEUTRAL, rejects FUNC-PFX@L8 → NEUTRAL (the artifact layer), and marks FUNC-PFX@L16 → NEUTRAL as ambiguous. The defensible cross-model interpretation is therefore: **Gemma 2 has cross-condition-transfer-compatible arity directions at validated early layers, while OLMo 2's tested arity directions do not transfer across notations at L7**. Both models have within-notation geometric centroid attractors at the operator-anchored position; OLMo 2's centroid attractor is in raw magnitude substantially stronger (NEUTRAL delta +0.0564, FUNC-PFX delta +0.0806 at L7) than Gemma 2's (+0.0093, +0.0068 at the validated focus layers). The Gemma 2 per-layer functional-prefix trajectory has plausibly two transferable arity stages (an early stage at L2 that cross-transfers cleanly to NEUTRAL, and a late stage at L16-17 with weaker transfer validation that should be treated as candidate) separated by intervening layers (L6-L12) whose within-condition probes pick up surface features and do not cross-transfer. This is qualitatively different from OLMo 2 7B's monotonic within-condition unary-mass plateau. The cross-model contrast is now measured along a six-component substrate-invariance battery (scripts 17–20): **M1** within-condition probe CV (baseline only); **M2** bidirectional canonical-transfer gate; **M3** cross-notation directional angle on the binary unary-vs-binary axis with bootstrap 95% CIs; **M4a** invented unary mass; **M4b** intended-arity agreement (does each invented word land in the canonical matching its intended arity?); **M4c** canonical catchment concentration (single-canonical collapse vs distributed). Gemma 2 L4 N→F is M2+M3-tight + M4a-high (80% mass) but M4b-FAIL (`bliq` intended-binary → "necessarily"; 4 of 5 invented → "necessarily") and M4c-collapsed — a generic-unary/modifier catchment basin rather than arity-respecting classification. OLMo 2 L7 fails M2 asymmetrically (1.000 / 0.212). The previously-unidentified OLMo 2 L10 satisfies M2 bidirectionally (0.800 / 0.688) but fails M3 (probe 74° wide) and M4a (mean 8.6%, *below* the 40% random-by-arity baseline; N→F predictions 100% "implies", actively binary-classified). **At no tested (model, layer) pairing does the per-invented-word predicted canonical track intended arity.** Bootstrap 95% CIs are 2–6° wide across all 19b measurements; the cross-model angle separation is statistically tight. The refined Phase 1 entry headline is therefore: **(a)** within-condition arity-region attractors are cross-model robust (Phase 0 finding survives at scale and across model families); **(b)** cross-notation transfer of a "necessarily"-dominated catchment basin exists in Gemma 2 at L4-L8 — suggestively monotone with depth across three tested points in the N→F direction — and is *not* arity-respecting; **(c)** no tested OLMo 2 pairing shows above-baseline bidirectional arity-respecting invented transfer, even at the L10 gate-passing layer; **(d)** cross-notation arity-respecting transfer at the operator-anchored position used in scripts 17-20 has not been demonstrated in any tested model at any tested layer. **Scripts 21-22 together demonstrate** that the result depends on both anchor position and the *type* of cross-notation transfer being tested. The full anchor × layer sweep (script 22b, 160 cells across both models) identifies **four cells satisfying the M2-arity-PASS + M4b-PASS + M4c-distributed + M4a-central + not-lucky-default conjunction**: (i) OLMo 2 N→F sente→close L10 (§3.7.9 cell, M4b = 90%, M2-arity = 1.000, M2-canonical = 0.616 AMBIG — the dissociation cell where the arity axis transfers but binary-canonical-identity does not, with within-arity or/implies → and confusions); (ii) Gemma 2 N→F sente→first L8 (M4b = 73.6%, M2-arity = 1.000); (iii) Gemma 2 N→F sente→opera L4 (M4b = 72.8%, M2-arity = 1.000, reinstating the Gemma cross-notation transfer claim from scripts 17-18 that was lost at the operator-after anchor); (iv) OLMo 2 F→N first→opera L7 (M4b = 65.6%, M2-canonical = 0.980 + M2-arity = 0.984 — the first observed cell with simultaneous canonical-identity AND arity-axis transfer, in the reverse direction, with the most distributed per-word predictions in the sweep). The §3.7.9 cell is confirmed unique-strongest by composite criterion. Structural commonality across all 4 PASS-arity cells: training at a post-call anchor (NEUT `sentence-final` or FUNC `first-arg`) is necessary; training at the operator-after position consistently produces lucky-default catchment basins (40 of 160 cells). The M2 gate has been split into M2-canonical (5-class) and M2-arity (binary-vs-unary, coarsened) in §3.7.5; the §3.7.9 candidate is classified as a confirmed *arity-respecting* transfer under M2-arity but a non-PASS *canonical-identity* transfer under M2-canonical; three additional cells extend the finding. **Bootstrap CIs (script 22a extension, §3.7.12) initially confirmed dual-PASS (M2-canonical AND M2-arity) at three new candidate cells, but the two-axis stimulus-expansion falsification (scripts 22c + 22d, §3.7.13 + §3.7.14) systematically retracts the cross-notation arity-respecting transfer claim for *novel* operators across all four originally-PASS-arity cells.** Script 22c expanded the invented set from 5 to 16 words (8 intended-binary + 8 intended-unary, up from 3+2 in the original set), retracting 2 of 4 PASS-arity cells (Gemma sente→first L8 and OLMo first→opera L7 dropped to M4b ≈ 0.56-0.57, below the 0.65 PASS threshold under bootstrap). Script 22d then expanded the canonical set from 5 to 10 (adding binary `xor`, `nand` and unary `possibly`, `always`, `negate`), retracting the remaining 2 survivors: the §3.7.9 OLMo sente→close L10 cell collapses entirely to M4b = 0.500 with 100% mass on `nand`, and the Gemma sente→opera L4 cell drops to M4b = 0.625 with 87.5% on `nand` + 12.5% on `negate` and lucky-default-flag firing. **The mechanism revealed by 22d is "default-to-rarest-canonical": the probe routes invented-operator activations to whichever canonical sits in the highest-entropy / lowest-training-prior region of the decision space. With 5 canonicals that target was {`and`, `necessarily`, `implies`}; with 10 canonicals adding near-zero-frequency `nand` / `negate`, the target shifts wholesale to them.** The previously-observed "arity-respecting" behaviour was coincidence between the 5-canonical default-target and the invented set's arity distribution — neither strict-logical-arity transfer nor generic-modifier-basin transfer; rather, a compressive low-confidence routing pattern that shifts target as the canonical set changes. **What IS robustly demonstrated across both models, multiple anchors, both directions, AND both stimulus-set expansions, is cross-notation transfer of *canonical-operator* identity and arity**: at the §3.7.9 OLMo cell and Gemma sente→opera L4 cell, M2-canonical reaches 0.812 (10-class, ~8× chance) and 1.000 respectively under canonical-set expansion; M2-arity = 1.000 in both. Substrate-invariance in current open LMs is therefore *operator-set-bound*: the model's logical-operator geometry transfers across notations for the canonical operators it was trained to recognize, but does not generalize to novel operators in an arity-respecting way. The post-§3.7.14 Phase 1 headline is "**cross-notation canonical-operator substrate-invariance is real and robust; cross-notation novel-operator substrate-invariance is not demonstrated and instead reduces to a default-to-rarest-canonical compression mechanism**". This is a sharper finding than either the previous "partial Platonic" or "full Platonic" framings; Section 4.1 develops the implications for the Platonic Representation Hypothesis. Per-anchor catchment basins differ within the same (model, layer) — Gemma 2 L4 has a "necessarily" basin at operator-after and a "not" basin at close-paren — indicating positional readout is a first-class methodological variable; same applies to NEUTRAL training-anchor choice. "Necessarily" is also a generic modal/adverbial modifier in natural language as well as a logical-unary canonical, so the Gemma 2 catchment basin may be acting at the generic-modifier level rather than the strict logical-unary level — disambiguating this requires additional unary canonicals with different lexical/grammatical profiles (`possibly`, `always`, `negate`, `is_false`) in a follow-up experiment.

**Phase 2 (cross-family replication, script 23).** Pythia 6.9B-deduped (EleutherAI, GPT-NeoX architecture with RoPE, trained on the deduplicated Pile) is the third model family tested. Single-cache v5-expanded-canonical extraction (~6 min at 276 tok/s MPS fp16) plus full anchor × layer sweep at three nested scopes (v3: 5+5, v4: 5+16, v5: 10+16) reproduces the operator-set-bound finding on all three Phase 2 predictions: **(P1)** M2-canonical PASS at 31/80 v5 cells (best 1.000, 10× chance on 10-class) — canonical-operator substrate-invariance replicates strongly across the three-model suite; **(P2)** 3 v3 PASS-arity candidates (more than OLMo's 1 or Gemma's 1), all in N→F direction with `operator-after → close-paren` anchor pair across three depths (L4, L7, L16); 2 of 3 survive v4 invented-set expansion (the most v4-robust set of any model) but all 3 retract at v5 canonical-set expansion — the v4→v5 retraction signature is cross-family stable; **(P3)** v5 default-to-rarest-canonical targets 70.0% of invented mass to the three multi-subword NEW canonicals (`nand` 27.9%, `xor` 22.3%, `negate` 19.8%) rather than collapsing to a single canonical as OLMo did. The cross-family synthesis: three model families, three training corpora (Pile / Dolma / Google proprietary), three architectures (GPT-NeoX / modified Llama / soft-capped Gemma), three tokenizers, the same two-part operator-set-bound finding. The mechanism is also cross-family stable in direction (compression toward low-frequency canonicals) but model-specific in target (single-canonical collapse in OLMo, two-canonical split in Gemma, three-canonical distribution in Pythia, with multi-subword tokenization correlating with attraction strength). All three Pythia candidates converge on `close-paren` as the test anchor (post-call position in functional notation), consistent with OLMo's §3.7.9 cell and Gemma's near-§3.7.13 cells in being post-call-test anchored. **The Phase 2 verdict is that the operator-set-bound substrate-invariance finding is a property of mid-scale (6.9-9B) open base language models with current-generation tokenizers and architectures, not an OLMo-or-Gemma-specific quirk.**

**Phase 2 follow-up: embedding-similarity probe rejects the contextual-semantic-neighborhood single-factor mechanism (script 25b).** Cosine similarity between mean invented-word activations and mean canonical activations at the focus layer was tested as the natural operationalisation of §3.7.16's tentatively-named third factor ("contextual semantic neighborhood"). Across 80 v6 sweep cells per model and three model families, identity-level agreement between the cosine argmax and the probe's per-word top canonical is at 11.6% / 26.6% / 24.0% (Gemma / OLMo / Pythia, bootstrap 95% CIs all below 30%); arity-conditioned identity agreement is at 11.4% / 21.5% / 19.0% — well below the §6 pre-specified `≥ 60%` "mechanism gap closed" threshold and at-or-near the within-arity chance baseline. At distributed cells (M4c < 0.7, the methodologically interesting regime), agreement drops to 11-16% across all three models, barely above the 6.7% chance baseline. The cleanest failure-mode signature: cosine argmax collapses every invented word to `nand` (binary) or `negate` (unary-intended under arity-conditioning), most likely a canonical-magnitude-and-idiosyncrasy effect rather than semantic structure. **The residual third factor in the default mechanism is therefore probe-decision-boundary geometry, not raw cosine similarity** — the LogisticRegression boundary captures per-word residual-stream structure that mean-pooling wipes out. Two additional findings: (a) L 0 embedding-layer cosine agreement is at floor (0-1%) in all three models, re-confirming the script 14 H1-construction-during-forward-pass finding under v6 across three model families; (b) mid-layer peaks at L 4 (Gemma) and L 10 (OLMo, Pythia) are consistent with cross-notation arity-region structure being *constructed* at mid-depth and *decoded into operator/lexical identities* at late layers in a way orthogonal to canonical mean activations. See §3.7.18.

**Phase 2 follow-up: causal grounding of Fact 1 and adjudication of the Gemma v6 emergence (script 25a).** Activation-patching at three target cells — Gemma 2 9B L2 close-paren under two source anchors and the OLMo 2 7B §3.7.9 Fact-1 anchor — using the mean NEUTRAL canonical activation as the patch source produces three distinct verdicts. **Gemma 2 9B `opera→close L 2` is causally arity-respecting**: probe-causality 100% / 100%, behavioural ΔKL = +0.048 / +0.038 on `not` / `and` reference axes (15-of-16 and 16-of-16 invented words positive), with arity-flip 8/0/8 in both directions (ΔKL = +0.033 for intended-unary patched with `and`; +0.061 for intended-binary patched with `not`) — a single tightly-scoped, causally validated model-specific exception to operator-set-bound in Gemma 2 9B. **Gemma 2 9B `sente→close L 2` is probe-only**: identical 100% / 100% probe-causality at the same layer and target anchor, but flat-or-negative behavioural ΔKL (random-norm control exceeds the targeted patch) — confirming the §3.7.16 methodological-caveat reading for this second cell. **OLMo 2 7B `sente→close L 10` — the project-flagship Fact-1 anchor — is probe-readable but not causally load-bearing**: probe-causality 100% / 100% (the patch reaches the residual), but ΔKL = −0.012 / −0.017 with random-norm strongly positive at +0.019 / +0.012, and arity-flip 1/7/8 and 3/5/8 (both directions fail). Three findings emerge: (i) Fact 1's geometric cross-notation transfer is not uniformly causally load-bearing — the §3.7.9 OLMo cell is geometric/probe-readable but causally inert under NEUTRAL-sentence-final patching, refining the original "linear probes only" limit to a concrete answer; (ii) the Gemma v6 emergent PASS-arity finding splits 1:1 (one cell causally validated, one cell confirmed as M4b-granularity-only), so operator-set-bound substrate-invariance holds across three model families at all tested cells in OLMo 2 7B and Pythia 6.9B-d under both linear-probe and causal-patching tests, with exactly one tightly-scoped causally-validated exception in Gemma 2 9B at L2 close-paren when sourced from NEUTRAL operator-after; (iii) source anchor is a first-class causal variable independent of probe-causality — same target, same layer, same probe reading, opposite causal verdicts at the two Gemma cells. Probe-causality is necessary but not sufficient for causal load-bearingness; future probe-based substrate-invariance work should include same-target / different-source causal tests as part of the standard battery. See §3.7.17.

**Phase 2 follow-up: pre-registered v6 canonical-set expansion (script 24).** Following external review of the §3.7.14 / §3.7.15 default-to-rarest framing, a frozen pre-registration document (`experiments/preregistration_v6.md`) was written *before* any v6 cache extraction, specifying three competing single-axis predictions for which canonicals should attract novel-operator mass under a 15-class readout (P_FREQ: low training-corpus frequency; P_SUBWORD: multi-subword tokenization; P_INTERACTION: combined with mid-frequency controls). The canonical set was expanded from 10 to 15 by adding `nor` (1pc, low-frequency), `iff` (target multi-pc/very-low; caught OOD by the audit gate — actual 1pc in all three tokenizers), `unless` (1pc, MF control), `definitely` (1pc, MF control), and `unprovably` (multi-pc, very-low-frequency). Three outcomes follow. **(1) Fact 1 strengthens** to M2-canonical = 1.000 with bootstrap 95% CI [1.000, 1.000] at the same N→F opera→opera L4 cell in all three model families under 15-class readout — cross-notation canonical-operator transfer is now ceiling-confirmed under three independent expansions (v4 invented-set, v5 10-class readout, v6 15-class readout including pre-registered novel canonicals). **(2) P_RETRACT splits 2:1.** OLMo 2 7B and Pythia 6.9B-d show zero PASS-arity cells at v6 (P_RETRACT holds). Gemma 2 9B shows two emergent v6 PASS-arity cells at N→F close-paren L 2 (M4b = 82.2% and 66.2%) that were *not* survivors from earlier scopes — M4b at these cells trajectories 60% → 56% → 50% → 82% across v3 → v4 → v5 → v6 *without any change to the underlying activations*. M2-arity at these cells is constant 0.78-1.00 across all four scopes; only M4b changes. This is the cleanest in-data demonstration that **M4b is a threshold test sensitive to canonical-readout granularity**, not a Boolean test of arity-respecting structure. The Gemma v6 emergence is reported as a methodological caveat on M4b's granularity-sensitivity, with causal-patching follow-up (script 25a) flagged as the highest-priority remaining experiment. **(3) All three single-axis predictions fail in all three models.** P_FREQ aggregates fall at 30.6% / 8.0% / 22.9% (below the 35% threshold); P_SUBWORD multi-pc aggregates at 14.8% / 44.2% / 63.4% (below 70%); MF controls slip in Gemma (`definitely` = 7.5%) and Pythia (`definitely` = 5.5%). The cleanest single anomaly is OLMo 2 routing 15.9% of v6 invented mass to `implies` — a v3/v4/v5 carryover canonical (1pc, mid-frequency) that none of the three predictions targeted. The §3.7.14 framing of "default to the rarest canonical" is empirically too coarse; the more accurate framing is "the model routes novel-operator activations to a low-prior canonical in a model-specific way that depends on at least three factors: training-corpus frequency, subword shape, and per-invented-word semantic neighborhood". The remaining mechanism gap (which low-prior canonical wins for each invented word, given that single-axis reductions fail) is the target of the script 25b embedding-similarity follow-up. **The pre-registration was usefully constraining**: the audit gate caught an unforeseeable tokenization failure (`iff` 1pc in all three tokenizers despite our 2-3pc design target), the multi-scope analysis is now disciplined by a frozen specification, and the falsification verdicts ("NONE supported") are clean and unambiguous, demonstrating that pre-registration discipline produces informative outcomes regardless of whether the data confirm or reject the pre-registered hypotheses.

**Methodological contribution.** Alongside the empirical operator-set-bound finding, the project's principal methodological contribution is the **lucky-default detector**: `min(per_word_top_pct) ≥ 0.95` as a required pre-registered measurement alongside the aggregated M4c canonical-concentration metric. The detector catches a specific false-positive pattern — `4-of-N-at-ceiling-plus-1-escape` — where every invented word's predictions are deterministically routed to a single canonical, yielding a Herfindahl-aggregated `M4c` that looks "distributed" while the per-word evidence is concentrated. This pattern produces inflated apparent intended-arity agreement (`M4b`) when the model's default canonical happens to align with the test set's arity distribution. The detector reclassified 4 of 8 originally-flagged PASS-arity cells as lucky-default false positives in §3.7.11 (and would have caught the rest under the §3.7.13 / §3.7.14 stimulus-expansion battery). The refinement is directly transferable to any probe-based substrate-invariance, compositional-generalization, or in-context-binding study that reports per-word intended-class agreement; see §4.4 for the full methodological treatment. This generalizes the methodological insight beyond our specific substrate-invariance setting.

## 1. Background and motivation

The Platonic Representation Hypothesis (Huh, Isola, et al., 2024) holds that sufficiently large neural networks trained on natural data converge on representations of a shared underlying structure that exists independently of the substrate (text, image, modality) used to express it. If true, this predicts that surface-form changes that preserve structure should produce aligned internal representations — *substrate-invariance*.

The hypothesis admits a particularly clean operationalization for language models: take a structurally-equivalent set of prompts, rewrite their surface form along controlled axes (rename variables, rename operators, distort the alphabet), and measure whether the internal representations remain aligned. A positive result is evidence the model has abstracted the structure; a negative result is evidence that the model's representation is bound to the lexical surface.

The literature on the Platonic Representation Hypothesis has predominantly examined *cross-modal* convergence (e.g., text vs vision encoders learning aligned representations of the same concept). The cross-surface question within a single language model is comparatively underexplored — and is the natural intermediate test case: the same model, the same modality, but a controlled perturbation of the substrate.

This work probes that intermediate case in OLMo 2 1B, an open-weights, open-training-data model from AI2 selected for its full audit-ability of training corpus. We use propositional-logic prompts as the structural primitive (well-defined, compositional, with a small operator alphabet) and measure substrate-invariance under two controlled axes of surface perturbation.

## 2. Methods

### 2.1 Models and infrastructure

All experiments use two scales of OLMo 2 (AI2): the 1B variant (`allenai/OLMo-2-0425-1B`, 16 transformer layers, 2048 hidden dim) and the 7B variant (`allenai/OLMo-2-1124-7B`, 32 transformer layers, 4096 hidden dim). Both share the same 100,278-token BPE vocabulary and are loaded in fp16. Activations are extracted via the HuggingFace `transformers` library's `output_hidden_states=True` flag, which yields the residual stream at each layer including the embedding layer (17 layers reported for 1B, 33 for 7B). All extraction is single-prompt to avoid padding artifacts. Compute is on an Apple M4 with 48 GB unified memory; the MPS backend achieved 527 tok/s in batched throughput tests without CPU fallbacks on 1B, and 7B forward passes complete at acceptable rates (~91s for 800 prompts on script 09).

### 2.2 Stimuli

We construct stimuli via a small set of propositional-logic templates parameterized by two variables `{p}` and `{q}`. Templates use English-language framing with logical operators appearing 1–3 times each: e.g., `"If {p} and {q} are both true, then {p} and {q} is true."`

For each canonical prompt, we generate five surface-form variants:

| Condition | Description | Variables | Operators |
|---|---|---|---|
| A | Canonical | p, q, r, s | and, or, not, implies |
| B | Variable-renamed | α, β, γ, δ | unchanged |
| B' | Operator-renamed | unchanged | bliq, dren, vusp, molex |
| B'' | Both renamed | α, β, γ, δ | bliq, dren, vusp, molex |
| C | Token-shuffled (scrambled negative control) | shuffled | shuffled |
| D | Unrelated natural language (calibration baseline) | n/a | n/a |

Greek lowercase variables are Tier-1 tokens in the OLMo 2 vocabulary (single-token, low Dolma-corpus frequency, semantically inert). Invented operator words are Tier-2 (consistent multi-token tokenization, no prior semantic association). All operator words tokenize as exactly 2 subwords: `bliq` → `[' bli', 'q']`, `dren` → `[' d', 'ren']`, `vusp` → `[' v', 'usp']`, `molex` → `[' mole', 'x']`.

For probe experiments we use N=200 stimuli with 50 per operator class, balanced across templates.

### 2.3 Tokenization screening

Before designing substrate-invariance stimuli, we screened candidate "alien" symbol sets against the OLMo 2 tokenizer to identify which symbols survive as single tokens (Tier 1, methodologically cleanest) versus fragment into byte-fallback subwords (Tier 2/3, requiring anchor-aggregation methodology).

Of 70 candidate symbols across 11 categories, only 23 (33%) were Tier 1. Notably:

- Greek lowercase: 11/12 Tier 1 (only `ζ` fragments)
- Standard logic operators (∧ ∨ ¬ → ↔ ⊕ ⊻): only `¬` and `→` are Tier 1; all others byte-fragment
- Linear B, Tifinagh, mathematical script, fraktur, alchemical, and Unicode Private Use Area glyphs: **all** byte-fragment

This refutes the hypothesis (held implicitly in early planning) that exotic Unicode scripts would provide a large Tier-1 reservoir of single-token alien glyphs. The substrate-invariance probe must therefore use invented Tier-2 words rather than alien single-glyph operators.

### 2.4 Representational analysis: three pooling strategies

Representations are aggregated per-prompt via three strategies, reported side-by-side. The choice of pooling strategy materially changes the substantive conclusion (see §3.2); reporting only one is unsafe.

1. **Last-token pooling**: residual stream at the final input position (typically `"."`). Sensitive to compositional structure that propagates to the sentence end; biased toward representations of *near-final* tokens.
2. **Mean-pool**: arithmetic mean of residual streams across all token positions. Sensitive to token-bag composition; dominated by token overlap at shallow layers.
3. **Operator-anchored pooling**: residual stream at the token position immediately following the last subword of the *first* operator occurrence in the prompt. Position is determined by tokenizer-aware string matching. This is the principled measurement for operator substrate-invariance specifically — the position must integrate the operator's effect on subsequent processing.

For control conditions C (scrambled) and D (unrelated) where no operator position is defined, the operator-anchored fallback is the last token position.

### 2.5 Similarity and identity metrics

We report two complementary metrics:

**Linear CKA** (Kornblith et al., 2019): a centered, normalized kernel-based similarity between two activation matrices, bounded in [0, 1]. We report CKA between each condition's representations and the canonical condition's representations at each layer. The substrate-invariance gap is defined as `CKA(A, *) − CKA(A, scrambled)`.

**Linear probe accuracy** (logistic regression, lbfgs solver, standardized features): a probe is trained on canonical-condition activations to classify operator identity (4 classes: and/or/not/implies). For condition A, accuracy is reported as the mean of 5-fold stratified CV. For test conditions, the probe trained on all 200 A activations is evaluated on each held-out condition's activations. Probe accuracy is more directly interpretable than CKA: it answers "does this position carry information sufficient to identify the operator?"

### 2.6 Pre-registered hypotheses

We pre-registered three predictions for operator substrate-invariance prior to running script 09:

1. **Strong Platonic**: probe accuracy on B' ≈ probe accuracy on B ≈ probe accuracy on A (≥0.95) at middle-to-late layers. Model treats invented operators as fully equivalent to canonical via in-context binding.
2. **Partial recovery**: probe accuracy on B' rises with depth, reaching 0.6–0.8 in late layers but never matching B.
3. **Syntactic-only**: probe accuracy on B' stays near chance (0.25) at all layers. Operator-anchored position has compositional structure but no operator-specific semantics.

A fourth outcome — *systematically below chance* — was not pre-registered; this is the outcome we observed (§3.4).

## 3. Results

### 3.1 Variable substrate-invariance: complete at both scales

Under operator-anchored pooling, variable renaming (condition B) preserves nearly complete representational alignment with canonical (A) at every transformer layer at both 1B and 7B scales:

| Scale | Layer | CKA(A, B) | Probe acc(B) |
|---|---|---|---|
| 1B | 1 | 0.946 | 0.895 |
| 1B | 4 | 0.973 | 1.000 |
| 1B | 8 | 0.983 | 1.000 |
| 1B | 16 | 0.966 | 1.000 |
| 7B | 1 | 0.932 | 0.880 |
| 7B | 7 | 0.987 | 1.000 |
| 7B | 16 | 0.970 | 1.000 |
| 7B | 32 | 0.951 | 0.970 |

A linear probe trained on canonical activations achieves perfect (or near-perfect) classification of operator identity in variable-renamed prompts from layer 2 onward at both scales. CKA stays in the 0.93–0.99 band throughout the network in both models. This is the *positive control* result: substrate-invariance is detectable when it is present, and Greek-lowercase Tier-1 variables are essentially interchangeable with English letters in OLMo 2's representations regardless of scale.

### 3.2 Methodological finding: pooling choice changes the conclusion

An early-iteration version of this experiment (script 07) reported CKA(A, B') ≈ 0.999 at layer 1 and ≈ 1.000 at layers 2–15 under last-token pooling — apparently strong substrate-invariance for operator renaming. The corresponding B-vs-B' "operator cost" was essentially zero (`+0.001`) across all layers.

This result is an artifact of last-token pooling combined with template structure. Operators appear in the middle of the prompt; the last token (`"."`) has little reason to attend back to operator positions through the network's residual stream geometry. Last-token pooling therefore systematically under-weights the influence of operator changes that occur far from the sentence end.

Running the same conditions under three pooling strategies simultaneously (script 08) exposed the artifact and revealed a different picture:

| Pooling strategy | CKA(A, B') at layer 10 | B-vs-B' gap at layer 16 |
|---|---|---|
| Last-token | 0.999 | +0.062 |
| Mean-pool | 0.964 | +0.093 |
| **Operator-anchored** | **0.752** | **+0.259** |

The diagnostic gap is two orders of magnitude larger under the principled (operator-anchored) measurement than under last-token. **Single-pooling pre-registration is unsafe in representational analysis**; the methodology must triangulate across at least two pooling strategies and demand consistent signal before claiming a representational result.

### 3.3 Operator-anchored CKA shows a depth-resolved U-shape at both scales

CKA(A, B') across the network depth is non-monotonic under operator-anchored pooling, with the same qualitative shape — shallow plateau, mid-network dip, partial recovery, final-layer drop — at both 1B and 7B scales.

**OLMo 2 1B (17 layers):**

| Layer | 0 | 2 | 4 | 6 | 8 | **10** | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| CKA(A, B') | 0.981 | 0.979 | 0.895 | 0.822 | 0.806 | **0.752** | 0.839 | 0.882 | 0.707 |

**OLMo 2 7B (33 layers):**

| Layer | 0 | 2 | 4 | 7 | **9** | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CKA(A, B') | 0.978 | 0.997 | 0.932 | 0.831 | **0.793** | 0.807 | 0.794 | 0.839 | 0.875 | 0.858 | 0.811 |

The minimum sits at layer 10 in 1B (0.752, ~63% of network depth) and at layer 9 in 7B (0.793, ~28% of network depth). The dip is at a *proportionally earlier* layer at 7B — suggesting the semantic-integration computation that distinguishes canonical from invented operators happens earlier when more depth is available, freeing later layers for a longer "abstract-to-binary-infix-operator" recovery phase. Variable-renaming CKA(A, B) under the same pooling stays in the 0.93–0.99 band over all layers at both scales.

A plausible computational story: at shallow layers, the post-operator position is dominated by the embedding of the token at that position (which is identical across canonical and operator-renamed conditions, since the next token is always a variable). In middle layers, the model computes the semantic effect of the preceding operator and writes it into the residual stream — and `and` carries a known semantic load that `bliq` does not. Late layers show partial recovery, consistent with the model abstracting to a more generic "binary infix operator" category. The final-layer divergence reflects commitment to specific next-token predictions, which are operator-driven.

### 3.4 Operator substrate-invariance is materially absent at both scales

The linear operator-identity probe trained on canonical activations classifies the operator with perfect accuracy on canonical (A) and variable-renamed (B) prompts from layer 1 onward at both scales. On operator-renamed prompts (B'), it achieves materially lower accuracy with a clear depth-resolved structure.

**OLMo 2 1B (chance = 0.250):**

| Layer | A (CV) | B | **B'** | B'' | gap (A − B') |
|---|---|---|---|---|---|
| 0 | 0.360 | 0.250 | 0.410 | 0.250 | −0.050 |
| 1 | 1.000 | 0.895 | 0.535 | 0.315 | +0.465 |
| 3 | 1.000 | 0.925 | **0.265** | 0.240 | +0.735 |
| **4** | 1.000 | 1.000 | **0.165** | 0.190 | **+0.835** |
| 5 | 1.000 | 0.980 | 0.315 | 0.410 | +0.685 |
| 10 | 1.000 | 1.000 | 0.365 | 0.445 | +0.635 |
| 16 | 0.980 | 1.000 | 0.210 | 0.355 | +0.770 |

**OLMo 2 7B (chance = 0.250):**

| Layer | A (CV) | B | **B'** | B'' | gap (A − B') |
|---|---|---|---|---|---|
| 0 | 0.360 | 0.250 | 0.410 | 0.250 | −0.050 |
| 1 | 1.000 | 0.880 | 0.300 | 0.260 | +0.700 |
| 4 | 1.000 | 1.000 | 0.335 | 0.250 | +0.665 |
| **7** | 1.000 | 1.000 | **0.290** | 0.250 | **+0.710** |
| 12 | 1.000 | 1.000 | 0.430 | 0.490 | +0.570 |
| 17 | 1.000 | 1.000 | 0.460 | **0.510** | +0.540 |
| **18** | 1.000 | 1.000 | **0.520** | 0.490 | +0.480 |
| 24 | 0.990 | 1.000 | 0.365 | 0.385 | +0.625 |
| 32 | 0.990 | 0.970 | 0.345 | 0.170 | +0.645 |

**Critical comparison:**

| Quantity | 1B | 7B |
|---|---|---|
| Min B' probe accuracy | 0.165 at layer 4 | 0.290 at layer 7 |
| Below-chance pathology | **Present** (acc < 0.250) | **Absent** (acc ≥ 0.290 throughout) |
| Peak B' probe accuracy | ~0.42 at layer 11 | **0.520 at layer 18** |
| Peak A-B' gap | **0.835** at layer 4 | **0.710** at layer 7 |
| Final-layer gap | 0.770 | 0.645 |

The 4× scale-up from 1B to 7B produces only modest improvement in operator substrate-invariance: peak B' probe accuracy rises from 0.42 to 0.52, and the worst-layer gap shrinks from 0.835 to 0.710. **Variable substrate-invariance was already at the ceiling (1.000) at 1B — there is no headroom for it to improve.** The asymmetry between the two roles persists at scale, with similar magnitude.

A notable finding at 7B: the below-chance pathology observed at 1B does *not* replicate. The minimum B' probe accuracy at 7B is 0.290, slightly above the chance baseline of 0.250. The 7B's failure mode is a graceful default (see §3.4.1), not active miscalibration. Whether to read this as "scaling produced a more disciplined failure" or "the 1B pathology was small-model numerical instability" is unclear from these data alone, but the qualitative shift is real.

### 3.4.1 Confusion-matrix mechanism: uniform default to `not`

Confusion-matrix analysis at 7B layer 7 (the peak-gap layer) reveals a striking single-column-dominant failure mode. The probe trained on canonical operators classifies *almost every invented-operator prompt* as `not`, regardless of which canonical operator the invented word actually replaces.

**B' (operator-renamed) at 7B layer 7:**

|              | predicted: and | predicted: or | predicted: not | predicted: implies | row total |
|--------------|---:|---:|---:|---:|---:|
| true: and (bliq)    | 0 | 5 | **44** | 1 | 50 |
| true: or  (dren)    | 0 | 8 | **42** | 0 | 50 |
| true: not (vusp)    | 0 | 0 | **50** | 0 | 50 |
| true: implies (molex) | 0 | 0 | **50** | 0 | 50 |

Overall accuracy: 58/200 = 0.290. **186 of 200 predictions are `not`.** The 50 correct `not` classifications occur trivially because `vusp` replaces `not` in our mapping; absent that coincidence, the probe would have achieved 8/200 = 0.04 — far below chance.

**B'' (both renamed) at 7B layer 7:**

|              | predicted: and | predicted: or | predicted: not | predicted: implies | row total |
|--------------|---:|---:|---:|---:|---:|
| true: and    | 0 | 0 | **50** | 0 | 50 |
| true: or     | 0 | 0 | **50** | 0 | 50 |
| true: not    | 0 | 0 | **50** | 0 | 50 |
| true: implies | 0 | 0 | **50** | 0 | 50 |

Overall accuracy: 50/200 = 0.250 (exactly chance). **200 of 200 predictions are `not`.**

This is the *uniform-default-mapping* hypothesis confirmed — but with a different default than originally predicted. We expected the model would fall back to `and` (the highest-frequency canonical operator in training). Instead it falls back to `not`, the *only unary operator* in our canonical set.

Two candidate mechanisms remain to be distinguished:

1. **Structural defaulting.** `not` is the only unary operator in our canonical set. The post-operator position has structurally different properties for unary vs binary operators (the variable that follows `not` is the operand itself; the variable that follows binary operators is the second operand). When the model cannot resolve operator semantics for an invented word, it may default to the "unary operand" representation as a syntactic fallback.
2. **Tokenization-position effect.** Invented operators are 2 BPE subwords; canonical operators are 1. The "right after the last subword" position for a 2-token operator may sit one token further into the sentence than the corresponding canonical position. That geometric shift may happen to land closest to the canonical `not`-position cluster.

The two hypotheses make distinct predictions for follow-up experiments: hypothesis 1 predicts that adding a second canonical unary operator (e.g., "the square of") should distribute the default mapping between unary classes; hypothesis 2 predicts that varying invented-operator subword length (1, 2, 3, 4 subwords) should shift the default. Both are tractable Phase 0/1 experiments.

By layer 16 (mid-network), the failure pattern relaxes somewhat — 28 of 50 `bliq` instances are correctly classified as `and`, though `dren` and `molex` remain misclassified as `not`. Late-network behavior under operator renaming is therefore heterogeneous: the model partially recovers operator identity for *some* invented words while keeping others collapsed to the unary default.

### 3.5 The notation coherence effect (B'' > B'): partial replication at scale

At 1B, jointly renaming both variables *and* operators (B'') consistently outperformed renaming operators alone (B'). The effect was on the order of 15 percentage points across multiple layers and across both CKA and probe measurements:

| Measurement (1B) | B' | B'' | Δ |
|---|---|---|---|
| CKA(A, *) at layer 10 (op-anchored) | 0.752 | 0.894 | +0.142 |
| Probe accuracy at layer 11 | 0.355 | 0.505 | +0.150 |
| Probe accuracy at layer 16 | 0.210 | 0.355 | +0.145 |

At 7B, the pattern does **not** replicate cleanly. The B''-vs-B' relationship is depth-dependent and qualitatively different across the network:

| Measurement (7B) | B' | B'' | Δ |
|---|---|---|---|
| Probe accuracy at layer 7 (peak gap) | 0.290 | 0.250 | −0.040 |
| Probe accuracy at layer 12 | 0.430 | 0.490 | +0.060 |
| Probe accuracy at layer 17 | 0.460 | 0.510 | +0.050 |
| Probe accuracy at layer 24 | 0.365 | 0.385 | +0.020 |
| Probe accuracy at layer 32 | 0.345 | **0.170** | **−0.175** |

At 7B:
- Early layers (1–9): B'' is *worse* than B' (e.g., layer 7: B' = 0.290, B'' = 0.250 — exactly chance).
- Middle layers (10–20): B'' slightly exceeds B' by about 5 percentage points — much smaller than the ~15-point effect at 1B.
- Final layer (32): B'' degrades sharply to 0.170 (all 200 predictions → `not`) while B' remains at 0.345.

The notation-coherence hypothesis from the 1B analysis therefore weakens substantially. At 7B, the joint-renaming condition does not consistently recover more substrate-invariance than operator-renaming alone, and at the final layer it actively performs worse — the model commits *more strongly* to the `not` default when both axes are alien.

This is worth flagging as a non-replication. The 1B-only effect should be treated as a candidate finding that requires further investigation rather than a confirmed phenomenon. Plausible interpretations:

- The 1B effect was a small-model artifact that disappears with scale.
- The effect is real but depth-dependent in a way that 1B's 16 layers couldn't fully express; 7B's 32 layers reveal the full curve, which is more complex than a uniform B'' > B' advantage.
- The effect interacts with the failure-mode mechanism — at 7B's final layer the `not` default dominates B'' completely; at 1B's final layer no such single-default existed (the 1B confusion matrix was not measured).

The non-replication is itself a useful finding: it indicates that *some* substrate-invariance effects observed at 1B may not generalize to larger scales, which raises the bar for any 1B-only claim in the eventual paper.

### 3.6 Asymmetric morphospace boundary: the headline finding

Combining §3.1, §3.3, §3.4, and §3.4.1–5, substrate-invariance in OLMo 2 is *not a uniform property* — and the operator-side failure has a four-channel mechanistic structure that fully accounts for the observed probe behaviour.

**The asymmetry survives a 4× parameter scale-up from 1B to 7B:**

| Syntactic role | Surface change | Substrate-invariance at 1B | Substrate-invariance at 7B | Scale-robust? |
|---|---|---|---|---|
| Variables | Greek-lowercase Tier-1 | Complete (probe 1.0, CKA 0.95+) | Complete (probe 1.0, CKA 0.95+) | Yes |
| Operators | Tier-2 invented words | Effectively absent (probe 0.165 < chance) | Effectively absent (probe 0.290, uniform default to `not`) | Yes |

The boundary is structured, scale-robust, and admits direct measurement. The 4× scale-up produces only modest quantitative improvement in operator substrate-invariance (peak B' accuracy 0.42 → 0.52, peak gap 0.835 → 0.710) and does not bridge the qualitative asymmetry between the two roles.

**The operator-side failure is decomposed into four mechanisms** (§3.4.5): a structural default to `not` specifically (H1, dominant), a rejected tokenization-position effect (H2), a word-specific embedding-similarity escape channel (H3), and a template-wide lexical-context pull channel (H4). The four-channel framework is the principal mechanistic contribution of Phase 0 and provides a quantitative model of what the model is doing when it appears to "fail" at operator substrate-invariance.

This is the primary thesis-relevant finding of Phase 0: the morphospace of substrate-invariance in OLMo 2 is *role-bound* (variables vs operators are fundamentally different) and *mechanistically structured* (the operator-side failure is not random; it has identifiable attractors and pull channels). Variables and operators occupy fundamentally different positions in the model's representational geometry — placeholder roles are interchangeable across surface forms, content-bearing roles are not, and within content-bearing roles a specific canonical (`not`) serves as the default attractor for unresolvable inputs.

## 4. Discussion

### 4.1 What the asymmetry implies for the Platonic Representation Hypothesis

#### 4.1.1 Phase 0 precursor: substrate-invariance is role-specific

The Platonic Representation Hypothesis (PRH), in its strongest form, predicts uniform substrate-invariance across structurally-equivalent perturbations of the surface form. The Phase 0 result on OLMo 2 (1B and 7B) was that this strong form does not hold: *variables* show near-ceiling substrate-invariance under renaming, *operators* show little to none under the same instrument. The asymmetry persisted across a 7× scale-up and the failure mode (uniform default to `not` for invented operators) sharpened with scale rather than dissolved. The strong PRH was rejected at this scale; a weaker role-specific PRH (variables substrate-invariant, operators not) was the best fit to the Phase 0 data.

The Phase 0 finding was clean but incomplete. It told us *that* operators fail substrate-invariance under our renaming instrument; it did not tell us *what kind of substrate-invariance the model does have for operators*. That is the question Phase 1 was designed to answer, and the answer turns out to be substantially more interesting than the Phase 0 framing anticipated.

#### 4.1.2 The Phase 1 finding in plain English: operator-set-bound substrate-invariance

Phase 1 added a cross-notation axis (NEUTRAL metalinguistic templates vs FUNC-PFX function-call templates) and an explicit canonical-vs-invented operator distinction. After the full M1-M4 + bootstrap + sweep + invented-expansion + canonical-expansion methodology stack (scripts 17-22d on OLMo 2 7B and Gemma 2 9B, replicated on Pythia 6.9B-deduped in script 23, §3.7.15), the picture that resolves is:

**The model has Platonic-like cross-notation transfer for the specific canonical operators it was trained to recognize as logical connectives. It does not have a Platonic-like representation of "logical-operator-ness" that extends to novel instances. This pattern is cross-family stable across three model families with different training corpora, architectures, and tokenizers.**

Concretely, two empirical facts ground this claim, each now established at three model families:

**Fact 1 — Cross-notation canonical-operator structure is robust.** At multiple cells across all three models (OLMo 2 sente→close L10, Gemma 2 sente→opera L4, Pythia 6.9B-d opera→close at L4/L7/L16, with similar evidence at other cells), a probe trained on canonical-operator activations at one notation correctly classifies canonical-operator activations at the other notation. M2-canonical reaches 0.812 (OLMo, 10-class, ~8× chance) and 1.000 (Gemma and Pythia, 10× chance) at our anchor-survivor cells under canonical-set expansion; M2-arity (the coarsened binary-vs-unary projection) reaches 1.000 in all three. Pythia adds the strongest cell-density evidence: 31/80 v5 cells PASS M2-canonical at ≥ 0.65, vs handfuls in OLMo and Gemma. The 5-class operator geometry — the relative positions of `and`, `or`, `not`, `implies`, `necessarily`, plus the 5 new canonicals `xor`, `nand`, `possibly`, `always`, `negate` — is preserved across the notation boundary in every tested model. This is the strongest PRH-supporting finding in the project, now established at three model families.

**Fact 2 — That cross-notation structure does not extend to novel operators in an arity-respecting way.** All four originally-PASS-arity cells from the script 22b full sweep (§3.7.11, OLMo + Gemma), and all three v3 PASS-arity candidates in Pythia (§3.7.15, opera→close at L4 / L7 / L16) — cells where the per-invented-word predicted canonical appeared to track intended arity — have been retracted under either invented-set expansion (scripts 22c on OLMo/Gemma; v4 scope in 23 on Pythia) or canonical-set expansion (scripts 22d; v5 scope in 23). The mechanism revealed by canonical-set expansion is **default-to-rarest-canonical**: when shown an invented-operator activation, the probe routes it to whichever canonical sits in the highest-entropy decision region of the canonical-operator manifold. With 5 canonicals that target was the cell-specific default (`and`, `necessarily`, `implies`, `not`, or `implies` depending on the cell); with 10 canonicals adding near-zero-frequency `xor`, `nand`, `possibly`, `always`, `negate`, the target shifts wholesale to the multi-subword new canonicals. The previously-observed arity-respecting pattern was coincidence between the 5-canonical default and the invented set's intended-arity distribution; it does not survive the addition of more canonicals in any of the three models tested.

The cross-family pattern in the v5 default target is itself interesting: **OLMo 2 collapses to `nand` (100% on its surviving cell), Gemma 2 splits between `nand` (87.5%) and `negate` (12.5%), and Pythia distributes across `nand` (27.9%), `xor` (22.3%), and `negate` (19.8%) for an aggregate 70% of invented mass on the three multi-subword new canonicals.** The *direction* of the mechanism (toward low-frequency multi-subword canonicals) is cross-family invariant; the *specific target distribution* varies by model. This is the signature one would expect from a compressive routing pattern over a model-specific softmax: which low-prior canonical wins depends on the local geometry, but the routing always prefers low-prior canonicals over high-prior ones.

Both facts together yield: **the morphospace of an LLM contains the specific canonical operators it was trained on, robustly across notation substrates and across the three model families we have tested. It does not contain "the abstract logical operator" as a category that generalizes to novel instances in any of those models.** This is what we mean by *operator-set-bound substrate-invariance*. The cross-family stability of both halves (Fact 1 + Fact 2) is the strongest evidence we have that the finding is a property of mid-scale open language models at the 6.9-9B parameter range, not an artifact of any single model's training corpus, architecture, or tokenizer.

#### 4.1.3 Why this refines rather than rejects the PRH

A pure-compression null would not predict Fact 1. If the model were merely curve-fitting next-token surface statistics, there would be no reason for the geometric arrangement of `and`/`or`/`not`/`implies`/`necessarily` (and the new five) to align across NEUTRAL `Consider the word and in the sentence` templates and FUNC-PFX `The function and(p, q)` templates. These are syntactically different positions, different roles (noun-mention vs verb-of-function-application), and use different distributional contexts in the training data. Yet a single linear probe direction transfers between them at 8-10× chance accuracy under bootstrap stim-resampling AND canonical-set expansion. Something more than surface compression is happening — there is a substrate-independent geometric structure for the canonical logical operators.

A pure-Platonic claim would not predict Fact 2. If the model had abstracted "logical operator" as a category with internal arity structure, the same substrate-invariance should extend to novel words occupying logical-operator positions. It does not. The arity axis transfers cleanly for canonical operators (M2-arity = 1.000 at multiple cells under all expansions) and does not transfer for novel operators (M4b retracts at all four PASS-arity cells under either expansion). The Platonic abstraction stops at the boundary of the trained operator vocabulary.

The synthesis is more constrained than either extreme:

| Claim | Status after §3.7.14 |
|---|---|
| Strong PRH: uniform substrate-invariance across structurally-equivalent perturbations | **Rejected.** Variables yes; canonical operators yes-across-notations; novel operators no. |
| Pure-compression null: no structure beyond next-token surface statistics | **Rejected.** Canonical-operator geometry transfers across notations; this is non-trivial structure. |
| **Operator-set-bound PRH (our finding)**: substrate-invariance holds for the operator set the model was trained to recognize, but does not generalize to novel instances | **Supported.** Fact 1 + Fact 2 across two models, two anchors, both directions, both stimulus expansions. |
| Hierarchical / "partial Platonic" (arity yes, identity no, c.f. §3.7.12 / Dan Lutalo) | **Falsified for novel operators.** Real for canonical operators (M2-arity + M2-canonical co-pass at multiple cells under expansion). |
| Modifier-basin (apparent unary attraction is actually generic-modifier attraction) | **Falsified.** When `possibly` and `always` are added as modal-adverbial unaries, mass migrates to `nand` and `negate`, not to the modal-adverbial subset. The "necessarily basin" of §3.7.13 dissolves. |

#### 4.1.4 The morphospace edge: between trained-operator structure and novel-operator generalization

In the language of the original Path B framing — probing the boundaries of the morphospace using non-human prompts — our data locates a specific morphospace edge:

**Inside the edge** (high-confidence morphospace interior): the model has a substrate-independent representation of the logical operators it was trained on. The geometry of `and` / `or` / `not` / `implies` / `necessarily` / `xor` / `nand` / `possibly` / `always` / `negate` — including their binary-vs-unary structure — is stable across notation substrates within each model. Cross-model, the geometry is non-identical but each model's internal geometry is internally consistent across substrates.

**Outside the edge** (failure region): novel operators inserted into logical-operator positions are not routed to the appropriate region of the canonical-operator manifold by an arity-respecting mechanism. They are routed to whichever canonical happens to sit closest to the model's "low-confidence / high-entropy" default region — a position that shifts depending on what other canonicals are in the readout vocabulary. This is a compressive behaviour: route uncertain inputs to the highest-uncertainty bucket. It is not a Platonic abstraction of logical structure.

The edge sits between *trained-operator geometry* and *novel-operator generalization*. The PRH-supporting evidence is on the inside of this edge; the compression-favouring evidence is on the outside.

#### 4.1.5 Why this matters for the field debate

The PRH literature has tended to argue at the extremes: either networks converge on a universal substrate-independent geometry (Huh et al. 2024), or they are purely compressive and apparent structure is curve-fitting (skeptical position). Our data fits neither extreme. It supports a middle ground that is empirically discriminating: substrate-invariance is a *vocabulary-bound* property in current open LMs at the 7-9B scale. Tested operators transfer; novel operators do not.

For the alignment / safety literature, this is actionable: the model's apparent ability to handle abstract logical structure is, to the extent we can measure it with linear probes, bound to the operators in its training vocabulary. Inserting a novel logical primitive does not get an arity-respecting treatment; it gets a default-routing treatment that depends on the model's softmax over canonicals. This is the kind of substrate-invariance failure mode that matters for capability extrapolation: a model that handles `and` / `or` correctly cannot be assumed to handle a novel binary connective correctly by extension.

For the mechanistic-interpretability literature, the methodological contribution is the five-layer expansion battery (§3.7.14). Within-condition probe CV + cross-condition transfer + canonical-transfer gate + per-word breakdown are all necessary but not sufficient. The discriminating tests are *stimulus-expansion stability*: does the M4b survive when the invented set is expanded? does it survive when the canonical set is expanded? Both are required before claiming arity-respecting transfer to novel operators. Our project's headline retracted twice (§3.7.13, §3.7.14) under these expansions, each time revealing a stimulus-sample-specific artifact in the previous reading.

#### 4.1.6 What we cannot say

Three limitations on the scope of this conclusion are worth being explicit about:

1. **Three model families, but a single parameter range and a single structural domain.** Our positive finding (cross-notation canonical-operator transfer + novel-operator generalization failure) is now demonstrated at three model families: OLMo 2 7B (Dolma training, modified-Llama architecture, 32 layers); Gemma 2 9B (Google proprietary training with soft-capping, 42 layers, SentencePiece tokenizer); Pythia 6.9B-deduped (Pile training, GPT-NeoX with RoPE, 32 layers, standard BPE). The three-model replication establishes the finding as cross-family stable in this parameter range (6.9-9B), but two scope limitations remain. **(i) Single parameter range.** The thesis claim should be scoped to mid-scale (~7-9B) open base models. Whether the operator-set-bound pattern persists at 70B+ frontier scale, or dissolves into a more genuinely-Platonic abstraction as scale increases, is not testable from our M4-hardware setup; this is the principal scaling question for follow-up work. **(ii) All base models, no instruction-tuned variants.** All three models are pre-instruction-tuning base checkpoints. Instruction-tuning has well-documented effects on probe-readable structure (especially the structured / "neat" representation of natural-language tasks) and could shift the operator-set-bound vs novel-generalization balance in either direction. A Pythia-Chat / OLMo-2-Instruct / Gemma-2-IT replication is a clean follow-up. Other open model families (Qwen, Mistral, Llama 3) remain untested but are likely to fall in the same parameter range and would add weight to the cross-family claim without changing the scope; the cross-family stability we already see across three corpora and three architectures is strong evidence the finding generalizes within the 6.9-9B base-model range.

2. **Linear probes only.** All claims here are about what linear probes can read out from residual stream activations. A model with rich non-linear representations of "abstract logical operator" that linear probes cannot decode would look identical to our negative result. Causal-intervention / activation-patching tests are required before we can claim the model *lacks* novel-operator structure, as opposed to *lacking a linear-probe-decodable* novel-operator structure.

3. **One structural domain.** Propositional logic only. Substrate-invariance for set theory, simple type theory, or finite-group arithmetic might exhibit a different operator-set-bound vs. operator-set-unbound balance. The thesis claim should be scoped to propositional-logic operators until replication on other domains lands.

#### 4.1.7 A concise summary suitable for the introduction / abstract

**The thesis-claim form**: Cross-notation substrate-invariance in mid-scale (6.9-9B) open base language models is *operator-set-bound*: the model has substrate-independent geometric structure for the canonical operators it was trained to recognize as logical connectives, with that structure transferring across notation boundaries and surviving canonical-set expansion. The same structure does not extend to novel operators in an arity-respecting way; novel operators are instead routed by a default-to-rarest-canonical compression mechanism that shifts target as the canonical set changes. This pattern is cross-family stable across three model families (OLMo 2, Gemma 2, Pythia) with different training corpora (Dolma, Google proprietary, Pile), architectures (modified Llama, soft-capped Gemma, GPT-NeoX), and tokenizers. This locates a specific empirical edge of the Platonic Representation Hypothesis: it holds in vocabulary-bound form for trained operators, and fails in the generalization to novel operators, with the failure mechanism cross-family stable in *direction* (compression toward low-prior canonicals) and model-specific in *target* (which low-prior canonical wins varies by model).

#### 4.1.8 v6 update: Fact 1 strengthens, Fact 2 holds with one methodological caveat, default mechanism is multi-factor

The pre-registered v6 canonical-set expansion (§3.7.16, script 24) tested the operator-set-bound finding under a 15-class readout (8 binary + 7 unary canonicals) across all three model families simultaneously, with the analysis plan frozen *before* any v6 cache extraction. Three updates to §4.1's framing follow.

**Fact 1 strengthens.** Cross-notation canonical-operator transfer at M2-canonical = 1.000 with bootstrap 95% CI = [1.000, 1.000] is now demonstrated under 15-class readout (≈ 15× chance) at the same `operator-after → operator-after L 4 N→F` cell in *all three* model families. The Phase 1 / Phase 2 positive result is the most robust it has ever been: adding five canonicals to the readout vocabulary — three with materially different frequency and subword profiles — does not degrade the cross-notation probe transfer for the operators the model was already trained to recognize. The "operator-set-bound" half of the finding (the trained-operator geometry) is now bootstrap-confirmed at ceiling under three independent scope expansions (v4: invented-set expansion; v5: 10-canonical readout; v6: 15-canonical readout including pre-registered new operators).

**Fact 2 holds in two of three models; one methodological caveat in the third, now causally adjudicated.** OLMo 2 7B and Pythia 6.9B-deduped both have *zero* PASS-arity cells at v6, fully consistent with the pre-registered P_RETRACT prediction. Gemma 2 9B has *two* emergent v6 PASS-arity cells (N→F opera→close L 2 and N→F sente→close L 2), with M4b jumping from 50% at v5 to 82.2% / 66.2% at v6 *without any change to the underlying activations* — only the canonical-readout sub-selection differs between scopes. M2-arity at these cells is constant across v3-v4-v5-v6 (0.78-1.00); only M4b changes. The follow-up causal patching test (§3.7.17, script 25a) splits the two cells: at `opera→close L 2`, patching the L2 close-paren residual with a NEUTRAL-`operator-after`-sourced canonical activation produces clean arity-flip behaviour under intervention (8/0/8 unary-intended words shift toward FUNC-PFX-`and` when patched with `and`, ΔKL = +0.033; symmetric 8/0/8 binary-intended → FUNC-PFX-`not`, ΔKL = +0.061), so this cell is a **causally validated single-model exception** to operator-set-bound substrate-invariance in Gemma 2 9B. At `sente→close L 2`, patching from NEUTRAL-`sentence-final` produces 100% probe-causality but flat-or-negative behavioural ΔKL (the random-norm control actually exceeds the targeted patch), confirming this second cell is a **probe-only artifact of M4b's threshold-sensitivity to readout granularity** rather than a substantive arity-respecting structure. M2-arity (partition-invariant once the binary/unary split is fixed) remains the primary arity-axis measurement; M4b should be reported alongside but flagged as granularity-sensitive. The headline for §4.1 is therefore: operator-set-bound substrate-invariance holds across three model families at all tested cells in OLMo 2 7B and Pythia 6.9B-d (under both linear-probe and causal-patching tests), with exactly one tightly-scoped, causally-validated exception in Gemma 2 9B at L2 close-paren when sourced from NEUTRAL operator-after.

**The default mechanism is multi-factor, not "rarest canonical" alone — and the residual third factor is probe-decision-boundary geometry, not raw cosine similarity.** The v6 pre-registration laid out three competing single-axis predictions for which canonicals would attract novel-operator mass: P_FREQ (route to low-frequency canonicals), P_SUBWORD (route to multi-subword canonicals), and P_INTERACTION (combine both with controls). **All three predictions fail in all three models.** Per-model details in §3.7.16, but the headline anomalies that resist any single-axis reading are: Gemma routes 30.6% of invented mass to the three NEW LF canonicals (below the pre-reg's 35% threshold for P_FREQ); OLMo routes 15.9% to `implies` — a v3/v4/v5 carryover canonical that none of the three predictions targeted — and only 8.0% to the new LF canonicals combined; Pythia distributes 22.9% across the new LF canonicals but loses 21.9% to the v5 `nand` attractor that the pre-reg expected to be displaced. The mechanism cannot be reduced to either frequency or subword shape. §3.7.18 (script 25b) tested the most natural third-factor candidate — *contextual semantic neighborhood* operationalised as cosine similarity between each invented word's mean activation and each canonical's mean activation at the focus layer — and found it does not close the gap either: at distributed cells across all three models, identity-level agreement between cosine argmax and probe prediction is barely above the 6.7% chance baseline (Gemma 11.6%, OLMo 14.8%, Pythia 16.2%), and arity-conditioned agreement is at within-arity chance. The residual third factor is therefore **probe-decision-boundary geometry**: the probe routes invented words via learned discriminative weights that capture per-word residual-stream structure not preserved by mean-pooling. The honest abstract / introduction characterisation is: "the model routes novel-operator activations to one of the low-prior canonicals in a model-specific way that depends on frequency, subword tokenization, and per-word residual-stream structure that is not captured by either mean-pooled cosine similarity or any single-axis training-distribution statistic; no published single-axis reduction of this mechanism is consistent with the v6 + script 25b data."

### 4.2 The role of tokenization (and what scale tells us about it)

### 4.2 The role of tokenization (and what scale tells us about it)

Operators in our setup tokenize as Tier-2 byte-fallback subwords with no prior semantic associations to logical operators. Variables tokenize as Tier-1 single tokens (Greek lowercase) with no prior semantic associations beyond their conventional mathematical use as placeholders.

We had pre-registered two interpretations of the asymmetry. The 7B replication has substantially adjudicated between them:

- **Capacity-bound interpretation (REJECTED)**: at 1B scale the model cannot do in-context binding of multi-subword operators because the byte-fallback subwords lack the embedding-space signal required to bridge to the canonical operator. *Prediction*: at 7B, the gap should close substantially or disappear. *Observed*: gap shrinks from 0.835 to 0.710 (~15% reduction). The qualitative asymmetry persists; the failure mode at 7B layer 7 is *sharper* than at 1B (200/200 uniform default for B''). This rejects the strong capacity-bound interpretation, though a weaker version (further capacity helps but at sub-linear returns) is consistent with the data.
- **Role-bound interpretation (SUPPORTED)**: operators are fundamentally harder to substrate-invariantly represent than variables, because operators *carry* semantic content while variables are placeholders. *Prediction*: asymmetry should persist at 7B with similar magnitude. *Observed*: variable substrate-invariance was at the ceiling at 1B and remains so at 7B; operator substrate-invariance gap reduces only modestly. The asymmetry between roles is preserved.

The remaining open question is whether the role-bound asymmetry is *fundamental* (would persist at 70B+ frontier scale) or merely *robust to modest scale-ups* (would yield to enough parameters). The current data cannot answer this, but the trajectory (~15% gap reduction for 4× parameters) does not project a clean closure at frontier scale.

A subsidiary tokenization observation worth recording: the canonical operators in our set are 1-token (`and`, `or`, `not`, `implies`), while invented operators are uniformly 2-token. This length difference is itself a confound for the operator-anchored measurement, because the post-operator position sits at slightly different sentence positions in canonical vs invented conditions. Disentangling "tokenization-length cost" from "operator-semantics cost" requires a follow-up experiment that varies invented-operator subword count systematically (see §6).

### 4.3 The four-channel mechanistic story

At 7B layer 7, the canonical-operator probe trained on A produces predictions on operator-renamed inputs (B') that are fully characterized by a four-mechanism factorial picture (§3.4.5). The pre-registered hypothesis space has been fully adjudicated by Phase 0:

- **H1 (structural defaulting to `not` specifically), CONFIRMED.** Subword-length variation (§3.4.2) shows the default persists across L ∈ {1, 2, 3, 4} with peak gaps in 0.71–0.75. The second-canonical-unary probe (§3.4.4) further establishes that the default is to `not` *specifically*, not to the unary class generically: even `perph` (the invented replacement for canonical `necessarily`) goes 84% to `not` and only 16% to `necessarily`. The H1 attractor is not a generic-unary attractor; it is a `not`-attractor.
- **H2 (tokenization-position effect), REJECTED.** H2's monotonic prediction (L=1 ≈ canonical, L=4 worst) is straightforwardly inconsistent with the data. L=1 is not measurably more substrate-invariant than L=4.
- **H3 (embedding-similarity escape channel), CONFIRMED.** The `bar` → `or` recovery follows `bar` regardless of which canonical slot it occupies (74% recovery in the and-slot, 82% in the or-slot; §3.4.3). This is a word-specific channel: `bar` has individual embedding-space proximity to `or`. Most invented words don't have this property and default to `not` (H1); a small minority do, and recover their target.
- **H4 (template-lexical-context pull channel), CONFIRMED.** A previously unrecognized fourth channel: the probe trained on canonical operators learns a direction defined partly by the surrounding template's lexical content, not purely by the post-operator semantic load. When B' substitutes invented operators into the same templates, attention from the operator-anchored position still flows to the template's context words ("either"/"neither"/"disjunction" for or-templates; "must"/"always"/"every situation" for modal templates), and the probe partially predicts the corresponding canonical. This applies to every invented word in a given template, with magnitude ~8–22 of 50 toward the template-relevant canonical.

The 1B below-chance pathology (acc 0.165 < chance 0.250 at layer 4 in script 09's earlier reading) is best read in light of these results as a noisier form of the same H1+H4 combination: at 1B the model was already collapsing invented operators to wrong canonical classes, but with the H4 template-context channel producing more chaotic per-template effects than the cleaner 7B pattern.

**Predictive formula.** For any (invented-word, template, layer-7) triple, the probe's prediction distribution is well-approximated by:

```
P(canonical = c | invented_word, template) ≈
    H1 default attractor strength toward `not`
  + H3 embedding-similarity contribution toward the canonical that
    the invented word's embedding is closest to (often zero)
  + H4 template-context contribution toward the canonical that
    the template's lexical content encodes
  + small residual
```

**Methodological consequence: H4 is a confound that must be reported.** The H4 channel means that operator-renaming probe accuracy *systematically overestimates* the model's actual binding capability for invented operators. The portion of "correct" predictions driven by H4 is not capturing in-context binding; it is capturing the probe's reliance on template lexical context. Any Phase 1 substrate-invariance result that does not control for H4 will publish an inflated recovery number. The control is straightforward: report per-template confusion matrices, not just aggregated ones, so the template-context effect is visible.

**Theoretical consequence: this is the right story for the Phase 1 paper.** The original framing "OLMo 2 fails at operator substrate-invariance" understates the finding. The accurate framing is *"OLMo 2 has a structural default-to-`not` attractor that is perturbed by two distinct content-driven pull channels (embedding-similarity and template-context); the four-channel decomposition fully accounts for the model's behaviour on invented operators."* This is a stronger, more mechanistic claim and is publishable in a venue that values mechanistic findings (e.g., NeurIPS Interpretability workshop, ICLR mainstream, or a mechanistic-interp-focused journal).

### 3.4.2 Subword-length variation: H2 rejected, H1 supported, H3 emerges

We ran the subword-length variation experiment (`experiments/10_subword_length_probe.py`) on OLMo 2 7B, holding the canonical operator set fixed and constructing four invented-operator sets at BPE subword counts L ∈ {1, 2, 3, 4}. The L=2 set was pinned to script 09's `{bliq, dren, vusp, molex}` for direct comparability.

Selected invented words (validated against the live OLMo tokenizer):
- L=1: `foo`, `bar`, `baz`, `fred`
- L=2: `bliq`, `dren`, `vusp`, `molex`
- L=3: `thwack`, `shplork`, `snurk`, `vepth`
- L=4: `qibblist`, `vusplark`, `snurklon`, `blaroxxen`

**Result 1: peak A-B' gap is essentially independent of subword length.**

| L | Min B'_L probe acc | Max gap | Peak-gap layer |
|---|---:|---:|---:|
| 1 | 0.265 | **0.735** | 1 |
| 2 | 0.290 | **0.710** | 7 |
| 3 | 0.270 | **0.730** | 5 |
| 4 | 0.250 | **0.750** | 1 |

The peak gaps are within a 4-percentage-point spread across a 4× variation in subword count. H2's monotonic prediction (L=1 ≈ canonical, L=4 worst) is straightforwardly rejected. H1 is correspondingly supported: the operator substrate-invariance failure is not tokenization-position-driven.

**Result 2: default-to-`not` rate is constant across L (with one anomaly).**

Confusion matrices at layer 7 (the script-09 fixed diagnostic):

| L | invented set | total predictions → `not` | % to `not` |
|---|---|---:|---:|
| 1 | foo / bar / baz / fred | 158/200 | 79% |
| 2 | bliq / dren / vusp / molex | 186/200 | 93% |
| 3 | thwack / shplork / snurk / vepth | 181/200 | 91% |
| 4 | qibblist / vusplark / snurklon / blaroxxen | 183/200 | 92% |

L=2, L=3, L=4 are statistically indistinguishable (91–93% → `not`). L=1 is the outlier at 79%, but the 14-percentage-point gap is entirely driven by one invented word (see Result 3 below). Excluding that word, the L=1 default-to-`not` rate is 149/150 = **99%** — higher than all other L conditions.

**Result 3: the `bar` → `or` anomaly suggests a third mechanism (H3).**

The L=1 confusion matrix at layer 7 contains one row that breaks the default pattern entirely:

```
true = or (replaced by 'bar'):  and=0   or=41   not=9   implies=0
```

`bar` recovers `or` at 82% accuracy (41/50). No other non-trivial invented-word → canonical-operator mapping in the entire experiment exceeds 34%. The other three L=1 words behave as H1 predicts:
- `foo` → not 98% (default)
- `baz` → not 100% (trivially "correct", baz replaces `not`)
- `fred` → not 100% (default)

This suggests a third, embedding-driven channel:

- **H3 (embedding-similarity channel):** when an invented operator's token embedding happens to be close to a specific canonical operator's embedding, the probe recovers that canonical operator from B'. Most invented words don't have this property and default to `not` (H1). A small minority (`bar`) do, and recover their target. Candidate explanation for `bar` specifically: it can function as a connector in English ("all numbers *bar* one" = "all numbers *except* one", a quasi-disjunction), or it appears in connector-like syntactic positions in programmer placeholder usage (`foo bar baz`).

H3 is not a competitor to H1; it's an *escape hatch* on top of H1. The structural default to `not` is the rule. Embedding accidents are the exception, and they're rare enough that across 16 invented operators × N=50 each, exactly one (`bar`) showed a strong recovery.

**Result 4: peak-gap layer is per-L, not universal.**

The depth at which the operator-renaming "costs the most" varies with L:

| L | Peak-gap layer | Network depth fraction |
|---|---:|---:|
| 1 | 1 | ~3% (embedding-adjacent) |
| 2 | 7 | ~22% |
| 3 | 5 | ~15% |
| 4 | 1 | ~3% (embedding-adjacent) |

L=1 and L=4 (the subword-count extremes) collapse to default immediately at layer 1. L=2 and L=3 attempt operator resolution for several layers before collapsing. This is an unexplained pattern — it does not fit cleanly with either H1 or H3 — and is worth a brief follow-up.

**Overall verdict.** The "uniform default to `not`" finding from script 09 is robust to varying invented-operator subword count by a factor of 4. The failure mode is structural (H1), not tokenization-positional (H2). There is a small but mechanistically interesting embedding-similarity escape hatch (H3) that recovers individual invented-canonical pairs when the surface forms happen to be close in embedding space.

The H1-supporting result substantially strengthens the role-bound interpretation of the variable/operator asymmetry: the model genuinely treats invented operators as instances of the unique canonical unary class, regardless of how those invented operators are tokenized. Adding a second canonical unary (e.g., a "negation-of" prefix) is now the priority follow-up to test whether the default is to *unary class generically* or to *`not` specifically*.

### 3.4.3 L=1 bar-anomaly follow-up: H3 confirmed, H4 emerges

We ran the focused L=1 follow-up (`experiments/11_l1_bar_anomaly_probe.py`) on OLMo 2 7B with three L=1 invented-word sets, holding everything else fixed: canonical operator set, templates, probe training data, diagnostic layer.

**Results at layer 7:**

| Set | Mapping | bar's prediction | Overall acc |
|---|---|---|---:|
| A (original) | bar replaces `or` | 41/50 → or (82%) | 0.455 |
| B (bar moved) | bar replaces `and` | **37/50 → or (74%)** | 0.360 |
| C (no bar) | bar absent; pop/zap/ping/huh | n/a | 0.405 |

**The decisive measurement.** In Set B, `bar` is in the `and` slot (and-templates have no `or`-related lexical context), yet bar-tagged prompts are *still* predicted as `or` 74% of the time. The recovery follows `bar`, not the slot. H3 (embedding-similarity escape channel) is decisively confirmed: `bar`'s token embedding is sufficiently close to `or` that the probe predicts `or` for bar-prompts regardless of structural role or template context.

**An unexpected fourth channel (H4).** Set C's L=1 set is `{pop, zap, ping, huh}`. The probe's predictions on these prompts at layer 7:

- `pop` (and-slot) → 47/50 → not (clean default-to-`not`)
- `ping` (not-slot) → 50/50 → not (trivially correct, ping replaces `not`)
- `huh` (implies-slot) → 48/50 → not (clean default-to-`not`)
- `zap` (or-slot) → 31/50 → **or** (62% recovery)

`zap` has no obvious embedding similarity to `or`. Yet 31/50 of zap-prompts are predicted as `or`. Cross-referencing with Set B (`foo` in or-slot → 22/50 → or):

| Word in or-slot | → or | inferred |
|---|---:|---|
| `foo` | 22/50 (44%) | H4 floor — pure template-context pull |
| `zap` | 31/50 (62%) | H4 + small H3 contribution |
| `bar` | 41/50 (82%) | H4 + large H3 contribution |
| `bar` (in and-slot, off H4) | 37/50 (74%) | H3 alone, no H4 |

This decomposes the operator-renaming probe predictions into mechanisms:

- **H4 (template-lexical-context pull)** — a previously unrecognized channel. The `or` templates contain lexically dense `or`-related context ("either", "neither", "at least one of", "disjunction"). The probe trained on canonical `or` learns a direction defined partly by this template context — not purely by the post-`or` semantic load. When B' substitutes an invented operator into the same template, the surrounding context is unchanged, attention from the operator-anchored position still flows to the context words, and the probe partially predicts `or` for *any* invented word in the `or` slot. This is a methodological confound: the probe is not measuring "in-context binding of invented operators to canonical operators"; it is partially measuring "consistency of template lexical context with each canonical class."
- **H3 (embedding-similarity escape)** — additive on top of H4 for words whose embedding is independently close to a canonical.

The clean H3 measurement is the bar-off-slot row (74%): when H4 is absent (bar in and-slot), H3 produces 74/50 → or all on its own.

### 3.4.4 Second-canonical-unary probe: H1a appears to win, but is later refined by script 13

**Note (May 2026):** the H1a/H1b conclusion from this script is *partially* superseded by script 13's factorial H4 quantification — see §3.4.5. The script 12 probe was trained on multi-operator templates, which we now understand produces a probe with a less sharply-defined `necessarily` direction than the single-operator-template probe of script 13. The script 12 numbers below are still accurate measurements of *that probe*, but the inferred "H1 is `not`-specific" claim does not survive the better-instrumented script 13 measurement.

We ran the 5-class probe with `necessarily` added as a second canonical unary (`experiments/12_second_unary_probe.py`). All five canonical operators are replaced in B' with 2-subword invented words: `{bliq, dren, vusp, molex, perph}`.

**Sanity check.** Canonical operators tokenize as: and (1), or (1), not (1), implies (1), necessarily (1). All single-token. The 5-class probe achieves 1.000 CV accuracy on A from layer 2 onward and 1.000 held-out accuracy on B (var-renamed) at layer 7. The probe distinguishes `not` from `necessarily` perfectly when given canonical inputs — so the question "does the model treat invented operators as `not` specifically or as a generic unary" is well-posed.

**Layer-7 confusion matrix (the headline result):**

| True (input) | → and | → or | → not | → implies | → necessarily | n |
|---|---:|---:|---:|---:|---:|---:|
| and (bliq) | 0 | 11 | **29** | 7 | 3 | 50 |
| or (dren) | 0 | 16 | **34** | 0 | 0 | 50 |
| not (vusp) | 0 | 0 | **50** | 0 | 0 | 50 |
| implies (molex) | 0 | 0 | **42** | 0 | 8 | 50 |
| necessarily (perph) | 0 | 0 | **42** | 0 | 8 | 50 |

**H1a vs H1b verdict.** `not` dominates `necessarily` roughly 5:1 as the structural attractor across all invented-input rows. The cleanest single piece of evidence is the `perph` row: `perph` is the invented replacement for canonical `necessarily`, yet 84% of perph-prompts are classified as `not` and only 16% as `necessarily`. If H1b (unary-class-generic default) were correct, perph should split roughly 50/50; it doesn't.

**H1a is confirmed**: the structural default is to `not` *specifically*, not to the unary class generically.

**The 16% `necessarily` predictions are H4, not H1b.** The implies-templates and necessarily-templates both contain modal-logic language ("must be true", "must hold", "always", "in every situation", "every case", "without exception"). When the probe is trained on canonical `necessarily`, the post-`necessarily` representation accumulates this modal-context signal via attention. When B' substitutes an invented operator into the same template, the modal context remains and the probe partially predicts `necessarily`. The 8/50 → necessarily counts on both `molex` (implies-templates) and `perph` (necessarily-templates) are explained by H4 — both template families contain modal language that pulls predictions toward `necessarily`.

The `dren` row (or-templates) shows no `necessarily` predictions (0/50) — because the or-templates contain no modal context. Instead, the H4 channel for or-templates pulls toward `or`, giving the 16/50 → or count. This is the same H4 mechanism observed in script 11.

**Combined-unary-class share is high, but not equal-split.** Across the four invented-binary rows, the combined "unary class" share (not + necessarily) is 64%, 68%, 100%, 100%. The remaining 32% and 36% goes to `or` (H4 from or-templates) and to other non-unary classes. Within the unary class, `not` captures 58–84% and `necessarily` captures 0–16%. This is the *partial-H1b* pattern the script's reading guide warned about, and it is fully consistent with H1a + H4 rather than H1b proper.

**Conclusion (provisional, superseded by §3.4.5).** As measured by the script 12 probe, the structural default-to-unary appears `not`-specific. Script 13 shows this is an artifact of the multi-op template training: with single-operator templates the within-unary distribution is template- and word-dependent and `necessarily` becomes a substantial competing attractor. The corrected H1 reading is *unary-region-attractor* with within-region modulation by H3 and H4 (§3.4.5).

### 3.4.5 The four-channel picture (refined by script 13)

Script 13 (`experiments/13_template_context_quantification.py`) quantified the H4 channel with a factorial design: 5 invented words × 5 template families × 50 stimuli, single-operator templates throughout, 5-class probe trained on canonical A at the same template form. The result substantially refined what we thought we knew about H1.

**The headline numerical finding.** Across all 25 (word × template) cells × 50 stimuli = 1250 invented-operator inputs, the probe predicted:

| Canonical class | Total predictions | Share |
|---|---:|---:|
| `and` | 0/1250 | **0.0%** |
| `or` | 17/1250 | 1.4% |
| `not` | 629/1250 | 50.3% |
| `implies` | 50/1250 | 4.0% |
| `necessarily` | 554/1250 | 44.3% |

**The empty `and` column is the decisive finding.** Across 1250 invented-operator inputs covering 5 template families and 5 invented words, the probe predicted `and` zero times. This rules out the strong corpus-frequency hypothesis (`and` is by far the highest-frequency canonical operator in English text; if frequency drove defaults, `and` would dominate). Together with the near-empty `or` and `implies` columns, the data show the default attractor is *specifically* unary-class — the model has a structural prior on "what does an operator look like at this position" that is unary, not binary.

**Reinterpretation of H1: unary-region attractor, not `not`-specific.** Per-word totals across all templates reveal substantial heterogeneity within the unary class:

| Invented word | → `not` (1250) | → `necessarily` (1250) | Net lean |
|---|---:|---:|---|
| bliq | 220/250 (88%) | 23/250 (9%) | strongly `not` |
| dren | 186/250 (74%) | 64/250 (26%) | `not` |
| vusp | 85/250 (34%) | 115/250 (46%) | mixed |
| molex | 72/250 (29%) | 178/250 (71%) | strongly `necessarily` |
| perph | 66/250 (26%) | 179/250 (72%) | strongly `necessarily` |

The script-12 conclusion that the default was `not`-specific (H1a) was an artifact of the probe being trained on lexically-richer multi-operator templates, which dilutes the per-operator signal. With single-operator templates and a probe trained on them (script 13), the unary-region structure becomes visible: `not` and `necessarily` are both anchor points within the same default-attractor region, and individual invented words land closer to one or the other based on embedding-space proximity.

**Refined H1 formulation.** *H1 (revised):* invented operators are pulled into a **unary-class region** in activation space at the operator-anchored position, with `not` and `necessarily` as the two probe-recoverable anchor points. The default attractor is specifically unary, not specifically `not`; the not/necessarily split is determined by H3 (word-embedding) and H4 (template-context).

**H4 quantification result.** The H4_pull metric was computed as `avg_W [in-template rate]` − `avg_{W, T'≠C} [out-of-template rate]` per canonical class:

| Canonical | In-template rate | Out-of-template rate | H4_pull |
|---|---:|---:|---:|
| and | 0.000 | 0.000 | +0.000 |
| or | 0.048 | 0.010 | +0.038 |
| not | 0.648 | 0.467 | **+0.181** |
| implies | 0.080 | 0.020 | +0.060 |
| necessarily | 0.580 | 0.414 | **+0.166** |

**H4 operates within the unary region, not toward arbitrary canonicals.** H4_pull is substantial for `not` (+0.181) and `necessarily` (+0.166), weak for `or` and `implies` (<+0.07), and zero for `and`. The template-context channel can shift the not/necessarily split within the unary region by ~18 percentage points but does not pull predictions outside the unary region. This is more constrained than the original H4 hypothesis (which assumed templates pulled toward whatever canonical they "owned"); the actual H4 channel is narrower and unary-bound.

**The H1/H3/H4 picture, finalized after scripts 14 and 15.**

| Channel | Refined description | Phase 0 status |
|---|---|---|
| **H1** | **Default to a unary-class region** in activation space. Mass: 99.6% in the cleanest (neutral-train × neutral-test) measurement (script 15); 94.6% in script 13's factorial; 91–93% in script 10's subword-length sweep. *Not derived from layer-0 embedding geometry* — all script-13 invented words have peak layer-0 cosine similarity to `and`/`or` (script 14), and the network actively moves them into the unary region during forward pass. | **Confirmed (multiple independent probe instruments).** |
| **H2** | Tokenization-position effect | Rejected (script 10) |
| **H3** | Word-specific bias in two regimes: cross-class escape (`bar` → `or`) and within-unary modulation (`molex` → `necessarily`). **Both regimes are mostly NOT layer-0 phenomena** — script 14 shows layer-0 embedding similarity predicts the script-13 top landing for only 1 of 5 words, and `bar`'s layer-0 sim(or)−sim(not) margin (+0.021) is too small to fully explain its 74–82% probe recovery. The H3 channel is partly word-embedding-derived but largely *constructed by layers 1–7*. | Confirmed (multiple distinct forms), but mechanism is not layer-0. |
| **H4** | **Reframed as probe-instrument artifact (script 15).** Script-13's measurement of "template-context pulls toward the template's owned canonical" is conditional on the probe being trained on rich templates of the same form. With a neutral-trained probe, the same residual-stream activations produce predictions dominated by template syntactic scaffolding (e.g., "If... then..." → `implies`), not by the H4 pattern of script 13. The underlying residual stream carries both unary-attractor and implication-scaffolding signals; H4 as originally formulated is a probe read-out artifact. | Reframed: scripts 11–13 results are valid probe-internal measurements but should not be interpreted as residual-stream-level properties. |

The factorial decomposition for the operator-anchored probe's prediction at layer 7 of OLMo 2 7B:

```
P(canonical = c | invented_word, template) ≈

  H1: ~95-100% of probability mass goes to {not, necessarily}
      (the empty and/or/implies columns mean H1 ≈ unary-region attractor)

  H3: word-embedding bias shifts the not↔necessarily split based on
      embedding-space proximity of the invented word to one anchor or
      the other (or, rarely, escapes the unary region entirely for words
      like `bar` with strong cross-class embedding similarity)

  H4: template-context bias shifts the not↔necessarily split by ~18 pp
      toward `not` in non-modal templates and toward `necessarily` in
      modal templates

  residual: very small, mostly to `or` (~1.4%) and `implies` (~4%) in cases
            where attention to specific template context words leaks beyond
            the unary region
```

This is the principal mechanistic contribution of Phase 0: the model has a *hierarchical encoding* of operator structure at the operator-anchored position. The arity distinction (unary vs binary) is robustly encoded — invented operators are recognized as unary-shaped or binary-shaped, with invented inputs reliably landing in the unary region. Within unary, the specific operator identity is much more fragile and is determined by content channels (H3 word-embedding, H4 template-context). Within binary, the specific operator identity is completely absent — invented operators are indistinguishable from unary at this position, suggesting the binary-operator semantics requires more than just the operator-anchored representation.

### 3.4.6 Embedding-similarity audit (script 14): H1 is constructed by the network, not inherited from embeddings

To distinguish "H1 is a tokenizer/embedding artifact" from "H1 is a real computational structure built by the network", we compute the cosine similarity of each invented word's layer-0 (token-embedding) representation to each canonical operator's layer-0 representation (`experiments/14_embedding_similarity_audit.py`).

**Setup.** For each canonical (and, or, not, implies, necessarily — all 1-subword) and each invented word (the script-13 set bliq, dren, vusp, molex, perph — each 2-subword), we tokenize with leading space (matching probe-extraction conventions) and look up embedding-layer vectors directly. For multi-subword words we report three pooling strategies: mean, first-subword, last-subword. The L=1 invented-word sets from scripts 10/11 (foo, bar, baz, fred, qux, quux, thud, pop, zap, ping, huh) are included to test the cross-class H3 bar→or finding.

**Canonical baseline similarities.** Pairwise cosine similarities among the five canonicals at layer 0:

| | and | or | not | implies | necessarily |
|---|---:|---:|---:|---:|---:|
| and | 1.00 | 0.55 | 0.24 | 0.08 | 0.09 |
| or | 0.55 | 1.00 | 0.26 | 0.07 | 0.11 |
| not | 0.24 | 0.26 | 1.00 | 0.07 | 0.18 |
| implies | 0.08 | 0.07 | 0.07 | 1.00 | 0.15 |
| necessarily | 0.09 | 0.11 | 0.18 | 0.15 | 1.00 |

This already shows a high-frequency cluster {and, or, not} (pairwise cosines 0.24–0.55) and a low-frequency outgroup {implies, necessarily}. *Notably, `not` is in the high-frequency cluster, not in any "unary" cluster with `necessarily`* — at layer 0, the unary category is not a primitive of embedding-space geometry.

**Invented-word embedding similarities to canonicals (mean-pool, script-13 set).**

| Invented word | and | or | not | implies | necessarily | Top canonical |
|---|---:|---:|---:|---:|---:|---|
| bliq | +0.013 | +0.008 | +0.046 | +0.031 | +0.037 | **not** |
| dren | +0.095 | +0.079 | +0.065 | +0.025 | -0.003 | **and** |
| vusp | +0.032 | +0.046 | +0.042 | +0.027 | -0.009 | **or** |
| molex | +0.043 | +0.067 | +0.037 | +0.040 | +0.007 | **or** |
| perph | +0.086 | +0.120 | +0.090 | +0.050 | +0.043 | **or** |

**Comparison with script-13 layer-7 probe landings.**

| Word | Layer-0 top (mean) | Layer-7 observed top (script 13) | Layer-0 sim(not) > sim(necc)? | Layer-7 landing not > necessarily? |
|---|---|---|---|---|
| bliq | not | not (88%) | YES | YES ✓ |
| dren | and | not (74%) | YES | YES ✓ |
| vusp | or | necessarily (46%) | YES | NO ✗ |
| molex | or | necessarily (71%) | YES | NO ✗ |
| perph | or | necessarily (72%) | YES | NO ✗ |

**Two structural findings.**

1. **Layer-0 top canonical predicts script-13 top landing for only 1 of 5 words** (bliq). The other 4 words start the forward pass closest to a binary canonical (`and` or `or`) at layer 0 and end at a unary canonical (`not` or `necessarily`) at layer 7. *The network is actively moving invented-operator representations from the high-frequency binary region into the lower-frequency unary region.*

2. **Within-unary not-vs-necessarily sign-agreement is 2 of 5** (only bliq and dren). For molex, perph, and vusp, the layer-0 ranking gives `not` higher similarity than `necessarily`, but the layer-7 probe routes them to `necessarily`. **H3 within-unary is NOT a layer-0 phenomenon** — it is constructed by intermediate layers.

**Cross-class H3 check: bar's anomaly.** Among the L=1 invented words, `bar` has the highest layer-0 sim(or)−sim(not) difference (+0.021), consistent with the script-11 bar→or recovery. However, `qux` (+0.015), `foo` (+0.013), and `baz` (+0.010) are close behind, while none of them produce the 74–82% cross-class recovery that bar does. The H3 cross-class regime has *some* layer-0 component (bar IS the top sim(or)−sim(not) word) but the magnitude is insufficient to explain bar's outsized probe behaviour. H3 cross-class is also partly constructed.

**Implication for the thesis.** The H1 unary-region attractor is a genuine computational mechanism built by attention/MLP processing across layers 1–7. Layer 0 does not bias invented operators toward the unary region; if anything, it biases them toward `and`/`or`. The unary attractor is therefore not a tokenizer artifact, not a token-frequency artifact, and not derivable from layer-0 geometry. This is what makes the hierarchical-arity finding worth pursuing as the Phase 1 paper centerpiece.

### 3.4.7 Template-neutral probe (script 15): H1 confirmed at 99.6% mass; H4 reframed as probe artifact

The remaining alternative explanation for H1 is that the probe's "unary direction" was constructed by training on canonical-rich templates and is not a real property of the residual stream. We rule this out by training the probe on **canonical-neutral templates** (`experiments/15_template_neutral_probe.py`).

**Setup.** Neutral templates treat the operator as a quoted word and contain no canonical-specific lexical content: "Consider the word {op} in this sentence." / "The word {op} is shown in the figure." / "Replace the word {op} with a synonym." — 50 such templates, all syntactically valid for any of the five canonicals with no implication, modal, conjunction, or disjunction scaffolding. The probe is trained 5-class on canonical A in these neutral templates and then evaluated on three test conditions.

**Result 1: probe quality.** CV accuracy on canonical A_neutral at layer 7 is **0.996**. Cross-template generalisation to canonical A_rich (script-13 lexically-rich templates) is **0.944**. *Canonical operators are perfectly distinguishable at the operator-anchored position even in template-neutral context, and the probe direction generalises across template families.*

**Result 2: Test 1 — invented operators in neutral templates (the cleanest H1 measurement).**

| Word | and | or | not | implies | necessarily | unary mass |
|---|---:|---:|---:|---:|---:|---:|
| bliq | 0 | 0 | 26 (52%) | 0 | 24 (48%) | **100%** |
| dren | 0 | 0 | 24 (48%) | 0 | 26 (52%) | **100%** |
| vusp | 0 | 0 | 28 (56%) | 0 | 22 (44%) | **100%** |
| molex | 0 | 0 | 0 (0%) | 1 (2%) | 49 (98%) | **98%** |
| perph | 0 | 0 | 31 (62%) | 0 | 19 (38%) | **100%** |

**Mean unary-region mass: 99.6%.** This is the cleanest single measurement of H1 in the entire Phase 0 dataset. With template-context ablated to the maximum extent possible while keeping syntactic well-formedness, invented operators land in the unary region at essentially 100% — no `and`, no `or`, no `implies`, all 50/per-word predictions to either `not` or `necessarily`. **The unary-region attractor is real and robust.**

**Within-unary distribution is roughly 50/50 in the cleanest measurement.** Four of five words (bliq, dren, vusp, perph) split between `not` and `necessarily` at ratios between 38/62 and 62/38 — i.e., close to uniform. Only molex shows strong within-unary bias (98% necessarily), and that bias is *not* derived from layer-0 embedding similarity (script 14: molex has the lowest layer-0 sim(necc) of the five invented words). Within-unary distribution structure is therefore both probe-instrument-dependent (script-13's rich-trained probe gave 88/9 for bliq; this neutral-trained probe gives 52/48 for the same word) and word-specific (molex's bias survives the probe-instrument change).

**Result 3: Test 2 — invented operators in script-13 rich templates (probed by neutral-trained probe).**

| Word | and | or | not | implies | necessarily | unary mass |
|---|---:|---:|---:|---:|---:|---:|
| bliq (in and-template) | 1 | 0 | 0 | **45 (90%)** | 4 | 8% |
| dren (in or-template) | 0 | 0 | 0 | **45 (90%)** | 5 | 10% |
| vusp (in not-template) | 5 | 0 | 8 (16%) | 26 (52%) | 11 (22%) | 38% |
| molex (in implies-template) | 0 | 0 | 0 | **45 (90%)** | 5 | 10% |
| perph (in necessarily-template) | 4 | 0 | 0 | 26 (52%) | 20 (40%) | 40% |

**Mean unary-region mass: 21.2%.** *Most predictions are `implies`*, even for bliq-in-and-template and dren-in-or-template. This is the opposite of script 13's pattern (94.6% unary mass with rich-trained probe).

**Why?** The script-13 rich templates all share "If... then..." or "true when..." or "is asserted" scaffolding — implication-like syntactic structures. When the canonical operator is present, the residual stream at the operator-anchored position carries both (a) the operator's identity signature and (b) the template's scaffolding signature. A probe trained on rich templates *learns to ignore the scaffolding* (since all five canonical classes share it in training) and focus on operator identity. A probe trained on neutral templates *has not seen the scaffolding* and conflates it with the canonical that most resembles it semantically — `implies`.

**Implications for the H4 channel.** The H4 effect of script 13 ("template-context pulls toward the template's owned canonical") is not a deep property of the residual stream. It is a probe-instrument-conditional response: when the probe is trained on rich templates of the same form, it produces the H4 pattern; when trained on neutral templates, it produces the implies-scaffolding pattern instead. The underlying residual-stream activation at the operator-anchored position carries both unary-attractor and implication-scaffolding signals; which one a probe reads out depends on its training distribution. The script-13 H4 quantification numbers are therefore valid as a probe-internal report but should not be interpreted as a representation-level property.

**The hierarchical-arity finding, finalised.**

| Substrate axis | Invariance | Probe accuracy or mass |
|---|---|---:|
| Variable renaming | Complete | 1.000 |
| Operator-arity (unary vs binary) | Complete and probe-instrument-invariant | 99.6% unary mass on B'_neutral |
| Operator identity within unary (not vs necessarily) | Effectively absent in cleanest measurement | ~50/50 split for 4/5 words |
| Operator identity within binary (and vs or vs implies) | Absent | 0% of invented operators classified as any binary canonical in B'_neutral |

This is the single strongest empirical signature of the Phase 0 program and is the centerpiece of the Phase 1 paper.

### 3.4.8 Syntactic-confound stress test (script 16): the arity attractor survives prefix/infix dissociation

After completing scripts 9–15, an external review (May 2026) identified two confounds that the existing data could not rule out:

- **Prefix vs infix syntactic confound.** In all scripts 7–15, binary canonicals (`and`, `or`, `implies`) appear in *infix* position ("p and q") and unary canonicals (`not`, `necessarily`) appear in *prefix* position ("not p"). The "unary attractor" might therefore be a "prefix-position attractor" — invented operators in infix slots fail to bind and retreat to the nearest representational cluster, which happens to be the prefix-position canonicals.
- **Metalinguistic POS prior.** Script 15's neutral templates ("Consider the word {op}.") frame the operator as a quoted-word noun-object. In natural English, `not` and `necessarily` can grammatically appear as standalone adverbial modifiers, whereas `and`/`or`/`implies` cannot. The 99.6% unary mass might reflect a "this token is being referenced as a word, route to the standalone-modifier canonicals" heuristic rather than a deep arity encoding.

`experiments/16_syntactic_confound_stress_test.py` addresses both. The decisive design uses **functional-prefix notation**: every canonical appears as `op(p)` (unary) or `op(p, q)` (binary), all in prefix position with the identical preceding context "The function ". This places every canonical in the same syntactic role and removes the metalinguistic-quotation frame; the only structural variable distinguishing canonicals is the operator's intrinsic representation plus the post-operator argument count, which is not visible at the operator-anchored position (causal LM).

**Two conditions are run side-by-side within the same script** to control for environmental differences:

- **Condition 1 (reference):** script-15 neutral metalinguistic templates, included to verify the script-15 result replicates in this run.
- **Condition 2 (critical):** functional-prefix notation, all canonicals in prefix-function-call position.

**Result.**

| Metric | Cond 1 (reference) | Cond 2 (critical) | Δ |
|---|---:|---:|---:|
| Probe CV accuracy on canonicals at L7 | 0.992 | **1.000** | +0.008 |
| Mean unary-region mass on invented operators | 99.6% | **100.0%** | +0.4 pp |
| ... on binary-replacement words specifically | 99.3% | **100.0%** | +0.7 pp |
| Mean binary-canonical mass on binary-replacement words | 0.7% | **0.0%** | −0.7 pp |

**Condition 2 per-word distribution at layer 7:**

| Invented word (replaces) | and | or | not | implies | necessarily |
|---|---:|---:|---:|---:|---:|
| bliq (~and, binary) | 0 | 0 | **50** | 0 | 0 |
| dren (~or, binary) | 0 | 0 | **50** | 0 | 0 |
| vusp (~not, unary) | 0 | 0 | **50** | 0 | 0 |
| molex (~implies, binary) | 0 | 0 | **50** | 0 | 0 |
| perph (~necessarily, unary) | 0 | 0 | **50** | 0 | 0 |

Across 250 invented-operator stimuli in functional-prefix notation, **all 250 are classified as `not`.** Zero `and`, zero `or`, zero `implies`, zero `necessarily`.

**Three interpretive consequences.**

1. **The prefix vs infix syntactic confound is decisively refuted.** Binary-replacement invented words (bliq, dren, molex) in *prefix-binary* functional position (`bliq(p, q)`, `dren(p, q)`, `molex(p, q)`) — exactly the position where binary canonicals naturally live in functional notation, syntactically indistinguishable from unary `not(p)` and `necessarily(p)` — are still classified 100% as unary `not`. Not a single one of 150 binary-replacement stimuli is classified as a binary canonical. The arity encoding is not an artifact of the prefix-position frame.
2. **The metalinguistic POS-prior hypothesis is weakened.** In Condition 2, the operator appears in function-call name position, where the standalone-adverb-modifier reading is structurally unavailable. The unary attractor appears even more sharply. The "POS-prior" hypothesis cannot fully explain the result.
3. **Within-arity identity is *more* fragile than scripts 9–15 indicated.** The same five invented words produce three qualitatively different within-unary distributions across conditions:

| Script | Probe-training notation | within-unary not:necessarily for bliq | for molex |
|---|---|---|---|
| 13 (rich) | natural-English multi-template | ~88:9 | ~29:71 |
| 15 (neutral) | metalinguistic single-template | 52:48 | 0:98 |
| 16 cond 2 (functional) | functional-prefix single-template | 100:0 | 100:0 |

The within-unary not-vs-necessarily split is therefore probe-instrument-dependent across a 0:98 to 100:0 range for the same word (molex) in different notations. The reviewer's specific "molex's mole+x composes to math-y → necessarily" hypothesis is partially refuted: if it were a robust subword-compositional property, molex should also go to necessarily in functional notation (which is *more* math-flavored than metalinguistic English). It doesn't.

**The refined Phase 1 claim.** The data now supports a narrower but bulletproof claim:

> OLMo 2 7B encodes operator arity (unary vs binary) at the operator-anchored position. The encoding is robust to (a) probe training-template family, (b) syntactic position (prefix vs infix), and (c) metalinguistic vs functional vs natural-English notation. Within-arity operator identity is *not* robustly encoded for invented operators; the within-unary `not`-vs-`necessarily` distribution is probe-instrument-dependent and varies from 50:50 to 100:0 across measurement conditions.

This is materially narrower than "hierarchical-arity substrate-invariance" because it admits the within-arity fragility explicitly. But it is also bulletproof in the sense that the arity-attractor finding has now survived three distinct probe instruments, two model scales, an embedding-layer geometric independence check, and a prefix/infix syntactic-position dissociation.

### 3.7 Phase 1 entry: cross-model replication (scripts 17–20)

Two scripts run at Phase 1 entry. Script 17 (`17_gemma2_cross_model_replication.py`) replicates the cleanest two Phase 0 probe instruments — script 15 NEUTRAL-metalinguistic and script 16 Condition 2 FUNCTIONAL-PREFIX — on Gemma 2 9B (`google/gemma-2-9b`). Gemma 2 9B was selected as the first cross-model target because it is deliberately different from OLMo 2 along every axis: different lab (Google DeepMind vs AI2), different training data, different architecture (GQA, sliding-window attention, no parallel residual), different tokenizer (SentencePiece 256k vs OLMo 2 BPE 100k), different precision requirement (bf16 vs fp16-OK), 42 layers vs 32, hidden dim 3584 vs 4096. Script 18 (`18_probe_artifact_diagnostics.py`) runs a four-diagnostic probe-artifact battery on both Gemma 2 9B and OLMo 2 7B to disambiguate within-condition probe-instrument artifact from genuine cross-model representational difference. Scripts 19/19b/20 layer in cross-notation directional-angle measurement, canonical-transfer gating, and gated invented-mass re-tests at pairings of interest.

> **CAVEAT BOX — anchor visibility constraint (added after script 20 peer review).**
>
> All scripts in §3.7 through §3.7.8 use the operator-anchored position `i + 1` after the last operator subword. In functional-prefix notation, this anchor lands on the `(` token *before* any arguments. In a causal LM the residual stream at that position cannot attend to future tokens (`p`, `,`, `q`, `)`). For canonical operators the model may still encode arity at this anchor from its pre-training lexical prior (billions of `and(...)` examples). For *invented* operators no such prior exists *and* no in-context evidence has been integrated yet. The cross-notation arity-respecting transfer tests in §3.7.8 are therefore strong evidence about **early post-operator catchment geometry** but only weak evidence about **full functional-call arity induction**. A post-call anchor re-test (script 21, in build) is required before any final negative claim about cross-notation arity-respecting transfer is defensible. All §3.7 claims should be read with this caveat in mind; specific implications are flagged inline where they bite.
>
> **REPRODUCIBILITY NOTE — stable seeding fix (applied 2026-05-19, scripts 19/19b/20 → v2).** Scripts 19/19b/20 originally used Python's built-in `hash()` to derive per-stimulus RNG seeds. `hash()` is per-process salted unless `PYTHONHASHSEED` is fixed, so the stimulus set was technically non-reproducible across runs. Replaced with `hashlib.blake2b`-based stable seeding (verified identical across processes with different `PYTHONHASHSEED` values). The 19b disk cache filename now includes a `_v2-stable-seeds` suffix and metadata fields (`stimulus_version`, `anchor_mode`, `canon_prompts_hash`, `inv_prompts_hash`, `dtype_before_cache`); the loader hard-rejects any cache that doesn't match. **Re-running 19b and 20 under the v2 seeds: most numbers shift by ≤ 0.5° or ≤ 1pp, but two boundary cases shifted enough to record**: (a) Gemma 2 L17 gate F→N moved from 0.644 (AMBIG, 0.006 below threshold) to 0.652 (PASS, 0.002 above) — the L17 same-layer pairing is therefore at the limit of the threshold-based gate verdict and a future revision should report gate accuracy with a bootstrap CI; (b) OLMo 2 L7 N→F invented unary mass moved from 20% (v1) to 40% (v2), and from "and 80% / necessarily 20%" to "and 60% / not 20% / necessarily 20%" in the per-canonical breakdown. OLMo 2 L7 N→F is therefore not a clean attractor; it is a *diffuse / noisy zone* where stimulus-set variation flips per-word predictions across canonicals. The L7 baseline finding ("0% unary in script 18") should be read as "0-40% unary depending on stimulus set, with no clean attractor canonical". All other §3.7 numbers are reproducible to within stimulus-sampling noise; the headline qualitative findings (L10 outcome (i), Gemma 2 catchment basin growth with depth, per-word arity non-tracking) are robust.

#### 3.7.1 Script 17: NEUTRAL replicates, FUNCTIONAL-PREFIX shows a non-monotonic per-layer trajectory

Headline numbers at the per-condition focus layer:

| Condition | OLMo 2 7B (focus L7) | Gemma 2 9B (focus L8 fixed-reference; condition-specific peak in parentheses) |
|---|---|---|
| NEUTRAL: probe CV / invented unary mass | 1.000 / 99.6% | 0.996 / 96.0% (peak 97.6% at L4) |
| FUNC-PFX: probe CV / invented unary mass | 1.000 / 100.0% | 1.000 / 40.0% (peak 100% at L2 and L16-17, trough 0% at L6-12) |

NEUTRAL replicates strongly. FUNCTIONAL-PREFIX at the fixed-reference layer 8 fails the pre-registered 80% replication threshold (40% < 80%) but the per-layer trajectory is non-monotonic with two local peaks of 100% unary mass at layers 2 and 16-17, with zero unary mass in between. This is qualitatively different from OLMo 2's monotonic rise-to-plateau (script 16, layers 4-16). Two additional observations made the FUNCTIONAL-PREFIX trajectory hard to interpret from script 17 alone:

- Within-condition probe CV accuracy on canonicals is 1.000 at every single layer 1-42 in Gemma 2 FUNC-PFX, consistent with the probe reading propagated previous-token identity at the operator-anchored position rather than structural arity.
- Per-word landings at the fixed-reference layer 8 split cleanly by last-subword identity (` q`, ` dren` → `not`; `usp`, `lex`, `ph` → `implies`), not by the invented word's role-in-test arity. Strong surface-form signature in what should have been a structural measurement.

These three patterns (peak CV at all layers, oscillating per-layer unary mass, last-subword-aligned per-word predictions at the artifact layer) are individually consistent with two mutually-exclusive hypotheses: (a) a within-condition probe-instrument artifact reading propagated token identity through the residual stream; (b) a genuine cross-model representational difference where Gemma 2 has a multi-stage arity-encoding mechanism with surface-feature leakage at intermediate layers. Script 18 was written to disambiguate.

#### 3.7.2 Script 18: four-diagnostic disambiguation

Four diagnostics applied to both models (full per-layer breakdown and raw landings in `experiments/outputs/18_*.log`). The headline numbers at each model's per-condition focus layer:

**Diagnostic A — cross-condition probe transfer.** Train a probe on canonical activations in one condition; evaluate on invented activations in the *other* condition. A real arity direction in residual stream transfers across notations; a within-condition probe overfitted to surface features does not.

| Direction | OLMo 2 7B | Gemma 2 9B |
|---|---|---|
| NEUTRAL-probe → FUNC-PFX invented (at L4 / L7) | **0.0% unary** (all 250 to `and`) | **100.0% unary** (all 250 to `necessarily`) |
| FUNC-PFX-probe → NEUTRAL invented (at L2/L7/L16) | **0.0% unary** (all 250 to `implies`) | 86.8% / 17% / 99.6% unary across the three FUNC-PFX peak layers |

OLMo 2's cross-condition transfer **fails in both directions** — the same residual-stream activations that one probe calls "unary" are called "binary" by the other-notation probe. Gemma 2's cross-condition transfer **succeeds at most layers** (the L4 NEUTRAL probe transfers cleanly to FUNC-PFX; the L2 and L16 FUNC-PFX probes transfer cleanly to NEUTRAL). The exception is Gemma 2 FUNC-PFX at the L7-L12 trough layers, where cross-transfer collapses to 12-17% unary — confirming that script 17's non-monotonic trajectory at those layers is a real surface-feature artifact within the FUNC-PFX condition rather than a structural feature.

**Diagnostic B — held-out canonical probe.** Train a probe on four canonicals; evaluate on the held-out fifth. A structural probe predicts the held-out canonical's arity-class partner(s); a surface-overfit probe predicts uniformly over the four trained classes (25% chance).

| Model / Condition | Focus layer | Held-out unary → other unary | Held-out binary → other binary |
|---|---|---|---|
| Gemma 2 / NEUTRAL | 4 | **97.0%** (≈4× chance) | 66.7% (2.7× chance) |
| Gemma 2 / FUNC-PFX | 2 | 50.0% (2× chance) | 66.7% (2.7× chance) |
| OLMo 2 / NEUTRAL | 7 | **100.0%** (4× chance) | 72.7% (2.9× chance) |
| OLMo 2 / FUNC-PFX | 7 | 50.0% (2× chance) | 66.7% (2.7× chance) |

Both NEUTRAL probes have strong structural arity signal under held-out generalisation. Both FUNCTIONAL-PREFIX probes have weaker (but still above-chance) structural signal, consistent with the within-condition FUNC-PFX probes being partially structural classifiers with substantial surface-feature contamination. The NEUTRAL probe is therefore the more reliable substrate-invariance instrument; the FUNC-PFX within-condition probe should be treated as a partial structural classifier and its results should be cross-checked with Diagnostic A transfer before being reported as substrate-invariance findings.

**Diagnostic C — probe-free geometric centroid delta.** For each invented word, compute cosine similarity to the unary-region centroid (mean of `not` and `necessarily` per-canonical centroids) minus cosine similarity to the binary-region centroid (mean of `and`, `or`, `implies`). Positive delta means the invented word is geometrically closer to unary canonicals.

| Model / Condition | Focus layer | Mean unary-binary delta | n / 5 closer to unary |
|---|---|---|---|
| Gemma 2 / NEUTRAL | 4 | +0.0093 | 5 / 5 |
| Gemma 2 / FUNC-PFX | 2 | +0.0068 | 5 / 5 |
| OLMo 2 / NEUTRAL | 7 | +0.0564 | 5 / 5 |
| OLMo 2 / FUNC-PFX | 7 | +0.0806 | 5 / 5 |

Every model × condition has a positive geometric arity attractor and 5/5 invented words sit closer to the unary centroid. The probe-free measurement confirms the arity attractor is real at the centroid-geometry level independently of any probe. OLMo 2's centroid attractor is in raw magnitude substantially stronger than Gemma 2's (≈ 6× in NEUTRAL, ≈ 12× in FUNC-PFX), which is a separate cross-model observation discussed in §3.7.4.

**Diagnostic D — last-subword embedding baseline.** For each invented word, find the canonical whose layer-0 embedding is closest to the invented word's last-subword embedding. Compare to the within-condition probe's per-word predictions at the focus layer. High match rate = last-subword identity is being propagated to the focus layer; low match rate = the focus-layer prediction is not a propagated-token-identity artifact.

For OLMo 2 the within-condition focus-layer predictions match last-subword embedding closeness for 2 of 5 invented words in NEUTRAL and 2 of 5 in FUNC-PFX. For Gemma 2 the match rate is 0 of 5 at the L2 NEUTRAL peak, 0 of 5 at the L16 FUNC-PFX peak, and 2 of 5 at the L8 fixed-reference (the artifact layer). The L2 and L16 Gemma 2 peaks are therefore *not* surface-token artifacts. The L8 trough behaviour is partially explained by last-subword leakage.

#### 3.7.3 The unified cross-model picture

Combining diagnostics A–D produces a single coherent interpretation, with explicit calibration of confidence at each Gemma 2 layer.

**Both models have within-notation arity-region attractors at the operator-anchored position in both notations.** All four (model × condition) combinations have 5/5 invented words geometrically closer to the unary centroid than to the binary centroid (Diagnostic C). The within-condition probe is a partially-structural classifier in both notations (Diagnostic B). The within-notation arity attractor is real and replicates across models.

**What differs across models is the cross-context stability of the arity direction.** The defensible cross-model contrast is:

- Gemma 2 9B has cross-condition-transfer-compatible unary/modifier catchment basins at validated early layers (NEUTRAL L4 and FUNC-PFX L2). At those layers, the probe direction trained on canonicals in one notation places invented operators in the unary region of the other notation, AND the same probe accurately classifies canonicals across the notation boundary (the canonical-transfer gate). Per-invented-word breakdowns (script 20) show invented operators are mapped to a single dominant unary canonical ("necessarily" at L4-L8 in N→F) rather than to the canonical matching their intended arity; this is therefore catchment-basin transfer rather than arity-respecting transfer.
- OLMo 2 7B's tested arity directions at L7 do not transfer across notations. The NEUTRAL probe direction (Diagnostic B-validated structural; 4× chance on held-out unary) does not transfer to FUNCTIONAL-PREFIX invented activations (0% unary mass, all to `and`). The reverse transfer also fails (0% unary, all to `implies`). Both within-condition probes are structural in their own notation (Diagnostic B), and both find arity-aligned centroid deltas (Diagnostic C, +0.0564 and +0.0806). But the two arity directions are not the same direction in residual stream space.

**Per-layer evidence calibration for Gemma 2** (with the canonical-transfer gate applied):

| Train / Test pairing | Cross-canonical 5-class accuracy | Invented unary mass | Status |
|---|---|---|---|
| NEUTRAL@L4 → FUNC-PFX@L4 | 1.000 | 100.0% | clean cross-notation arity transfer |
| FUNC-PFX@L2 → NEUTRAL@L2 | 0.756 | 86.8% | good but imperfect reverse support |
| FUNC-PFX@L2 → NEUTRAL@L4 | 0.672 | 88.8% | good but imperfect reverse support |
| FUNC-PFX@L8 → NEUTRAL@L* | — | 12-17% | non-transportable / surface-readout artifact |
| FUNC-PFX@L16 → NEUTRAL@L16 | 0.564 | 99.6% | candidate late re-emergence (ambiguous) |
| FUNC-PFX@L16 → NEUTRAL@L4 | 0.200 | 100.0% | candidate late re-emergence (chance-level canonical transfer) |

The L8 row in this table is the methodological asset of script 18: the within-condition probe at L8 is excellent (CV 1.000) but its direction does not transfer to NEUTRAL canonicals or to NEUTRAL invented activations. Within-condition probe success is therefore *not by itself* a substrate-invariance signal — cross-condition transfer with a canonical-transfer gate is required. The L16 row is the calibration warning the reviewer flagged: the invented unary mass is high (99.6-100%) but the underlying cross-canonical transfer is at or near chance (0.564 within-layer, 0.200 cross-layer to L4), so the "100% unary at L16" claim partly reflects decision-boundary bias rather than a clean transferable arity representation. The defensible Gemma 2 cross-model story is therefore: a clean early cross-notation arity-transfer stage (NEUTRAL L4 ↔ FUNC-PFX L2-L4), a non-transportable middle (L6-L12), and a candidate late re-emergence (L16-17) requiring stronger canonical-transfer validation before it can be claimed.

**The script 17 fixed-reference layer 8 in Gemma 2 is the artifact layer (methodologically useful).** Diagnostic A and Diagnostic D together explain it: the within-condition probe at L8 reads propagated last-subword identity (Diagnostic D, 2/5 match; per-word predictions split by ` q`, ` dren`, `usp`, `lex`, `ph` rather than by intended arity), and the resulting probe direction does not cross-transfer to NEUTRAL invented activations (12-17%). Choosing L8 because it matched OLMo 2's depth-fraction was the wrong heuristic; Gemma 2's validated arity-encoding layer for cross-context transfer is L4 in NEUTRAL or L2 in FUNC-PFX, not L8. The L8 result is reframed as the cleanest single illustration of why cross-condition transfer is the gold-standard substrate-invariance measurement (§3.7.5).

#### 3.7.4 Cross-model observations beyond the headline

Two observations worth flagging beyond the headline cross-context-stability finding:

1. **OLMo 2's geometric arity attractor is in raw magnitude stronger than Gemma 2's.** The centroid delta is +0.0564 vs +0.0093 in NEUTRAL (6× larger in OLMo 2), and +0.0806 vs +0.0068 in FUNC-PFX (12× larger). This is not yet interpretable without normalisation (the two models have different residual-stream norms and different baseline cosine geometries), and the directional-angle follow-up (script 19) is intended to add a scale-invariant comparison. Pending that, the qualitative pattern is robust: OLMo 2's unary canonicals form a tighter cluster relative to its binary canonicals than Gemma 2's do. This is reminiscent of the well-known observation that smaller / less-trained models often have geometrically "sharper" categorical clusters than larger / better-trained ones (e.g., Heimersheim & Turner 2023 on the role of "anti-monosemantic" superposition with scale), but a cleaner causal account is not yet available.
2. **Gemma 2's per-layer FUNC-PFX trajectory is suggestively non-monotonic but should be characterised carefully.** The within-condition unary-mass curve has peaks at L2 and L16-17 with intervening near-zero values; cross-canonical-transfer gate analysis (§3.7.3) accepts L2 cleanly, marks L16 as ambiguous, and rejects L6-L12. A "Gemma does multi-stage arity processing" story is plausible (early-construction stage at L2, middle surface-feature stage L6-L12, candidate late re-emergence at L16-17) but the late-stage claim is currently weak. Phase 1 follow-up: stronger canonical-transfer validation of the L16 stage and / or causal intervention by activation patching at L2 vs L16 to determine whether the two stages are *the same* arity direction reasserted late or *a different* arity-class encoding that happens to land in the same unary-canonical decision region. Gemma 2 9B uses alternating-block attention (every other layer is sliding-window); the L2/L16 rough periodicity is suggestive of an attention-pattern-modulated mechanism, but within-block detail does not cleanly track the SWA / full-attention alternation. Flagged for the Phase 1 → Phase 2 mechanistic trace.

#### 3.7.5 Methodological lesson: cross-condition probe transfer with a canonical-transfer gate is the gold-standard substrate-invariance instrument

Five probe-instrument failure modes have now been characterised across Phase 0 and Phase 1 entry:

1. **Within-condition probes can hit CV=1.000 at every layer in functional-prefix notation** (Gemma 2 FUNC-PFX in script 17; partial in OLMo 2 FUNC-PFX in script 16). Propagated previous-token identity at the operator-anchored position dominates the residual stream, so a probe can always find a separating hyperplane regardless of structural content. Within-condition probe accuracy is therefore not by itself a substrate-invariance signal in functional-prefix notation.
2. **Within-condition probes can be structural in one notation and non-transferable to another** (OLMo 2 NEUTRAL probe → 0% unary on FUNC-PFX invented activations, script 18 Diagnostic A). A probe direction that is structural in its own notation is not necessarily the *same* direction as the structural-in-that-other-notation probe. Cross-condition probe transfer is required to distinguish the two cases.
3. **High invented-unary mass under cross-condition transfer is not by itself evidence of transferable arity structure if the same train/test pairing has chance-level canonical 5-class transfer accuracy.** The Gemma 2 FUNC-PFX@L16 → NEUTRAL@L4 pairing in script 18 has 100% invented unary mass but cross-canonical accuracy 0.200 — at or near chance for 5-class classification. The high unary mass at that pairing therefore partly reflects decision-boundary bias rather than a cleanly transferable arity representation. A canonical-transfer gate is required to distinguish the two.
4. **The canonical-transfer gate can pass asymmetrically: one direction at near-1.0, the other at near-chance** (OLMo 2 L7 in script 19b: NEUTRAL@L7 → FUNC-PFX@L7 canonical-transfer accuracy 1.000; reverse direction 0.212, at 5-class chance). The NEUTRAL@L7 probe finds features that exist in FUNC-PFX@L7 (gate-PASS one way), but the FUNC-PFX@L7 probe finds features that don't exist in NEUTRAL@L7 (gate-FAIL the other way). The underlying mechanism is *representational asymmetry*: the NEUTRAL probe at L7 encodes broader / more shared features (e.g., abstract operator-class information); the FUNC-PFX probe at L7 encodes narrower / more surface-specific features (propagated previous-token identity, position-specific patterns) that don't have NEUTRAL counterparts. Asymmetric gate failure is therefore not bilateral orthogonality but a one-way feature subset relation. A bidirectional gate check (both train/test directions, not just one) is required to detect this case.
5. **The canonical-transfer gate can PASS while the binary unary-vs-binary direction remains cross-notation-wide and invented-unary mass remains low** (OLMo 2 L10 in script 20, the cleanest illustration: gate N→F = 0.800 PASS, gate F→N = 0.688 PASS; centroid 71.3°, probe 74.4° — wide; invented unary mass at L10 same-layer = 0.0% / 17.2%, mean 8.6%, *below* the 40% by-arity random baseline, with N→F invented predictions 100% "implies" — the invented words are actively binary-classified). The 5-class canonical probe distinguishes individual operators (`and`, `or`, `not`, `implies`, `necessarily`) by some feature combination that transfers across notations, but the specific axis encoding *unary-vs-binary as a class* is not aligned across notations and invented words do not land in the unary region. The 5-canonical-discrimination axis and the binary-arity axis are distinct geometric objects; transferability of one does not imply transferability of the other.
6. **Invented unary mass can be high without arity-respecting classification: the "necessarily catchment basin" effect** (Gemma 2 NEUTRAL → FUNC-PFX cross-notation transfer in script 20: L2 invented unary mass = 40% (split NOT-attractor and implies-attractor); L4 = 80% (mostly "necessarily"); L8 = 100% (all "necessarily")). The "catchment basin" expands monotonically with depth in the N→F direction, *anti-correlated* with directional-angle alignment (L2 is tightest at 47° but invented mass is lowest at 40%; L8 is widest at 67° but invented mass is maximal at 100%). The per-invented-word breakdown at L2 shows invented words land in unary by tokenization / subword cues, not intended arity (intended-binary `bliq` → predicted NOT 100%; intended-unary `vusp` → predicted implies 100%). High invented unary mass under cross-notation transfer therefore measures *catchment basin width into a specific unary canonical* rather than *cross-condition arity-respecting classification*. The right cross-notation arity-transfer measurement requires either (a) gate-PASS + tight directional angle + invented mass *distributed across unary canonicals* (not single-canonical-dominated), or (b) a held-out-canonical generalisation test on the *invented* set (Diagnostic B extended).

Phase 1 finding: **no single measurement is sufficient. The substrate-invariance instrument is now a four-measurement battery (M1-M4), with M4 itself split into three sub-measurements (M4a-M4c) after the script 20 peer review identified that invented unary mass alone conflates three distinct geometric questions.** The battery is:

| | Measurement | What it establishes | Sufficient by itself? |
|---|---|---|---|
| **M1** | Within-condition probe CV accuracy | Can a probe classify canonical operators inside one notation? | No — Phase 0 baseline only; no longer treated as substrate-invariance evidence in isolation |
| **M2-canonical** | Bidirectional 5-class canonical-transfer gate | Does a 5-class canonical-operator classifier survive the notation shift in *both* directions? | No — necessary but not sufficient (OLMo 2 L10 illustration); also dissociates from M2-arity when the binary-canonical-identity substructure does not survive the notation shift even though the arity-class axis does (see §3.7.10) |
| **M2-arity** | Bidirectional binary-vs-unary gate, coarsened from the same 5-class probe | Does the *arity-class* membership of canonicals (binary vs unary) survive the notation shift, separately from whether the specific canonical identity is preserved? | No, but is the appropriate gate for the *arity-respecting* transfer claim (introduced post-script-22a after OLMo close-paren L10 dissociation). Chance-by-arity floor is ~0.60 (the lucky-default predict-all-one-binary-canonical baseline), so M2-arity ≥ 0.65 is the defensible threshold. |
| **M3** | Cross-notation directional angle on the binary unary-vs-binary axis (centroid + probe; bootstrap 95% CIs) | Are the unary-vs-binary axes geometrically aligned in residual stream coordinates across notations? | No — Gemma 2 L2 has the tightest angle but only modest invented mass |
| **M4a** | Invented unary mass | How much of the invented population enters the unary region under cross-condition readout? | No — high mass can be a single-canonical catchment basin |
| **M4b** | Intended-arity agreement | Does each invented word land in the arity class it was assigned to replace (e.g., `bliq` intended-binary → predicted "and"/"or"/"implies")? | No — but the *strongest* arity-respecting test |
| **M4c** | Canonical catchment concentration | Is the invented mass distributed across unary canonicals or collapsed into a single label (Herfindahl-like)? | No — but the single-canonical-collapse vs distributed distinction is the cleanest catchment-basin diagnostic |

**The headline claim** has been refined post-script-22a into two distinct conjunctions:

- **Cross-notation arity-respecting transfer** = M2-arity-PASS + M3-tight + M4a-central + M4b-tracks-intended + M4c-distributed (binary-vs-unary axis transfers; canonical identity may or may not).
- **Cross-notation canonical-identity transfer** = M2-canonical-PASS (5-class) + M3-tight + M4-distributed across multiple canonicals (the same canonical identities are preserved across notations).

The §3.7.9 candidate cell (OLMo 2 close-paren L10 N→F) satisfies the *arity-respecting* conjunction (M2-arity = 1.000 PASS, M4b = 90%, M4a = 30% central, M4c = 0.57 distributed) but fails the *canonical-identity* conjunction (M2-canonical = 0.616 AMBIG with bootstrap-PASS probability only 7%; or, implies → and collapse). This is the project's first positive cross-notation arity-respecting transfer result; it is not a positive canonical-identity transfer result. Both at the operator-anchored position used in scripts 17-20, no tested (model, layer) pairing satisfies either conjunction.

| Pairing | M2 (gate) | M3 (angle) | M4a (mass) | M4b (agreement) | M4c (concentration) |
|---|---|---|---|---|---|
| Gemma 2 L4 N→F | PASS 1.000 / 0.956 | 55° | 80% high | FAILS — `bliq` intended-binary → "necessarily"; 4 of 5 invented → "necessarily" | COLLAPSED — single canonical 80% (Herfindahl ≈ 0.68) |
| Gemma 2 L2 N→F | PASS 1.000 / 0.884 | 47° tight | 40% moderate | FAILS — `bliq` intended-binary → NOT 100%; `vusp` intended-unary → implies 100% | distributed across NOT (40%) and implies (60%) |
| Gemma 2 L8 N→F | PASS 1.000 / 0.864 | 67° wide | 100% maximal | FAILS — all 5 invented → "necessarily" | COLLAPSED — single canonical 100% (Herfindahl = 1.0) |
| OLMo 2 L7 N→F | PASS but asym (1.000/0.212) | 75° wide | 20% low | FAILS — invented → "and" 80% | collapsed (single binary) |
| OLMo 2 L10 N→F | PASS 0.800 / 0.688 | 73° wide | 0% floor | FAILS — invented → "implies" 100% | COLLAPSED — single binary canonical 100% |

**Empirical vindication of the M4 split (post-script-21).** Script 21 produces two cells where M4b looks high (≥ 0.70) in isolation but fails the conjunction: OLMo 2 operator-after L7 N→F at M4b = 80% has M4a = 20% (floor) and M4c = 0.68 (collapsed to "and"); Gemma 2 first-arg L8 N→F at M4b = 73.6% has M4a = 36.8% and M4c = 0.51 with the per-word predictions defaulting to "and" 100% for 3 of 5 invented words. In both cases the high M4b is a *coincidental* match between the model's default-canonical prediction and the intended arities of the invented-word set (3 of 5 invented words are intended-binary and the model's default for "unknown operator" happens to be the binary "and"). The M4a/M4b/M4c separation introduced in §3.7.5 after the script 20 peer review is the exact instrument that detects this — M4b in isolation would have triggered a false-positive "cross-notation arity transfer detected" claim. The OLMo 2 close-paren L10 N→F cell (§3.7.9) is structurally different: M4b = 90% with M4a = 30% (central, not floor not ceiling) AND M4c = 0.57 (mass distributed across "and" + "necessarily", not collapsed) AND per-word predictions tracking intended arity (4 of 5 invented words cleanly correct, 5th genuinely split). This is the conjunction the M4 split was designed to flag as a candidate arity-respecting transfer.

The Phase 1 finding is therefore reframed as: **within-condition arity-region attractors are cross-model robust (both OLMo 2 and Gemma 2 reproduce them); cross-notation arity-respecting transfer (M2+M3+M4a+M4b+M4c-distributed) is currently demonstrated only as a borderline candidate at OLMo 2 close-paren L10 N→F (M2 = 0.616 AMBIG, M4b = 90%); at the operator-anchored position used in scripts 17-20 it has not been demonstrated in any tested model at any tested layer**. The "at the operator-anchored position" qualifier is load-bearing: in functional-prefix notation that anchor lands *before* the argument list, so a causal LM cannot have integrated the call-structure evidence required for invented-operator arity inference at this position (see CAVEAT BOX at the top of §3.7). The post-call-anchor re-test in script 21 is required to determine whether the negative result on M4b survives once the model has had access to the argument list, or whether the script 17-20 measurement was systematically blind to a cross-notation arity-respecting representation that exists at later anchor positions. This is the principal open empirical question.

#### 3.7.6 Directional-angle quantification (script 19)

Script 19 (`19_directional_angle_analysis.py`) directly measures the geometric question raised by script 18's cross-condition probe transfer asymmetry: at the operator-anchored position, how aligned in residual stream space is the arity direction in NEUTRAL with the arity direction in FUNCTIONAL-PREFIX, in each model? Two operationalisations per layer, both unit-normalised in raw residual stream coordinates:

- **Centroid-based unary direction.** `mean(unary canonical centroids) - mean(binary canonical centroids)`. Probe-free.
- **Probe-based unary direction.** Unit-normalised weight of a binary (unary-vs-binary) logistic-regression probe trained on raw canonical activations.

Cosine angle between the two condition-specific directions is computed for both operationalisations. Random-unit-vector baseline (200 Gaussian-sampled directions at the layer's dim) lands at 90.02 ± 0.99° for Gemma 2's 3584-dim and 90.02 ± 0.84° for OLMo 2's 4096-dim — i.e., a precisely-orthogonal direction in this hidden dim is 90° on the nose.

**Headline per-layer table:**

| Model | Layer | Centroid angle | Probe angle | Centroid cos | Probe cos |
|---|---|---|---|---|---|
| Gemma 2 9B | L2 | **50.83°** | **45.88°** | +0.63 | **+0.70** |
| Gemma 2 9B | L4 (script-17/18 focus) | 56.46° | 54.33° | +0.55 | +0.58 |
| Gemma 2 9B | L8 (artifact layer) | 67.98° | 66.81° | +0.37 | +0.39 |
| Gemma 2 9B | L16 (script-18 "candidate") | 63.89° | 70.41° | +0.44 | +0.34 |
| Gemma 2 9B | L17 | 66.98° | 70.58° | +0.39 | +0.33 |
| OLMo 2 7B | L4 | 74.14° | 71.36° | +0.27 | +0.32 |
| OLMo 2 7B | L7 (focus) | 75.56° | 74.94° | +0.25 | +0.26 |
| OLMo 2 7B | L10 | 71.32° | 74.43° | +0.32 | +0.27 |
| OLMo 2 7B | L16 | 74.78° | 75.86° | +0.26 | +0.24 |
| OLMo 2 7B | L24 | 74.92° | 72.18° | +0.26 | +0.31 |

Within-notation centroid-vs-probe sanity check (probe direction and centroid direction agree on what arity is within each notation) ranges 18–35° at most layers in both models, confirming that the angular measurements compare the same underlying arity geometry. Centroid and probe cross-notation angles agree to within 1–7° in both models, validating the angular measurement.

**Four findings:**

**(a) The reviewer's calibrated framing is empirically supported, with explicit numbers.** Gemma 2's best cross-notation cosine is +0.70 (L2 probe direction); OLMo 2's best is +0.32. Both models are well below the random baseline of cos 0 (so neither is fully orthogonal), but Gemma 2's cross-notation alignment is ≈ 2× tighter in cosine terms and ≈ 20° tighter in angular terms. The cross-condition probe transfer asymmetry in script 18 (Gemma 2 100% / OLMo 2 0% invented unary mass) is therefore underwritten by a real geometric difference, but the binary "globally aligned vs notation-local" framing oversimplifies. The defensible cross-model statement is: **Gemma 2 has substantially tighter (but not perfect) cross-notation alignment of its arity direction than OLMo 2 has, with a clean early sweet-spot layer and a monotonic decay of cross-notation alignment with depth**.

**(b) Gemma 2 L2 is the cross-notation sweet spot, not L4.** The script-17 focus layer (L4) was a defensible choice but L2 has materially better cross-notation alignment (probe cos +0.70 vs +0.58; centroid cos +0.63 vs +0.55). This is consistent with script 17's "FUNC-PFX layer-2 peak" and explains why script 18's reverse-direction transfer FUNC-PFX@L2 → NEUTRAL succeeded so cleanly (86.8% invented unary mass with cross-canonical accuracy 0.756). The clean early cross-notation arity stage in Gemma 2 is L2; L4 is a slightly fading residue of that signal. The script-17/18 narrative of a "NEUTRAL @ L4 focus layer" should be retained as a defensible-but-suboptimal choice; future Gemma 2 work on this question should treat L2 as the canonical early-construction layer.

**(c) The L16 "candidate late re-emergence" calibration concern is fully validated empirically.** Gemma 2 L16 cross-notation probe cosine is +0.33–0.34 — barely higher than OLMo 2's notation-local cos +0.26 at L16. The script-18 "99.6% invented unary mass at L16" is now exposed as exactly what the peer reviewer warned about: a decision-boundary bias from a probe that is only marginally cross-notation-aligned. The Gemma 2 per-layer trajectory in this directional-angle measurement is *monotonically decreasing cross-notation alignment with depth* (L2 best, L17 worst), not biphasic. A "two transferable arity stages" claim is therefore not defensible from the current data. The "candidate late re-emergence" framing of §3.7.3 is correct and should not be upgraded.

**(d) OLMo 2 7B is remarkably flat across the full tested depth.** Cross-notation alignment is cos +0.24–0.32 from L4 through L24. No layer has a "sweet spot". The notation-local arity directions in OLMo 2 are a property of the entire post-early-layer residual stream, not just L7. The minimum cross-notation angle in OLMo 2 (L10 centroid 71.32°) is essentially identical to the median, so there is no useful "if-we'd-picked-a-different-OLMo-layer" alternative.

**Implications for the cross-model claim.** OLMo 2 is *not* at the random baseline (cos +0.25–0.32, ~ 20° below random), so it has some non-trivial cross-notation alignment — just not enough for the script-18 NEUTRAL probe's labels to land in the unary class on FUNC-PFX activations. This implies a threshold around cos 0.4–0.5 at which probe-label cross-transfer flips from "0% unary" to "100% unary". A third model (Pythia, planned next) would help characterise where this threshold actually sits, and whether models cluster bimodally around it or sit on a smooth continuum.

#### 3.7.7 Gated directional-angle analysis with bootstrap CIs (script 19b)

Script 19b (`19b_directional_angle_gated.py`) extends script 19 with three additions: (i) canonical-transfer gate column in both directions at every layer pair, (ii) bootstrap 95% CIs on centroid and probe angles via 100 within-class resamples per pairing, and (iii) cross-layer pairings that reproduce the script-18 critical-pairings table with directional angles layered on top. Disk-caches activations to `experiments/outputs/cache/` so re-runs on the same model/condition skip extraction.

**Headline cross-model table (same-layer focus-layer pairing):**

| Model | Focus L | NEUT CV | FUNC CV | Gate N→F | Gate F→N | Centroid deg (95% CI) | Probe deg (95% CI) | Verdict |
|---|---|---|---|---|---|---|---|---|
| Gemma 2 9B | L4 | 1.000 | 1.000 | 1.000 | 0.956 | 58.0 [56.8, 60.4] | 55.3 [53.8, 57.3] | **PASS** |
| OLMo 2 7B | L7 | 0.996 | 1.000 | 1.000 | **0.212** | 76.0 [75.0, 77.3] | 75.4 [74.6, 76.3] | **FAIL** |

The CIs do not overlap on either centroid or probe angle. The cross-model contrast is statistically tight, with PASS in Gemma 2 and FAIL in OLMo 2 at their respective focus layers. Six findings refine the picture from scripts 17–19:

**(a) OLMo 2 L7 fails the gate asymmetrically (= new failure mode 4 above).** The gate is PASS one direction (1.000), FAIL the other (0.212, at 5-class chance for OLMo 2's tested layers). The "OLMo 2 has notation-local arity directions" claim from §3.7.3 / §3.7.4 survives, but the *mechanism* is now sharper: NEUTRAL@L7 features partially exist in FUNC-PFX@L7 (so a NEUTRAL probe identifies FUNC-PFX canonicals), but FUNC-PFX@L7 features do not exist in NEUTRAL@L7 (so the FUNC-PFX probe is at chance on NEUTRAL). The "notation-local" framing should be amended to "asymmetric notation-local": OLMo 2 has a one-way feature subset relation at L7, not bilateral orthogonality.

**(b) OLMo 2 L10 is a previously-unidentified bidirectionally-gate-passing layer.** Same-layer pairing gate: N→F 0.800, F→N 0.688 — both PASS. This is the only OLMo 2 layer in the tested set (L4, L7, L10, L16, L24) where both gate directions PASS. Cross-layer FUNC-PFX@L10 → NEUTRAL@L7 also passes (0.664). Directional angles at L10 are still wide (centroid 71.8°, probe 74.7°), so by failure mode 5 above the gate-PASS is necessary but not sufficient for invented cross-transfer to work; the invented-unary mass at OLMo 2 L10 is *not yet measured*. This is the principal open empirical question at Phase 1 entry and is being closed by script 20 (`20_gated_invented_mass.py`).

**(c) Gemma 2 L8 same-layer test is direction-asymmetric (refined by script 20).** Same-layer gate at L8: N→F 1.000, F→N 0.864 — both PASS. Centroid 68.9°, probe 67.4° — wide. Script 20 invented unary mass: **N→F = 100% (all "necessarily"); F→N = 23.6%** — asymmetric in invented mass, not bidirectionally low. The F→N direction is the gate-PASS+angle-wide+invented-mass-LOW failure mode (= failure mode 5 of §3.7.5); the N→F direction is the necessarily-catchment-basin effect (= failure mode 6 of §3.7.5). Both directions exemplify why gate-PASS is not sufficient, but for different reasons in each direction. The script-18 reframing of L8 as "the cleanest single illustration of why cross-condition transfer is the gold-standard measurement" needs further revision: L8 illustrates that *each direction* of cross-condition probe transfer can fail for different reasons even when both directions pass the gate.

**(d) Gemma 2 L17 falls to AMBIG (gate F→N = 0.644, 0.006 below the 0.65 threshold).** The script-17 "L16-17 second peak" picture is now: L16 same-layer PASS but with wide angles (centroid 64.5°, probe 70.1°); L17 same-layer AMBIG. The late-stage cross-notation arity transfer claim weakens further under the gated analysis.

**(e) Gemma 2 FUNC-PFX@L16 → NEUTRAL@L4 cross-layer is AMBIG with 82° angles.** Gate 0.620 (just below threshold), centroid 82.6° [81.9, 83.5], probe 81.7° [81.3, 82.3] — 8° below the random baseline. This emphatically retracts the script-18 "100% invented unary mass at this pairing" finding: the directional alignment is almost orthogonal and the gate fails. The L16→L4 100% from script 18 was decision-boundary bias, exactly as the peer-review caution predicted.

**(f) Bootstrap CIs are tight (2-6° wide across all measurements), confirming the script 19 point estimates were robust.** Within-class resampling does not destabilise either centroid or probe direction estimation at N_PER_CLASS=50. The Gemma 2 ↔ OLMo 2 cross-model separation is statistically tight (no CI overlap on the focus-layer comparison). This rules out the possibility that the script 19 cross-model difference was an artifact of stimulus selection.

**Updated cross-model framing.** The four-claim picture from §3.7.6 is refined as:

1. **Both models reproduce the within-condition arity-region attractor in both notations** (script 17 confirmed; carried forward).
2. **Gemma 2 has a clean cross-notation unary-catchment stage at L2 / L4 with both gate-PASS and tight directional alignment** (L4 gate verdict PASS, centroid 58°, probe 55°; L2 has tighter probe angle 47° but only L4 is the focus layer for the headline comparison). This is the cleanest cross-notation unary-catchment transfer case in the project; script 20 shows it is not *arity-respecting* transfer (the invented mass is dominated by "necessarily" rather than tracking per-word intended arity).
3. **OLMo 2 L7 (the long-standing focus layer) fails the gate** (asymmetric failure mode 4), and OLMo 2 L10 is a candidate gate-passing layer with bidirectional PASS but wide directional angles (failure mode 5 territory). Whether OLMo 2 has *any* cross-notation arity transfer at L10 is the next empirical question.
4. **Gemma 2 L8 / L16 / L17 fall in the "necessary-but-not-sufficient" zone**: gate-PASS or near-PASS but directional-angle-wide. The script-18 invented-unary mass at L16→L4 (100%) was decision-boundary bias; the L16 same-layer (99.6%) is a closer call (L16 is bidirectionally gate-PASS at 0.800 / 0.728, with centroid 64.5° / probe 70.1° — wider than L4 but not as wide as L8). A causal intervention (activation patching along the L4 arity axis) would resolve whether L16 carries any independent arity information beyond a decision-boundary shadow of the L4 axis.

The cross-model headline that survives all of scripts 17–19b: **Gemma 2 has gate-PASS + tight-angle cross-notation arity transfer at L4 (the clean case); OLMo 2 has gate-FAIL at L7 (asymmetric) and a candidate gate-PASS at L10 with wide angles (still notation-local-ish even where the gate passes)**. Script 20 closes the L10 invented-mass question and the L8 invented-mass question; the cross-model headline survives but is refined further (§3.7.8).

#### 3.7.8 Gated invented-mass re-test (script 20)

Script 20 (`20_gated_invented_mass.py`) re-runs cross-condition probe transfer (script 18 Diagnostic A: train 5-class probe on source canonicals, predict on target invented activations, report unary mass) at the specific gate-passing pairings that 19b identified as the principal open questions. Uses the 19b disk cache exclusively — no model load, runs in ~6 s.

**Headline finding 1: OLMo 2 L10 outcome (i) — gate-PASS, angle-wide, invented mass at floor.**

| OLMo 2 7B pairing | Gate acc | Centroid° | Probe° | Invented unary mass | Dominant predicted canonical |
|---|---|---|---|---|---|
| L7 N→F (script-18 baseline) | 1.000 PASS | 75.6° | 74.9° | 20-40% (seed-sensitive)¹ | "and" 60-80%, with minor "not" / "necessarily" mass — diffuse |
| L7 F→N (script-18 baseline) | 0.21-0.28 FAIL² | 75.6° | 74.9° | 0.0% | "implies" 100% |
| **L10 same-layer N→F** | **0.800 PASS** | **71.3°** | **74.4°** | **0.0%** | **"implies" 100%** |
| **L10 same-layer F→N** | **0.688 PASS** | **71.3°** | **74.4°** | **17.2%** | "implies" 57%, "or" 26%, "not" 17% |
| L10 → L7 N→F (cross-layer) | 0.800 PASS | 80.0° | 80.2° | 0.0% | "implies" 100% |
| L10 → L7 F→N (cross-layer) | 0.664 PASS | 78.4° | 78.4° | 2.8% | "implies" 97% |

Mean L10 same-layer invented unary mass = **8.6%** — *below* the 40% random-by-arity baseline (2/5 canonicals are unary). The OLMo 2 L10 N→F probe actively *binary-classifies* invented words (100% "implies"). The "notation-local arity direction in OLMo 2" framing is therefore reinforced, not softened: even at the OLMo 2 best gate-passing layer L10, the binary unary-vs-binary axis does not transfer, and invented words land in a single binary canonical ("implies"). The gate-PASS at OLMo 2 L10 represents a 5-class canonical-discrimination axis (the 5 canonicals are individually recognisable across notations) but the specific axis encoding unary-vs-binary as a class is not aligned. This is the cleanest demonstration in the project so far that **gate-PASS is necessary but not sufficient** (= failure mode 5 of §3.7.5; OLMo 2 L10 replaces the earlier-claimed Gemma 2 L8 as the cleanest illustration).

¹ The OLMo 2 L7 N→F invented unary mass shifts noticeably across stimulus-seed regimes: script 18 ≈ 0%, script 20 v1 (unstable seeds) 20%, script 20 v2 (stable seeds) 40%. The per-canonical breakdown shifts in parallel: under v2 the 40% unary mass is split "and" 60% + "not" 20% + "necessarily" 20%, with per-word predictions splitting `bliq` → NOT 100%, `perph` → necessarily 100%, others → and. L7 in OLMo 2 N→F is therefore a *diffuse zone* with no clean canonical attractor under cross-notation probe transfer; the 0% → 40% range across seeds tracks the binary canonical that absorbs the dominant fraction (always "and") rather than a switch in unary-canonical preference. The headline "OLMo 2 L7 does not transfer to unary" is reproducible only in its weaker form: *no clean unary attractor exists at L7 N→F; the probe distributes invented activations across canonicals with a binary-canonical-dominant ("and") pattern that is consistent across seeds*.

² The OLMo 2 L7 F→N gate accuracy is consistently FAIL (0.21–0.28 across script 18 / script 20 v1 / v2) and the invented predictions are consistently 100% "implies"; this is the asymmetric-gate-failure direction described as failure mode 4 in §3.7.5, and it is robust to stimulus seed.

**Headline finding 2: Gemma 2 has a "necessarily catchment basin" that grows monotonically with depth in N→F (= failure mode 6 of §3.7.5).**

| Gemma 2 9B pairing | Gate acc | Centroid° | Probe° | Invented unary mass | Per-canonical breakdown |
|---|---|---|---|---|---|
| L4 N→F (baseline) | 1.000 PASS | 56.5° | 54.3° | **80.0%** | "necessarily" 80%, "and" 20% |
| L2 same-layer N→F | 1.000 PASS | 50.8° | 45.9° | **40.0%** | "not" 40%, "implies" 60% |
| L2 same-layer F→N | 0.884 PASS | 50.8° | 45.9° | 42.0% | "not" 24%, "necessarily" 18%, "implies" 57% |
| L8 same-layer N→F | 1.000 PASS | 68.0° | 66.8° | **100.0%** | "necessarily" 100% |
| L8 same-layer F→N | 0.864 PASS | 68.0° | 66.8° | 23.6% | "and" 38%, "implies" 39%, "not" 7%, "necessarily" 16% |

The invented unary mass and the directional-angle alignment are *anti-correlated* across the three tested points in the Gemma 2 N→F direction (tighter angle L2 at 47° → lower invented mass 40%; wider angle L8 at 67° → higher invented mass 100%). The per-canonical breakdown is suggestive of a mechanism: as depth increases, the *catchment basin* of the "necessarily" class appears to expand. This is currently only three data points in one transfer direction, so it should be described as a suggestive monotone pattern rather than a confirmed mechanism. A plausibly related Phase 0 effect (H4, the NEUTRAL-template pull toward "necessarily") may be contributing, but a stronger test requires (a) more tested depths with a denser scan, (b) the reverse direction (FUNC-PFX → NEUTRAL) for symmetry, and (c) additional unary canonicals with different lexical / grammatical profiles to disambiguate "logical-unary catchment" from "generic-modifier catchment" — the canonical "necessarily" has broad modal/adverbial usage in natural language and could be acting as a generic modifier attractor rather than as a clean logical-unary canonical.

**Headline finding 3: per-invented-word breakdown shows invented words do not follow intended arity, at any tested layer in either model.**

The W_TO_CANONICAL mapping assigns each invented word an intended canonical (and thus intended arity): `bliq`→and (binary), `dren`→or (binary), `vusp`→not (unary), `molex`→implies (binary), `perph`→necessarily (unary). Script 20's per-invented-word per-pairing detail shows the predicted canonical does *not* track this intended mapping at any tested (model, layer) pair:

- Gemma 2 L2 N→F: bliq (intended-binary) → NOT 100%; vusp (intended-unary) → implies 100%; perph (intended-unary) → NOT 100% (one correct out of three checked).
- Gemma 2 L4 N→F: 4 of 5 words → necessarily; dren (intended-binary) → and (the only non-necessarily prediction).
- Gemma 2 L8 N→F: 5 of 5 words → necessarily (no arity signal).
- OLMo 2 L10 N→F: 5 of 5 words → implies (no arity signal).

The invented words are mapped to canonicals by subword tokenization / structural-position cues (Phase 0 H3 finding), not by their intended arity. This means the "invented unary mass" measurement is not a substrate-invariance measurement in the strict sense (it does not measure whether the model recognises the intended arity of each invented word); it is a *catchment basin* measurement (it measures the fraction of invented activations that fall into the unary cluster *as a whole*, regardless of whether the within-unary distribution matches intended arity).

**Caveat on NEUTRAL vs FUNC-PFX as tests of arity-respecting transfer.** Intended arity is *expressed in the prompt only in functional-prefix notation*: `bliq(p, q)` makes binary arity visible to the model via the argument list; `vusp(p)` makes unary arity visible via the single argument. In NEUTRAL notation ("Consider the word bliq.") there is no in-prompt arity evidence — the intended-arity mapping exists only in the experimental design (`W_TO_CANONICAL`: `bliq` → `and`, etc.). Failure to track intended arity in NEUTRAL is therefore not a failure of *context-grounded arity induction*; it is a failure of *lexeme-mapping transfer* (does the model preserve the arbitrary `bliq` → `and` assignment given no in-context evidence?). These are different claims and should be reported separately:

- **Context-grounded arity transfer test** (FUNC-PFX → FUNC-PFX, or NEUTRAL → FUNC-PFX with FUNC-PFX as the test condition): does `bliq(p, q)` land in a binary canonical and `vusp(p)` land in a unary canonical, when the model has seen the argument list? *Only this test is a fair structural-substrate-invariance test.*
- **Lexeme-mapping transfer test** (FUNC-PFX → NEUTRAL, or any pairing where NEUTRAL is the test condition): does the model preserve the experimenter-imposed mapping `bliq` → `and` in a context with no arity cue? This is a weaker test more akin to novel lexical binding than to structural substrate-invariance.

The script 20 M4b "FAILS at every tested pairing" finding spans both test types but does not separate them. The headline negative claim — *cross-notation arity-respecting transfer has not been demonstrated* — is therefore strongest only for the context-grounded subset. The script 21 post-call anchor re-test should report context-grounded and lexeme-mapping results separately, with the former being the primary substrate-invariance instrument.

**Caveat on anchor visibility (compounding with the above).** In FUNC-PFX, the operator-anchored position used in scripts 17-20 lands on the `(` token *before* the argument list. A causal LM has not integrated argument-count information at this position. The script-20 M4b failures in the FUNC-PFX direction are therefore *not yet* clean evidence against context-grounded arity transfer, because the measurement may have been taken before the model had access to the relevant in-context evidence. Script 21 (post-call anchor re-test, completed) addresses this directly by measuring the full M1-M4 battery at four FUNC-PFX anchors (operator-after, first-arg, close-paren, sentence-final) and two NEUTRAL anchors (operator-after, sentence-final). The script 21 result, summarised in §3.7.9, softens the §3.7.8 negative-result headline: it identifies a *candidate* cross-notation arity-respecting transfer pattern at OLMo 2 close-paren L10 NEUTRAL → FUNC-PFX (M4b = 90%, with the per-word predicted canonical tracking intended arity for 4 of 5 invented words and the mass split across "and" (binary, 70%) + "necessarily" (unary, 30%)), but with the M2 canonical-transfer gate at AMBIG (0.616, just below the 0.65 PASS threshold) and the F→N direction failing. The negative-result headline is NOT retracted — it requires bootstrap calibration of the M2 threshold and replication with an expanded invented-word set and additional unary canonicals — but it is softened from "not demonstrated at any pairing" to "candidate borderline transfer at one (model, anchor, layer, direction) cell".

**Implication for the substrate-invariance instrument.** The §3.7.5 finding that the canonical-transfer gate is "necessary but not sufficient" is now refined into a four-measurement battery (see updated §3.7.5 paragraph). The cross-model claim survives but is sharper:

- Both OLMo 2 and Gemma 2 reproduce the *within-condition arity-region attractor* — invented words in NEUTRAL templates land in the unary region (Phase 0).
- Gemma 2 has *cross-notation transfer of a generic-unary/modifier catchment basin* at L4-L8, dominated by "necessarily" and growing monotonically with depth in the N→F direction across three tested points (L2 = 0% necessarily, L4 = 80% necessarily, L8 = 100% necessarily; the depth-monotone pattern is suggestive but is currently only three data points in one transfer direction). This is real cross-notation directional alignment plus a catchment-basin effect plausibly related to the Phase 0 H4 template-context finding; "necessarily" is a unary logical canonical but also a generic modal/adverbial modifier in natural language, so the catchment may be acting at the generic-modifier level rather than at the strict logical-unary level. Disambiguating this requires adding more unary canonicals with different lexical / grammatical profiles (e.g., `possibly`, `always`, `negate`, `is_false`); flagged as a Phase 1 → Phase 2 follow-up.
- OLMo 2 has gate-FAIL asymmetrically at L7 and gate-PASS-but-empty-of-invented-mass at L10. No tested OLMo 2 pairing shows above-baseline, bidirectional, arity-respecting invented transfer; even the L10 bidirectional gate-passing layer sends invented items mostly or entirely to a single binary canonical ("implies").
- *Cross-notation arity-respecting transfer* (each invented word correctly assigned to its intended arity class) has NOT been demonstrated cleanly in any tested model at any tested layer at the operator-anchored position; **a candidate borderline pattern emerges at OLMo 2 close-paren L10 N→F when the anchor is moved post-arguments (see §3.7.9)** but requires bootstrap M2 calibration and replication before being upgraded from candidate to confirmed.

The thesis claim is therefore reframed: **cross-model arity-region attractors exist within each notation, cross-notation arity-respecting transfer is the principal open question, and the answer depends sensitively on the residual stream readout position; a single (model, anchor, layer, direction) cell shows a candidate arity-respecting pattern but at a borderline gate accuracy that requires further validation**. This is a stronger thesis position than the previous "cross-model contrast" framing because it (a) identifies a falsifiable open question that the field has not yet addressed, (b) shows that positional readout is a first-class methodological variable, and (c) gives a specific candidate cell to attack with follow-up experiments.

#### 3.7.9 Multi-anchor M1-M4 battery (script 21)

Script 21 (`21_multi_anchor_battery.py`) re-runs the full M1-M4 battery at four FUNC-PFX anchors (`operator-after` = baseline matching scripts 17-20; `first-arg`; `close-paren`; `sentence-final`) and two NEUTRAL anchors (`operator-after`, `sentence-final`). One forward pass per stimulus extracts all layer activations at every anchor position; cache shape `(n_anchors, n_stim, n_layers, dim)` at fp16; STIMULUS_VERSION = `v3-multi-anchor`. Stable-seed stimulus generation reused from 19b v2 via importlib. Focus layers identical to 19b's diagnostic layers (Gemma 2: L2/L4/L8/L16/L17; OLMo 2: L4/L7/L10/L16/L24). Compute: 322s for Gemma 2 + 234s for OLMo 2 on MPS; total 556s.

**Three principal findings:**

**(1) The §3.7.8 negative-result headline is anchor-bound.** The script-20 conclusion ("cross-notation arity-respecting transfer not demonstrated at any pairing") was measured exclusively at the operator-after anchor. Script 21 confirms that conclusion *at the operator-after anchor* across both models and all focus layers (M4b is at or below the random-by-arity baseline of ~52% in all operator-after cells where M4a is non-trivial). But the same measurement at later anchors gives a different picture:

| Best M4b across all script-21 cells | Model | Anchor | L | Direction | M4b | M4a | M4c | M2 gate | M2 verdict |
|---|---|---|---|---|---|---|---|---|---|
| #1 | OLMo 2 7B | close-paren | 10 | N→F | **90.0%** | 30.0% | 0.57 | 0.616 | AMBIG |
| #2 | OLMo 2 7B | operator-after | 7 | N→F | 80.0%* | 20.0% | 0.68 | 1.000 / 0.212 | FAIL (asym) |
| #3 | Gemma 2 9B | first-arg | 8 | N→F | 73.6%* | 36.8% | 0.51 | 1.000 / 0.200 | FAIL (asym) |
| #4 | OLMo 2 7B | close-paren | 16 | N→F | 71.6%* | 11.6% | 0.79 | 0.768 / 0.268 | FAIL (asym) |

\* = "lucky-default" artifact (see §3.7.5 callout): the model is defaulting to a single binary canonical and one of the invented words happens to look modifier-y, so M4b is high without arity discrimination.

The #1 row (OLMo 2 7B close-paren L10 N→F) is qualitatively different from #2-#4. Per-invented-word breakdown at this cell:

| word | intended arity | predicted top canonical | unary share per word | arity match? |
|---|---|---|---|---|
| bliq | binary | "and" | 0% | ✓ binary→binary |
| dren | binary | "and" | 0% | ✓ binary→binary |
| vusp | unary | "necessarily" | 100% | ✓ unary→unary |
| molex | binary | "and" | 0% | ✓ binary→binary |
| perph | unary | "and" 50% / "necessarily" 50% | 50% | partial |

4 of 5 invented words land cleanly in the correct arity class; the 5th (perph) is genuinely split 50/50. The mass is distributed across TWO canonicals ("and" 70% binary, "necessarily" 30% unary) rather than collapsing into one (Herfindahl 0.57, vs single-canonical-collapse value of 1.0). This is **the cleanest per-word arity-respecting pattern observed in the project to date**. It is the strongest single piece of evidence so far for cross-notation arity-respecting transfer.

**Three caveats kept this a candidate rather than a confirmed finding initially. Script 22a (§3.7.10) addresses the first two:**

- M2 canonical-transfer gate is at 0.616 (AMBIG, 0.034 below the 0.65 PASS threshold). A bootstrap CI on the gate accuracy is needed before the verdict is defensible — the cell is at the gate threshold and threshold-based verdicts are fragile there (see §3.7 reproducibility note for the parallel Gemma 2 L17 gate flip). **[Resolved in §3.7.10: bootstrap CI [0.432, 0.682], P(M2-canonical ≥ 0.65) = 7.0%. The strict 5-class M2 gate is robustly below threshold, NOT a borderline-PASS. However, M2-arity = 1.000 (perfect binary-vs-unary classification across notations), and the §3.7.5 substrate-invariance battery has been updated to use M2-arity as the appropriate gate for arity-respecting transfer claims.]**
- The F→N direction at the same pairing FAILS cleanly: M2 = 0.332, M4a = 17.2%, M4b = 48.4%, M4c = 0.42. The candidate transfer is unidirectional. **[Confirmed in §3.7.10: F→N reverse direction bootstrap M2-canonical = 0.330 (CI [0.308, 0.346]), M2-arity = 0.544 (only +0.044 above the 0.50 by-arity floor for predict-all-not-which-happens-here). The transfer is genuinely unidirectional.]**
- The result exists at *exactly one* (model, anchor, layer, direction) cell across the 80-cell battery. Without replication on an expanded invented-word set or with additional unary canonicals, it could be a 5-word sampling fluke. **[Open: scripts 22b (full anchor sweep at OLMo 2 with M2-arity scoring) and 22c (expanded invented + canonical sets) queued.]**

**(2) "Lucky-default" cells reveal a real methodological hazard.** The M4b = 80% at OLMo 2 operator-after L7 N→F looks superficially high but the per-word breakdown shows the mechanism is *default-to-"and"-with-one-escape*:

| word | predicted | arity match? |
|---|---|---|
| bliq (B) | "and" 100% | ✓ |
| dren (B) | "and" 100% | ✓ |
| vusp (U) | "and" 100% | ✗ MISMATCH |
| molex (B) | "and" 100% | ✓ |
| perph (U) | "necessarily" 100% | ✓ |

3 of 4 binary-intended words land in "and" (matching by default, not by discrimination) and 1 of 2 unary-intended escapes to "necessarily" (matching by lexical resemblance / subword cue). M4a = 20% (low), M4c = 0.68 (collapsed to "and"), M2 FAILS asymmetrically (1.000 / 0.212). This is the exact failure mode the M4a/M4b/M4c split (introduced in §3.7.5 after the script 20 peer review) was designed to detect: **M4b is sensitive to coincidental alignment between the model's default-canonical prediction and the intended arities of the invented-word set**. Gemma 2 first-arg L8 N→F (M4b = 73.6%) has the same signature: M4a = 36.8%, M4c = 0.51, M2 FAILS asymmetrically; 3 of 3 binary-intended → "and", 1 of 2 unary-intended → "necessarily", same lucky-default pattern.

**Implication.** M4b in isolation is not a sufficient signal for arity-respecting transfer. The full conjunction is needed: M4b high AND M4a in the central range (not floor, not ceiling) AND M4c < ~0.7 (mass not collapsed into one canonical) AND M2 PASS bidirectionally. The OLMo 2 close-paren L10 N→F cell uniquely satisfies M4b high + M4a central + M4c low + M2 borderline among all 80 script-21 cells.

**(3) Per-anchor catchment basins differ within the same (model, layer).** The dominant catchment canonical changes across anchors at the same model and layer, indicating that different positional readouts read from *different* representations rather than the same representation queried at different points. Gemma 2 at L4:

| anchor | dominant canonical | invented unary share |
|---|---|---|
| operator-after | "necessarily" | 100% (all 5 → necessarily) |
| first-arg | "and" 12-34% | low (mostly binary) |
| close-paren | **"not"** | 100% (all 5 → not) |
| sentence-final | "implies" 0-26% | low (mostly binary) |

The Gemma 2 operator-after / close-paren basin asymmetry (necessarily-basin at operator-after, not-basin at close-paren) is striking. The same model at the same layer produces two distinct unary-catchment basins at two different positions in the same call. This is consistent with the H1 / H4 Phase 0 picture — different residual stream positions encode different aspects of the operator's role — but is the cleanest demonstration yet that the positional readout is itself shaping the apparent arity geometry. Anchor choice is a first-class methodological variable.

OLMo 2 at the close-paren anchor shows a depth-dependent basin migration:

| layer | dominant predicted canonical | profile |
|---|---|---|
| L4 | "not" (100% for 4 of 5 words) | unary catchment basin |
| L7 | "not" (100% for 4 of 5 words) | unary catchment basin |
| L10 | "and" + "necessarily" split | **arity-respecting candidate** |
| L16 | "or" (0% unary for 4 of 5 words) | binary catchment basin |
| L24 | "or" (0% unary for all 5 words) | binary catchment basin |

L10 is the *only* OLMo 2 close-paren layer with mass distributed across both arities; L4-L7 are unary-collapsed and L16-L24 are binary-collapsed. This makes the L10 candidate look structurally special: it sits at a transition between two collapse regimes. Whether the L10 mass-distribution is the "real" arity representation or just a layer-of-balance between two competing collapse mechanisms is the principal question for follow-up.

**Net effect on the headline.** The §3.7.8 negative-result headline is *softened* but not retracted:

- *Before script 21*: "Cross-notation arity-respecting transfer has not been demonstrated in any tested model at any tested layer, at the operator-anchored position."
- *After script 21*: "Cross-notation arity-respecting transfer has not been cleanly demonstrated, but a candidate borderline pattern emerges in OLMo 2 at the close-paren anchor, layer 10, NEUTRAL → FUNC-PFX direction only. M4b = 90% (4 of 5 invented words track intended arity) with mass distributed across "and" (binary) and "necessarily" (unary). M2 canonical-transfer gate is at 0.616, AMBIG. The candidate is unidirectional, gate-borderline, and exists at one cell in an 80-cell battery. Confirmation requires (a) bootstrap CI on M2 at this cell, (b) replication with an expanded invented-word set, and (c) additional unary canonicals to disambiguate the 'necessarily as generic modifier' from logical-unary."
- *After script 22a*: "Cross-notation arity-respecting transfer **is demonstrated under the M2-arity gate** (M2-arity = 1.000, P = 100% PASS under bootstrap) at OLMo 2 close-paren L10 N→F, but **canonical-identity transfer is not** (M2-canonical = 0.616, bootstrap-PASS probability only 7%). The 5-class canonical probe trained at NEUTRAL sentence-final maps FUNC-PFX 'or' and 'implies' activations into the 'and' decision region (within-arity confusion), but preserves the binary-vs-unary axis perfectly. This is a partial-transfer finding: arity-respecting cross-notation transfer is real but it is *not* the same as canonical-identity cross-notation transfer. The two gates dissociate at exactly this cell; both transfer at the alternative training anchor (NEUT op-after) but at the cost of catchment collapse to 'necessarily' on invented words. Validation queued: expanded invented + canonical sets."

#### 3.7.10 Bootstrap M2 gate + M2-arity introduction (script 22a)

Script 22a (`22a_bootstrap_m2_gate.py`) puts confidence intervals on the M2 canonical-transfer gate accuracy at the script-21 candidate and control cells using stim-resampling bootstrap (B = 500, with-replacement resampling within each canonical class on the training side, fixed test set). The script reads script 21's `v3-multi-anchor` cache; no model loading is required. Runtime ~2.5 min. Two extensions were added during the run after the primary result revealed a deeper structural finding:

1. **M2-arity bootstrap**: train the same 5-class canonical probe used for M2, then score predictions on the coarsened binary-vs-unary partition. Tests whether arity-class membership transfers across the notation shift separately from canonical identity.
2. **M4 re-computation at each cell's actual train anchor**: script 21 used a fixed anchor mapping (FUNC-PFX `close-paren` ↔ NEUTRAL `sentence-final`). Script 22a additionally tests NEUTRAL `operator-after` as the training anchor for the same FUNC-PFX `close-paren` test position — an alternative readout choice that turns out to give a structurally different result.

**Cells tested and bootstrap results:**

| Cell | M2-canonical (95% CI; P PASS) | M2-arity (95% CI) | M4a / M4b / M4c | Per-word predicted top canonical (% unary share) |
|---|---|---|---|---|
| **§3.7.9 PRIMARY: OLMo (NEUT sf → FUNC cp) L10 N→F** | 0.616 [0.432, 0.682]; **7.0% PASS** | **1.000** [0.800, 1.000] | 30.0% / **90.0%** / 0.57 | bliq→and(0%U), dren→and(0%U), vusp→necessarily(100%U), molex→and(0%U), perph→and(50%U) |
| ALT-TRAIN: OLMo (NEUT op-aft → FUNC cp) L10 N→F | 0.924 [0.800, 1.000]; **100% PASS** | 0.924 [0.800, 1.000] | 100.0% / 40.0% / 1.00 | all 5 → necessarily(100%U) — catchment basin |
| REV: OLMo (FUNC cp → NEUT sf) L10 F→N (neg control) | 0.332 [0.308, 0.346]; 0% PASS | 0.544 [0.516, 0.560] | 96.0% / 40.8% / 0.92 | all 5 → not(94-98%U) |
| REV: OLMo (FUNC cp → NEUT op-aft) L10 F→N | 0.396 [0.376, 0.408]; 0% PASS | 0.500 [0.488, 0.512] | 100.0% / 40.0% / 0.95 | all 5 → not(100%U) |
| POS: OLMo (NEUT op-aft → FUNC op-aft) L10 N→F | 0.800 [0.800, 0.800]; 100% PASS | 0.800 [0.800, 0.800] | 0.0% / 60.0% / 1.00 | all 5 → implies(0%U) |
| BOUNDARY: Gemma op-aft L17 F→N (19b v2 PASS at 0.652) | 0.640 [0.604, 0.684]; **53.6% PASS** | 0.860 [0.856, 0.868] | 99.2% / 40.0% / 0.98 | all 5 → not(98-100%U) |
| LUCKY-NEG: OLMo (op-aft) L7 F→N | 0.212 [0.212, 0.212]; 0% PASS | 0.612 [0.612, 0.612] | 0.0% / 60.0% / 1.00 | all 5 → implies(0%U) |
| LUCKY-NEG: Gemma (first-arg→sf) L8 F→N | 0.200 [0.200, 0.204]; 0% PASS | 0.600 [0.600, 0.604] | 0.0% / 60.0% / 1.00 | all 5 → implies(0%U) |

**Three principal findings:**

**(1) The §3.7.9 candidate's strict M2-canonical gate is robustly AMBIG, not borderline.** Bootstrap mean 0.602, std 0.062, 95% CI [0.432, 0.682], P(M2 ≥ 0.65) = 7.0%. The CI does extend across the threshold but the mass is concentrated below it. This is *not* a "just barely missed PASS" — the strict 5-class canonical-transfer at this cell is robustly below the threshold. **The §3.7.9 candidate is therefore not upgraded to "PASS within statistical noise" on M2-canonical.**

**(2) But M2-arity = 1.000 at the same cell.** The confusion matrix from the 5-class probe trained at NEUTRAL sentence-final and tested on FUNC-PFX close-paren shows the structure of the partial transfer:

```
                  predicted ->  and    or    not   implies  necessarily
true canonical:
  and             50    0     0     0        0           (correct)
  or              50    0     0     0        0           (or  -> and: WRONG, within-arity confusion)
  not              0    0    50     0        0           (correct)
  implies         46    0     0     4        0           (implies -> and: mostly WRONG, within-arity)
  necessarily      0    0     0     0       50           (correct)
```

All errors are within-arity (or → and, implies → and; both binary→binary). Zero cross-arity errors. The probe has lost the binary-canonical-identity substructure across the notation shift but has perfectly preserved the binary-vs-unary axis. M2-arity = (50+50+50+50+50)/250 = 1.000.

This is the project's first piece of direct evidence that **arity-axis transfer is structurally distinct from canonical-identity transfer**, and the two can dissociate at a single cell.

**(3) The training-anchor choice changes the readout fundamentally.** Same test activations (FUNC-PFX close-paren L10 invented words), two different NEUTRAL training anchors:

| | NEUTRAL `sentence-final` train (§3.7.9 cell) | NEUTRAL `operator-after` train (ALT-TRAIN cell) |
|---|---|---|
| M2-canonical | 0.616 AMBIG | 0.924 PASS |
| M2-arity | 1.000 perfect | 0.924 PASS (necessarily collapse causes 19 unary errors) |
| Invented unary mass (M4a) | 30.0% (split) | 100.0% (all "necessarily") |
| Per-word agreement (M4b) | 90.0% (4 of 5 correct arity) | 40.0% (catchment collapse) |
| Mass concentration (M4c) | 0.57 (distributed) | 1.00 (single canonical) |
| Per-word reading | bliq, dren, molex → "and" (binary); vusp → "necessarily" (unary); perph → split | all 5 → "necessarily" 100% |

The same FUNC-PFX close-paren residual stream is read by two probes trained on different NEUTRAL anchors. The operator-after-trained probe sees a "necessarily catchment basin" (all invented words → "necessarily" 100%); the sentence-final-trained probe sees an arity-respecting decomposition (binary-intended → "and", unary-intended → "necessarily"). **The model's residual stream at FUNC-PFX close-paren L10 evidently contains both a fine-grained canonical-identity axis (visible to the operator-after-trained probe) and a coarse arity axis (visible to the sentence-final-trained probe); which axis dominates the readout depends on which NEUTRAL position the probe was trained at.**

This is consistent with a multi-axis representation hypothesis: late in OLMo 2 (L10), the model maintains parallel axes for both fine-grained canonical identity and coarse arity, and the choice of training anchor selects which axis the cross-notation transfer reads from. The arity axis transfers cleanly across notations (M2-arity = 1.000); the canonical-identity axis transfers cleanly only when both train and test anchors are at canonical-rich positions (M2-canonical PASS only at op-aft → op-aft, op-aft → close-paren).

**(4) Lucky-default cells have M2-arity ≈ 0.60 — the chance-by-arity floor.** When the probe defaults to a single binary canonical (M2-canonical FAILS at 0.20), M2-arity becomes (50+50+50)/250 = 0.60 by accidentally matching the 3 binary intended classes. This sets the **defensible M2-arity threshold at ≥ 0.65** (slightly above the predict-all-one-binary-canonical chance floor); the §3.7.9 cell at M2-arity = 1.000 is 0.40 above this floor.

**(5) The Gemma 2 L17 boundary case is genuinely 50/50 under bootstrap.** P(M2-canonical ≥ 0.65) = 53.6%, 95% CI [0.604, 0.684]. M2-arity at the same cell = 0.860, which is well above the chance-by-arity floor. The Gemma 2 L17 cell is therefore a *clean* M2-arity PASS but a *threshold-fragile* M2-canonical case — same dissociation pattern as the §3.7.9 cell, but at smaller magnitude.

**Net effect on the headline (after script 22a):**

- *Before script 22a*: "Cross-notation arity-respecting transfer is demonstrated only as a candidate borderline at OLMo 2 close-paren L10 N→F (M2 = 0.616 AMBIG, M4b = 90%)."
- *After script 22a*: "Cross-notation arity-respecting transfer is **demonstrated under the M2-arity gate** at OLMo 2 (NEUTRAL sentence-final → FUNC-PFX close-paren) L10 N→F: M2-arity = 1.000 (CI [0.800, 1.000]), M4b = 90% (4 of 5 invented words track intended arity), mass distributed across 'and' (binary) and 'necessarily' (unary). Cross-notation canonical-identity transfer is **not** demonstrated at this cell (M2-canonical = 0.616 AMBIG, P(M2 ≥ 0.65) = 7%); within-arity confusions (or → and, implies → and) are responsible for the canonical-identity gap. The arity axis and the canonical-identity axis are structurally distinct in OLMo 2's residual stream at this layer, and the choice of NEUTRAL training anchor selects which axis the cross-notation readout reads from."

The original M2 split into M2-canonical + M2-arity (added to §3.7.5) is the principal methodological contribution of this script. Validation in script 22b (full anchor × layer sweep, see §3.7.11) extends the analysis to all 160 cells across both models; the §3.7.9 cell is confirmed unique-strongest and three additional PASS-arity cells are identified.

#### 3.7.11 Full anchor × layer sweep (script 22b)

Script 22b (`22b_anchor_layer_sweep.py`) runs the full M1-M4 + M2-arity battery at every (NEUTRAL train anchor × FUNC-PFX test anchor × focus layer × direction) cell across OLMo 2 7B and Gemma 2 9B. The sweep enumerates 160 cells (2 models × 5 focus layers × (2 NEUT × 4 FUNC) N→F + (4 FUNC × 2 NEUT) F→N = 16 anchor pairings/layer/model). Cache-only against the script-21 v3-multi-anchor activations; runtime ~2 min. Each cell reports M1 (within-cond CV at train and test), M2-canonical (5-class accuracy), M2-arity (binary-vs-unary accuracy coarsened from the same 5-class probe), M3 (centroid and probe arity-direction cosine angles), M4a (invented unary mass), M4b (intended-arity agreement), M4c (canonical catchment concentration / Herfindahl), per-word top canonical with within-word concentration, and per-word minimum top concentration (`pwmin`).

**Lucky-default detector refinement.** The script's initial verdict logic used M4c ≥ 0.85 as the lucky-default Herfindahl threshold, but this missed the *escape* pattern where 4 of 5 invented words land at one canonical (100% per-word concentration) and 1 invented word escapes to a different canonical (also 100% per-word concentration), giving M4c = 0.8² + 0.2² = 0.68 — below the original threshold but mechanistically still a lucky-default. The refined detector uses **min(per_word_top_pct) ≥ 0.95**: a cell is lucky-default if every invented word's predictions are essentially deterministic on a single canonical. The §3.7.9 cell escapes this detector because perph has within-word top concentration of 0.50 (genuine split between "and" and "necessarily"). The refined detector reclassified 4 of 8 originally-flagged PASS-arity cells as lucky-default-escapes (3 OLMo, 1 Gemma).

**Headline counts (post-refinement):**

| Verdict | Count | Meaning |
|---|---|---|
| PASS-arity | **4** | M2-arity ≥ 0.65 + M4b ≥ 0.65 + M4c < 0.70 + M4a ∈ [0.10, 0.90] + not lucky-default |
| ARITY-AXIS-ONLY | 1 | M2-arity + M4b PASS but M4c collapsed (binary catchment) |
| Lucky-default | 40 | min per-word concentration ≥ 0.95, default-canonical-plus-escape pattern |
| FAIL | 115 | At least one M2/M3/M4 criterion below threshold |

**The 4 PASS-arity cells:**

| Rank | Cell | Model | Direction | Train→Test anchor | Layer | M2-cano | M2-arity | M4a | M4b | M4c | pwmin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | §3.7.9 | OLMo 2 7B | N→F | sente → close | L10 | 0.616 | **1.000** | 30.0% | **90.0%** | 0.57 | 0.50 |
| 2 | new (L8) | Gemma 2 9B | N→F | sente → first | L8 | 1.000 | 1.000 | 36.8% | 73.6% | 0.51 | 0.66 |
| 3 | new (L4) | Gemma 2 9B | N→F | sente → opera | L4 | 1.000 | 1.000 | 57.6% | 72.8% | 0.51 | 0.76 |
| 4 | new (F→N) | OLMo 2 7B | F→N | first → opera | L7 | **0.980** | 0.984 | 55.2% | 65.6% | **0.35** | 0.36 |

**Per-cell mechanism:**

1. **OLMo 2 7B N→F sente→close L10 (§3.7.9 cell, confirmed unique strongest).** Same per-word pattern as documented in §3.7.10: bliq/dren/molex → "and" 100% (binary), vusp → "necessarily" 100% (unary), perph → "and" 50% / "necessarily" 50% (genuine within-word split that distinguishes this cell from lucky-default-escapes). Confusion-matrix mechanism: probe trained at NEUTRAL sentence-final maps FUNC-PFX close-paren "or" and "implies" canonicals into the "and" decision region, but preserves binary-vs-unary cleanly. **The §3.7.9-cell-vs-all-alternatives diagnostic confirms no challenger has both higher M2-arity AND higher M4b.**

2. **Gemma 2 9B N→F sente→first L8 (new finding).** Per-word: bliq/dren → "and" 66-76% binary; molex → "and" 88% binary; vusp → "necessarily" 92% unary; perph → "and" 66% (MISMATCH — intended-unary predicted binary). 4 of 5 words match intended arity in their top canonical; per-word concentrations range 66-92% (distributed within-word, distributed across canonicals). M4c = 0.51 — well distributed. The Gemma 2 cross-notation arity transfer claim from scripts 17-18 is **reinstated** for this specific anchor pairing — it was lost in scripts 19/20 because the operator-after anchor at L8 collapses to a "necessarily" catchment basin, but the (sente train → first-arg test) pairing preserves the arity-respecting transfer.

3. **Gemma 2 9B N→F sente→opera L4 (Gemma reinstatement, complementary cell).** Per-word: bliq → "and" 88% binary (12% unary leakage); molex → "and" 100% binary; vusp → "necessarily" 76% unary; perph → "necessarily" 100% unary; **dren → "necessarily" 100% (MISMATCH — intended-binary predicted unary).** 4 of 5 words match intended arity, dren is the failure (it's the second binary-intended invented word, with subwords that may pull it into the "necessarily" basin). M4c = 0.51 — distributed; M4a = 57.6% — central. **This is the strongest Gemma 2 transfer cell observed in the project, supersedes the L4 operator-after finding from scripts 17-18.**

4. **OLMo 2 7B F→N first→opera L7 (NEW: first F→N PASS, first dual M2 PASS).** Per-word: bliq/dren → "implies" 54-58% (binary, with 42-44% unary leakage); molex → "implies" 60%; vusp → "necessarily" 36% top but 70% unary share; perph → "necessarily" 66% top, 82% unary share. **All five words have within-word top concentrations 36-66% — most distributed of any cell in the sweep (M4c = 0.35).** 4 of 5 words match intended arity in their top canonical. **Critical: this is the first cell where M2-canonical AND M2-arity BOTH PASS (0.980 / 0.984).** The canonical-identity axis and the arity axis transfer simultaneously at this cell, in the F→N direction — the reverse of the §3.7.9 direction and a direction that previous scripts had never observed positive results in. The per-word pattern is also qualitatively different from the §3.7.9 cell: noisy but on-average arity-respecting, rather than decisive arity assignment with one genuine partial case.

**Structural commonality across all 4 PASS-arity cells.** None of them use NEUT `operator-after` as the training anchor. Three use NEUT `sentence-final` (a "post-call" position where the entire NEUTRAL stimulus context has been integrated); one uses FUNC `first-arg`. **The training anchor matters as much as the test anchor.** The operator-after anchor on NEUTRAL is consistently associated with lucky-default catchment basins (40 of the 160 cells are lucky-default, with the operator-after-trained pairings dominating that count). Training at a post-call position preserves the integrated arity-class representation; training at the operator-after position learns a representation dominated by the operator token itself, which collapses to single-canonical decision regions on cross-notation transfer.

**Per-(layer, direction) verdict grid.** The full grid (see script-22b output for both models) shows that PASS-arity cells are concentrated at:
- OLMo 2 sentence-final-trained pairings in N→F (most L10 cells; one ARITY-AXIS-ONLY at L16)
- OLMo 2 first-arg-trained pairings in F→N (only L7; PASS does not generalise to other OLMo layers in F→N)
- Gemma 2 sentence-final-trained pairings in N→F at L4 and L8

**No PASS-arity cell exists at OLMo 2 L4 (any direction), Gemma 2 L17 (any direction without lucky-default), or any Gemma 2 F→N pairing.** The transfer is layer-specific, direction-specific, and anchor-specific.

**Largest M2-canonical / M2-arity dissociation gaps (excluding lucky-default cells):**

| Cell | M2-cano | M2-arity | Gap | Mechanism |
|---|---|---|---|---|
| Gemma 2 N→F opera→sente L2 | 0.252 | 0.652 | +0.400 | arity-axis transfers; canonical-identity within-arity confusion |
| OLMo 2 N→F sente→close L10 (§3.7.9) | 0.616 | 1.000 | +0.384 | as documented in §3.7.10 |
| OLMo 2 F→N first→sente L16 | 0.308 | 0.692 | +0.384 | arity-axis transfers; canonical-identity within-arity confusion |

The dissociation pattern (M2-arity high, M2-canonical low, mechanism = within-arity confusion) **generalises beyond the §3.7.9 cell**. There are at least 3 cells where the binary-vs-unary axis transfers cleanly but the binary-canonical-identity axis collapses — though only the §3.7.9 cell also has the M4b + M4c + M4a + non-lucky-default conjunction needed for the full PASS-arity verdict.

**Net effect on the Phase 1 headline (after script 22b):**

- *Before script 22b*: "Cross-notation arity-respecting transfer is demonstrated under the M2-arity gate at one cell: OLMo 2 (NEUTRAL sentence-final → FUNC-PFX close-paren) L10 N→F."
- *After script 22b*: "Cross-notation arity-respecting transfer is demonstrated under the M2-arity gate at **four cells across both models**, all using post-call training anchors: (i) OLMo 2 N→F sente→close L10 (§3.7.9 cell, unique strongest, M4b = 90%); (ii) Gemma 2 N→F sente→first L8 (M4b = 73.6%); (iii) Gemma 2 N→F sente→opera L4 (M4b = 72.8%); (iv) OLMo 2 F→N first→opera L7 (M4b = 65.6%, with simultaneous M2-canonical PASS at 0.980 — the first observed cell with both axes transferring). The Gemma 2 cross-notation arity transfer claim from scripts 17-18 is reinstated for the sente→first L8 and sente→opera L4 pairings; the operator-after-anchor results in scripts 17-20 were anchor-bound and missed these cells. Structural commonality: training at a post-call position (NEUT sentence-final or FUNC first-arg) is necessary for arity-respecting transfer; training at the operator-after position consistently produces lucky-default catchment basins."

The §3.7.9 cell remains the headline because (a) it has the highest M4b (90% vs 65.6-73.6%) and (b) its per-word reading is qualitatively cleaner (decisive arity assignment with one genuine partial case, rather than noisy on-average arity assignment with within-word leakage). But it is no longer a singular finding — it is the strongest example of a phenomenon that generalises across both models at three other cells.

**Open questions for scripts 22c (expanded invented set) and 22d (expanded canonical set):**

- Does the 4-of-5 invented-word arity-tracking pattern persist with 15-20 invented words? If yes, the finding is real; if it drops sharply, the 4-of-5 pattern was a 5-word sampling artifact.
- Does the OLMo 2 F→N first→opera L7 simultaneous-M2-canonical-PASS survive expanded canonicals? With 5 canonicals it is bootstrap-confirmed (CI [0.964, 0.992], §3.7.12); expanded canonicals will reveal whether the fine-grained canonical-identity axis is robust across the expanded set or specific to the 5-word canonical partition.
- Does the Gemma 2 sente→opera L4 transfer survive added unary canonicals (possibly, always, negate)? Point-perfect M2-canonical = 1.000 with bootstrap CI [1.000, 1.000] (§3.7.12); the dren mismatch (intended-binary → "necessarily" 100%) suggests dren has subword affinity for "necessarily" — with more unary canonicals, the mismatch may distribute (good for the transfer claim, as it shows the model is using a broad unary representation) or sharpen onto "necessarily" specifically (revealing word-specific subword binding).

#### 3.7.12 Bootstrap CIs on the four PASS-arity cells (script 22a extension)

The point estimates from the script 22b sweep identified four PASS-arity cells; their dual-PASS status under bootstrap stim-resampling (500 resamples, the same protocol as §3.7.10) is needed for the cells to be defensible as headline findings rather than point-estimate-fragile cells. Script 22a was extended to add three new candidate cells alongside the original §3.7.9 cell. Cache-only run (~9 min, all activations reused from script 21 cache `v3-multi-anchor`).

| Cell | Direction | Point M2-canonical | Bootstrap M2-canonical (95% CI) | P(≥ 0.65) | Point M2-arity | Bootstrap M2-arity (95% CI) | Verdict |
|---|---|---|---|---|---|---|---|
| **OLMo 2 N→F sente→close L10 (§3.7.9)** | N→F | 0.616 | [0.432, 0.682] | 7.0% | **1.000** | [0.800, 1.000] | M2-arity PASS, M2-canonical robustly AMBIG (as in §3.7.10) |
| **OLMo 2 F→N first→opera L7** | F→N | **0.980** | **[0.964, 0.992]** | **100.0%** | **0.984** | [0.980, 0.996] | **DUAL PASS confirmed (M2-canonical AND M2-arity)** |
| **Gemma 2 N→F sente→opera L4** | N→F | **1.000** | **[1.000, 1.000]** | **100.0%** | **1.000** | [1.000, 1.000] | **DUAL PASS confirmed, point-perfect** |
| **Gemma 2 N→F sente→first L8** | N→F | **1.000** | [0.888, 1.000] | **100.0%** | **1.000** | [0.956, 1.000] | **DUAL PASS confirmed** |

**Headline:** The three additional PASS-arity cells identified by the script 22b sweep are *all* confirmed dual-PASS (M2-canonical AND M2-arity) under bootstrap stim-resampling. The §3.7.9 cell remains the unique M2-arity-only cell — perfect M2-arity transfer with M2-canonical robustly below threshold, the dissociation pattern documented in §3.7.10.

**Implications:**

1. **Four PASS-arity cells, three of them dual-PASS.** The §3.7.9 cell's M2-arity-only profile is *not* the prevailing pattern. The three new cells all transfer both axes simultaneously — fine-grained 5-class canonical identity AND coarse binary-vs-unary arity. Cross-notation substrate-invariance for canonical identity is therefore *demonstrated* at three cells across both models, not zero as we had concluded after scripts 19/19b/20 at the operator-anchored position.

2. **Gemma 2 cross-notation transfer is doubly-reinstated.** Both Gemma 2 PASS-arity cells (sente→opera L4 and sente→first L8) are *also* point-perfect M2-canonical = 1.000 with bootstrap floors at [1.000, 1.000] and [0.888, 1.000] respectively. The Gemma 2 cross-notation transfer claim from scripts 17-18 is reinstated in the strongest form: not only does the arity axis transfer, but the full 5-class canonical-identity axis also transfers at these cells. The operator-after-anchor results in scripts 17-20 were anchor-bound and missed these cells.

3. **The OLMo 2 F→N first→opera L7 cell is the first robust simultaneous-canonical-identity-and-arity transfer cell observed in OLMo 2 — on canonical operators only.** Bootstrap M2-canonical CI [0.964, 0.992] is tight and well above the 0.65 PASS threshold; M2-arity CI [0.980, 0.996] is similarly tight. This is the first OLMo 2 cell that has *any* form of cross-notation positive result on either axis. It is in the reverse direction (F→N) from the §3.7.9 cell. **[Caveat from §3.7.13: at the 16-word expanded invented set this cell's M4b drops from 0.656 to 0.573 (CI [0.539, 0.604], P(≥ 0.65) = 0%). The canonical-only M2 transfer is real; the invented-operator arity-respecting transfer is retracted as a 5-word sampling artifact. The Gemma 2 sente→first L8 cell is similarly retracted by §3.7.13.]**

4. **The dissociation pattern (M2-arity high, M2-canonical low) is the exception, not the rule.** At three of four PASS-arity cells the two metrics agree (M2-arity − M2-canonical ≈ +0.000 or +0.004). Only the §3.7.9 cell has a large positive gap (+0.384). The dissociation pattern that motivated the introduction of M2-arity in §3.7.10 is *not* a generic property of cross-notation transfer — it is a specific feature of the OLMo 2 N→F sente→close L10 cell, mechanistically traceable to within-arity canonical-identity confusion (or → and, implies → and) at that specific cell.

5. **Methodological lesson.** Without the script 22b full sweep, the §3.7.10 conclusion would have been "the M2-canonical / M2-arity dissociation is the principal cross-notation pattern". With the sweep + bootstrap on all four cells, the corrected conclusion is "M2-canonical and M2-arity usually co-pass; the §3.7.9 cell is a *specific* dissociation case with within-arity-confusion mechanism, not a generic pattern". The point-vs-bootstrap-vs-sweep methodology stack is the principal methodological contribution of scripts 22a + 22b.

**Net effect on the Phase 1 headline (after script 22a + 22b + the extension):**

- *Before §3.7.12*: "Cross-notation arity-respecting transfer is demonstrated at four cells across both models. The §3.7.9 cell is the headline because of its perfect M2-arity and clean per-word pattern."
- *After §3.7.12*: "Cross-notation arity-respecting transfer is demonstrated at four cells across both models, **three of which are also full canonical-identity transfer (dual M2-canonical + M2-arity PASS) under bootstrap stim-resampling**." (Note: superseded by §3.7.13 below.)
- *After §3.7.13 (expanded invented-set falsification)*: "Cross-notation arity-respecting transfer is demonstrated at **two cells across two models** under stimulus expansion to a balanced 8-binary / 8-unary invented set: (i) **OLMo 2 sente→close L10 N→F** — M4b = 0.796 robust (CI [0.772, 0.819]) with distributed mass across "and" / "necessarily" / "not", the cleanest positive result in the project (M2-arity-only, M2-canonical AMBIG); and (ii) **Gemma 2 sente→opera L4 N→F** — M4b = 0.669 borderline (CI [0.659, 0.676]) with mass concentrated in a "necessarily" basin (full M2-canonical PASS but mechanism partly modifier-basin). The §3.7.12 "first OLMo 2 simultaneous canonical + invented arity transfer" claim at first→opera L7 is **retracted on the invented-mass axis** (M4b drops to 0.573 at 16 words); the M2-canonical/M2-arity numbers at that cell are real on canonical operators but do not extend to invented operators. The Gemma 2 sente→first L8 cell is also retracted on M4b at 16 words."

This is still a meaningful upgrade relative to the §3.7.8 negative-result headline ("cross-notation arity-respecting transfer is not demonstrated in any tested model, layer, or pairing"), but a more conservative one than §3.7.12 alone suggested. The cross-notation arity-respecting transfer finding survives in tightened form (2 cells across 2 models, both robust to bootstrap stim-resample with 16 invented words), but the M2-canonical/M2-arity dissociation is the operative axis of substrate-invariance for invented operators in the §3.7.9 cell (the project's strongest positive), and the modifier-basin alternative remains a live confound for the Gemma 2 cell pending script 22d.

#### 3.7.13 Expanded-invented falsification (script 22c)

The §3.7.11/§3.7.12 PASS-arity headline rests on per-word patterns observed under a 5-word invented set (`bliq`, `dren`, `vusp`, `molex`, `perph`; 3 intended-binary, 2 intended-unary). With only 5 words, the M4b ≥ 0.65 criterion can be satisfied by sampling luck plus near-coincidental alignment between the model's default canonical and the binary-intended majority of the set. Script 22c expands the invented set to **16 words (8 intended-binary, 8 intended-unary)** — the original 5 plus 11 new phonotactically plausible Latin-script tokens audited for subword decomposition under both tokenizers (mostly 2-subword; one 3-subword `drelth`; one model-specific 1-subword `sond`/`dren`). Re-extraction at the same anchors as script 21 produced a fresh `v4-expanded-invented` cache (~15 min total on MPS for both models). The M1-M4 + M2-arity battery and a bootstrap M4b CI (500 resamples, with-replacement within each invented word so the 8/8 binary/unary partition is preserved) were re-run at the four PASS-arity cells of §3.7.11 plus the two lucky-default negative controls of §3.7.10/§3.7.11.

**Verdict: 2 of 4 PASS-arity cells survive expansion. The other 2 retract as 5-word sampling artifacts.**

| Cell | M4b (5-word) | M4b (16-word) | Bootstrap 16w CI | P(≥ 0.65) | Verdict |
|---|---|---|---|---|---|
| **§3.7.9 OLMo 2 sente→close L10 N→F** | 0.900 | **0.796** | [0.772, 0.819] | **100%** | **SURVIVES (PASS)** |
| **Gemma 2 sente→opera L4 N→F** | 0.728 | **0.669** | [0.659, 0.676] | **100%** | **SURVIVES (PASS, borderline)** |
| Gemma 2 sente→first L8 N→F | 0.736 | 0.561 | [0.532, 0.589] | 0% | RETRACTED (5-word artifact) |
| OLMo 2 first→opera L7 F→N | 0.656 | 0.573 | [0.539, 0.604] | 0% | RETRACTED (5-word artifact) |
| [LUCKY-NEG] OLMo opa→opa L7 F→N | 0.800 | **0.500** | [0.500, 0.500] | 0% | confirmed lucky-default |
| [LUCKY-NEG] Gemma first→sente L8 F→N | 0.736 | **0.500** | [0.500, 0.500] | 0% | confirmed lucky-default |

**Five findings:**

**(1) The §3.7.9 cell is the unique-strongest finding and survives expansion robustly.** Per-word breakdown at 16 words: all 8 binary-intended words → "and" with within-word concentration 76-100% (8/8 binary correct); 5 of 8 unary-intended words → unary canonicals ("necessarily" or "not") with concentrations 54-100% (5/8 unary correct); 3 unary-intended words mis-classify (perph → and 50/50 split; gleph → and 54%; vrith → and 64%). Total intended-binary-correct: 379/400 (94.75%); intended-unary-correct: 258/400 (64.5%). Cross-canonical mass at 16 words: 65.1% "and" / 27.4% "necessarily" / 7.5% "not" / 0% "or" / 0% "implies" (distributed across 3 canonicals, M4c = 0.50). **The §3.7.9 cell is the clearest evidence in the project for cross-notation arity-respecting transfer that generalises beyond the original 5-word set.**

**(2) Gemma 2 sente→opera L4 survives borderline, with a "necessarily-basin"-flavoured mechanism.** M4b = 0.669 (P(≥ 0.65) = 100% under bootstrap, but only 1.9 percentage points above threshold; CI floor 0.659 just clears 0.65). Per-word breakdown at 16 words is mechanistically *not* the same as the §3.7.9 cell — 4 of 8 binary-intended words go to "and" (correct binary: bliq, molex, glin, twiv), 4 of 8 go to "necessarily" (incorrect arity: dren, krev, sond, fump); 7 of 8 unary-intended go to "necessarily" (correct, plus zorf → and at 100%). Cross-canonical mass: 32.1% "and" / 67.9% "necessarily" / 0% all others (M4c = 0.56, bimodal between two canonicals). **The Gemma 2 PASS-arity is mechanistically partly a "necessarily catchment basin" + a partial binary-to-"and" prior; it survives the M4b ≥ 0.65 threshold but the per-word reading is qualitatively weaker than the §3.7.9 cell.** This empirically confirms the peer-review concern (Dan Lutalo, §3.7.5 caveat box) that "necessarily" is both a logical unary canonical and a generic English modifier, and that the Gemma 2 unary mass may be a generic-modifier abstraction rather than a strict logical-unary abstraction.

**(3) The OLMo 2 F→N first→opera L7 dual-M2-PASS cell from §3.7.12 retracts on M4b.** This is the most consequential retraction in §3.7.13. §3.7.12 reported this cell as the project's first simultaneous M2-canonical + M2-arity PASS under bootstrap (point M2-canonical = 0.980, CI [0.964, 0.992]; M2-arity = 0.984, CI [0.980, 0.996]). At 16 invented words, M4b drops from 0.656 to 0.573 (bootstrap CI [0.539, 0.604], P(≥ 0.65) = 0%). The 5-word M4b of 0.656 was barely above threshold; the expansion exposes that the 4-of-5 arity-tracking was coincidence. **Per-word breakdown at 16 words shows distributed predictions** (M4c = 0.35, the most distributed of any cell in the sweep) — 6 of 8 binary-intended words correctly go to a binary canonical ("implies" or "necessarily"; twiv is exception), but only 4 of 8 unary-intended words correctly go to a unary canonical. The M2-canonical = 0.980 / M2-arity = 0.984 numbers remain real on canonical operators; the *invented* mass simply does not respect intended arity above chance. **The §3.7.12 "first OLMo 2 simultaneous canonical+arity transfer in invented operators" claim is therefore retracted. The §3.7.12 M2-canonical and M2-arity numbers are downgraded to "canonical-only transfer" — invented operators do not inherit this transfer above chance.**

**(4) Lucky-default negative controls drop cleanly to M4b = 0.500 with 16 words, validating the methodology.** Both [LUCKY-NEG] cells at 5 words had M4b in the 0.736-0.800 range (where 3-of-5 = 60% intended-binary + 2-of-5 = 40% intended-unary intersected with a default-to-one-binary-canonical pattern produced an inflated apparent agreement). At 8/8 invented-binary/unary, the same default-canonical pattern yields exactly 8/16 = 50% agreement, which is exactly what we observe (CI [0.500, 0.500] at both lucky-default cells; M4c = 1.00 at both; predict-everything-"implies" at both). **This is the strongest possible validation of the M4b ≥ 0.65 threshold and the M4a/M4b/M4c separation introduced in §3.7.5: with a balanced 8/8 invented set, lucky-default cells collapse to chance exactly, and PASS-arity cells stay above threshold.** The methodology is empirically discriminating.

**(5) The Phase 1 headline tightens to "two cells across two models with confirmed cross-notation arity-respecting transfer under stimulus expansion".** Before §3.7.13: 4 cells, of which §3.7.12 confirmed 3 as dual-PASS. After §3.7.13: 2 cells (§3.7.9 OLMo sente→close L10 + Gemma sente→opera L4), with the §3.7.9 cell as unique-strongest (clean distributed-across-3-canonicals mechanism) and the Gemma cell as borderline + mechanistically partly modifier-basin. The cross-model cross-notation arity-respecting transfer finding survives in tightened form — one cell each in OLMo 2 and Gemma 2 — but is no longer claimed as a generic property across multiple anchors and layers; it is anchor-specific, layer-specific, and stimulus-sample-sensitive. The §3.7.12 "first OLMo 2 simultaneous canonical+arity invented transfer" claim is retracted.

**Implication for the Platonic Representation Hypothesis framing (forward pointer to §4.1).** With §3.7.13 in hand, the cleanest synthesis is:

- *Strong PRH ("full Platonic")* — predicted by both M2-canonical AND M4b passing simultaneously at a cell. Observed at 0 of 2 surviving cells (Gemma sente→opera L4 has M2-canonical = 1.000 but M4b = 0.669 borderline and mechanistically necessarily-basin-leaning; OLMo §3.7.9 cell has M4b = 0.796 robust but M2-canonical = 0.616 AMBIG).
- *Partial PRH ("hierarchical abstraction: arity yes, canonical-identity no")* — predicted by M2-arity passing with M2-canonical below threshold AND M4b passing. **Observed at the §3.7.9 OLMo cell** (M2-arity = 1.000, M2-canonical AMBIG, M4b = 0.796). This is now the cleanest single positive result in the project.
- *Modifier-basin / generic-prior null* — predicted by M4a → unary with within-arity mass collapsing to "necessarily". **Observed at the Gemma 2 sente→opera L4 cell** in tightened form (67.9% necessarily / 32.1% and; modifier-leaning).
- *Pure compression / catchment null* — predicted by lucky-default behaviour with no arity structure. Observed at all 40 lucky-default cells from §3.7.11 + both negative-control cells in §3.7.13.

**The §3.7.9 cell is the cleanest current example of partial-Platonic / hierarchical-abstraction substrate-invariance in the project; the Gemma 2 sente→opera L4 cell shows that some cross-notation transfer also exists in a different model but its strict-logical-arity interpretation is contested by the modifier-basin alternative.** Section 4.1 will be written to operationalise this distinction once script 22d (expanded canonical set with additional unary canonicals: possibly, always, negate) lands, since 22d is what distinguishes the modifier-basin from the strict-arity reading. **[Update from §3.7.14: script 22d retracts both §3.7.13 survivors on M4b. The "modifier-basin vs strict-arity" disambiguation produces a third answer that neither anticipated — a *default-to-rarest-canonical* compression mechanism. See §3.7.14.]**

#### 3.7.14 Expanded-canonical falsification (script 22d)

Both §3.7.13 survivors were tested under canonical-set expansion (5 → 10 canonicals: original 5 plus binary `xor`, `nand` and unary `possibly`, `always`, `negate`; invented set unchanged from 22c) to disambiguate strict-logical-arity transfer from generic-modifier-basin transfer. The result falsifies *both* hypotheses in favour of a third mechanism.

**Verdict: both §3.7.13 survivors retract on M4b under canonical-set expansion. All four originally-PASS-arity cells from §3.7.11 are now retracted in some form.**

| Cell | M4b (5-cano, 22c) | M4b (10-cano, 22d) | Bootstrap 10-cano CI | P(≥ 0.65) | dominant canonical |
|---|---|---|---|---|---|
| §3.7.9 OLMo sente→close L10 | **0.796** | **0.500** | [0.500, 0.500] | 0% | `nand` 100% (single-canonical collapse) |
| Gemma sente→opera L4 | **0.669** | 0.625 | [0.625, 0.625] | 0% | `nand` 87.5% + `negate` 12.5% (lucky-default flag YES) |
| [LUCKY-NEG] OLMo opa→opa L7 | 0.500 | 0.500 | [0.500, 0.500] | 0% | `negate` 100% (target-shifted from `implies`) |
| [LUCKY-NEG] Gemma first→sf L8 | 0.500 | 0.510 | [0.480, 0.544] | 0% | `implies` 69.6% + `negate` 30.2% |

**Five findings:**

**(1) The §3.7.9 cell — previously the project's "cleanest" cross-notation arity-respecting transfer — retracts completely under canonical-set expansion.** With 10 canonicals, ALL 16 invented words land on `nand` at 100% within-word concentration (lucky-default-flag YES). M4b = 0.500, exactly chance. The 8-of-8 intended-binary correctness from 22c (where binary-intended words landed on `and`) was a coincidence between the 5-canonical default-target (`and`) and the intended-arity distribution of the invented set (8 of 16 binary). When `nand` is added (a near-zero-frequency English word, the highest-entropy canonical), the default-target shifts wholesale to it. The cell now has M4a = 0% (zero unary mass), M4c = 1.00 (total single-canonical collapse), modal-adverbial fraction undefined. **The cell's previously-distributed mass across "and" + "necessarily" + "not" was specific to the 5-canonical readout; it does not survive expansion.**

**(2) The Gemma sente→opera L4 cell also retracts on M4b, and the "necessarily basin" of 22c is revealed as a *5-canonical-specific default* rather than a stable representation.** With 10 canonicals, all 8 binary-intended invented words land on `nand` (binary-correct by coincidence) and 2 of 8 unary-intended words land on `negate` (unary-correct by coincidence); the remaining 6 unary-intended words *also* land on `nand` (binary, arity-wrong). Total M4b = 0.625, below threshold. **Critically, the `necessarily` canonical — which received 67.9% of invented mass in 22c — receives 0% mass in 22d.** The "necessarily basin" did not move to another modal-adverbial unary (`possibly`, `always`); it disappeared. Mass migrated to `nand` (binary) and `negate` (unary, operational). Neither the strict-arity hypothesis (uniform across 5 unaries) nor the modifier-basin hypothesis (mass concentrated on `necessarily` + `possibly` + `always`) is supported; both are falsified by the actual readout. The lucky-default-flag fires on this cell at 22d.

**(3) The mechanism is "default-to-rarest-canonical".** Both surviving cells, both lucky-default negative controls, and (one assumes) the cells retracted by 22c all share the same underlying behaviour: the probe routes invented-operator activations to whichever canonical sits in the most reachable / highest-entropy decision region. With 5 canonicals (and / or / not / implies / necessarily) that region is occupied by `implies` (low English frequency) or `and` / `necessarily` (high prior weight default targets, varying by cell and training anchor). With 10 canonicals adding `xor`, `nand`, `possibly`, `always`, `negate` — all near-zero-frequency in natural English — the highest-entropy canonical becomes `nand` or `negate`. The probe's invented-operator predictions shift wholesale to those new targets. **This is a compressive / low-confidence behaviour, not a structural-substrate-invariance behaviour.** It is exactly what a sophisticated compression-only model would do: route uncertain inputs to the bucket with the lowest training-prior weight, which under softmax behaviour has the largest log-margin to gain when the activation is far from any high-confidence cluster.

**(4) Crucially, canonical-operator M2 transfer remains robust under canonical-set expansion.** The two surviving cells have:
- **Gemma sente→opera L4 N→F**: M2-canonical = 1.000 (10-class accuracy on test canonicals; chance = 0.10; ~10× chance), M2-arity = 1.000. A probe trained on NEUTRAL canonical operators *perfectly* classifies FUNC-PFX canonical operators including the 5 new ones it has never seen at the test side, transferred across notation boundaries.
- **§3.7.9 OLMo sente→close L10 N→F**: M2-canonical = 0.812 (10-class accuracy; ~8× chance), M2-arity = 1.000. Same pattern: canonical-operator structure transfers cleanly across notations even when the canonical set is expanded.

**What does NOT transfer is the model's ability to extend its canonical-operator geometry to novel operators in an arity-respecting way.** The M2 / M4 dissociation is now a *canonical-vs-novel* dissociation: canonical-operator substrate-invariance is robust; novel-operator substrate-invariance is illusory under the default-to-rarest mechanism.

**(5) The OLMo `nand` tokenization confound is real but does not change the conclusion.** In OLMo's BPE, `nand` → `[' n', 'and']` (in Gemma, `nand` → `[' nand']` single subword). At the operator-anchored position (last subword) in OLMo, the residual stream for `nand` is computed at the `'and'` token — so the §3.7.9 cell's "100% to `nand`" reading is partially contaminated by reading the `'and'` subword's representation directly. However, **the same `nand`-dominance appears in Gemma**, where there is no tokenization confound (`nand` is a single token). The default-to-rarest-canonical mechanism is therefore real and cross-model; the OLMo-`nand` confound only makes the OLMo cell's M4b retraction *more* certain (the cell wasn't really classifying invented words as `nand`; it was classifying them as `and`-the-subword, which under expanded canonicals gets a `nand` label).

**The reframed Phase 1 headline (after §3.7.14):**

- *Before §3.7.11*: "Cross-notation arity-respecting transfer is not demonstrated in any tested model at the operator-anchored position." (Negative result, anchor-bound.)
- *After §3.7.11/§3.7.12*: "Cross-notation arity-respecting transfer is demonstrated at four cells (3 dual-PASS, 1 arity-only) under bootstrap." (Strong positive, four cells.)
- *After §3.7.13*: "Cross-notation arity-respecting transfer survives invented-set expansion at two cells (1 OLMo, 1 Gemma); two cells retract as 5-word artifacts." (Tightened positive, two cells.)
- ***After §3.7.14***: "**Cross-notation arity-respecting transfer to *novel* operators is not demonstrated under both invented-set AND canonical-set expansions.** All four originally-PASS-arity cells are retracted in some form. The apparent positive results at 22c's two surviving cells were specific to the 5-canonical readout — they reflect a 'default-to-rarest-canonical' compression mechanism that shifts target as the canonical set expands. **What IS demonstrated, robustly across both models at multiple anchors, is cross-notation transfer of canonical-operator identity and arity (M2-canonical and M2-arity PASS under bootstrap stim-resample AND canonical-set expansion).** Substrate-invariance in current open LMs is operator-set-bound: the model's logical-operator geometry transfers across notations for the specific canonical operators it knows, but does not generalize to novel operators in an arity-respecting way."

**Implications for the Section 4.1 PRH framing:** The reviewer's "partial Platonic" framing — arity transfers, identity does not — is not supported by 22d. The data supports a different and arguably sharper synthesis:

- **Operator-set-bound substrate-invariance is real**: cross-notation transfer of canonical-operator structure is demonstrated robustly across both models, both anchors, both directions, and both stimulus expansions. This is a meaningful PRH-supporting finding restricted to known operators.
- **Novel-operator substrate-invariance is not demonstrated**: the model cannot extend its canonical-operator geometry to invented words in an arity-respecting way. The cross-notation behaviour for invented operators is dominated by a low-confidence default-to-rarest-canonical compression mechanism.
- **The "morphospace edge"** is therefore located between *learned-operator structural invariance* and *novel-operator structural generalization*. The model has a Platonic-like representation of *the logical operators it was trained on as logical operators*; it does not have a Platonic-like representation of "logical-operator-ness" that extends to novel instances. This is closer to a "compression-favouring with operator-set memorisation" reading than to a "Platonic abstraction of logical structure" reading, but it is more nuanced than pure compression: the operator-set geometry transfers across notation boundaries, which a flat compressive model would not predict.

This is a Section 4.1 worth writing.

**Methodological contribution:** The point-vs-bootstrap-vs-invented-expansion-vs-canonical-expansion methodology stack is the central methodological contribution of scripts 22a-d. Each layer catches a different class of false positive:
- **Bootstrap on M2** (§3.7.10): catches stim-sampling fragility on the canonical-transfer gate
- **Sweep across cells** (§3.7.11): catches cell-specific effects vs structural commonalities; introduces the refined lucky-default detector
- **Bootstrap CI on PASS-arity cells** (§3.7.12): confirms point estimates
- **Invented-set expansion** (§3.7.13): catches 5-word stimulus-sampling artifacts in per-word patterns
- **Canonical-set expansion** (§3.7.14): catches default-to-rarest-canonical artifacts where high M4b is achieved by the model routing invented mass to whatever canonical happens to align with the invented set's arity distribution

**All five layers are necessary.** Without 22c, we would have reported 4 PASS-arity cells (3 retracted by 22c, 2 more falsified by 22d). Without 22d, we would have reported the §3.7.9 cell as the cleanest cross-notation arity-respecting transfer in the project (now retracted). The full M2-canonical + M2-arity + M4a + M4b + M4c + per-word + lucky-default-detector + invented-expansion + canonical-expansion battery is the empirical instrument for cross-notation substrate-invariance claims, and it is materially stronger than any of its components in isolation. **Future substrate-invariance studies should pre-register all five layers, not just M1-M4.**

#### 3.7.15 Pythia 6.9B-deduped Phase 2 replication (script 23)

The operator-set-bound substrate-invariance finding established on OLMo 2 7B and Gemma 2 9B (Phase 1) was replicated on a third model family — **Pythia 6.9B-deduped** (EleutherAI, GPT-NeoX architecture with RoPE positional encoding, trained on the deduplicated Pile). Pythia was chosen as the cross-family replication target because it diverges from both Phase 1 models on three axes simultaneously: different training corpus (Pile vs Dolma vs Google's proprietary mix), different architecture (GPT-NeoX vs OLMo 2's modified Llama vs Gemma 2's soft-capping architecture), and different tokenizer (Pythia's standard BPE vs OLMo 2's Dolma BPE vs Gemma 2's SentencePiece). Script 23 extracts a single v5-expanded-canonical cache (10 canonicals + 16 invented at all four FUNC-PFX anchors × 32 layers + two NEUTRAL anchors × 32 layers, ~1.9 GB compressed) and runs three nested anchor × layer × direction sweeps from it (v3-scope: 5 canonicals + 5 invented, comparable to script 22b; v4-scope: 5 canonicals + 16 invented, comparable to 22c; v5-scope: 10 canonicals + 16 invented, comparable to 22d). Total runtime ~16 min on M4 MPS fp16 (~6 min extraction at 276 tok/s + ~8 min sweep × 3 scopes). The single-cache + multi-scope analysis cleanly reproduces all three predictions of the operator-set-bound finding.

**Pythia tokenization audit.** `nand` → `[' n', 'and']` (same 2-subword split as OLMo, unlike Gemma's single-token `nand`); `xor` → `[' x', 'or']` (new 2-subword split, neither OLMo nor Gemma exhibits this); `negate` → `[' neg', 'ate']`. All other canonicals are single-subword. All 16 invented words are 2-3 subwords. The `xor` split creates a potential operator-anchored confound analogous to OLMo's `nand` (the `'or'` subword sits one position before the operator-after anchor for canonical `xor`); the `nand` confound is identical to OLMo's. Both confounds are bounded to the operator-after anchor; the other three FUNC-PFX anchors (`first-arg`, `close-paren`, `sentence-final`) are not affected.

**Phase 2 question P1 — M2-canonical PASS across notations: replicated at high density.**

| Scope | Canonicals | Threshold | Cells passing | Best |
|---|---|---|---|---|
| v3 | 5 (chance = 0.20) | ≥ 0.65 | 42 / 80 | 1.000 at opera→opera L4 (and 10+ other cells) |
| v4 | 5 (chance = 0.20) | ≥ 0.65 | 42 / 80 | 1.000 at opera→opera L4 |
| v5 | 10 (chance = 0.10) | ≥ 0.65 | 31 / 80 | 1.000 at opera→opera L4 |

At v5 (10-class, ~10× chance), Pythia has **31 cells** with M2-canonical PASS, distributed across all 5 focus layers [4, 7, 10, 16, 24] and both N→F and F→N directions. This is materially denser than OLMo 2 7B (which has handful of M2-cano PASS cells at v5) and comparable to Gemma 2 9B (similar density). The canonical-operator substrate-invariance half of the headline replicates strongly.

**Phase 2 question P2 — PASS-arity cells exist at v3 and retract under stimulus expansion: replicated, with novel structural features.**

Pythia has **3 v3-scope PASS-arity candidates**, all in N→F direction with `operator-after` as the training anchor and `close-paren` as the test anchor, distributed across three depths:

| Cell | M2-cano (v3) | M2-arity (v3) | M4b (v3) | M4c (v3) | verdict |
|---|---|---|---|---|---|
| v3 N→F opera→close L4 | 0.820 | 0.920 | 66.8% | 0.44 | PASS-arity |
| v3 N→F opera→close L7 | 1.000 | 1.000 | 79.6% | 0.55 | PASS-arity |
| v3 N→F opera→close L16 | 1.000 | 1.000 | 75.6% | 0.64 | PASS-arity |

This is **more v3-PASS candidates than either OLMo 2 (1 cell, §3.7.9) or Gemma 2 (1 cell, sente→opera L4)**, and the structural commonality is striking: all three Pythia candidates share the same anchor pair (operator-after → close-paren) and emerge at three different depths (L4, L7, L16). OLMo 2 7B's §3.7.9 cell uses `sentence-final → close-paren` (a different train anchor); Gemma 2 9B's §3.7.13 cell uses `sentence-final → operator-after` (different both anchors). The three models converge on `close-paren` as the *test* anchor (the post-call position in functional notation) but diverge on the *training* anchor — a methodologically informative finding for future cross-model substrate-invariance work.

The within-Pythia falsification chain (script 23 PHASE D):

| Cell | M4b v3 | M4b v4 (16 inv) | M4b v5 (10 canon) | Verdict |
|---|---|---|---|---|
| v3 N→F opera→close L4 | 0.668 PASS | 0.699 PASS | 0.466 RETRACT | survives v4, retracted by v5 |
| v3 N→F opera→close L7 | 0.796 PASS | 0.736 PASS | 0.286 RETRACT | survives v4, retracted by v5 |
| v3 N→F opera→close L16 | 0.756 PASS | 0.583 RETRACT | 0.500 RETRACT | retracted by v4 |

**Two of three Pythia candidates survive v4 expansion (16 invented words) but all three retract at v5 (10-canonical expansion).** This is a stronger pattern than OLMo 2 (where the §3.7.9 candidate survived v4 with M4b = 0.796 but retracted at v5 to nand) and Gemma 2 (where the sente→opera L4 candidate survived v4 at M4b = 0.669 borderline, retracted at v5). All three model families show the same v4→v5 retraction signature, but Pythia's "robustness to invented-set expansion" before its v5 retraction is the most pronounced — interesting in itself, but does not save the operator-set-bound conclusion. **All cross-family v3 PASS-arity cells retract at v5; the canonical-set-expansion test is the operator-set-bound finding's principal falsifier.**

**Phase 2 question P3 — default-to-rarest-canonical mechanism at v5, with cross-family target diversity.**

Pythia's v5 aggregate per-word top-canonical distribution (across all 80 sweep cells × 16 invented words = 1280 invented-word readouts):

| Canonical | Pythia v5 | Tokenization in Pythia | Comparison |
|---|---|---|---|
| `nand` (NEW) | 27.9% | `[' n', 'and']` | dominant in OLMo (~100% at single cell); shared dominance in Gemma (~87% + negate 12.5%) |
| `xor` (NEW) | 22.3% | `[' x', 'or']` | not previously a default target in OLMo or Gemma |
| `negate` (NEW) | 19.8% | `[' neg', 'ate']` | secondary target in Gemma; not in OLMo |
| `and` | 10.1% | `[' and']` | — |
| `implies` | 6.6% | `[' implies']` | — |
| `necessarily` | 5.6% | `[' necessarily']` | — |
| `or` | 4.8% | `[' or']` | — |
| `possibly` (NEW) | 1.3% | `[' possibly']` | — |
| `always` (NEW) | 1.1% | `[' always']` | — |
| `not` | 0.5% | `[' not']` | — |

**70.0% of all invented-word v5 routings go to the three multi-subword NEW canonicals (nand + xor + negate).** This is a cleaner version of the mechanism than either OLMo or Gemma showed: rather than concentrating on one new canonical (OLMo's `nand`), Pythia distributes invented mass across all three high-entropy multi-subword binaries simultaneously. The single-subword NEW unaries (`possibly`, `always`) are essentially never selected (1.3% + 1.1% = 2.4% combined).

**Two new observations from this distribution:**

1. **The default-to-rarest-canonical mechanism is cross-family stable in *direction* (toward low-frequency canonicals) but model-specific in *which low-frequency canonical wins*.** OLMo collapses to `nand` singly; Gemma splits between `nand` and `negate`; Pythia splits broadly across `nand`, `xor`, `negate`. The mechanism is not "route to the rarest one canonical" but "route to whichever of the rare canonicals has the highest-entropy decision region at this specific (model × anchor × layer) cell". Pythia's broader distribution is consistent with the model having multiple equi-low-prior canonicals at most cells; OLMo's concentrated distribution is consistent with one dominant rare-canonical attractor per cell.

2. **Multi-subword tokenization of the new canonicals correlates with their attraction strength.** In Pythia, the three dominant default targets (`nand` 2-subword, `xor` 2-subword, `negate` 2-subword) sum to 70.0% of v5 invented mass; the two single-subword new canonicals (`possibly` 1-subword, `always` 1-subword) sum to 2.4%. This is consistent with two non-mutually-exclusive readings: **(i)** invented words are 2-3-subword tokens whose residual-stream representation at the operator-after anchor has token-boundary geometry similar to multi-subword canonicals (the H2/H3 channel from Phase 0 propagating into the v5 readout); **(ii)** multi-subword canonicals are rarer in training data (`nand`, `xor`, `negate` are formal-logic and programming terms; `possibly` and `always` are common English adverbs), so the rare-canonical attractor is partly a frequency effect, partly a tokenization-shape effect, with these two confounded in our current data. Distinguishing them would require a future test with single-subword rare canonicals (if any exist for logical-operator semantics) or multi-subword high-frequency canonicals.

The xor tokenization confound is bounded: it only affects the operator-after anchor, where the `'or'` subword sits at the same position the canonical `xor` activation is read from. The close-paren and sentence-final anchors are unaffected. Pythia's xor-attraction at operator-after specifically should be noted as confounded; xor-attraction at other anchors (where it also appears) is real.

**Bootstrap M4b on the top-M2c v5 cell (PHASE D2):**

At Pythia's strongest v5 M2-canonical cell (v5 N→F opera→opera L4, M2-cano = 1.000, M2-arity = 1.000), the point M4b = 0.562 and the 500-resample bootstrap yields CI [0.562, 0.562] with std = 0.000 (every invented word has within-word concentration = 100% on a single canonical, so resampling-within-words is a no-op). P(M4b ≥ 0.65) = 0.0%. This is the cleanest illustration of the default-to-rarest mechanism in the project: invented mass routes deterministically to whichever new canonical the cell prefers, with zero per-stimulus variance.

**Cross-family synthesis.** Three model families (OLMo 2 7B, Gemma 2 9B, Pythia 6.9B-deduped), three training corpora (Dolma, Google proprietary, Pile), three architectures (modified Llama, soft-capped Gemma, GPT-NeoX), and the same two-part finding replicates:

1. **Canonical-operator cross-notation substrate-invariance is robust** (M2-canonical PASS at multiple cells in every model; M2-arity = 1.000 at the strongest cells; v5 10-class accuracy = 1.000 at 1+ cells in every model).

2. **Novel-operator generalization to invented words fails under joint invented-set and canonical-set expansion** (every initially-positive PASS-arity cell across all three models retracts under either v4 or v5 expansion; the mechanism is default-to-rarest-canonical, with the specific target varying by model but the *direction* (toward low-frequency canonicals) being cross-family invariant).

The Phase 2 verdict is that the operator-set-bound substrate-invariance finding is a property of mid-scale open language models at the 6.9-9B parameter range with current-generation tokenizers and architectures, not an OLMo-specific or Gemma-specific quirk. The mechanism (compression toward low-prior canonicals) is also cross-family stable, with model-specific noise in which low-prior canonical wins. **§4.1 is updated below to incorporate Pythia.**

**Limitations specific to script 23:**

- Pythia 6.9B-deduped is a base model without instruction tuning, like OLMo 2 7B and Gemma 2 9B base. We have not tested whether instruction-tuned variants (Pythia-Chat, OLMo 2 Instruct, Gemma 2 IT) show the same pattern. Instruction tuning could change the novel-operator behaviour either direction.
- 32 layers is the same depth as OLMo 2 7B; we have not tested Pythia 12B (which would push memory limits on the M4). The depth-distribution of PASS-arity candidates (L4, L7, L16) hints that arity-readable structure persists across more depth in Pythia than in the other two models, but a Pythia 12B test would be needed to claim a scale-trend.
- The xor tokenization confound at operator-after is partial; the close-paren-anchored PASS-arity cells are clean of this issue but the v5 aggregate xor-percentage includes some operator-after cells that are confounded. The 22.3% xor figure should be interpreted with this caveat; the *direction* of the finding (multi-subword rare canonicals dominate) is robust.
- All three Pythia candidate cells are in N→F direction; F→N is consistently weaker, mirroring OLMo and Gemma. The "directional asymmetry" of cross-notation transfer is now a three-model observation worth highlighting.

#### 3.7.16 Pre-registered v6 canonical-set expansion (script 24)

The v6 pre-registration (`experiments/preregistration_v6.md`) was the response to a reviewer concern that the M2-arity / lucky-default / multi-scope (v3-v4-v5) framework was developed iteratively on the same data: a garden-of-forking-paths risk. Script 24 (`experiments/24_v6_canonical_expansion.py`) was built and run with the analysis plan frozen *before* any v6 cache extraction. The pre-registered set adds five new canonicals (`nor`, `iff`, `unless`, `definitely`, `unprovably`) chosen to disentangle two confounded explanatory dimensions of the §3.7.14 / §3.7.15 default-to-rarest mechanism — token frequency in the training corpus, and subword-tokenization shape — against three MF (mid-frequency) controls (`unless`, `definitely`) that should not attract default mass on either reading. The full canonical set at v6 is 15 (8 binary + 7 unary). The invented set remains the same 16 words from v4/v5. Total cross-model compute on M4 MPS was 4834 s (≈ 81 min) for fresh extraction + sweep across Gemma 2 9B, OLMo 2 7B, and Pythia 6.9B-deduped (carryover + held-out canonical caches; ~5.5 GB of new caches).

**Audit gate outcome: `iff` failed its target profile in all three tokenizers.** The pre-reg specified `iff` as 2-3-subword (target multi-piece), but Gemma's SentencePiece tokenizer maps it to `[' iff']` (1pc), OLMo 2's BPE maps it to `[' iff']` (1pc), and Pythia's BPE maps it to `[' iff']` (1pc). Per pre-reg §1, the analysis proceeds with `iff` flagged OUT-OF-DESIGN — the multi-pc-LF arm of the disentanglement test relies on `unprovably` alone (`[' un', 'prov', 'ably']` 3pc in all three tokenizers, in-design). The other four NEW canonicals are in-design across all three tokenizers: `nor` (1pc, low-frequency), `unless` (1pc, mid-frequency), `definitely` (1pc, mid-frequency), and `unprovably` (multi-pc, very-low-frequency). The audit-gate result is therefore "4 of 5 NEW canonicals in-design, analysis proceeds with `iff` caveat", below the pre-reg's "more than 2 OOD" abort threshold.

**M1heldout sanity check.** Each model's v6 carryover-trained probe was evaluated on a syntactically disjoint held-out template family at every focus layer and anchor. The headline pattern is consistent across all three models: M1tr (carryover) is at ceiling 0.95-1.00 everywhere; M1heldout drops by 5-40 percentage points but stays above the pre-reg's structural threshold (0.65) at the cross-notation Fact-1 cells. **Specifically, at FUNC-PFX operator-after at the early focus layers (Gemma L2-L8, OLMo L4-L10, Pythia L4-L10), M1heldout = 0.94-1.00** — the principal Fact-1 cells are template-leakage-clean. Sentence-final and deep-layer cells show larger degradation (Gemma FUNC-PFX sentence-final L8 = 0.30; OLMo FUNC-PFX close-paren L24 = 0.37) — these cells are not the locus of any §3.7.15 / §3.7.16 positive finding, so the template-leakage exposure does not affect headline claims.

**Phase 2 question 1 (Fact 1 under 15-class readout): replicated cleanly across all three models.**

| Model | Best v6 M2-canonical (15-class chance = 0.067) | Cell | Best v6 M2-arity | Bootstrap M2-cano 95% CI |
|---|---|---|---|---|
| Gemma 2 9B | 1.000 | v6 N→F opera→opera L4 | 1.000 | [1.000, 1.000] |
| OLMo 2 7B | 1.000 | v6 N→F opera→opera L4 | 1.000 | [1.000, 1.000] |
| Pythia 6.9B-d | 1.000 | v6 N→F opera→opera L4 | 1.000 | [1.000, 1.000] |

All three models reach M2-canonical = 1.000 (≈ 15× chance) on the 15-class readout at the same `operator-after → operator-after L4` cell, with bootstrap 95% CIs tight at [1.000, 1.000] under 500 stim-resamples. **Cross-notation canonical-operator substrate-invariance under the expanded readout is not a 10-class artifact**: adding five more canonicals to the readout vocabulary, three of them with materially different frequency / subword profiles, does not degrade the cross-notation probe transfer for the operators the model was already trained on. Fact 1 (the principal positive finding) is the strongest it has ever been: bootstrap-CI-confirmed at [1.000, 1.000] under 15-class readout across three model families.

**Phase 2 question 2 (default-to-rarest disentanglement): all three pre-registered single-axis predictions fail; the mechanism is multi-factor.**

The v6 aggregate per-word top-canonical distribution per model (across all 80 sweep cells × 16 invented words = 1280 readouts):

| Canonical | Tok | Freq | Gemma v6 | OLMo v6 | Pythia v6 |
|---|---|---|---|---|---|
| `and` (B) | 1pc | high | 0.4% | 0.2% | 2.3% |
| `or` (B) | 1pc | high | 0.8% | 1.2% | 3.8% |
| `implies` (B) | 1pc | mid | 1.6% | **15.9%** | 5.4% |
| `xor` (B) | 1pc/2pc/2pc | very-low | 22.5% | 9.3% | 16.6% |
| `nand` (B) | 1pc/2pc/2pc | very-low | 17.4% | **40.5%** | 21.9% |
| `not` (U) | 1pc | high | 4.9% | 2.4% | 0.8% |
| `necessarily` (U) | 1pc | mid | 0.2% | 0.5% | 4.1% |
| `possibly` (U) | 1pc | high | 1.6% | 3.5% | 0.0% |
| `always` (U) | 1pc | high | 0.0% | 0.1% | 0.0% |
| `negate` (U) | 1pc/1pc/2pc | low | 12.1% | 16.6% | 14.2% |
| **`nor` (B, NEW)** | 1pc | low | 5.1% | 1.2% | 0.2% |
| **`iff` (B, NEW, OOD)** | 1pc | very-low | 10.8% | 3.2% | 12.1% |
| **`unless` (B, NEW)** | 1pc | mid | 0.4% | 0.3% | 2.4% |
| **`definitely` (U, NEW)** | 1pc | mid | 7.5% | 1.3% | 5.5% |
| **`unprovably` (U, NEW)** | multi-pc | very-low | 14.8% | 3.7% | 10.6% |

Per-model verdicts against the pre-registered thresholds (P_FREQ requires aggregate {nor + iff + unprovably} ≥ 35% with each individual ≥ 10%; P_SUBWORD requires multi-pc aggregate ≥ 70%; P_INTERACTION requires `nor` ∈ [5%, 15%] with MF controls ≤ 5%):

- **Gemma 2 9B**: P_FREQ.6 fails (aggregate new-LF = 30.6%, below 35%); P_SUBWORD.4 fails (multi-pc aggregate = 14.8% — only `unprovably` is multi-pc in Gemma); P_INTERACTION.3 fails (`definitely` = 7.5%, above 5%). All three single-axis predictions reject. Notable: `iff` attracts 10.8% (matches P_FREQ.2 threshold individually but `iff` is OOD).
- **OLMo 2 7B**: All thresholds fail. The MF controls hold cleanly (`unless` 0.3%, `definitely` 1.3%), but neither LF prediction is met (`nor` 1.2%, `iff` 3.2%, `unprovably` 3.7%). **OLMo's #3 attractor at 15.9% is `implies` — a v3/v4/v5 carryover canonical** (1pc, mid-frequency, in the readout since the very first script in this project). Neither P_FREQ nor P_SUBWORD predicts this; the only single-axis reading consistent with it would be a "default to the next-rarest binary canonical after `nand`" reading, which collapses to a frequency mechanism but with a much wider definition of "rarest" than the pre-reg envisioned.
- **Pythia 6.9B-d**: P_FREQ.6 fails (aggregate new-LF = 22.9%, below 35%); P_SUBWORD.4 fails (multi-pc aggregate = 63.4%, below 70%); P_INT.2 fails (`nor` = 0.2%, below the [5%, 15%] band). MF controls hold tightly (`unless` 2.4%, `definitely` 5.5% — borderline-fail on `definitely`). The strongest individual NEW attractor in Pythia is `iff` (12.1%), but `iff` is OOD in Pythia's tokenization.

**Adjudication**: none of P_FREQ / P_SUBWORD / P_INTERACTION cleanly passes in any model. The pre-registered single-axis dichotomy is empirically rejected. The §3.7.14 / §3.7.15 framing of the default mechanism — "default to the rarest canonical" — is **not a complete description**. The mechanism appears to be a model-specific mixture of at least three factors: training-corpus frequency (consistent with v5 attractors being `nand` / `xor` / `negate`), subword shape (correlated but not dominant; `iff` ended up 1pc and still attracts > 10% in Gemma and Pythia), and what we will tentatively call **contextual semantic neighborhood** — OLMo's 16% routing to `implies` is the cleanest hint of this third factor: at OLMo's softmax-over-canonicals geometry, the in-distribution canonical most similar to "binary-but-not-strongly-affirmative" is `implies`, and that similarity persists even when `xor` / `nand` / `iff` are in the readout. The follow-up experiment that closes the mechanism gap is an embedding-similarity probe at the early extraction layers, controlling for arity, predicting each invented word's v6 default attractor from cosine similarity to the v6 canonicals' embeddings (script 25b, see §6).

**Phase 2 question 3 (P_RETRACT): splits 2:1, with the Gemma exception interpreted as a methodological caveat rather than a substantive retraction.**

Pre-registered prediction P_RETRACT was "zero PASS-arity cells in any model at v6". Outcomes:

- **OLMo 2 7B**: 0 PASS-arity cells at v6 (P_RETRACT holds).
- **Pythia 6.9B-deduped**: 0 PASS-arity cells at v6 (P_RETRACT holds).
- **Gemma 2 9B**: 2 PASS-arity cells at v6 (P_SURVIVE per pre-reg §8). Specifically:
  - `v6 N→F opera→close L 2`: M2c = 0.420, M2a = 0.841, **M4b = 82.2%**, M4c = 0.42, pwmin = 0.44.
  - `v6 N→F sente→close L 2`: M2c = 0.512, M2a = 0.832, **M4b = 66.2%**, M4c = 0.58, pwmin = 0.60.

These two Gemma cells are **emergent at v6**, not survivors from earlier scopes. Their cross-scope chain:

| Cell | M2-arity v3 → v4 → v5 → v6 | M4b v3 → v4 → v5 → v6 | Verdict chain |
|---|---|---|---|
| Gemma 2 `N→F opera→close L 2` | 0.884 → 0.884 → 0.804 → 0.841 | **59.6% → 55.9% → 50.0% → 82.2%** | M2A-ONLY → M2A-ONLY → M2A-ONLY → **PASS-arity** |
| Gemma 2 `N→F sente→close L 2` | 1.000 → 1.000 → 0.776 → 0.832 | 44.0% → 45.9% → 50.0% → **66.2%** | M2A-ONLY → M2A-ONLY → LUCKY-NEG → **PASS-arity** |

M2-arity is approximately constant across scopes (0.78-1.00); only M4b changes, jumping at v6. **The arity-axis structure at these cells has been visible across all four scopes** (M2-arity ≥ 0.78 throughout). What v6 added was *more within-arity readout buckets* (8 binary + 7 unary canonicals in v6 vs 3 binary + 2 unary in v3): the larger per-arity readout pool let the same underlying arity-respecting structure express itself with a higher M4b. The first cell's M4b trajectory makes this especially clear — 60% → 56% → 50% → 82%, jumping by 32 percentage points at v6 alone, without any change to the underlying activations (cache content is identical; only the scope's `canonicals` and `invented_set` sub-selections differ).

**Interpretation: this is a methodological caveat about M4b threshold-sensitivity, not a substantive retraction of operator-set-bound substrate-invariance.** The M4b ≥ 0.65 threshold is sensitive to the granularity of the canonical readout. At v3 (5 canonicals, 3B + 2U), a cell with genuine arity structure may have a per-word top-canonical concentration that happens to land just below 0.65 because the 3-binary readout pool is too coarse to express the structure cleanly. At v6 (15 canonicals, 8B + 7U), the same cell's per-word predictions can spread across more within-arity buckets, raising M4b without any change to the underlying residual stream. **The v3→v6 trajectory for these two Gemma cells is the cleanest in-data demonstration that M4b is not a Boolean test of arity-respecting structure** — it is a threshold test on a metric that is sensitive to readout granularity. The Phase 1 / Phase 2 framework should treat M2-arity (which is invariant to readout granularity, since the binary/unary partition is fixed) as the primary arity-axis measurement, and treat M4b only as an additional concentration-respecting check. We leave the §3.7.14 / §3.7.15 "operator-set-bound" framing in place — the operator-set-bound finding is principally a claim about novel-operator generalization (Fact 2), and it remains true under v6 for OLMo and Pythia. The Gemma v6 PASS-arity cells deserve causal-patching follow-up (script 25a, see §6) to test whether the arity-respecting reading at L2 close-paren is causally load-bearing or merely a probe-readable artifact. Until that lands, the v6 Gemma emergence is reported as a methodological caveat on M4b, not as a positive demonstration that Gemma exhibits arity-respecting novel-operator transfer.

**Bootstrap on the top-M2c v6 cell per model.** At each model's strongest v6 M2-canonical cell (which is the same cell across all three: `v6 N→F opera→opera L 4`), 500-resample bootstrap yields:

| Model | M2c point / mean / 95% CI | M4b point / mean / 95% CI | P(M4b ≥ 0.65) |
|---|---|---|---|
| Gemma 2 9B | 1.000 / 1.000 / [1.000, 1.000] | 0.562 / 0.562 / [0.562, 0.562] | 0.0% |
| OLMo 2 7B | 1.000 / 1.000 / [1.000, 1.000] | 0.500 / 0.500 / [0.500, 0.500] | 0.0% |
| Pythia 6.9B-d | 1.000 / 1.000 / [1.000, 1.000] | 0.500 / 0.500 / [0.500, 0.500] | 0.0% |

All three models hit M2c = 1.000 deterministically at this cell (every bootstrap resample reproduces 100% canonical-class accuracy on the FUNC-PFX test set; the canonical-operator geometry is rock-solid). At the same cell, M4b is at chance for the 16-invented set (50% for binary chance, 56.2% in Gemma reflecting the v6-arity-split chance of 53.3% plus small residual). Per-word concentration is at ceiling (pwmin = 1.00 in OLMo and Pythia; LUCKY-NEG flag fires); Gemma's pwmin = 0.86 is just below the lucky-default threshold but the cell is M2A-ONLY (M4b = 56.2% below the PASS gate). **This is the cleanest demonstration in the project of the §3.7.14 default mechanism**: at the cell where canonical-operator transfer is at ceiling, every single invented word is routed to a single attractor canonical with 100% within-word concentration, yielding M4b at chance.

**The OLMo §3.7.9 anchor (sente→close L10) under v5+v6.** This is the cell that anchored Phase 1's principal positive finding (§3.7.9 / §3.7.10 / §3.7.13). Its full retraction chain through v6:

| Scope | M2-arity | M4b | M4c | pwmin | Verdict |
|---|---|---|---|---|---|
| v3 (5 canon + 5 inv) | 1.000 | 0.880 | 0.57 | 0.56 | PASS-arity |
| v4 (5 canon + 16 inv) | 1.000 | 0.777 | 0.53 | 0.52 | PASS-arity |
| v5 (10 canon + 16 inv) | 1.000 | 0.500 | 1.00 | **1.00** | **LUCKY-NEG** |
| v6 (15 canon + 16 inv) | 1.000 | 0.500 | 1.00 | **1.00** | **LUCKY-NEG** |

M2-arity stays at 1.000 across all four scopes — the binary/unary axis at this cell is rock-solid. M4b drops to chance and pwmin pegs at 1.00 once the canonical set is expanded to include `nand`: 100% of invented words are routed to the single `nand` attractor. The §3.7.9 reading at v3 was a clean lucky-default — the 5-canonical readout happened to land on a unary canonical (`not`) for the intended-unary words and on `and` for the intended-binary words, yielding M4b = 0.880 that looked like arity-respecting transfer. The v5+v6 expansion exposes the underlying mechanism. **The lucky-default detector flags this cell correctly under v5 and v6**, and the §3.7.11 refinement of the detector (`pwmin ≥ 0.95`) is vindicated as the methodological filter that prevented us from publishing this cell as a positive finding. See §4.4.

**Tokenization audit findings worth recording.** The audit gate caught one design failure (`iff` 1pc in all three tokenizers). Beyond `iff`, the audit also surfaced two model-specific tokenizations of v5 carryover canonicals: Gemma 2 collapses `nand` to `[' nand']` (1pc, unlike OLMo's `[' n', 'and']` 2pc and Pythia's `[' n', 'and']` 2pc), and Gemma 2 collapses `xor` to `[' xor']` (1pc, unlike OLMo's `[' xor']` 1pc and Pythia's `[' x', 'or']` 2pc). The `negate` canonical is `[' negate']` (1pc) in Gemma and OLMo but `[' neg', 'ate']` (2pc) in Pythia. These per-tokenizer differences are now logged in the v6 audit; they explain part of the model-specific target-distribution finding in §3.7.15 / §3.7.16 P3 (Gemma's 1pc `nand` attracts less mass than OLMo's 2pc `nand`; Pythia's 2pc `xor` attracts more than Gemma's 1pc `xor`), but do not explain OLMo's 15.9% routing to `implies`, which is 1pc in all three tokenizers.

**Net §3.7.16 verdict.** (1) Fact 1 strengthens: cross-notation canonical-operator transfer holds at M2c = 1.000, bootstrap [1.000, 1.000], under 15-class readout across three model families. (2) Fact 2 holds in two of three models (OLMo and Pythia); the Gemma v6 emergent PASS-arity is reported as a methodological caveat on M4b threshold-sensitivity, with causal-patching follow-up flagged as the highest-priority remaining experiment. (3) The default-to-rarest mechanism is multi-factor (frequency × subword × semantic neighborhood), not single-axis; the embedding-similarity follow-up will close the mechanism gap. (4) The lucky-default detector is further vindicated by the v5+v6 OLMo sente→close L10 trajectory. (5) The pre-registration was usefully constraining: the audit gate caught an unforeseeable tokenization failure (`iff` everywhere 1pc), and the falsification verdicts ("NONE supported") are clean and unambiguous, demonstrating that pre-registration discipline does not depend on confirmatory outcomes to be informative.

#### 3.7.17 Causal patching at the v6 emergent + Fact-1 anchor cells (script 25a)

The §3.7.16 v6 expansion produced two findings that linear probes alone could not adjudicate: (Q1) cross-notation canonical-operator transfer — Fact 1, bootstrap-CI [1.000, 1.000] under 15-class readout — is a correlational linear-probe finding, leaving the §5 "linear probes only" caveat open at the principal positive cells; and (Q2) the two Gemma 2 9B v6 emergent PASS-arity cells (`N→F opera→close L 2`, `N→F sente→close L 2`) jumped M4b from chance at v5 to PASS at v6 *without any change to the underlying activations*, with the §3.7.16 reading being that this is a methodological caveat on M4b's threshold-sensitivity to readout granularity rather than a substantive retraction of operator-set-bound substrate-invariance. Script 25a (`experiments/25a_causal_patching.py`) addresses both questions with the same instrument: activation patching with a forward hook on `model.model.layers[L−1]` that replaces the FUNC-PFX target-anchor residual at layer L with the mean NEUTRAL canonical activation at the source anchor, then measures (i) the patched-residual probe readout for sanity, (ii) the behavioural KL shift in the sentence-final next-token distribution against a no-patch FUNC-PFX canonical reference, and (iii) the wrong-arity flip test (intended-unary invented words patched with `and` should ΔKL-positive against the FUNC-PFX-`and` reference iff the cell is causally arity-respecting). Three cells × four conditions (`BASELINE`, `PATCH_not`, `PATCH_and`, `RANDOM_NORM`) × 16 invented words × 10 stimuli per word = 1920 patched forward passes + 60 reference passes. Total runtime ~7.5 min compute (model loads excluded). The v6 carryover NPZ caches from script 24 are reused for source extraction and probe training; no new caches written.

**Methodological note: a one-line hook bug surfaced and was fixed before headline interpretation.** The v1 implementation of script 25a read the patched residual from `out.hidden_states[layer]` (i.e., the captured hidden-states tuple from `output_hidden_states=True`). On MPS with current-generation HF transformers (Gemma 2 + OLMo 2), the forward hook's return value correctly replaced the layer's effective output for downstream computation (next-token logits shifted), but the hidden-states tuple captured the pre-hook output. Result: probe-causality readings of 0.0% across all conditions in v1, while behavioural KL shifts were nonetheless real. The v2 fix replaces the hidden-states read with a capture inside the same forward hook — `_make_patch_capture_hook` directly appends the post-patch slice at `[0, position, :]` into a Python list — guaranteeing the probe sees exactly what subsequent layers see. The v2 sanity check passes cleanly: `P(probe → not | PATCH_not) = 100%` and `P(probe → and | PATCH_and) = 100%` at all three cells (table below). Critically, the v2 behavioural KL block is byte-for-byte identical to the v1 block (same forward pass, just a different read), so all behavioural conclusions are confirmed against the v1 run.

**Cross-cell synthesis table.** Per-cell metrics (`P(probe → not)` is on PATCH_not stimuli; `ΔKL(not)` is mean over 16 invented words of `KL(BASELINE || ref_not) − KL(PATCH_not || ref_not)`; positive = patch pulled behaviour toward the canonical reference; arity-flip ΔKL is the same computed on the wrong-arity-patched subset only):

| Cell | M2-cano | M2-arity | P→not\|p_not | P→and\|p_and | ΔKL(not) | ΔKL(and) | RANDOM ΔKL(not) | RANDOM ΔKL(and) | Arity-flip U→and | Arity-flip B→not |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 9B `opera→close L 2` | 0.420 | 0.841 | 100% | 100% | **+0.048** (15/16+) | **+0.038** (16/16+) | +0.020 (11/16+) | −0.001 (7/16+) | **+0.033** (8/0/8) | **+0.061** (8/0/8) |
| Gemma 9B `sente→close L 2` | 0.512 | 0.832 | 100% | 100% | −0.020 (6/16+) | −0.012 (7/16+) | +0.019 (11/16+) | −0.002 (6/16+) | −0.018 (4/4/8) | −0.004 (3/5/8) |
| OLMo  7B `sente→close L 10` | 0.736 | 1.000 | 100% | 100% | −0.012 (4/16+) | −0.017 (3/16+) | **+0.019 (14/16+)** | **+0.012 (14/16+)** | −0.020 (1/7/8) | −0.006 (3/5/8) |

Three distinct causal verdicts emerge from the table:

**Cell 1 — Gemma 2 9B `opera→close L 2`: causally arity-respecting.** All four indicators line up. Probe-causality is 100% (sanity check, expected if the hook is working). Behavioural ΔKL is positive on both reference axes (+0.048 against `ref_not`, 15/16 words; +0.038 against `ref_and`, *all 16 words positive*) and substantially exceeds the RANDOM_NORM control on both axes (the random-norm baseline against `ref_and` is essentially zero: −0.001, 7/16 positive — clean separation). The decisive datum is the arity-flip block (E): when we patch an intended-unary invented word (e.g., `vusp`, `perph`) with `and` (a canonical binary), all 8 of 8 unary-intended words shift their next-token distribution toward the FUNC-PFX-`and` reference (mean Δ = +0.033, 8/0/8); symmetrically, all 8 of 8 intended-binary words patched with `not` shift toward FUNC-PFX-`not` (mean Δ = +0.061, 8/0/8). The L2 close-paren position in Gemma 2 9B, when patched with the NEUTRAL-`operator-after`-sourced canonical activation, **causally controls the downstream arity-respecting behaviour**. The v6 emergent PASS-arity at this cell is real under causal intervention, not an M4b-granularity artifact.

**Cell 2 — Gemma 2 9B `sente→close L 2`: probe-readable, not causally load-bearing.** Probe-causality is 100% (the residual *is* being modified), but the behavioural KL block is essentially flat: ΔKL(not) = −0.020 (only 6/16 words positive), ΔKL(and) = −0.012 (7/16 positive). The RANDOM_NORM control ΔKL(not) = +0.019 (11/16 positive) *exceeds* the targeted patch ΔKL(not), which is the cleanest possible null signal: a random direction of matched norm causes more behavioural movement toward the `not` reference than the precisely-targeted NEUTRAL-`not` source vector. The arity-flip block is also at chance: unary-intended words patched with `and` shift with mean Δ = −0.018 (4 positive, 4 negative out of 8). **The L2 close-paren position cannot be causally driven from a NEUTRAL-sentence-final source at this cell** — the M4b reading at v6 (M4b = 66.2%) is a probe-readable threshold-crossing, not a causally arity-respecting structure. The §3.7.16 methodological-caveat reading is confirmed for this cell.

**Cell 3 — OLMo 2 7B `sente→close L 10` (the §3.7.9 Fact-1 anchor): probe-readable, not causally load-bearing.** This is the project-flagship cell where M2-arity = 1.000 with M2-canonical = 0.736 under 15-class readout, the cell that anchored Phase 1 / Phase 2's principal positive finding (operator-set-bound canonical-operator transfer). Probe-causality after patching is 100% (the patch reaches the residual cleanly), but the behavioural KL signal is *negative*: ΔKL(not) = −0.012 (4/16 positive), ΔKL(and) = −0.017 (3/16 positive). The RANDOM_NORM control is *strongly positive* (ΔKL(not) = +0.019, 14/16 positive; ΔKL(and) = +0.012, 14/16 positive). Both arity-flip directions fail (U→and = −0.020, 1/7/8; B→not = −0.006, 3/5/8). **The L10 close-paren position in OLMo 2 7B carries arity-discriminative information in linear-probe space (M2-arity = 1.000) but is not on the causal path for downstream next-token computation when patched from NEUTRAL-sentence-final.**

**Three findings worth recording for the paper:**

1. **Fact 1's geometric cross-notation transfer is not uniformly causally load-bearing.** The §3.7.9 OLMo cell — Phase 1's flagship Fact-1 anchor and the cell where the project's lucky-default detector was first vindicated — is causally inert under patching from NEUTRAL-sentence-final. The 100% probe-causality (sanity check passes) confirms the patch reaches the residual; the negative-or-flat behavioural ΔKL confirms downstream computation does not use the patched value at this position. This *does not refute* Fact 1 as a geometric statement about cross-notation linear-probe readouts — that geometric structure remains real, bootstrap-confirmed at three model families. But it does refine §5's "linear probes only" caveat to a concrete answer: cross-notation canonical-operator structure at this cell is *geometric, not load-bearing*. A more comprehensive Fact-1 causal test would patch at multiple (source, target) anchor pairs to identify which combinations are load-bearing.

2. **The Gemma v6 emergent PASS-arity finding splits 1:1.** The `opera→close L 2` cell is a genuine model-specific exception to operator-set-bound substrate-invariance: at this cell in Gemma 2 9B, the residual stream representation at L2 close-paren causally encodes the operator's arity in a way that transfers from NEUTRAL operator-after, and patching with the wrong-arity canonical produces clean arity-flip behaviour (8/0/8 in both directions, ΔKL = +0.033 / +0.061). The `sente→close L 2` cell is a probe-only artifact of M4b granularity, as §3.7.16 hypothesised. The operator-set-bound headline therefore needs one tightly-scoped Gemma-specific caveat in §4.1, not a wholesale retraction: **Gemma 2 9B has a single causally-validated cell at L2 close-paren (operator-after sourced) where novel-operator activations behave arity-respectingly under intervention**, while OLMo 2 7B and Pythia 6.9B-d remain operator-set-bound at all tested cells under both linear-probe and causal-patching tests.

3. **Source anchor is a first-class causal variable.** Both Gemma cells use the same target anchor (FUNC-PFX close-paren), same layer (L 2), and both achieve identical probe-causality (100% / 100% — the patches reach the residual equally cleanly). They differ *only* in the source: NEUTRAL operator-after vs NEUTRAL sentence-final. The behavioural verdicts are opposite. The cleanest reading: at L2 in Gemma 2 9B, the operator-after position carries arity-discriminative information that is *geometrically aligned* with FUNC-PFX close-paren L2 (transplants cleanly and downstream layers use it); the sentence-final position at L2 carries similar information in linear-probe space (the probe-causality test shows both source vectors fall in the same canonical decision regions) but in a *different residual-stream coordinate system* that downstream layers do not parse as the canonical's arity. This dissociation — same target, same layer, same probe reading, opposite causal verdicts — is a methodological signature future causal-intervention studies should anticipate: probe-causality is necessary but not sufficient for causal load-bearingness.

**Net §3.7.17 verdict.** (1) Operator-set-bound substrate-invariance remains the headline finding at three model families across linear-probe transfer, but Fact 1's causal grounding is anchor-pair-dependent: the OLMo flagship cell is geometric/probe-readable but not causally load-bearing under NEUTRAL-sentence-final patching. (2) The Gemma v6 emergent PASS-arity at `opera→close L 2` is causally real and constitutes a single tightly-scoped model-specific exception to operator-set-bound for Gemma 2 9B; the other emergent cell (`sente→close L 2`) is M4b-granularity-only as §3.7.16 hypothesised. (3) The principal new methodological finding is that source-anchor is a first-class causal variable independent of probe-causality; future probe-based substrate-invariance work should include same-target / different-source causal tests as part of the standard battery. The §6 next-experiment block now pivots to the embedding-similarity probe (script 25b) to close the multi-factor default-mechanism gap from §3.7.16; once 25b lands, the paper draft is in scope.

#### 3.7.18 Embedding-similarity probe vs probe per-word routing (script 25b)

The §3.7.16 v6 disentanglement test rejected all three pre-registered single-axis readings of the default-to-rarest mechanism (P_FREQ, P_SUBWORD, P_INTERACTION) in all three model families and tentatively named a third factor — *contextual semantic neighborhood* — as the most likely missing mechanism component. The script 25b mechanism-closing test directly operationalises that hypothesis: for each of three models × 80 v6 sweep cells, compute the cosine-similarity argmax between each invented word's mean activation and each canonical's mean activation at the same `(target_cond, target_anchor, layer)` coordinate the probe predicts on, and ask whether that argmax matches the probe's empirical per-word top canonical. Two variants: unconstrained `sim-all` (chance 1/15 ≈ 6.7%) and arity-conditioned `sim-arity` (chance ≈ 13.3% averaged over binary/unary-intended words). Plus a coarser `arity-match` metric: does the unconstrained sim-all top have the same arity as the probe top? (Chance ≈ 53.3% under the majority-arity baseline.) Bootstrap 95% CIs by resampling the 16-word invented set with replacement, B = 200, mirror script 22a's protocol. Cache-only on script 24's v6 carryover NPZ caches; runtime 32 s (Gemma) + 18 s (OLMo) + 42 s (Pythia) = 1.5 min plus the L0 add-on; no model loads.

**Cross-model headline table** (mean over all 80 sweep cells per model at the focus layers, with bootstrap 95% CIs):

| Model | `agree-all` | `agree-arity` | `arity-match` |
|---|---|---|---|
| Gemma 2 9B    | **11.6%** [9.6, 13.8]  | 11.4% [9.1, 13.4]   | 54.1% [49.8, 58.2] |
| OLMo 2 7B     | **26.6%** [24.1, 28.6] | 21.5% [18.4, 24.6]  | 66.4% [64.0, 68.4] |
| Pythia 6.9B-d | **24.0%** [21.4, 26.5] | 19.0% [15.6, 22.5]  | 60.2% [57.7, 62.6] |

The pre-registered §6 threshold for "the mechanism gap is closed" was `agree-arity ≥ 60%`. **No model meets this threshold; all three are well below**: Gemma's CI [9.1, 13.4] does not even reach the ~13.3% within-arity chance baseline, and OLMo and Pythia are at modest 1.4-1.6× chance. The §3.7.16 multi-factor mechanism does **not** reduce to "softmax over arity-conditioned semantic neighborhood" at mean-pooled-residual-cosine resolution.

**Three findings worth recording for the paper:**

1. **Mean-pooled cosine similarity captures arity but not the per-canonical routing.** All three models have `arity-match > agree-all > agree-arity`, with the gap between `arity-match` (~54-66%) and `agree-arity` (~11-22%) being the principal positive signal. The unconstrained cosine argmax tends to pick a canonical of the same arity as the probe's top, but conditioning on intended arity destroys the predictive power — within-arity, the probe distinguishes between canonicals via decision-boundary geometry that mean-pooled cosine cannot reproduce. The headline interpretation: similarity is *necessary* but not *sufficient* for the v6 default mechanism.

2. **The collapsed/distributed partition is the cleanest informative result.** At collapsed cells (M4c ≥ 0.7, where one canonical dominates ≥ 84% of invented words), cosine often picks the same canonical: OLMo 45.6%, Pythia 39.4% identity-agreement. At distributed cells (M4c < 0.7, the methodologically interesting regime where the probe spreads predictions across multiple canonicals), agreement drops to floor:

| Model | distributed n | distributed `agree-all` | distributed `agree-arity` | distributed `arity-match` |
|---|---|---|---|---|
| Gemma 2 9B    | 65 | 11.6% | 10.7% | 53.3% |
| OLMo 2 7B     | 49 | 14.8% | 15.6% | 62.4% |
| Pythia 6.9B-d | 53 | 16.2% | 13.9% | 58.5% |

   At distributed cells across all three models, identity-level cosine agreement is barely above the 6.7% chance baseline and arity-conditioned agreement is at within-arity chance. The probe's *per-word-specific* routing at these cells is uniformly **not** captured by mean-pooled-cosine similarity. The per-word breakdown shows the failure mode concretely: at every distributed cell across all three models, the cosine argmax collapses every invented word to a single canonical (`nand` for the unconstrained variant, `nand` for binary-intended and `negate` for unary-intended under arity-conditioning), while the probe routes per-word to a much wider set (`not`, `unprovably`, `negate`, `xor`, `implies`, `or`, etc.) in word-specific patterns. The default-to-rarest mechanism therefore involves at least one factor beyond mean-pooled-residual cosine geometry — most likely *probe-decision-boundary geometry* shaped by the LogisticRegression's learned weights, which is sensitive to per-word residual-stream structure that the mean-pool wipes out.

3. **L0 confirms the script 14 finding under v6 across three model families.** At the embedding layer L 0, all three models collapse to floor agreement: Gemma `agree-all` = 0.8%, OLMo = 0.0%, Pythia = 0.0%; arity-match = 16.4% / 1.6% / 2.3% (at-or-below chance). The probe's per-word routing structure is built during forward pass, not inherited from token-embedding geometry — a clean v6 cross-family re-confirmation of script 14's original OLMo-7B finding that "H1 is constructed by attention/MLP processing in layers 1-7, not inherited from layer-0 geometry." This generalises the script 14 finding from single-model-single-layer to three-model-multi-layer under the v6 readout vocabulary, and it strengthens §3.7.16's interpretation: contextual semantic neighborhood, to the extent it captures the mechanism at all, lives at intermediate layers and is not present at L 0.

**Per-layer / per-direction structure.** OLMo and Pythia both peak at L 10 (`agree-all` 41.0% / 38.7%, `arity-match` 77.0% / 69.1%), well above their respective Fact-1 cells at L 4 (32.8% / 26.6%). Gemma peaks at L 4 (`agree-all` 27.0%, `arity-match` 66.4%) and degrades sharply at L 8-L 17. All three models show late-layer (L 24 / L 16-17) collapse on identity-agreement (`agree-all` 2-7%) with `arity-match` remaining elevated (63-83% in OLMo/Pythia, 41-44% in Gemma). The mid-layer peak is consistent with the script 14 / §3.7 picture: the cross-notation arity-region structure is *constructed* at mid-depth (L 4-L 10) and *decoded into specific operator/lexical identities* at late layers (L 16-L 24), with the late-layer residual stream's identity content being orthogonal to canonical mean activations under cosine. The N→F direction has uniformly higher agreement than F→N (Gemma 15.5% vs 7.7%, OLMo 40.9% vs 12.5%, Pythia 31.2% vs 16.7% on `agree-all`) — consistent with FUNC-PFX being the more structurally-cleaner notation for cross-notation transfer at the canonical-operator level.

**A specific geometric anomaly worth naming.** Across all three models at distributed cells, the unconstrained cosine argmax collapses to **`nand` for binary-intended words** with near-universal frequency. The mechanism is most likely a tokenization-driven structural signature: `nand` is multi-subword in OLMo (`[' n', 'and']`) and Pythia (`[' n', 'and']`) but 1pc in Gemma (`[' nand']`), and yet Gemma still shows `nand` as the consistent unconstrained-cosine attractor at distributed cells. The more general pattern is that `nand`'s mean canonical activation is consistently *further from the canonical centroid* than higher-frequency canonicals (because `nand` has the most idiosyncratic per-stimulus activations in the v6 set), so under cosine similarity invented words — which are also far from any sensible centroid — end up closest to `nand`. This is a near-tautological geometric structure that does not reflect the probe's actual decision boundary: the LogisticRegression boundary is optimized to discriminate canonicals from one another, and is robust to the magnitude-driven cosine collapse. The script 25b finding therefore additionally constitutes a methodological caveat on raw cosine-similarity readouts of substrate-invariance: **mean-pooled cosine on residual-stream activations is dominated by canonical-magnitude-and-idiosyncrasy effects that distort the apparent semantic-neighborhood structure**.

**Net §3.7.18 verdict.** The §3.7.16 mechanism gap is **not** closed by mean-pooled cosine similarity at the focus layers. The default-to-rarest mechanism's third factor (beyond frequency and subword shape) is *probe-decision-boundary geometry*, not raw geometric proximity — i.e., the probe routes invented words to canonicals via *learned discriminative weights* that capture per-word residual-stream structure not preserved by mean-pooling. This is itself a substantive empirical finding with two paper-relevant consequences: (a) the §3.7.16 pre-registration's tentative third factor ("contextual semantic neighborhood") is empirically falsified in its mean-pooled-cosine operationalisation — what remains is "probe-decision-boundary geometry," a more honest and less reducible characterisation; (b) the script 14 H1-construction finding generalises cleanly to three model families under v6, with L 0 cosine agreement at floor and mid-layer agreement peaking at L 4-L 10. With §3.7.16, §3.7.17, and §3.7.18 complete, the experimental programme is essentially closed and paper.md drafting is in scope. The remaining open mechanism question — what specifically about the LogisticRegression's learned weights produces per-word-specific routing that mean-pooled cosine cannot reproduce — is a follow-up for sparse-autoencoder feature labelling (see §6) and is not blocking the paper.

#### 3.7.19 Two reviewer-round-1 follow-up cells: source-anchor direction-specificity disambiguation (script 25a-extra)

The paper.md v1 draft hedged the §3.7.17 source-anchor finding as "first-class causal variable but with insufficient sample for the strongest reading" — three cells, all three sentence-final-sourced patches failing and the one operator-after-sourced passing. External reviewer round 1 flagged the ambiguity directly: the data are consistent both with the "operator-after sources reliably drive causal effects, sentence-final sources reliably fail" reading *and* with a "joint-source-target product" reading where neither source anchor is deterministic and the per-cell verdict is the product of source and target idiosyncrasy. Script 25a was extended in place by adding two new `PATCH_CELLS` entries (`CELL_FILTER=extra` runs only these) to disambiguate.

**The two new cells.** (1) **OLMo 2 7B `opera→close L 10`**: same target as the inert §3.7.17 OLMo flagship cell, but with operator-after source instead of sentence-final — a within-target source-anchor flip. (2) **Gemma 2 9B `opera→opera L 4`**: the principal Fact-1 anchor where M2c = 1.000 with bootstrap CI [1.000, 1.000] across all three model families — a causal-grounding test at the cell that anchors the paper's principal positive result.

**Per-cell metrics.**

| Cell | M2c | M2a | P→not\|p_not | P→and\|p_and | ΔKL(not) | ΔKL(and) | RND(not) | RND(and) | U→and (mean / +//−//n) | B→not (mean / +//−//n) |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma `opera→opera L 4` | 1.000 | 1.000 | 100.0% | 100.0% | **+0.033** (13/16+) | **+0.027** (12/16+) | +0.017 | +0.021 | +0.028 / 6/2/8 | +0.039 / 7/1/8 |
| OLMo `opera→close L 10` | 0.508 | 0.872 | 100.0% | 100.0% | +0.013 (15/16+) | +0.003 (10/16+) | **+0.023** | **+0.016** | +0.001 / 5/3/8 | +0.021 / 8/0/8 |

**Gemma `opera→opera L 4`: WEAK PASS.** ΔKL is positive on both reference axes (+0.033 / +0.027) and exceeds RANDOM_NORM by 1.94× on `not` and 1.29× on `and`. Arity-flip is moderately positive in both directions (75% / 87.5% positive). This is a *real but modest* causal effect — substantially weaker than the §3.7.17 Gemma `opera→close L 2` PASS where RANDOM_NORM was near zero and arity-flip was 8/0/8 cleanly. The signal is real (the geometric Fact 1 structure at this cell does causally drive downstream behaviour to some degree) but the noise floor is also positive, indicating that mean-pool patching at L 4 operator-after produces modest generic-disruption effects on top of any specific-canonical signal. **Verdict: the principal cross-family Fact-1 cell is causally load-bearing at ~1.5-2× the RANDOM_NORM floor.**

**OLMo `opera→close L 10`: FAIL.** Same fail signature as the original sente-sourced version of this target cell: RANDOM exceeds targeted on both axes (RND `not` = +0.023 vs targeted +0.013; RND `and` = +0.016 vs targeted +0.003). The arity-flip block is split: unary→and is at chance (5/3/8); binary→not is 8/0/8 positive *but this aligns with the model's natural default* at this cell (BASELINE routes 100% of invented mass to `nand`, the binary default). **Verdict: OLMo L 10 close-paren is causally inert under BOTH tested source anchors** — the within-target source-anchor flip does not rescue this target.

**Cross-cell 5-cell synthesis (3 original + 2 extras).**

| Cell | Source | Target | ΔKL(not) | ΔKL(and) | RND(not) | RND(and) | Arity-flip | **Verdict** |
|---|---|---|---|---|---|---|---|---|
| Gemma `opera→close L 2` | opera | close L 2 | +0.048 | +0.038 | ~0 | ~0 | 8/0/8 both | **CLEAN PASS** |
| Gemma `opera→opera L 4` ★ | opera | opera L 4 | +0.033 | +0.027 | +0.017 | +0.021 | 6/2/8, 7/1/8 | **WEAK PASS** |
| Gemma `sente→close L 2` | sente | close L 2 | −0.020 | −0.012 | +0.019 | ? | chance | **FAIL** |
| OLMo `sente→close L 10` | sente | close L 10 | −0.012 | −0.017 | +0.019 | +0.012 | 1/7/8, 3/5/8 | **FAIL** |
| OLMo `opera→close L 10` ★ | opera | close L 10 | +0.013 | +0.003 | +0.023 | +0.016 | 5/3/8, 8/0/8 | **FAIL** |

★ = new from §3.7.19.

**Three revised findings for the paper.**

1. **Source-anchor direction-specificity is NOT deterministic.** The §3.7.17 hedge — "all sentence-final-sourced patches in our three-cell sample failed; the one operator-after-sourced passed" — is updated by the new evidence. Operator-after sources do not reliably produce causal effects across targets: Gemma `opera→opera L 4` is only WEAK PASS (1.5-2× RANDOM), and OLMo `opera→close L 10` is clean FAIL. The "operator-after sources reliably PASS, sentence-final sources reliably FAIL" reading is empirically rejected; the cleanest dissociation we have is the within-target source-anchor flip at Gemma L 2 close-paren (opera passes cleanly, sente fails cleanly at same target/layer), and that dissociation **does not generalise across targets**.

2. **OLMo L 10 close-paren is causally inert regardless of source.** The within-target source-anchor flip (the new OLMo cell) yields the same FAIL verdict as the original. This is a *stronger* version of §3.7.17's OLMo finding. The reviewer's protocol-confound (b) — "L 10 close-paren may be read out by L 11-32 through a FUNC-PFX-specific attention pattern that does not propagate a NEUTRAL-sourced patched signal" — now has clean empirical support: the probe-causality is 100% under both source anchors (the patch *does* propagate the source-canonical signal to the residual), but the FUNC-PFX downstream computation does not use that signal regardless of which NEUTRAL source it was. **The §3.7.9 / §4.2 OLMo cell is probe-readable but the target position itself is causally inert under our patching protocol, independent of source anchor.**

3. **Fact 1's principal geometric cell is only weakly causally load-bearing.** Gemma `opera→opera L 4` is the project-flagship cell — M2c = 1.000 with bootstrap CI [1.000, 1.000] across all three model families. Its causal signal is real (ΔKL positive on both axes, 12-13/16 invented words positive, RANDOM-exceeded by 1.3-2×) but *substantially weaker* than the Gemma `opera→close L 2` PASS. **The only cell with clean causally-arity-respecting behaviour in the project is the v6 emergent Gemma L 2 close-paren cell**, not the Fact-1 anchor. The geometric Fact 1 result (M2c = 1.000) does not translate into clean causal load-bearingness even at its strongest cell, and the Phase 1 / Phase 2 / v6 "operator-set-bound" headline is therefore more accurately read as a *geometric / linear-probe-readable* substrate-invariance claim, with causal grounding holding cleanly at one specific (source, target, layer) joint product (Gemma `opera→close L 2`) and weakly or not at all elsewhere.

**Methodological note.** The RANDOM_NORM control's positive ΔKL at three of the five cells (Gemma L 4 opera-source; OLMo L 10 under both sources) is informative about the protocol's noise floor. At cells where the targeted patch is cleanly causally arity-respecting (Gemma L 2 close-paren opera-source), RANDOM_NORM is near zero. At cells where the targeted patch is only weakly arity-respecting or inert, RANDOM_NORM produces a modest positive ΔKL of similar magnitude (~0.015-0.023). The most likely reading: norm-matched random vectors at certain (anchor, layer) coordinates produce a generic "disruption" effect that pulls behaviour toward a default canonical (`definitely` reads the RANDOM probe top consistently — a unary default). When the specific-canonical signal is absent, only the generic-disruption effect is visible. When the specific-canonical signal is strong, it dwarfs the generic effect. This makes the RANDOM_NORM control more useful than a pure null — it implicitly characterises the cell's *cleanness* by the targeted-vs-random ratio rather than just the targeted-vs-zero signal.

**Net §3.7.19 verdict.** The 5-cell evidence supports a more nuanced picture than the v1 draft hedge. (a) Source-anchor is a non-trivial causal variable at one specific cell (Gemma L 2 close-paren) but is not directionally deterministic across cells; the verdict is the joint product of source, target, and layer. (b) OLMo L 10 close-paren is now demonstrated causally inert under both tested source anchors, strengthening the §3.7.9 / §4.2 Fact-1-anchor reading from "probe-readable but causally inert under NEUTRAL-sentence-final patching" to "probe-readable but causally inert under either NEUTRAL source anchor tested". (c) Fact 1's principal cross-family cell at Gemma L 4 operator-after is weakly causally load-bearing (1.3-2× RANDOM); the only clean causal PASS in the project is the Gemma L 2 close-paren v6 emergent cell. The paper.md headline should accordingly soften "causal grounding for Fact 1" claims and lean into the more interesting "geometric Fact 1 does not uniformly imply causal load-bearingness" reading. The §3.7.17 source-anchor finding remains a methodological contribution — within-target source-anchor flip *can* produce opposite verdicts — but is downgraded from "first-class causal variable" framing to "joint-source-target product" framing.

#### 3.7.20 Corpus-frequency / lexical-identity control (script 25c): Fact 1 reframes from operator-set-bound to trained-vocabulary-bound

External reviewer round 1's primary scientific concern: Fact 1's headline `M2-canonical = 1.000` (15-class, ceiling under bootstrap CI `[1.000, 1.000]`) might be a *generic* substrate-independent lexical-identity signal — the probe finds a per-token-identity hyperplane that would also reach ceiling for any 15-word readout vocabulary, not specifically a logical-operator-class abstraction. The pre-registered control: re-run M1-M4 with 15 heterogeneous non-operator content words inserted into syntactically-identical NEUTRAL and FUNC-PFX templates. **The control triggers REFRAME in all three model families at the principal Fact-1 cell.**

**Stimulus design.** 15 content words spanning subword-length range matched to the v6 canonical set: 8 intended-binary (`house`, `water`, `music`, `light`, `paper`, `pattern`, `theory`, `system`) and 7 intended-unary (`region`, `period`, `archipelago`, `mosaic`, `plinth`, `ledger`, `cassowary`). Multi-piece tokenization rate: 2/15 (Gemma SentencePiece: `plinth` 2pc, `cassowary` 3pc), 3/15 (OLMo BPE: `archipelago` 3pc, `plinth` 2pc, `cassowary` 3pc), 4/15 (Pythia BPE: `archipelago` 3pc, `plinth` 2pc, `ledger` 2pc, `cassowary` 3pc) — comparable to v6's 1/15 multi-piece (`unprovably`). NEUTRAL and FUNC-PFX templates are byte-identical to the v6 canonical battery; only the operator-position word is substituted. The "intended-arity" assignment is a syntactic-position match against the v6 binary-vs-unary split (not a semantic claim — `house` is not a binary operator in English) so that the FUNC-PFX template emits `house(p, q)` and `region(p)` in the same syntactic position the v6 canonicals occupy.

**Pre-registered adjudication.** If `M2-canonical ≥ 0.65` on content words at the principal `N→F operator-after → operator-after L 4` cell in *any* of the three models, REFRAME the paper.md framing from "operator-set-bound" to "trained-vocabulary-bound". The 0.65 threshold is identical to the §3.7.5 substrate-invariance battery's M2-canonical gate (≈ 10× the 15-class chance of 0.067). The reframe was specified *before* any 25c cache extraction.

**Results (M2-canonical, content words, with bootstrap 95% CI from `B = 500` stim-resamples).**

| Model | Cell | M2c (content) | CI (content) | M2c (v6 canonical, §3.7.16) | Verdict |
|---|---|---|---|---|---|
| Gemma 2 9B | `N→F opera→opera L 4` | **1.000** | [1.000, 1.000] | 1.000 [1.000, 1.000] | REFRAME |
| Gemma 2 9B | `N→F opera→opera L 8` | 1.000 | [1.000, 1.000] | — | REFRAME |
| Gemma 2 9B | `F→N opera→opera L 4` | 0.836 | [0.820, 0.852] | — | REFRAME |
| OLMo 2 7B | `N→F opera→opera L 4` | **1.000** | [1.000, 1.000] | 1.000 [1.000, 1.000] | REFRAME |
| OLMo 2 7B | `N→F opera→opera L 10` | 1.000 | [1.000, 1.000] | — | REFRAME |
| OLMo 2 7B | `F→N opera→opera L 4` | 0.376 | [0.357, 0.396] | — | operator-class-asymmetric |
| Pythia 6.9B-d | `N→F opera→opera L 4` | **1.000** | [1.000, 1.000] | 1.000 [1.000, 1.000] | REFRAME |
| Pythia 6.9B-d | `N→F opera→opera L 10` | 1.000 | [1.000, 1.000] | — | REFRAME |
| Pythia 6.9B-d | `F→N opera→opera L 4` | 0.943 | [0.932, 0.956] | — | REFRAME |

Probe within-condition CV at the same cells: `M1n` = 0.999 / 0.959 / 1.000; `M1f` = 1.000 / 1.000 / 1.000 (Gemma / OLMo / Pythia). The content-word probe is structural and the cross-notation transfer is exactly as clean as for canonical operators.

**Two findings worth recording.**

(1) **The principal `N→F operator-after → operator-after` direction is at ceiling for content words across all three model families, byte-identically to the v6 canonical-operator result.** This is the load-bearing observation. The pre-registered reframe trigger fires unambiguously.

(2) **The reverse `F→N opera→opera L 4` direction is asymmetrically weaker for content words, model-specifically.** Gemma drops to 0.836 (clean PASS), Pythia to 0.943 (clean PASS), OLMo to 0.376 (clean FAIL under the 0.65 threshold). The OLMo content-word F→N drop is a model-specific phenomenon that does not appear at the same cell with canonical operators — operator F→N transfer in OLMo at this cell sits above 0.7 per the v6 sweep. The reading: OLMo 2 7B's FUNC-PFX content-word representations have a noisier per-class structure than its FUNC-PFX operator representations, such that a probe trained on FUNC-PFX content-word activations does not generalise to NEUTRAL content-word activations cleanly. This may reflect higher per-stimulus context-driven variability for common English nouns in function-call syntax compared to logical operators in the same syntax, in OLMo's BPE-tokenizer activation distribution specifically. The asymmetry is interesting but does not weaken the principal REFRAME trigger (N→F is the principal Fact-1 direction).

**Reframe consequences.** The headline framing throughout the paper updates from "operator-set-bound substrate-invariance" to "trained-vocabulary-bound substrate-invariance". Specifically: Fact 1 generalises to any in-vocabulary word at the operator-after anchor under FUNC-PFX↔NEUTRAL notation swap, not just to logical operators. Fact 2 (novel-operator generalisation failure under canonical-set expansion) is preserved as-stated — the negative result is specifically about *novel* invented words, and we have not tested novel content words for the same retraction pattern. The combined two-part finding becomes: *trained-vocabulary substrate-invariance holds at ceiling for the model's known vocabulary; novel-operator generalisation fails by default-to-rarest-canonical compression*. This is a strictly more general positive claim (the substrate-invariance applies to any known word, not just operators), and a strictly equally-scoped negative claim (we have only tested novel operators, not novel content words; the latter is an open follow-up).

**Methodological note.** The §4.1.1 control is a directly transferable methodological device: any probe-based substrate-invariance study reporting ceiling-level cross-context transfer should include a within-vocabulary content-word control to dissociate domain-specific vs generic lexical-identity hyperplanes. Without the control, the v1 draft's "operator-set-bound" framing was a step beyond the data. With it, the claim correctly reframes to "trained-vocabulary-bound" — a broader and somewhat less surprising claim, but the one actually supported. The reviewer was correct.

**Net §3.7.20 verdict.** Fact 1 reframes to *trained-vocabulary substrate-invariance*: cross-notation linear-probe geometry transfers at ceiling for any well-tokenized word in the model's training vocabulary, at the operator-after anchor in the early-to-mid layers (L 4 in Gemma, L 4-L 10 in OLMo and Pythia), in the N→F direction, in all three model families. The "operator-set-bound" framing is updated; the negative Fact 2 (novel-operator generalisation failure) is preserved. The paper.md title, abstract, §1 intro, §4.1, §5.1, §6, and §7 require corresponding updates.

#### 3.7.21 Pre-registration criterion-drift reconciliation (script 24b)

**Trigger.** External-review round 1 (2026-05-21) flagged a mismatch between the v6 pre-registration's PASS-arity criterion and the running v6 sweep code's PASS-arity criterion. The reviewer's observation: "a reader comparing the public preregistration_v6.md against paper.md §3.5 could conclude that the preregistered threshold was rewritten after analysis". The mismatch is real.

**The two criteria.**

- **Frozen pre-registered criterion (`experiments/preregistration_v6.md` §5, dated 2026-05-20, before any v6 extraction):**

  `M2-arity ≥ 0.65 ∧ M4b ≥ 0.65 ∧ max_c p_c ≤ 0.85 ∧ M4a ∈ [0.20, 0.80] ∧ pwmin < 0.95`

  where `max_c p_c` is the maximum share of invented mass on any single canonical (a max-fraction concentration statistic).

- **Running-code criterion (`experiments/24_v6_canonical_expansion.py` lines 230-233, 941-943):**

  `M2-arity ≥ 0.65 ∧ M4b ≥ 0.65 ∧ HHI < 0.70 ∧ M4a ∈ [0.10, 0.90] ∧ pwmin < 0.95`

  where `HHI = Σ_c p_c²` is the Herfindahl-Hirschman concentration index (which equals `1/K ≈ 0.067` at uniform routing for K = 15 canonicals and 1 at single-canonical collapse, with intermediate "distributed" routing flagged at `< 0.70`).

The drift between the two criteria affects two of the five PASS-arity conjuncts: the M4c statistic (max-fraction vs HHI) and the M4a balanced-arity band ([0.20, 0.80] vs [0.10, 0.90]). The drift was discovered post-extraction at external-review time, not pre-extraction; the most likely root cause is a refactor during script-24 development from the script-23 v5 codepath (which used max-fraction) to the v6 codepath (which adopted HHI as a tighter "distributed routing" concentration measure), without the running code being explicitly synced to the §3.5 prose of the manuscript.

**The reconciliation experiment (script 24b).** `experiments/24b_frozen_criterion_rederivation.py` reuses script 24's `Scope`, `SweepCell`, `run_cell`, and `enumerate_cells` machinery unchanged (so probe fits, M4 breakdowns, lucky-default detection, and cell-cell ordering are byte-identical to the principal v6 sweep) and re-applies BOTH criteria to every cell. Output: `experiments/outputs/24b_20260521_120258.log`. Runtime: ~30 min total on CPU across the three models, dominated by sklearn LogisticRegression refits.

**Headline result.** The v6 P_RETRACT verdict is identical under both criteria in all three model families:

| Model | v6 PASS-arity cells (running) | v6 PASS-arity cells (frozen) | Disagreement |
|---|---|---|---|
| Gemma 2 9B | 2 (`N→F opera→close L 2` M4b=82.3%, `N→F sente→close L 2` M4b=66.2%) | 2 (same cells) | 0 |
| OLMo 2 7B | 0 | 0 | 0 |
| Pythia 6.9B-d | 0 | 0 | 0 |

The two Gemma emergent PASS-arity cells (§3.7.16 / §4.5 Cell 1 + Cell 3) are PASS-arity under both criteria. The OLMo and Pythia v6 "zero PASS-arity cells" P_RETRACT verdict holds under both. The headline of the paper is unchanged.

**The one verdict disagreement anywhere in the 4-scope × 3-model sweep.** Pythia v3 `N→F opera→close L 16` is PASS-arity under the running criterion (M4a = 0.192 is within [0.10, 0.90]) but ARITY-AXIS-ONLY under the frozen criterion (M4a = 0.192 < 0.20). M2-arity = 1.000, M4b = 0.744, max-c = 0.778, HHI = 0.64, pwmin = 0.84 — none of these differ between criteria; the verdict flip is driven entirely by the M4a band width.

The disagreement is therefore **driven by the M4a balanced-arity band, not the M4c concentration definition**. Substantively: the pre-registration's [0.20, 0.80] band is the tighter "balanced-arity routing" requirement (invented mass must be at least 20% / at most 80% in either arity class for the cell to count as substantively-balanced), while the running code's [0.10, 0.90] band admits cells with up to 10% / 90% arity-class skew. Pythia v3 L 16 sits in the 10%-19% unary-mass corner of this difference: the cell routes 80.8% of invented mass to binary canonicals and only 19.2% to unary canonicals — formally "balanced enough" under running, "unary-mass-deficient" under frozen.

**Cross-scope retraction-chain comparison.** All v3 PASS-arity candidates retract by v5 or v6 under both criteria:

- **Gemma 2 9B**: 2 v3 PASS-arity cells (`opera→first L 4`, `sente→first L 8`) retract to M2A-ONLY by v4 under both criteria; not survivors at v5 or v6.

- **OLMo 2 7B**: 3 v3 PASS-arity cells (`F→N first→opera L 7`, `N→F sente→close L 10`, `N→F opera→close L 24`). Under both criteria: `F→N first→opera L 7` retracts to M2A-ONLY at v4; `sente→close L 10` survives v4 PASS-arity then collapses to LUCKY-NEG at v5 (single-canonical `nand` collapse, the lucky-default-detector signature); `opera→close L 24` survives v4 then retracts to M2A-ONLY at v5.

- **Pythia 6.9B-d**: 3 frozen v3 PASS-arity cells (`opera→close L 4`, `opera→close L 7`, `sente→close L 10`) + 1 running-only cell (`opera→close L 16`). Under both criteria, all retract by v5: `opera→close L 4` → M2A-ONLY at v5; `opera→close L 7` → M2A-ONLY at v5; `sente→close L 10` → LUCKY-NEG at v5 (M4c = HHI 1.00 = max-c 1.00 single-canonical collapse, pwmin = 0.98) then M2A-ONLY at v6 (the v5→v6 differential is because v6 adds enough canonicals that the routing distributes again, but M4b stays at chance 0.500). The `opera→close L 16` running-only cell retracts to M2A-ONLY at v4 under both criteria.

**Headline retraction-chain candidate count.** Under the frozen pre-registered criterion: 2 Gemma + 3 OLMo + 3 Pythia v3 PASS-arity cells = **8 retracted PASS-arity candidates total**, all retracting by v5 or v6. Under the running-code criterion: 2 + 3 + 4 = 9 cells, with the additional cell being the Pythia v3 L 16 case. Paper.md §4.3 uses the frozen count (8 cells) as the criterion-of-record and reports the running-only L 16 cell as a §3.5 criterion-drift footnote.

**Reconciliation with the pre-pre-registration Phase 1 sweep (script 22b).** The Phase 1 (script 22b) sweep on its own pre-v6 caches flagged a related but not byte-identical four-cell candidate set: OLMo `N→F sente→close L 10`, Gemma `N→F sente→first L 8`, Gemma `N→F sente→opera L 4`, OLMo `F→N first→opera L 7`. Comparing against the v6-pipeline v3 replay (24b): 3 of 4 Phase 1 cells reappear (OLMo sente→close L 10 ✓, OLMo first→opera L 7 ✓, Gemma sente→first L 8 ✓); the fourth Phase 1 cell (Gemma `sente→opera L 4`, M4b = 0.669 borderline in 22c) does not pass the v6-pipeline frozen criterion at v3. The v6-pipeline replay additionally identifies cells the Phase 1 sweep missed: Gemma `opera→first L 4`, OLMo `N→F opera→close L 24`, and all 3 Pythia v3 cells (Phase 1 was OLMo + Gemma only). The lineage discrepancy almost certainly traces to stimulus-generation seed differences between script 22b's pre-pre-registration extraction and script 24's v6 extraction. The paper adopts the v6-pipeline list as the published headline (criterion-of-record) and retains the Phase 1 narrative only in this lab-notes lineage.

**Bottom-line interpretive verdict.** The paper's central empirical claims — Fact 1 (ceiling cross-notation transfer for trained vocabulary at the principal cell), Fact 2 (novel-operator generalisation failure under canonical-set expansion in all three models), the v6 P_RETRACT verdict (zero PASS-arity at v6 modulo the two Gemma L 2 emergent cells), and the Gemma 2 9B `opera→close L 2` causally-validated single-model exception — are all robust to which of the two criteria is applied. The criterion-drift, while a real provenance defect, does not change any headline result. The §3.5 amendment in paper.md acknowledges the drift explicitly, adopts the frozen criterion as the criterion-of-record going forward, and forward-points to this §3.7.21 reconciliation for full details.

**Methodological lesson.** Any pre-registered analysis script should be sanity-checked against the pre-registration text immediately after every code refactor, ideally via an automated assertion that compares the threshold constants in the running code against a parsed copy of the pre-registration. The script 24 → script 24b iteration is a low-cost retrofit of this idea — a cache-only "frozen-criterion replay" that takes ~30 min total on CPU and produces a one-page comparison report — and is a directly transferable methodological device for any project that pre-registers a multi-conjunct PASS verdict on a complex sweep.

#### 3.7.22 Δ_specific per-word bootstrap on Cell 2 (script 25d)

**Trigger.** External-review round 1 flagged the Cell-2 WEAK PASS verdict in paper.md §4.5 (Gemma 2 9B `opera→opera L 4`, the principal cross-family Fact-1 anchor) as sitting at the soft boundary of the 1.3-2× mean-ratio band: behavioural ΔKL exceeds RANDOM_NORM by approximately 1.94× on `not` and 1.29× on `and`, with the `and` ratio just above the lower edge of the heuristic band. The §6 "Causal evidence is partial" paragraph in paper.md flagged the cleaner statistic as `Δ_specific = ΔKL_targeted − ΔKL_random`, bootstrapped per-invented-word across the 16-word set.

**Script 25d.** `experiments/25d_delta_specific_bootstrap.py` parses the per-word ΔKL table from `outputs/25a_20260521_085745.log` (the reviewer-round-1 follow-up run containing Cell 2) and computes:

- per-word `Δ_specific(not) = ΔKL_targeted(not) − ΔKL_random_mean(not)` for each of the 16 invented words
- per-word `Δ_specific(and) = ΔKL_targeted(and) − ΔKL_random_mean(and)`
- 95% bootstrap CI on the mean Δ_specific (B = 500 with-replacement resamples over the 16 invented words)

The RANDOM_NORM baseline is the aggregate (16-word) mean from the 25a log's Section (D) summary block. This treats the random-norm baseline as a fixed offset; a more rigorous per-(word, stim) bootstrap would require re-running script 25a with extended per-word RANDOM_NORM logging, which is deferred. Cache-only; runtime < 5 s on CPU; log at `outputs/25d_20260521_132205.log`.

**Result.** Both axes' Δ_specific 95% CIs include zero, marginally on `not` and clearly on `and`:

| Axis | Targeted mean | RANDOM mean | Ratio | Δ_specific mean | Δ_specific 95% CI | Axis verdict |
|---|---|---|---|---|---|---|
| `not` | +0.0331 | +0.0170 | 1.95× | +0.0161 | **[−0.001, +0.031]** | borderline (CI just touches 0) |
| `and` | +0.0274 | +0.0210 | 1.30× | +0.0064 | **[−0.005, +0.020]** | clearly includes 0 |

Per-word breakdown: 13/16 invented words have positive Δ_specific(not); 11/16 have positive Δ_specific(and). Two unary-intended words (`perph`, `kelm`) are strong negative outliers on both axes (−0.056 / −0.031 on `not`, −0.028 / −0.023 on `and`). The mean ratios match the previously-reported 1.94× / 1.29× to two decimals, confirming the parse is correct.

**Implication.** **Cell 2's WEAK PASS verdict does not firm up under per-word Δ_specific bootstrap.** Both axes' 95% CIs straddle zero, marginally on `not` and clearly on `and`. The mean-ratio heuristic that produced the WEAK PASS classification was partly driven by a small number of high-magnitude positive words (`molex`, `krev`, `drelth`, `vrith`); the per-word distribution as a whole does not statistically separate from the RANDOM_NORM baseline at α = 0.05. Under the cleaner statistic, **Cell 2 is AMBIG: real but borderline on `not`, clearly inert on `and`.**

**Reframing the 5-cell sweep headline.** With Cell 2 reclassified from WEAK PASS to AMBIG, the cross-cell pattern becomes:

- **1 CLEAN PASS**: Gemma `opera→close L 2` (v6 emergent, not a Fact-1 anchor)
- **1 AMBIG**: Gemma `opera→opera L 4` (the principal cross-family Fact-1 anchor under the cleaner statistic; previously WEAK PASS under the mean-ratio heuristic)
- **3 FAILs**: Gemma `sente→close L 2`, OLMo `sente→close L 10`, OLMo `opera→close L 10`

This **strengthens** the §5.1 "Fact 1's geometric transfer does not uniformly imply causal load-bearingness — and is only weakly causally load-bearing even at its strongest cell" argument. Under the cleaner statistic, the principal cross-family Fact-1 cell does not even cleanly weakly pass: its causal effect on downstream computation is not statistically distinguishable from the norm-matched generic-disruption baseline. The headline becomes: *the cleanest causally-arity-respecting cell is the Gemma v6 emergent cell at L 2 close-paren, not any Fact-1 anchor — and the principal Fact-1 anchor's causal load-bearingness is not robust to a per-word bootstrap*.

**Caveat on the statistic.** The aggregate RANDOM_NORM mean used as the baseline is itself a 16-word sample mean (with its own ~±0.005-0.010 sampling variability). A more rigorous statistic would be a per-(word, stim) bootstrap where the RANDOM_NORM ΔKL is paired with the targeted ΔKL for the same word, controlling for per-word variability in the residual-stream norm and the random-direction draw. We expect this to slightly tighten the CIs (because the per-word random baseline is correlated with the per-word targeted effect through the residual-stream norm) but not to flip the verdict — the `and`-axis ratio is too close to 1.0 to firm up under any reasonable per-word baseline. The per-(word, stim) re-run is a ~6 min MPS follow-up that we list as a paper.md §6 follow-up but do not block submission on.

**Paper.md updates.** Table 5 verdict for Cell 2 updated from `WEAK PASS` to `AMBIG (mean ratio 1.3-2×; Δ_specific 95% CI includes 0 on both axes)`. §4.5 Cell-2 description rewrites the "approximately 1.94× on `not` and 1.29× on `and`" finding to add the Δ_specific bootstrap CI as the criterion-of-record. §5.1 and §7 update the "principal cross-family Fact-1 cell is weakly causally load-bearing" claim to "principal cross-family Fact-1 cell's causal load-bearingness is not robust under per-word Δ_specific bootstrap (95% CI [-0.001, +0.031] on `not`, [-0.005, +0.020] on `and`)". §6 "Causal evidence is partial" replaces the named follow-up with a reference to script 25d and lists the per-(word, stim) RANDOM_NORM re-run as the remaining cheap follow-up.

### 4.4 Methodological lessons

Six methodological findings worth flagging. The sixth (the lucky-default detector) is the principal methodological contribution of this project and warrants particular emphasis: it is the refinement that prevented us from publishing four false-positive PASS-arity cells in §3.7.11, and it is directly transferable to any probe-based substrate-invariance study that reports per-word intended-class agreement metrics.

1. **Single-pooling pre-registration is unsafe.** Last-token, mean-pool, and operator-anchored pooling produced materially different conclusions on the same data. Future work must triangulate across pooling strategies.
2. **Pre-registered success criteria require empirical refinement.** Our initial criterion (CKA > 0.7 absolute) was rendered uninformative by the bag-of-tokens behavior of mean-pool at shallow layers. The substrate-invariance *gap* (signed difference from scrambled baseline) is the more robust metric.
3. **Tokenization is itself the first morphospace boundary.** A surface-form change that fragments under byte-fallback BPE introduces a confound (tokenization length differences) on top of any representational change. Probes for substrate-invariance must control for this through anchor-aggregation methodology.
4. **Within-condition probe CV is insufficient as a substrate-invariance signal in functional-prefix notation** (Phase 1 entry, script 17–18). A probe can reach CV=1.000 at every single layer 1–42 by reading propagated previous-token identity at the operator-anchored position, with no structural arity content. Within-condition probe accuracy must be cross-validated against either (a) cross-condition probe transfer or (b) the held-out canonical generalisation test before being reported as a substrate-invariance measurement.
5. **Cross-condition probe transfer with a canonical-transfer gate is the gold-standard substrate-invariance instrument across notations and models.** The cross-condition transfer asymmetry (Gemma 2 100% at validated layers / OLMo 2 0%) reveals a model-level property — cross-context stability of the arity direction — that is invisible to within-condition probe accuracy and to probe-free centroid geometry, both of which give similar within-notation signals across both models. The canonical-transfer gate (require cross-canonical 5-class accuracy ≥ 0.65 at the same train/test pairing before accepting the invented-unary result as transportable) prevents a chance-level canonical-transfer with high invented-unary mass — i.e., decision-boundary bias rather than structural arity transfer — from being misreported as a substrate-invariance finding. Phase 1 / Phase 2 substrate-invariance studies should pre-register cross-condition transfer in both directions, with the canonical-transfer gate, as required measurements.
6. **The lucky-default detector: `min(per_word_top_pct) ≥ 0.95` catches a specific false-positive pattern that aggregated concentration metrics miss.** This is the principal methodological contribution of the project. **Setup**: when reporting M4b (intended-arity agreement: fraction of invented words whose top predicted canonical matches their intended arity class), a researcher's natural concentration check is Herfindahl-style — does the invented mass spread across multiple canonicals, or collapse to one? A threshold like *M4c (max-canonical-share) ≤ 0.85* superficially catches the single-canonical-collapse pattern. **Problem**: it does not catch the 4-of-N-at-ceiling-plus-1-escape pattern. Suppose 4 invented words land on canonical A at 100% within-word concentration, and 1 invented word lands on canonical B at 100% within-word concentration. Aggregated M4c = 0.8² + 0.2² = 0.68, comfortably below 0.85 — looks "distributed". But the per-word predictions are *deterministically* concentrated; no per-word probabilistic mixing is happening. If the 4-of-N-at-A canonical happens to match those 4 words' intended arity and the 1-of-N-at-B happens to match the escape word's intended arity, M4b will appear arity-respecting at 5/5 = 100% — a false positive driven by lucky alignment between the model's default choices and the test set's arity distribution. **Refinement**: define `pwmin = min over invented words of (within-word top-canonical concentration)`. A cell with `pwmin ≥ 0.95` has every single invented word deterministically routed; this is a lucky-default cell regardless of M4c. Combined verdict: M4b ≥ 0.65 AND `pwmin < 0.95` is required for PASS-arity. **Impact in our data**: introduced in §3.7.11 after the full anchor × layer sweep. Reclassified 4 of 8 originally-flagged PASS-arity cells as `LUCKY-NEG` (3 OLMo, 1 Gemma); only the §3.7.9 cell escaped the refined detector (per-word `perph` concentration 0.50 — genuine within-word split between "and" and "necessarily"). All 4 surviving PASS-arity cells from §3.7.11 then retracted under §3.7.13 / §3.7.14 stimulus expansion — but without the `pwmin` refinement, *eight* cells would have been retracted instead of four, and the original "lucky-default exposure" methodology contribution would not exist. **Generalization**: any probe experiment reporting per-word agreement metrics — for substrate-invariance, for compositional generalization, for analogical reasoning, for in-context binding — is vulnerable to the same false-positive pattern. The `min(per_word_top_pct) ≥ 0.95` lucky-default detector should be a pre-registered required measurement alongside the aggregated metric. **Validation**: at the 16-word invented-set expansion (§3.7.13), the refined detector cleanly identifies the lucky-default control cells (M4b → 0.500 at expansion; CI [0.500, 0.500]; `pwmin ≥ 0.95` at both) and distinguishes them from the surviving cells (§3.7.9 cell at `pwmin = 0.50`). At the 10-canonical expansion (§3.7.14), it triggers immediately on the §3.7.9 cell when all 16 invented words collapse to `nand` (every word at 100% concentration → `pwmin = 1.00` → LUCKY-NEG flag fires). At Pythia v5 (§3.7.15), all top-M2c cells have `pwmin = 1.00` because Pythia's default-to-rarest-canonical mechanism is even more deterministic than OLMo/Gemma's. At the pre-registered v6 expansion (§3.7.16), the detector continues to flag the §3.7.9 OLMo `sente→close L 10` cell as LUCKY-NEG at `pwmin = 1.00`, even as M2-arity stays at 1.000 — making this cell the canonical example for the detector across the entire project: a cell whose binary-vs-unary axis is genuinely structural (M2-arity = 1.000 at all four scopes) but whose invented-word readout is 100% lucky-default (all 16 words route to a single `nand` attractor at v5+v6). The detector is consistent across models, across stimulus-set expansions, and across readout-set expansions; we recommend it be adopted in any future probe-based substrate-invariance instrument as a pre-registered required measurement.

## 5. Limitations

- **Three model families across Phase 1 + Phase 2.** Phase 0 findings (scripts 06–16) are OLMo 2 1B and 7B only. Phase 1 entry (scripts 17-22d) adds Gemma 2 9B and confirms the operator-set-bound substrate-invariance pattern across both models in both notations. Phase 2 (script 23, §3.7.15) adds Pythia 6.9B-deduped (EleutherAI, Pile-trained, GPT-NeoX with RoPE) as the third model family and replicates all three operator-set-bound predictions: cross-notation canonical-operator transfer (M2-canonical = 1.000 at multiple cells), v3→v4→v5 PASS-arity retraction signature, and default-to-rarest-canonical at v5. The pre-registered v6 expansion (script 24, §3.7.16) tested the framework under a 15-class readout across all three models simultaneously; Fact 1 strengthens to bootstrap-CI = [1.000, 1.000] in all three models, and Fact 2 holds in OLMo and Pythia with one methodological caveat in Gemma (the v6 emergence is reported as M4b granularity-sensitivity rather than substantive retraction; see §3.7.16 and §4.1.8). The three-model replication spans three training corpora (Dolma, Google proprietary, Pile), three architectures (modified Llama, soft-capped Gemma, GPT-NeoX), and three tokenizers; this is meaningfully stronger evidence than the two-model Phase 1 result for treating the finding as a property of mid-scale open base LMs at the 6.9-9B parameter range. **Remaining scope limits**: single parameter range (we have not tested whether the pattern dissolves at 70B+ frontier scale); base models only (no instruction-tuned variants); Qwen, Mistral, and Llama 3 are untested. The "general property of language model representations" claim is now meaningfully closer to defensible at this scale but still requires the scale and instruction-tuning dimensions to close.
- **Default-to-rarest mechanism is not a clean single-axis function (new from §3.7.16 + §3.7.18).** The v6 pre-registration tested three competing single-axis readings of the default mechanism — token frequency in training data (P_FREQ), subword-tokenization shape (P_SUBWORD), and an interaction reading (P_INTERACTION). All three predictions fail in all three models. Script 25b (§3.7.18) then tested whether the residual factor could be operationalised as mean-pooled cosine similarity between invented-word and canonical activations at the focus layer ("contextual semantic neighborhood"); the answer is also no — at distributed (M4c < 0.7) cells, identity-level agreement between cosine argmax and probe prediction is at chance for identity (11.6% / 14.8% / 16.2% across Gemma / OLMo / Pythia, vs 6.7% chance baseline) and at chance for arity-conditioned identity. The residual third factor is **probe-decision-boundary geometry** — the probe routes invented words via learned discriminative weights that capture per-word residual-stream structure not preserved by mean-pooling — and is therefore not reducible to any single mean-pooled-activation statistic. The §3.7.14 / §3.7.15 framing of the mechanism as "default to the rarest canonical" is too coarse; the more accurate framing is "the model routes novel-operator activations to a low-prior canonical in a model-specific way that depends on frequency, subword shape, and per-word residual-stream structure not captured by mean-pooled cosine; closing this further would require sparse-autoencoder feature labelling at the focus layers (see §6)."

- **Cosine similarity on residual-stream activations is dominated by canonical-magnitude-and-idiosyncrasy effects (new from §3.7.18).** A specific failure mode surfaced in script 25b: at distributed cells across all three models, the unconstrained cosine argmax collapses every invented word to `nand` (or to `negate` for unary-intended words under arity-conditioning). The mechanism is most likely that `nand` has the most idiosyncratic per-stimulus activations in the v6 set — its mean canonical activation sits further from the canonical centroid than higher-frequency canonicals — so under cosine similarity invented words (which are also far from any sensible centroid) end up closest to `nand`. This is a near-tautological geometric structure that does not reflect the probe's actual decision boundary. The methodological caveat: **mean-pooled cosine on residual-stream activations as a "semantic-neighborhood" proxy is contaminated by canonical-magnitude effects** and should be reported alongside probe-based readouts, never as a standalone replacement.
- **M4b is sensitive to canonical-readout granularity (new from §3.7.16).** The Gemma 2 9B `N→F opera→close L 2` and `N→F sente→close L 2` cells exhibit M4b trajectories of 60% → 56% → 50% → 82% and 44% → 46% → 50% → 66% across the v3 → v4 → v5 → v6 scope expansions, with no change in the underlying activations (the same cache is sub-selected by canonical / invented sets at each scope). Their M2-arity stays at 0.78-1.00 across all four scopes; only M4b changes. This demonstrates that M4b is not a Boolean test of arity-respecting structure — it is a threshold test on a metric sensitive to the granularity of the within-arity readout pool. The Phase 1 / Phase 2 framework should treat M2-arity as the primary arity-axis measurement and M4b as an additional concentration check; reporting M4b on its own without a multi-scope sensitivity check is methodologically incomplete. The Gemma v6 emergent PASS-arity cells deserve causal-patching follow-up (script 25a) before any substantive Fact-2 retraction is claimed for that model.
- **v6 audit gate caught one pre-registered tokenization failure.** The pre-registration specified `iff` as 2-3 subwords (target multi-piece, very-low frequency), to provide a clean multi-pc-LF arm of the disentanglement test. Empirically `iff` ended up as 1pc in all three tokenizers (Gemma SentencePiece, OLMo BPE, Pythia BPE), reducing the multi-pc-LF design arm to `unprovably` alone. This is a real limit on the v6 disentanglement strength: the P_SUBWORD prediction would have been more discriminating with two in-design multi-pc-LF canonicals instead of one. The audit-gate caveat is reported in §3.7.16; future tokenization-targeted designs should pre-test the candidate canonicals against the actual tokenizer rather than relying on out-of-band frequency intuitions.
- **Single structural domain.** Propositional logic only. Substrate-invariance for set theory, algebra ($S_4$ and similar finite groups), or other formal domains is unverified.
- **Confounded operator-length.** Canonical operators are 1 BPE token (`and`, `or`, `not`, `implies`, `necessarily`); invented operators are uniformly 2 BPE tokens. The operator-renaming probe therefore conflates "operator semantics" with "operator token-length." Script 10 substantially de-confounded this by varying invented-operator length L ∈ {1, 2, 3, 4} and showing the failure pattern is length-independent, but the 1-token-canonical vs 2-token-invented asymmetry in our headline numbers remains a minor confound.
- **Resolved (and refined): single unary canonical operator.** Our original canonical set had exactly one unary operator (`not`); script 12 added `necessarily` as a second canonical unary and initially appeared to establish that the default is `not`-specific (H1a). Script 13 refined this: the default is to the unary *region* containing both `not` and `necessarily`, and the apparent `not`-dominance of script 12 was an artifact of the multi-operator-template probe instrument. The upgraded limitation is: **only two unary canonicals tested**, and the not-vs-necessarily landing distribution within the unary region is probe-context-dependent.
- **Probe-instrument dependence (refined by script 15).** The same residual-stream activation produces materially different probe predictions depending on whether the probe was trained on multi-operator, single-operator-rich, or canonical-neutral templates. Script 15 demonstrates this in the cleanest possible setup: the same B'_rich stimuli produce 94.6% unary mass with a rich-trained probe (script 13) and 21.2% unary mass with a neutral-trained probe (script 15 Test 2). The H1 unary-region attractor is robust across all three probe instruments when train and test are matched-template-family; it is fragile when train and test are mismatched. Phase 1 must report probe results across multiple training-template regimes as standard practice; the matched-train/test family results are the meaningful measurements.
- **H4 template-context channel reframed as probe artifact (script 15).** Scripts 11–13's H4 effect ("template-context pulls toward the template's owned canonical") is no longer interpreted as a residual-stream-level property. Script 15 Test 2 shows that with a neutral-trained probe, the same residual-stream activations produce predictions dominated by template syntactic scaffolding (e.g., "If... then..." → `implies`), not by H4 as defined in script 13. The H4 quantification numbers remain valid as probe-internal measurements but should not be used to argue for a deep template-context channel.
- **Embedding-vs-residual-stream divergence (new from script 14).** Token-embedding-layer cosine similarity does not predict the operator-anchored probe's behaviour at layer 7 for invented words. The H1 unary-region attractor is constructed by attention/MLP processing in layers 1–7, not inherited from layer-0 geometry. This is a positive result (it strengthens the "H1 is real" claim) but it limits any analysis that uses embedding-similarity as a screening tool for invented-operator selection (such pre-screens may not actually predict what the probe sees).
- **Within-arity identity is fragile (new from script 16).** The within-unary `not`-vs-`necessarily` distribution for the same invented word varies from 0:98 (molex in metalinguistic frame) to 100:0 (molex in functional-prefix frame) across notations. The Phase 1 paper cannot make any claim about which specific unary canonical an invented operator "really" maps to — only that the arity-class is robustly unary. The within-arity story is part of the Phase 0 *limitations*, not the headline finding.
- **The "intrinsic operator representation" claim is partially circumstantial.** The probe achieves 1.000 CV accuracy on canonicals in functional-prefix notation with identical preceding context. This is consistent with the model having operator-specific intrinsic representations, but it could also reflect very small subword-level differences (`and`/`or`/`not` are 1-subword high-frequency tokens, `implies`/`necessarily` are 1-subword lower-frequency tokens) that the probe leverages. A causal-patching follow-up at the operator-anchored position is needed to confirm that the model is using the operator's intrinsic representation rather than some leaked contextual signature.
- **Cross-context stability is measured at a single layer per model per condition.** The cross-condition probe transfer result (Diagnostic A) is reported at the script-17 / script-16 focus layers (L7 for OLMo 2; L4 for Gemma 2 NEUTRAL and L2/L16 for Gemma 2 FUNC-PFX). A full per-layer cross-transfer sweep would be a stronger basis for the "Gemma 2 globally aligned vs OLMo 2 notation-local" claim. Script 19 (directional-angle analysis) and the planned per-layer cross-transfer sweep are the immediate Phase 1 follow-ups that close this gap.
- **The L7 OLMo 2 cross-transfer collapse mode is not yet fully understood.** Cross-condition transfer in OLMo 2 collapses to 0% unary in both directions, but the *target* canonical of the collapse is notation-dependent (NEUTRAL → FUNC-PFX: all to `and`; FUNC-PFX → NEUTRAL: all to `implies`). This is consistent with each notation's probe being read by the wrong-notation activations as projecting onto a notation-specific "default binary canonical", but the mechanism is not characterised. Script 19's directional-angle and Phase 1's per-layer cross-transfer sweep should help here.
- **Sample size.** N=200–250 stimuli (50 per class) is adequate for probe training but small for fine-grained per-template analyses. Detecting H4 magnitudes per template requires N ≥ 50 per (template × operator-class) cell.
- **Pre-registration was iteratively refined during Phase 0.** Our original CKA threshold (absolute > 0.7, mean-pool) was retired after observing that mean-pool over largely-overlapping token bags produces uninformative absolute CKA values. The replacement metric (substrate-invariance gap with multi-pooling triangulation) is more robust but was determined post-hoc on the same data. A clean Phase 1 replication on novel templates with the final methodology fully pre-registered is required before any of these results should be considered confirmatory.
- **The notation-coherence effect (B'' > B') did not replicate at 7B.** This is included as a non-finding rather than a finding (§3.5). We should expect, going into Phase 1, that *some* of our 1B-only observations will not survive replication at 7B or on novel stimuli; the notation-coherence effect is the first clear instance.
- **Single tokenizer.** Findings about operator-renaming behavior are entangled with OLMo 2's BPE tokenizer specifically. Different tokenizers (e.g., SentencePiece, tiktoken) might handle invented operator words differently and produce qualitatively different failure modes.
- **Causal evidence is now partial (new from §3.7.17).** Script 25a applied activation patching at three target cells (Gemma 2 9B L2 close-paren under two source anchors; OLMo 2 7B L10 close-paren). Two findings update the original "no causal evidence" caveat. **(i)** At Gemma 2 9B `opera→close L 2`, patching is causally arity-respecting (8/0/8 arity-flip in both directions, ΔKL = +0.033 / +0.061 vs random-norm baseline near zero), confirming this single cell is a load-bearing model-specific exception to operator-set-bound. **(ii)** At Gemma 2 9B `sente→close L 2` and OLMo 2 7B `sente→close L 10`, patching achieves 100% probe-causality (the patch reaches the residual) but does *not* produce arity-respecting behavioural shifts (ΔKL flat or negative; random-norm control exceeds the targeted patch). The OLMo §3.7.9 anchor — the project-flagship Fact-1 cell — is therefore demonstrated to be **probe-readable but not causally load-bearing** under NEUTRAL-sentence-final patching. The original Fact 1 finding is not refuted (it remains a robust geometric statement about cross-notation linear-probe transfer at the canonical level, bootstrap-confirmed across three model families), but its causal grounding is now known to be anchor-pair-dependent. A more comprehensive Fact-1 causal test would patch at multiple (source, target) anchor pairs to identify which combinations are load-bearing; this is an open empirical question. The general claim "all results are correlational" is replaced by "causal grounding is established at one Gemma cell, falsified at two cells, and untested at every other cell".
- **Source anchor is a first-class causal variable (new from §3.7.17).** Both Gemma 2 9B L2 close-paren cells achieve identical 100% probe-causality under patching (the patches reach the residual equally cleanly), but their behavioural verdicts are opposite depending on whether the source vector comes from NEUTRAL operator-after or NEUTRAL sentence-final. Probe-causality is necessary but not sufficient for causal load-bearingness. Future probe-based substrate-invariance work should include same-target / different-source causal tests as part of the standard battery; reporting probe transfer at a single anchor pair systematically misses this dissociation.

## 6. Open questions and next experiments

**Completed since first draft:**

- 7B replication of scripts 08 and 09 → role-bound interpretation supported, capacity-bound interpretation rejected.
- Confusion-matrix analysis at 7B layer 7 → uniform-default-to-`not` mechanism identified.
- Subword-length variation probe (script 10) → H1 (structural defaulting) supported, H2 (tokenization-position) rejected; H3 (embedding-similarity escape hatch) emerged as a third mechanism.
- L=1 bar-anomaly probe (script 11) → H3 decisively confirmed (`bar`→`or` follows `bar` regardless of slot: 82% in or-slot, 74% in and-slot). A fourth channel H4 (template-lexical-context pull) emerged from observing that `zap` and `foo` recover partial `or` predictions in the or-slot driven by template lexical signals rather than embedding similarity.
- Second-canonical-unary probe (script 12) → H1a (default-to-`not`-specifically) provisionally confirmed; superseded by script 13.
- **H4 quantification (script 13) → original principal finding, since refined.** Factorial 5 W × 5 T design: across 1250 invented-operator inputs, the probe predicted `and` zero times, `or` 1.4%, `implies` 4.0%, `not` 50.3%, `necessarily` 44.3%. Established the unary-region attractor framing. H4_pull magnitudes (not = +0.181, necessarily = +0.166, others < +0.07) were interpreted as template-context modulation within the unary region. Script 15 has since reframed this interpretation — the H4 magnitudes are valid probe-internal measurements but reflect probe training-distribution, not residual-stream structure.
- **Embedding-similarity audit (script 14) → H1 not inherited from layer-0 embeddings.** All 5 script-13 invented words have peak layer-0 cosine similarity to `and` or `or`; only 1 of 5 layer-0 top-canonical predictions match the layer-7 probe landing; sign-agreement on the within-unary not-vs-necessarily ranking is 2/5. The unary-region attractor is constructed by attention/MLP processing in layers 1–7, not by token-embedding geometry.
- **Template-neutral probe (script 15) → first major H1 confirmation.** Probe CV accuracy on canonical A in neutral templates is 0.996; cross-template generalization to canonical A_rich is 0.944. Invented operators in neutral templates land in the unary region at **99.6% mass** with 0% predictions to any binary canonical and 0% to `or`. The H4 channel from script 13 is reframed as a probe-instrument artifact; the underlying residual-stream contains both unary-attractor and implication-scaffolding signals, and which one a probe reads out depends on its training distribution.
- **Syntactic-confound stress test (script 16) → external-review-driven falsification attempt.** Address two reviewer-flagged confounds: prefix-vs-infix syntactic position, and metalinguistic-POS-prior. Critical condition uses functional-prefix notation (`op(p, q)` for binaries, `op(p)` for unaries, identical preceding context "The function ") to place all canonicals in the same prefix-function-call syntactic role. Probe CV accuracy: 1.000. Invented operators: **100% of 250 stimuli classified as `not`** with 0 binary-canonical predictions. The arity attractor survives the syntactic-position dissociation. Within-arity identity is shown to be even more probe-instrument-fragile than scripts 9–15 indicated (molex: 0:98 → 100:0 across notations).
- **Cross-model replication on Gemma 2 9B (script 17) → NEUTRAL replicates cleanly; FUNCTIONAL-PREFIX shows non-monotonic per-layer trajectory.** Probe CV 0.996 / 1.000 on canonicals in both conditions; invented unary mass 97.6% peak in NEUTRAL (at L4) and 100% peaks at L2 and L16-17 in FUNC-PFX with 0% trough at L6-L12. Three observations made the FUNC-PFX result hard to interpret from script 17 alone: probe CV is 1.000 at every layer 1-42, last-subword identity dominates per-word landings at the fixed-reference L8, and the per-layer trajectory is qualitatively different from OLMo 2's monotonic plateau.
- **Gated directional-angle analysis with bootstrap CIs (script 19b) → necessary-but-not-sufficient picture.** Extends script 19 with bidirectional canonical-transfer gate at each layer pair, bootstrap 95% CIs on angles (100 within-class resamples), and cross-layer pairings. Headline: Gemma 2 L4 gate-PASS + tight angles (PASS verdict), OLMo 2 L7 gate-FAIL asymmetrically (1.000 one direction, 0.212 the other — chance-level). Two new findings: (i) **OLMo 2 L10 is a previously-unidentified bidirectionally-gate-passing layer** (0.800 / 0.688) with wide angles (71.8° / 74.7°) — the principal open empirical question (script 20 closes); (ii) **Gemma 2 L8 reveals a new failure mode**: bidirectional gate-PASS at 1.000 / 0.864 *coexisting* with wide angles (68.9° / 67.4°) and script-18 invented-unary mass of 12-17%, demonstrating that 5-class canonical-transfer and binary unary-vs-binary directional alignment are distinct geometric properties. The canonical-transfer gate is therefore necessary but not sufficient for cross-notation invented transfer. Gemma 2 L17 falls to AMBIG (gate 0.644, 0.006 below threshold); Gemma 2 FUNC-PFX@L16 → NEUTRAL@L4 cross-layer is AMBIG with 82° angles, retracting the script-18 100% invented-unary mass at that pairing as decision-boundary bias. Bootstrap CIs are 2-6° wide across the board; cross-model CI separation is non-overlapping.
- **Directional-angle quantification (script 19) → scale-invariant cross-model number.** Computes cosine angle between NEUTRAL and FUNCTIONAL-PREFIX arity directions per model, in two operationalisations (centroid-based and probe-based) and against a 200-sample random-unit-vector baseline of exactly 90.0°. Headline numbers: Gemma 2 best cross-notation probe cosine +0.70 at L2 (sweet spot, beats the script-17/18 focus layer L4 at +0.58); OLMo 2 cos +0.24–0.32 across the entire L4–L24 depth (flat, no sweet spot). The Gemma 2 L16 "candidate late re-emergence" lands at cos +0.33–0.34 — directionally weaker than the L2/L4 stage and only marginally above OLMo 2's notation-local baseline, validating the peer-review caution that L16 should remain "candidate" rather than confirmed. The Gemma 2 cross-notation alignment trajectory is monotonically decreasing with depth (L2 best, L17 worst), not biphasic — there is no second arity-direction emergence at late layers in residual stream space.
- **Probe-artifact diagnostic battery (script 18) → cross-condition probe transfer is the principal new finding.** Four diagnostics on both Gemma 2 9B and OLMo 2 7B: cross-condition probe transfer (A), held-out canonical generalisation (B), probe-free centroid geometry (C), last-subword embedding baseline (D). Both models pass B (NEUTRAL probes structural at 4× chance) and C (5/5 invented words closer to unary centroid in every model × condition). The two models diverge on A: Gemma 2's NEUTRAL probe transfers to FUNC-PFX at **100% unary mass**; OLMo 2's NEUTRAL probe transfers to FUNC-PFX at **0% unary mass** (all to `and`). The reverse direction also diverges (Gemma 2 86-100% across cross-transfer-validated layers; OLMo 2 0%). The Gemma 2 multi-stage per-layer FUNC-PFX trajectory is confirmed real (L2 and L16-17 cross-transfer-validated as structural; L7-L12 confirmed as a surface-feature artifact layer). Phase 1 entry verdict: **cross-model arity-region attractor with model-specific cross-context representational stability**.

**Phase 0 verdict: GO.** The arity-attractor finding stands on five independent legs:
1. Replication across two model scales (1B and 7B).
2. Replication across four probe instruments (multi-op-rich, single-op-rich, neutral-metalinguistic, functional-prefix).
3. Independence from layer-0 embedding geometry (confirmed by script 14).
4. Independence from prefix-vs-infix syntactic position (confirmed by script 16).
5. Independence from metalinguistic-vs-functional notation (confirmed by script 16).

The within-arity identity claim is *not* defensible — the within-unary distribution is fragile to probe instrument and notation. The refined Phase 1 claim is therefore narrower: arity is encoded, within-arity-identity is not. This is the form the Phase 1 paper should take.

**Immediate next steps (Phase 1 entry):**

- **✓ Done: Directional-angle quantification (script 19).** Confirmed quantitatively (Gemma 2 best cross-notation probe cos +0.70 at L2; OLMo 2 flat at +0.24–0.32 from L4 to L24; both well below the random-baseline cos 0 ≡ 90°). Refined picture: Gemma 2 ≈ 2× tighter cross-notation alignment than OLMo 2, with a clear L2 sweet spot, and a monotonically decreasing alignment with depth (no biphasic structure — the L16 "candidate late re-emergence" of script 18 is directionally weak, validating the peer-review caution).
- **✓ Done: Script 19b gated directional-angle analysis with bootstrap CIs.** Findings refine the cross-model picture: OLMo 2 L7 (long-standing focus) gate-FAILS asymmetrically; OLMo 2 L10 is a new bidirectionally-gate-passing layer (open empirical question); Gemma 2 L8 reveals gate-PASS-but-angle-wide failure mode (= gate is necessary not sufficient); Gemma 2 L17 drops to AMBIG; L16→L4 cross-layer is AMBIG with 82° angles (validates peer-review caution on L16). Cross-model CI separation is non-overlapping. Bootstrap CIs are 2-6° wide.
- **✓ Done: Script 20 gated-invented-mass re-test.** Three principal findings that further refine the cross-model picture: **(i)** OLMo 2 L10 outcome (i) confirmed — invented unary mass at floor (0% / 17%, mean 8.6%, *below* random-by-arity 40% baseline; N→F predictions 100% "implies"). OLMo 2 has no cross-notation arity-respecting transfer at any tested layer, *at the operator-anchored position*. **(ii)** Gemma 2 has a "necessarily" catchment basin that grows suggestively monotonically with depth in N→F (L2 = 0% necessarily of 40% unary, L4 = 80% necessarily, L8 = 100% necessarily; 3 tested points in one direction). The script-17/18 "Gemma 2 cross-notation arity transfer at L4" claim is refined to cross-notation transfer of a (possibly generic-modifier) catchment basin, not strict arity-respecting transfer. **(iii)** Per-invented-word breakdown shows the predicted canonical does NOT track intended-arity at any tested (model, layer) pairing — bliq (intended-binary) maps to NOT at Gemma 2 L2; vusp (intended-unary) maps to implies at Gemma 2 L2; etc.
- **✓ Done: Script 21 multi-anchor M1-M4 battery (post-call anchor re-test).** Built, smoke-tested, and run on MPS (Gemma 2 + OLMo 2, ~9 min total compute). The script-20 negative-result headline is anchor-bound: at the operator-after anchor (matching scripts 17-20) the conclusion survives unchanged in both models; at the close-paren anchor a *candidate* cross-notation arity-respecting transfer emerges in OLMo 2 L10 N→F (M4b = 90%, per-word: 4 of 5 invented words track intended arity, mass distributed across "and" 70% + "necessarily" 30%). M2 gate at this cell is AMBIG (0.616, just below threshold) and the F→N direction FAILS. Two "lucky-default" artifact cells empirically vindicate the M4a/M4b/M4c split (M4b high without arity discrimination at OLMo 2 op-after L7 + Gemma 2 first-arg L8, both with M4a floor and M4c collapsed). Per-anchor catchment basins differ within the same (model, layer) — Gemma 2 L4 has a "necessarily" basin at operator-after and a "not" basin at close-paren — indicating positional readout is a first-class variable. The §3.7.8 / §3.7.5 / abstract negative-result headline is SOFTENED (from "not demonstrated at any pairing" to "candidate borderline transfer at one cell") but NOT retracted. See §3.7.9.
- **✓ Done: Script 22a bootstrap M2 gate + M2-arity introduction.** Cache-only (~2.5 min). Bootstrap CI on the script-21 candidate cell yielded P(M2-canonical ≥ 0.65) = 7.0%, robustly AMBIG. The strict 5-class M2 gate is not borderline-PASS; the within-arity confusion pattern (or → and, implies → and) explains why. Introduced **M2-arity** (binary-vs-unary coarsened from the same 5-class probe) as a separate measurement. M2-arity = 1.000 at the §3.7.9 cell. The §3.7.5 substrate-invariance battery now has M2-canonical and M2-arity as distinct gates with distinct conjunctions for arity-respecting vs canonical-identity transfer. The §3.7.9 candidate is upgraded to "demonstrated under M2-arity" but remains "not demonstrated under M2-canonical". Also revealed: training-anchor choice (NEUTRAL sentence-final vs operator-after) selects different cross-notation readouts of the same FUNC-PFX residual stream; the M2-arity gate is robust under bootstrap (CI [0.800, 1.000]). See §3.7.10.
- **✓ Done: Script 22b full anchor × layer sweep at OLMo 2 + Gemma 2 (cache-only, ~2 min).** Ran M2-canonical + M2-arity + M3 + M4a + M4b + M4c at every (NEUTRAL train anchor × FUNC-PFX test anchor × focus layer × direction) cell across both models (160 cells total). Confirmed the §3.7.9 cell as unique-strongest by composite criterion; identified 3 additional PASS-arity cells (Gemma N→F sente→first L8, Gemma N→F sente→opera L4, OLMo F→N first→opera L7). Refined the lucky-default detector to use min(per_word_top_pct) ≥ 0.95 (catches the 4-of-5-at-ceiling + 1-escape pattern that M4c ≥ 0.85 missed). Identified the structural commonality: post-call training anchors required for PASS-arity, operator-after training anchors consistently produce lucky-default catchment basins (40 of 160 cells). The OLMo F→N L7 cell is the first observed simultaneous M2-canonical + M2-arity PASS, in the reverse direction. See §3.7.11.
- **✓ Done: Script 22a extension — bootstrap CIs on the four PASS-arity cells (cache-only, ~9 min).** Added the three new PASS-arity cells from the script 22b sweep to the existing script 22a bootstrap protocol (500 stim-resamples each, identical to §3.7.10). Result: **all three new cells are confirmed dual-PASS (M2-canonical AND M2-arity)** under bootstrap. Gemma 2 N→F sente→opera L4: M2-canonical CI [1.000, 1.000], M2-arity CI [1.000, 1.000], point-perfect. Gemma 2 N→F sente→first L8: M2-canonical CI [0.888, 1.000], M2-arity CI [0.956, 1.000], P(≥ 0.65) = 100%. OLMo 2 F→N first→opera L7: M2-canonical CI [0.964, 0.992], M2-arity CI [0.980, 0.996], P(≥ 0.65) = 100%. The §3.7.9 cell remains M2-arity-only (the dissociation pattern is *not* a generic property of cross-notation transfer; it is specific to the §3.7.9 cell's within-arity-confusion mechanism). The Phase 1 headline is upgraded: cross-notation arity-respecting transfer is demonstrated at four cells, three of which are also bootstrap-confirmed full canonical-identity transfer. See §3.7.12. **Note (post-§3.7.13): the M2-canonical/M2-arity numbers at the OLMo F→N L7 cell and Gemma sente→first L8 cell remain valid on *canonical* operators, but the *invented*-operator M4b extension at these cells does not survive the 16-word falsification — see §3.7.13 retraction.**
- **✓ Done: Script 22c expanded invented-word set falsification (re-extraction, ~15 min on MPS).** Expanded the invented set to 16 words (8 intended-binary + 8 intended-unary; original 5 + 11 new phonotactically plausible Latin-script tokens audited for subword decomposition under both tokenizers). Re-ran the M1-M4 + M2-arity battery with bootstrap M4b CI at the 4 PASS-arity cells + 2 lucky-default negative controls. **Result: 2 of 4 PASS-arity cells survive the M4b ≥ 0.65 threshold under bootstrap; 2 retract as 5-word sampling artifacts.** Survivors: §3.7.9 OLMo sente→close L10 (M4b = 0.796 with CI [0.772, 0.819], cleanest distributed-across-3-canonicals mechanism) + Gemma 2 sente→opera L4 (M4b = 0.669 borderline with CI [0.659, 0.676], necessarily-basin-leaning mechanism). Retracted: Gemma 2 sente→first L8 (M4b → 0.561) + OLMo 2 first→opera L7 F→N (M4b → 0.573 — its canonical-only M2 transfer remains valid). Lucky-default negative controls drop cleanly to M4b = 0.500 at 16 words, exactly validating the M4b ≥ 0.65 threshold. The cross-notation arity-respecting transfer finding survives in tightened form (2 cells across 2 models) but is no longer generic across 4 cells. See §3.7.13.
- **✓ Done: Script 22d expanded canonical set (full re-extraction, ~17 min on MPS).** Added binary `xor`, `nand` and unary `possibly`, `always`, `negate` to the canonical set (10 total, 5B + 5U). Invented set unchanged from 22c. **Result: both §3.7.13 survivors retract on M4b**. §3.7.9 OLMo: 100% of invented mass → `nand`, M4b = 0.500. Gemma sente→opera L4: 87.5% → `nand` + 12.5% → `negate`, M4b = 0.625 (below threshold), lucky-default flag YES. The 67.9% "necessarily basin" of 22c disappears entirely at 22d (0% on `necessarily`). **Mechanism revealed: default-to-rarest-canonical** — the probe routes invented activations to the highest-entropy / lowest-training-prior canonical, which shifts wholesale when the canonical set expands. Neither strict-arity nor modifier-basin reading is supported; both are falsified. **Critically, M2-canonical and M2-arity remain robust under canonical-set expansion** (Gemma L4: both 1.000; §3.7.9: 0.812 + 1.000) — canonical-operator substrate-invariance is real; novel-operator substrate-invariance is illusory. See §3.7.14. **All four originally-PASS-arity cells from §3.7.11 are now retracted in some form.** Net Phase 1 headline: operator-set-bound substrate-invariance is demonstrated; novel-operator generalization is not.
- **Defer: Probe-variant sensitivity table (raw LR vs StandardScaler+LR vs nearest centroid vs ridge).** Originally proposed as the next experiment post-script-21; now deferred until after the bootstrap M2 / expanded-invented / expanded-canonical results land, since the script 21 + 22a findings add specific cells (OLMo 2 close-paren L10) where the probe-variant sensitivity question is much more sharply posed. The script-18-vs-script-20 estimator discrepancy (Gemma L4 N→F invented mass: 99.6% in script 18 vs 80% in script 20) is documented in the §3.7 reproducibility note and is not currently blocking.
- **✓ Done: Stable seeding + cache reproducibility fix.** Scripts 19/19b/20 now use `hashlib.blake2b`-based stable seeding (verified identical across processes with different `PYTHONHASHSEED` values). 19b cache file format bumped to `_v2-stable-seeds`; cache metadata now stores `stimulus_version`, `anchor_mode`, `canon_prompts_hash`, `inv_prompts_hash`, `dtype_before_cache`; the loader hard-rejects any mismatch. Re-ran 19b and 20 under v2: most numbers reproducible to 0.1° / 1pp, two boundary cases shifted (Gemma 2 L17 gate F→N tipped to PASS at 0.652; OLMo 2 L7 N→F invented unary mass shifted from 20% to 40% — a diffuse-zone signature rather than a clean attractor). The §3.7 caveat box records these. Earlier scripts (13, 16, 17, 18) still use `hash()` and should be migrated when next rerun; deferred since their results are baked into paper_notes from runs that won't be regenerated.
- **Permutation / null calibration on the gate and angle (script 21 or 22 addition).** The random-unit-vector baseline (≈ 90°) is useful but doesn't test the canonical-partition specifically. Add: shuffled arity labels (does angle alignment survive arbitrary labels?), random binary/unary class split (is the unary partition special?), template-label permutation (template-leakage check), invented-word permutation (word-specific overfitting check). All are cache-only.
- **✓ Done: Cross-model replication on Pythia 6.9B-deduped (script 23, Phase 2 principal experiment).** Single-cache v5-expanded-canonical extraction (~6 min at 276 tok/s MPS fp16) plus three nested sweeps (v3 / v4 / v5) at 80 cells per scope. **All three predictions replicate**: (P1) M2-canonical PASS at 31/80 v5 cells with best = 1.000 (10-class, ~10× chance) — Pythia is the strongest cell-density evidence for canonical-operator substrate-invariance in the project; (P2) 3 v3 PASS-arity candidates (more than either OLMo or Gemma), all in N→F direction at the `operator-after → close-paren` anchor pair at three depths (L4, L7, L16); 2 of 3 survive v4 invented-set expansion (the most v4-robust set of any model), all 3 retract at v5 canonical-set expansion — the v4→v5 retraction signature is cross-family stable; (P3) 70.0% of v5 invented mass routes to the three multi-subword NEW canonicals (`nand` 27.9%, `xor` 22.3%, `negate` 19.8%), distributed broadly rather than collapsing to one canonical as OLMo did, with multi-subword tokenization correlating with attraction strength (the two single-subword NEW unaries `possibly` + `always` total 2.4%). Cross-family pattern: same direction (compression toward low-frequency canonicals), model-specific target distribution. Pythia tokenization caveats: `nand` → `[' n', 'and']`, `xor` → `[' x', 'or']`, `negate` → `[' neg', 'ate']`; the `xor` confound at the operator-after anchor is bounded but worth noting. See §3.7.15. **Net Phase 2 verdict: operator-set-bound substrate-invariance is now a three-model finding across three training corpora, three architectures, and three tokenizers.**
- **Per-layer cross-condition probe transfer sweep, both models.** The cross-context-stability story from script 18 is reported at single focus layers. A full per-layer cross-transfer sweep (Diagnostic A at every layer L = 0..N for each model) generalises the finding and identifies whether Gemma 2 is globally aligned at *every* layer or only at the attractor-construction layers. Builds directly on script 17's per-layer machinery; ≈ 30 minutes of compute per model.
- **Per-layer mechanism trace of the arity attractor (originally Phase 0 → Phase 1 handoff; now upgraded).** Combine the per-layer probe trace (script 17 already does this within-condition) with the per-layer cross-transfer sweep above. Plot (a) within-condition unary mass and (b) cross-condition unary mass at each layer. Identifies the "real" arity-encoding layers (cross-transfer-validated) vs the surface-feature-artifact layers (within-condition only).
- **Causal intervention by patching.** Patch canonical A's focus-layer operator-anchored activations into B' (any condition) and measure whether the probe predictions revert to the patched canonical's class. Confirms the operator-anchored position is *causally* responsible for the default-to-unary behavior in each model. Removes the "model is using leaked context signature" alternative explanation. Higher priority for OLMo 2 (where the within-condition probes are structural at chance levels not quite as strong as Gemma 2's, suggesting more contextual leakage).
- **Subword-semantic-load test (lower priority post script 16).** Script 16 partially refuted the reviewer's specific "molex's mole+x → math-y → necessarily" hypothesis by showing molex's necessarily-bias does not survive a context change. The broader methodological point still stands: Tier-2 invented words may have subword-level semantic associations that drive within-unary distribution.
- **Postfix-unary natural-English templates (optional).** A second syntactic-position test that further generalises script 16's prefix/infix dissociation. Lower urgency since script 16 already established the arity attractor's syntactic-position-independence.
- **Explain the L=1/L=4 vs L=2/L=3 depth pattern from script 10.** L=1 and L=4 have peak-gap at layer 1; L=2 and L=3 peak at layers 5–7. This length-parity effect remains unexplained.

**Phase 2 priorities (after the three-model replication + v6 disentanglement, ordered by expected impact):**

- **✓ Done: Multi-subword vs frequency disambiguation (pre-registered v6 canonical-set expansion, script 24, §3.7.16).** All three pre-registered single-axis predictions (P_FREQ, P_SUBWORD, P_INTERACTION) fail in all three models; the audit gate caught one tokenization failure (`iff` 1pc in all three tokenizers); P_RETRACT holds in OLMo and Pythia; Gemma exhibits two emergent v6 PASS-arity cells reported as a methodological caveat on M4b's threshold-sensitivity to readout granularity. Fact 1 strengthens to bootstrap-CI = [1.000, 1.000] in all three models under 15-class readout. The remaining mechanism gap (which canonical attracts a given invented word, given that single-axis predictions fail) is the target of the script 25b embedding-similarity probe.
- **✓ Done: Script 25a — Causal patching at the Gemma L2 close-paren v6 emergent PASS-arity cells + OLMo §3.7.9 Fact-1 anchor (§3.7.17).** Three cells × four conditions × 16 invented words × 10 stimuli = 1920 patched forward passes (~7.5 min compute, model loads excluded). **Three distinct verdicts, one per cell.** Gemma 2 9B `opera→close L 2` is **causally arity-respecting** (probe-causality 100%, ΔKL = +0.048 / +0.038, arity-flip 8/0/8 in both directions with ΔKL = +0.033 / +0.061): a single tightly-scoped, causally validated model-specific exception to operator-set-bound substrate-invariance in Gemma 2 9B. Gemma 2 9B `sente→close L 2` is **probe-only** (probe-causality 100%, ΔKL flat at −0.020 / −0.012, random-norm control exceeds targeted patch): §3.7.16's methodological-caveat reading is confirmed for this second emergent cell. OLMo 2 7B `sente→close L 10` — the project-flagship Fact-1 anchor — is also **probe-readable but not causally load-bearing** (probe-causality 100%, ΔKL = −0.012 / −0.017, random-norm strongly positive at +0.019 / +0.012). Three findings recorded in §3.7.17: (1) Fact 1's geometric cross-notation transfer is not uniformly causally load-bearing — the §3.7.9 OLMo cell is geometric/probe-readable but causally inert under this patching protocol, refining §5's "linear probes only" caveat to a concrete answer at this cell; (2) the Gemma v6 emergent PASS-arity finding splits 1:1, with one cell causally validated and one cell confirmed as M4b-granularity-only; (3) source anchor is a first-class causal variable independent of probe-causality — same target, same layer, same probe reading, opposite causal verdicts at the two Gemma cells. The headline for §4.1 needs one tightly-scoped Gemma-specific caveat, not a wholesale retraction.
- **Script 25c (optional follow-up to 25a, deferred).** A more comprehensive Fact-1 causal test would patch the §3.7.9 OLMo cell from *multiple* source anchors (operator-after, first-arg, close-paren) to identify which (source, target) anchor pairs are causally load-bearing at this cell. Single-cell × four sources × four conditions × 16 words ≈ 600 forward passes; ~5 min compute reusing script 24 caches. Deferred until after paper.md draft scope is decided — the current §3.7.17 evidence is sufficient for the "anchor-pair-dependent causal grounding" claim, but a fuller causal map would strengthen the paper's mechanistic story.
- **✓ Done: Script 25b — Embedding-similarity probe at the early extraction layers (§3.7.18).** The §3.7.16 P_FREQ / P_SUBWORD / P_INTERACTION adjudication showed no single-axis reduction of the default mechanism is consistent with the data. Script 25b tested whether the residual factor could be operationalised as mean-pooled cosine similarity ("contextual semantic neighborhood"). **Result: the gap is NOT closed.** Cross-model headline (mean over 80 cells per model, bootstrap 95% CIs): `agree-all` = 11.6% / 26.6% / 24.0%, `agree-arity` = 11.4% / 21.5% / 19.0%, `arity-match` = 54.1% / 66.4% / 60.2% (Gemma / OLMo / Pythia). No model meets the §6 pre-spec `agree-arity ≥ 60%` threshold. At distributed cells (M4c < 0.7, the methodologically interesting regime), `agree-all` drops to 11-16% across all three models — barely above the 6.7% chance baseline. L0 agreement is at floor (0-1%) in all three models, re-confirming the script 14 H1-construction finding under v6. The residual third factor is therefore *probe-decision-boundary geometry*, not raw cosine similarity — the LogisticRegression boundary captures per-word residual-stream structure that mean-pooling wipes out. A specific failure-mode footnote: at distributed cells across all three models, the unconstrained cosine argmax collapses every invented word to `nand` (or `negate` for unary-intended under arity-conditioning), most likely because `nand`'s mean activation sits further from the canonical centroid than higher-frequency canonicals. This is a methodological caveat on raw cosine-similarity readouts of substrate-invariance — they are contaminated by canonical-magnitude effects. With §3.7.16, §3.7.17, and §3.7.18 complete, the experimental programme is essentially closed; paper.md drafting is now the principal remaining task.
- **Sparse-autoencoder feature labelling (deferred follow-up to §3.7.18).** §3.7.18 identified probe-decision-boundary geometry as the residual third factor in the default mechanism, but did not characterise it. The natural next experiment is SAE feature labelling at the focus layers (L 4 in Gemma, L 10 in OLMo/Pythia where 25b agreement peaks) on either the open Gemma Scope SAEs or a small set trained on the v6 caches. The question: does the canonical-discriminating direction the probe learns decompose into a small set of feature directions, and do those features carry interpretable labels (frequency, subword shape, semantic neighbourhood, syntactic position)? Deferred to post-paper.md; the current §3.7.18 evidence is sufficient for the "probe-decision-boundary geometry" claim and the SAE follow-up would extend the mechanism story rather than close any current gap.
- **Instruction-tuned variant test.** All three Phase 2 models are base checkpoints. Run scripts 22-24 on Pythia-Chat, OLMo-2-Instruct, and Gemma-2-IT (same parameter count, same architecture). Does instruction tuning expand the operator-set the model can represent (Fact 1 stronger or extended to more cells), expand novel-operator generalization (Fact 2 partially refuted), or have no effect (the operator-set-bound finding is a pre-training-corpus property, not modified by SFT)? Single-cache extraction per model, ~10-15 min each on M4 fp16. Lower priority than 25a/25b because the base-model finding is already three-family replicated; instruction-tuning is a scope-expansion follow-up rather than a mechanism-closing one.
- **Sparse-autoencoder analysis on Gemma Scope.** SAE feature labeling at the cross-transfer-validated arity-encoding layers (L4 NEUTRAL and L2/L16 FUNC-PFX in Gemma 2). Does the arity direction decompose into a single "unary operator" feature, separable "not" and "necessarily" features, or a more complex factor structure? The molex-98%-necessarily anomaly from script 15 suggests at minimum that some within-unary structure exists beyond a simple unary-class encoding.
- **Operator-novelty (edge type 2)**: test substrate-invariance for *truly novel* operators with no canonical analog (e.g., a ternary majority connective, or a binary operator with an unfamiliar semantic table). The arity-attractor finding predicts the ternary canonical should *not* fit into the unary region; the question is whether the model has a corresponding "ternary" region, defaults novel binaries into the unary region (treating all novelty as unary-shaped), or develops some other failure mode. The cross-condition transfer instrument from Phase 1 entry is well-suited to this: train a probe on canonical-arity activations and evaluate it on novel-arity invented activations to see whether the novel arity has a separate region or collapses into one of the canonical arity-regions.
- **Causal intervention by activation patching** at the cross-transfer-validated arity-encoding layers. Patch canonical-A operator-anchored activations into B'_neutral and observe probe predictions and downstream next-token outputs. Confirms causality.
- **Test the arity-encoded-at-operator-position claim against position perturbation.** If unary/binary is encoded by the operator-anchored position specifically, post-operator residual streams should NOT carry the arity signal as cleanly. Probing post-operator and pre-operator positions for unary-vs-binary classification on canonical A is a direct test.
- **Stimulus expansion beyond propositional logic**: set theory, group theory ($S_4$), simple type theory. Each domain has its own arity/role structure; the arity-attractor finding makes a sharp testable prediction (within-arity confusability, across-arity discriminability).
- **Perplexity-matched control ladder**: shuffled, n-gram-sampled, high-temperature-sampled, and uniform-random comparison sequences as the full control class.

## 7. Reproducibility

All experiments are reproducible from the scripts in `experiments/`. Specific dependencies (sklearn ≥ 1.5 with the `multi_class` parameter removed; transformers ≥ 4.45 for OLMo 2 support) are pinned in `experiments/requirements.txt`. All random seeds are fixed (default 17). Stimulus generation is deterministic given seed and template set.

End-to-end runtime on the M4 (48 GB) for the complete Phase 0 experiment suite is approximately 5 minutes at 1B and approximately 45–55 minutes at 7B once weights are cached locally (script 13 alone is ~4 minutes; script 15 is ~7 min CPU / ~3 min MPS; script 16 is ~7 min CPU / ~3 min MPS due to ~1500 forward passes across both conditions). First-run 7B weight download is ~14 GB (~10 minutes on a fast connection). Memory usage peaks at ~14 GB RSS during 7B forward passes; comfortable within the M4's unified-memory budget with no swap-spill or MPS CPU-fallback observed. Switching the model between scales requires only editing the `MODEL_ID` constant at the top of each script.

Specific scripts and their outputs referenced in this document:

- §2.3 tokenization screening: `experiments/02_tokenization_screening.py`
- §2.1 MPS smoke test: `experiments/03_mps_smoke_test.py`
- §3.1 variable substrate-invariance: `experiments/06_substrate_invariance_v2.py`
- §3.2 pooling-choice methodological finding: `experiments/07_operator_renaming.py` (the artifact) and `experiments/08_pooling_comparison.py` (the correction)
- §3.3 operator-anchored U-shape: `experiments/08_pooling_comparison.py` (run at both 1B and 7B)
- §3.4 operator-identity probe results, §3.4.1 confusion-matrix mechanism, §3.5 notation coherence: `experiments/09_operator_identity_probe.py` (run at both 1B and 7B)
- §3.4.2 subword-length variation, H1/H2/H3 adjudication: `experiments/10_subword_length_probe.py` (run at 7B)
- §3.4.3 L=1 bar-anomaly H3 confirmation, H4 emergence: `experiments/11_l1_bar_anomaly_probe.py` (run at 7B)
- §3.4.4 second-canonical-unary probe (provisional H1a): `experiments/12_second_unary_probe.py` (run at 7B)
- §3.4.5 four-channel decomposition and H4 factorial quantification: `experiments/13_template_context_quantification.py` (run at 7B). The empty-and-column finding is from this script's prediction-count matrix. H4_pull values are computed directly by the script.
- §3.4.6 embedding-similarity audit (H3 layer-0 attribution test): `experiments/14_embedding_similarity_audit.py` (run at 7B, CPU-only since no forward passes needed; ~75 s including model load).
- §3.4.7 template-neutral probe (intermediate H1 confirmation): `experiments/15_template_neutral_probe.py` (run at 7B, ~7 min on CPU / ~3 min on MPS).
- §3.4.8 syntactic-confound stress test (definitive arity-attractor confirmation, prefix/infix dissociation): `experiments/16_syntactic_confound_stress_test.py` (run at 7B, ~7 min on CPU / ~3 min on MPS).
- §3.7.1 cross-model replication on Gemma 2 9B: `experiments/17_gemma2_cross_model_replication.py`. Requires gated-model authentication (`huggingface-cli login` against an account approved for `google/gemma-2-9b`). Runs bf16 on MPS; peak RSS ≈ 22 GB. Full run ≈ 12 min on M4 once weights are cached locally; first-run weight download ≈ 18 GB.
- §3.7.2–§3.7.3 four-diagnostic probe-artifact battery: `experiments/18_probe_artifact_diagnostics.py`. Runs both Gemma 2 9B and OLMo 2 7B sequentially with mid-run memory cleanup (deletes Gemma 2 before loading OLMo 2). Tees full stdout/stderr to a timestamped log file in `experiments/outputs/18_*.log` (set `NO_LOG=1` to disable). Full run ≈ 18–22 min on M4.
- §3.7.6 directional-angle quantification: `experiments/19_directional_angle_analysis.py`. Same dual-model load pattern as script 18 with mid-run memory cleanup. Computes centroid-based and probe-based cross-notation angles per layer, with a 200-sample random-unit-vector baseline. Tees to `experiments/outputs/19_*.log`. Full run ≈ 7–8 min on M4 once both models are cached.
- §3.7.7 gated directional-angle analysis: `experiments/19b_directional_angle_gated.py`. Extends script 19 with bootstrap 95% CIs on the angles (100 within-class resamples), a bidirectional canonical-transfer gate column at each layer pair, and cross-layer pairings (the script-18 critical pairings reproduced with directional-angle measurements layered on top). Disk-caches activations to `experiments/outputs/cache/` (~270 MB total across both models) so repeat runs skip extraction. Tees to `experiments/outputs/19b_*.log`. Full run ≈ 8 min on M4 first-run; ~3 min cache-hit.
- §3.7.8 gated invented-mass re-test: `experiments/20_gated_invented_mass.py`. Re-runs cross-condition probe transfer at the specific gate-passing pairings identified by script 19b. Uses the 19b disk cache exclusively — no model load required, runs in ~6 s. Tees to `experiments/outputs/20_*.log`. The 19b → 20 split (cache-then-probe) is now the recommended pattern for any follow-up invented-mass measurement at a new (model, layer) pairing: do the extraction once with 19b (or a 19b-style script), then probe rapidly with 20-style scripts.

## Appendix A: Working title candidates

1. *Asymmetric Substrate-Invariance: Variable and Operator Renaming Probe Different Morphospace Boundaries in OLMo 2*
2. *The Morphospace Has Structure: Role-Specific Substrate-Invariance in Open Language Models*
3. *Below Chance: A Diagnostic Signature of Substrate-Bound Operator Representation in OLMo 2*
4. *Three Pooling Strategies, Three Conclusions: A Methodological Note on Substrate-Invariance Measurement*

## Appendix B: Glossary

- **Substrate-invariance**: the property that structurally-equivalent inputs produce aligned internal representations regardless of surface form.
- **Morphospace**: the space of possible representational structures a model can express; borrowed from evolutionary biology (Raup, 1966). A *morphospace boundary* is the limit of structures the model can accommodate.
- **Platonic Representation Hypothesis**: the conjecture that sufficiently large neural networks converge on representations of a substrate-independent underlying structure (Huh et al., 2024).
- **Tier 1 / Tier 2 / Tier 3 tokenization**: classification of candidate symbols by how OLMo 2's BPE tokenizer handles them — single-token, consistent multi-token, or byte-fallback fragmentation respectively.
- **Operator-anchored pooling**: residual-stream extraction at the token position immediately following the last subword of the first operator occurrence.
- **Substrate-invariance gap**: signed difference between CKA(canonical, renamed) and CKA(canonical, scrambled) at a given layer. The principal robust metric for substrate-invariance under multi-pooling reporting.
- **A-B' probe gap**: the difference between linear-probe accuracy on canonical activations (A, measured by 5-fold cross-validation) and on operator-renamed activations (B', measured as held-out accuracy of the probe trained on A). The principal single-number metric for operator substrate-invariance.
- **Notation coherence effect**: empirical observation (1B-only; did not replicate at 7B) that jointly renaming variables and operators (B'') recovers more substrate-invariance than renaming operators alone (B').
- **Uniform-default-to-`not` mechanism**: failure mode observed at 7B layer 7 in which the canonical-operator probe assigns ≥93% of invented-operator instances to the `not` class regardless of which canonical operator the invented word actually replaces. Confirmed by subword-length variation (script 10) to persist across L ∈ {1, 2, 3, 4} at 91–93% rates, with one embedding-driven exception (the `bar`→`or` recovery).
- **Role-bound asymmetry**: the working hypothesis (supported by 1B→7B replication) that the difference between variable and operator substrate-invariance reflects the syntactic role of the symbol (placeholder vs content-bearing), not the model's parameter capacity.
- **H1 / H2 / H3 / H4**: the four mechanisms shaping operator-renaming probe predictions in OLMo 2 7B, as established by Phase 0 scripts 09–12. **H1** (structural default to `not` *specifically*, not unary-class-generically) is the dominant attractor — confirmed by script 12's 5-class probe showing 84% of invented-`necessarily` inputs go to `not`. **H2** (tokenization-position effect) is rejected — peak gaps across L ∈ {1, 2, 3, 4} are within 4 percentage points. **H3** (embedding-similarity escape channel) is a word-specific channel that allows rare invented words to escape the H1 default when their embedding is independently close to a non-`not` canonical — confirmed by `bar`→`or` recovery persisting across template contexts (74-82%). **H4** (template-lexical-context pull channel) is a template-wide channel by which the surrounding template's lexical content pulls predictions toward the canonical that the template encodes — confirmed by `foo`/`zap` recoveries in or-slots reaching 44-62%, and by 16% `necessarily` predictions on modal-template invented operators.
- **`bar`→`or` recovery**: empirical observation that the L=1 invented word `bar`, when used as a replacement for any canonical, is dominantly classified by the canonical-operator probe as `or` (82% in or-slot, 74% in and-slot). The principal piece of evidence for H3 isolated from H4.
- **Four-channel decomposition**: the working factorial picture for operator-renaming probe predictions: P(canonical=c) ≈ H1 attractor toward `not` + H3 word-specific embedding pull + H4 template-context pull + residual. Each channel has a measurable magnitude on canonical Phase-0 stimuli; the decomposition fully accounts for the observed confusion matrices.
- **Cross-condition probe transfer**: train a linear probe on canonical activations in one notation (e.g., NEUTRAL-metalinguistic), evaluate on invented-operator activations in a different notation (e.g., FUNCTIONAL-PREFIX). High cross-transfer unary mass = the probe direction captures a notation-invariant arity structure. Low cross-transfer unary mass with within-condition structural probe (verified by held-out canonical generalisation) = notation-local arity directions. Established as the gold-standard substrate-invariance instrument at Phase 1 entry (script 18 Diagnostic A).
- **Context-invariant arity direction**: a single direction in residual stream space that approximately encodes the unary-vs-binary distinction across multiple syntactic/notation contexts. Gemma 2 9B has one; OLMo 2 7B does not. Operationally measured by cross-condition probe transfer; quantified scale-invariantly by directional-angle analysis (script 19).
- **Notation-local arity direction**: a probe direction that captures arity within one notation but does not transfer to another. OLMo 2's NEUTRAL probe direction is notation-local: it is structural under within-notation held-out canonical generalisation, but its predictions on FUNCTIONAL-PREFIX invented activations are 0% unary (vs the within-notation FUNC-PFX probe's 100% unary). The two notation-local directions are different vectors in residual stream space.
- **Multi-stage arity processing**: the *within-condition* per-layer pattern observed in Gemma 2 9B FUNCTIONAL-PREFIX (script 17): unary-mass peaks at L2 and L16-17 with a trough at L6-L12. Cross-condition probe transfer (script 18) accepts L2 cleanly, marks L16 as ambiguous (chance-level canonical transfer at the canonical-transfer gate), and rejects L6-L12. Script 19's directional-angle measurement places the L16 cross-notation cosine at +0.33–0.34, only marginally above OLMo 2's notation-local baseline of +0.26, so the "multi-stage" framing is now downgraded to "an early cross-notation-transferable arity stage (L2) followed by monotonic decay with depth"; the late-stage emergence at L16-L17 in within-condition probe space is real but does not correspond to a re-aligned cross-notation arity direction.
- **Cross-notation alignment**: cosine similarity between condition-specific arity directions at a given layer in residual stream space. The principal scale-invariant cross-model measurement at Phase 1 entry (script 19). Operationalised as both a centroid-based direction (mean-unary minus mean-binary centroid) and a probe-based direction (binary logistic-regression weight on raw activations). Reference points: cos 0 ≡ 90° = random (= isotropically unrelated); cos +1 ≡ 0° = identical direction. Both Gemma 2 (best cos +0.70) and OLMo 2 (flat ≈ +0.27) are well above cos 0, so neither has fully orthogonal arity directions across notations. Cross-notation alignment is sufficient for the script-18 NEUTRAL probe's labels to transfer to FUNC-PFX activations in Gemma 2 but not in OLMo 2; the empirical transfer threshold appears to sit somewhere in cos 0.4–0.5 and Pythia results will help refine this.
- **Canonical-transfer gate**: a methodological filter applied to cross-condition probe transfer results. A same-layer pairing is accepted as "mutually transportable" only if a 5-class canonical classifier transfers across the notation boundary at accuracy ≥ 0.65 in *both* directions (≈ 3× chance for 5-class; bidirectional check added in script 19b after the OLMo 2 L7 case exposed unidirectional gate failure). PASS = both directions ≥ 0.65; AMBIG = one direction PASS / one in [0.30, 0.65]; FAIL = either direction below 0.30 or both in [0.30, 0.65]. Introduced after peer review of scripts 17–18 to flag the Gemma 2 FUNC-PFX@L16 → NEUTRAL@L4 case where 100% invented unary mass coexisted with 0.200 cross-canonical accuracy (chance-level). **Necessary but not sufficient**: script 20's OLMo 2 L10 finding (bidirectional gate PASS at 0.800 / 0.688, centroid 71.3°, probe 74.4°, invented unary mass 0% / 17% with N→F predictions 100% "implies") is the cleanest illustration in the project — invented words are *actively binary-classified* under bidirectional gate-PASS, demonstrating that the gate measures 5-class canonical discriminability across notations but not unary-vs-binary axis alignment. The full instrument pairs the gate with a directional-angle measurement on the binary unary-vs-binary axis (M3) and an invented-mass measurement with per-canonical breakdown (M4, the catchment-basin check); cross-notation arity-respecting transfer requires all four (M1-M4) to be in agreement.
- **Necessarily catchment basin**: a Gemma 2-specific phenomenon identified by script 20. In NEUTRAL → FUNC-PFX cross-condition probe transfer, invented words are mapped to the "necessarily" canonical at a fraction that grows monotonically with depth (L2 = 0%, L4 = 80%, L8 = 100%). The basin is *anti-correlated* with directional-angle alignment: tightest cross-notation angle (L2 at 47°) → smallest basin (0% necessarily); widest cross-notation angle (L8 at 67°) → maximal basin (100% necessarily). The mechanism is the H4 template-context effect (Phase 0): NEUTRAL templates ("Consider the word {op}...") give "necessarily" a wide decision-boundary catchment because it is the most syntactically generic unary modifier. The catchment basin propagates through depth in residual stream space, eventually absorbing all invented activations into the single canonical regardless of cross-notation directional alignment. The script-17/18 "Gemma 2 has cross-notation arity transfer at L4" claim is therefore refined to "Gemma 2 has cross-notation transfer of a necessarily-dominated catchment basin at L4-L8"; arity-respecting transfer (invented words assigned to the canonical that matches their intended arity) is not demonstrated.
