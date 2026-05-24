# Lucky Defaults: A Failure Mode in Probe-Based Generalisation Metrics

**Brendan Budworth** &nbsp;·&nbsp; Independent &nbsp;·&nbsp; `brendan.budworth@protonmail.com` &nbsp;·&nbsp; May 2026

Companion case study: [github.com/brendanbudworth-research/trained-vocab-substrate-invariance](https://github.com/brendanbudworth-research/trained-vocab-substrate-invariance)

---

## Abstract

A common pattern in probe-based generalisation studies is to train a many-class linear probe on in-distribution items, evaluate it on out-of-distribution items, and report a coarser *property-level* accuracy (binary vs. unary, positive vs. negative, singular vs. plural, language A vs. language B) as evidence that the underlying representation respects the property. We describe a failure mode that inflates this coarse accuracy without any genuine property-sensitive routing: every out-of-distribution item is deterministically mapped to a small set of in-distribution prototypes whose property labels happen to match the intended labels frequently enough. We call this *lucky-default* routing. We propose a per-item top-class concentration metric \\(p_{w,\min}\\) as a cheap diagnostic, illustrate it on a worked example, and report a case study from a recent substrate-invariance experiment on three mid-scale base language models. The detector was developed on an exploratory two-model sweep where it flagged four of eight initially-flagged "passing" generalisation cells as lucky-default routings; pre-registered and subsequently applied to a held-out four-scope sweep that included a previously-unseen model family, it flagged another two of eight cells. Once a cell is flagged, the property-level metric ceases to be sufficient evidence of property-sensitive generalisation without additional checks. The recommendation generalises beyond logical-operator vocabulary to any probe study where many-class predictions are coarsened to a property metric.

---

## 1. Introduction

A common evaluation pattern in probe-based interpretability proceeds as follows: train a linear probe on in-distribution items with a \\(K\\)-class readout, evaluate the probe on a set of held-out or out-of-distribution items, and report a coarsened *property-level* accuracy by mapping the \\(K\\)-class predictions through a deterministic partition \\(\pi: \{1,\ldots,K\} \to \{0,1\}\\) (binary vs. unary, singular vs. plural, positive vs. negative, language A vs. language B). High aggregate \\(\pi\\)-accuracy on out-of-distribution items is then taken as evidence that the representation has internalised the property in a way that generalises beyond the training items.

This pattern hides a failure mode that aggregate metrics do not surface. Suppose every out-of-distribution item's predictions concentrate on the same small set of in-distribution prototypes. Then the coarse \\(\pi\\)-accuracy reflects two facts about the readout vocabulary, not two facts about the representation: *(i)* which familiar prototype each novel item is routed to, and *(ii)* how the partition \\(\pi\\) happens to label the dominant prototype(s). When both facts align with the intended \\(\pi\\)-labels of the out-of-distribution items, \\(\pi\\)-accuracy looks high. The representation has not generalised in any property-sensitive way; the readout has collapsed onto a default. We call this routing pattern *lucky-default*.

The lucky-default failure mode is detected by a single per-item statistic: the minimum, over out-of-distribution items \\(w\\), of the fraction of \\(w\\)'s predictions concentrated on \\(w\\)'s modal predicted class. We write this minimum as \\(p_{w,\min}\\). When \\(p_{w,\min}\\) is at or near \\(1\\), every item is essentially deterministically routed to a single class, and any aggregate \\(\pi\\)-accuracy is a statement about routing targets plus the property partition, not about property-sensitive structure in the representation. The diagnostic is one line of code to add to any probe-evaluation pipeline that already records the full \\(K\\)-way prediction distribution per item.

Our contribution is threefold. First, we name and formalise the failure mode (Section 2) and the diagnostic (Section 3). Second, we present a worked anchoring example (Section 2, Table 1) and re-analyse a recent substrate-invariance experiment on three mid-scale base language models (Section 4). The diagnostic was developed on an exploratory two-model sweep (OLMo 2 7B, Gemma 2 9B), where it flagged four of eight initially-flagged "passing" cells as lucky-default routings. The same threshold, re-applied as a pre-registered PASS criterion to a held-out four-scope sweep including a previously-unseen model family (Pythia 6.9B-deduped), flagged another two of eight cells. The two passes provide development and held-out evidence respectively. A flag does not by itself prove that a result is mistaken; it shows that the property-level metric is no longer sufficient evidence for property-sensitive generalisation, and that additional checks (individual-stimulus patches, held-out template tests, or independent causal evidence) are needed before a generalisation claim is defensible. Third, we describe the regime in which the diagnostic is most useful (Section 5): any probe study where many-class predictions are coarsened to a property metric, and the evaluation set lies in a different distributional regime from the training set.

The diagnostic is closest in spirit to calibration [Guo et al. 2017] and out-of-distribution detection [Hendrycks & Gimpel 2017; Liang et al. 2018; Lee et al. 2018] but answers a slightly different question: in those literatures, the concern is that a classifier assigns high confidence to predictions that are wrong. Here, the concern is that a classifier assigns confident, repeatable, but *collapsed* predictions, and that downstream coarsening of the prediction space makes the collapse look like generalisation.

---

## 2. The lucky-default failure mode

### 2.1 Setup and notation

Let \\(f: \mathbb{R}^d \to \{1, \ldots, K\}\\) be a \\(K\\)-class linear probe trained on in-distribution activations. Let \\(\{w_1, \ldots, w_N\}\\) be a set of \\(N\\) out-of-distribution items, each represented by \\(S\\) stimuli with activations \\(x_{i,1}, \ldots, x_{i,S} \in \mathbb{R}^d\\). Let \\(\pi: \{1, \ldots, K\} \to \{0, 1\}\\) be a property partition over the \\(K\\) classes (e.g. binary vs. unary), and let \\(y_i \in \{0, 1\}\\) be the intended property label of item \\(w_i\\). The *coarse property accuracy* reported in many probing studies is

$$
A_\pi \;=\; \tfrac{1}{N S} \sum_{i=1}^{N} \sum_{s=1}^{S} \mathbf{1}\bigl[\pi(f(x_{i,s})) = y_i\bigr].
$$

A study reports a generalisation success when \\(A_\pi\\) clears some pre-specified threshold (e.g. \\(A_\pi \geq 0.65\\), or \\(A_\pi\\) significantly above the marginal-frequency baseline).

### 2.2 The failure mode

The lucky-default pattern is the joint occurrence of:

- **(LD1) Per-item collapse.** There exists a small set of classes \\(\mathcal{C}^\star \subset \{1, \ldots, K\}\\) with \\(|\mathcal{C}^\star| \ll K\\) such that for every item \\(w_i\\), nearly all of \\(w_i\\)'s \\(S\\) stimuli map to a single class in \\(\mathcal{C}^\star\\).
- **(LD2) Property alignment by coincidence.** The intended labels \\(\{y_i\}\\) happen to be distributed in such a way that the (few) classes in \\(\mathcal{C}^\star\\), under \\(\pi\\), are labelled in the same way as the majority of items.

Under (LD1) and (LD2), \\(A_\pi\\) can be arbitrarily close to \\(1\\) even though the probe is performing only two-prototype routing rather than \\(K\\)-way property-sensitive classification. No information about per-item structure is being read out; the coarse accuracy is a statement about which prototypes the routing defaults to, multiplied by the partition \\(\pi\\)'s label of those prototypes.

The pattern is invisible to several natural sanity checks. Aggregate per-class entropy can be high enough that no single class dominates the marginal prediction distribution (e.g. four items collapse to one binary class and four to one unary class, giving balanced marginals). Aggregate concentration measures over the marginal distribution — maximum-class share or the Herfindahl–Hirschman index [Hirschman 1964] — fail to fire when \\(\mathcal{C}^\star\\) has more than one element. The diagnostic must be per-item, not marginal. The following statement makes the asymmetry precise.

**Proposition (lucky-default routing inflates \\(A_\pi\\)).** Fix \\(K, m, N\\) with \\(1 \leq m < K\\) and \\(N \geq m\\). Let \\(\mathcal{C}^\star \subseteq \{1, \ldots, K\}\\) be a target set of size \\(m\\), and let \\(q: \mathcal{C}^\star \to [0, 1]\\) be a routing distribution with \\(\sum_c q_c = 1\\). Suppose

1. every item \\(w_i\\) has \\(p_{w,\text{top}}(i) = 1\\), with modal class \\(c^\star_i \in \mathcal{C}^\star\\) and item-to-class frequencies given by \\(q\\) (LD1');
2. every item's intended property label is the property label of its routing target: \\(y_i = \pi(c^\star_i)\\) for all \\(i = 1, \ldots, N\\) (LD2').

Then \\(p_{w,\min} = 1\\) and \\(A_\pi = 1\\). The marginal Herfindahl index is \\(\mathrm{HHI} = \sum_{c \in \mathcal{C}^\star} q_c^2 \in [1/m, 1]\\) and the marginal max-share is \\(\max_c p_c = \max_c q_c \in [1/m, 1]\\). When \\(q\\) is uniform over \\(\mathcal{C}^\star\\) both equal \\(1/m\\), which can be set arbitrarily small by increasing \\(m\\) while \\(A_\pi\\) remains at \\(1\\).

*Proof.* Routing is deterministic by (LD1'), so \\(p_{w,\min} = \min_i p_{w,\text{top}}(i) = 1\\) and every stimulus prediction is \\(f(x_{i,s}) = c^\star_i\\). Coarse property accuracy is \\(A_\pi = \tfrac{1}{N} \sum_i \mathbf{1}[\pi(c^\star_i) = y_i]\\), which equals \\(1\\) by (LD2'). The marginal distribution is \\(p_c = q_c\\) for \\(c \in \mathcal{C}^\star\\) and \\(p_c = 0\\) otherwise, so the marginal indices follow. \\(\square\\)

**Remark.** (LD2') is the *coincidence* condition: it says that whichever class each item is routed to happens to have a property label matching the item's intended label. In a generic out-of-distribution evaluation, items have intended labels drawn from some prior, and the routing target is selected by the classifier; (LD2') asks that these two independently-determined quantities agree on every item. The opposite extreme — every item routed to a class whose property label opposes its intended label — yields \\(A_\pi = 0\\). The proposition isolates the high-\\(A_\pi\\) case, which is the failure mode this paper diagnoses; the diagnostic's value lies in detecting (LD1') irrespective of (LD2'), because once (LD1') holds, \\(A_\pi\\) depends on \\(\pi\\) and the routing targets alone and is no longer evidence of property-sensitive generalisation.

The consequence is that for any fixed marginal-concentration threshold (e.g. \\(\mathrm{HHI} < 0.70\\) or \\(\max_c p_c \leq 0.85\\)), there is a target-set size \\(m\\) at which the constructed lucky-default cell passes the marginal check, is flagged by the per-item collapse diagnostic (\\(p_{w,\min} = 1\\)), and inflates \\(A_\pi\\) to its ceiling. Per-item \\(p_{w,\min}\\) catches the case the marginal indices cannot.

### 2.3 Anchoring example

The example we use throughout is taken from the substrate-invariance case study (Section 4; [Budworth 2026]). A linear probe is trained on activations from one of three mid-scale base language models (OLMo 2 7B) at a particular layer and token position, using a small in-distribution readout vocabulary of familiar logical operator words. The probe is then evaluated on \\(N = 16\\) "invented" (out-of-distribution) operator names, each appearing in \\(S\\) stimuli, with an intended-arity property label \\(y_i \in \{\text{binary}, \text{unary}\}\\) assigned by the experimenter.

Table 1 tracks one specific cell — OLMo 2 7B, notation transfer N→F, training anchor `sente`, target anchor `close`, layer 10 — across four pre-registered canonical-set sizes (5 to 15 readout classes, all extensions pre-specified before evaluation, §3.5 of [Budworth 2026]). At the smallest readout (5 readout classes × 5 invented words), the cell looks like a clean PASS-arity result: \\(A_\pi = 0.88\\), well above the \\(0.65\\) threshold. The per-item floor \\(p_{w,\min} = 0.50\\) at this scope is *not* itself a deterministic lucky-default verdict; it means each invented word's predictions split roughly evenly between two classes, which is compatible with either genuine per-item ambiguity or a precursor to deterministic collapse at larger readouts. On the largest pre-registered readout (15 classes × 16 invented words), every one of the 16 invented words routes deterministically to the single classifier label `nand` (\\(p_{w,\min} = 1.00\\)), and \\(A_\pi\\) is now exactly the marginal binary-arity rate of the test set (\\(0.50\\)). The diagnostic's role here is therefore not to flag the \\(5 \times 5\\) scope as lucky-default in isolation, but to make the cell's behaviour under readout expansion legible: the \\(5 \times 5\\) \\(A_\pi\\) of \\(0.88\\) is *unstable / non-evidential* once the readout is expanded, because expansion collapses the per-item routing onto a single canonical (\\(\mathcal{C}^\star = \{\texttt{nand}\}\\) at \\(K=15\\)) and the \\(5 \times 5\\) property accuracy fails to persist. The diagnostic does not by itself adjudicate whether this collapse reflects a meaningful nearest-prototype semantics or an artefact; it shows that \\(A_\pi\\) at the smaller readout is not, on its own, evidence of property-sensitive structure that survives canonical-set expansion.

**Table 1.** Anchoring example. One cell (OLMo 2 7B, N→F, layer 10, training anchor `sente`, target anchor `close`) tracked across four pre-registered readout scopes from [Budworth 2026]. \\(A_\pi\\) is the coarse binary-vs-unary property accuracy; HHI is the Herfindahl–Hirschman index of the marginal per-class invented-prediction share; \\(\max_c p_c\\) is its top-class share; \\(p_{w,\min}\\) is the proposed per-item top-class concentration diagnostic. *None* of the three aggregate diagnostics flags the \\(5 \times 5\\) scope. The per-item \\(p_{w,\min}\\) diagnostic flags the larger readouts as lucky-default; the \\(5 \times 5\\) scope is not itself flagged as deterministic collapse (\\(p_{w,\min} = 0.50\\) admits per-item ambiguity), but is rendered *non-evidential* by the readout-expansion trajectory: the apparent \\(A_\pi\\) does not survive when the readout is enlarged.

| Readout | \\(A_\pi\\) | HHI | \\(\max_c p_c\\) | \\(p_{w,\min}\\) | Verdict |
|:--|:-:|:-:|:-:|:-:|:--|
| \\(5 \times 5\\)   | 0.88 | 0.57 | 0.70 | 0.50 | PASS-arity (non-evidential) |
| \\(5 \times 16\\)  | 0.78 | 0.53 | 0.65 | ~0.50 | PASS-arity (non-evidential) |
| \\(10 \times 16\\) | 0.50 | 1.00 | 1.00 | 1.00 | **lucky-default** |
| \\(15 \times 16\\) | 0.50 | 1.00 | 1.00 | 1.00 | **lucky-default** |

---

## 3. Diagnostic: per-item top-class concentration

### 3.1 Definition

For item \\(w_i\\), define the modal predicted class \\(c_i^\star = \arg\max_{c \in \{1, \ldots, K\}} \sum_{s=1}^{S} \mathbf{1}[f(x_{i,s}) = c]\\), and the *per-item top-class concentration*

$$
p_{w,\text{top}}(i) \;=\; \frac{1}{S} \sum_{s=1}^{S} \mathbf{1}\bigl[f(x_{i,s}) = c_i^\star\bigr] \;\in\; [1/K, 1].
$$

\\(p_{w,\text{top}}(i)\\) is the maximum share of \\(w_i\\)'s \\(S\\) stimulus predictions landing on any single class. Define the diagnostic as the *floor* of \\(p_{w,\text{top}}\\) over items:

$$
p_{w,\min} \;=\; \min_{i \in \{1, \ldots, N\}} p_{w,\text{top}}(i) \;\in\; [1/K, 1].
$$

\\(p_{w,\min}\\) is the top-class concentration of the *least*-collapsed item: when \\(p_{w,\min}\\) is high, every item's predictions are highly concentrated on a single class, including the item whose distribution is most spread out. \\(p_{w,\min}\\) therefore answers a stronger question than "is some item routed deterministically?" — it asks whether *every* item is.

### 3.2 Threshold

We recommend \\(p_{w,\min} < 0.95\\) as a PASS gate (equivalently: \\(p_{w,\min} \geq 0.95\\) flags the cell as lucky-default). The threshold is loose enough to tolerate single-stimulus noise and modest within-item disagreement, and tight enough that values \\(\geq 0.95\\) imply *every* item's predictions are nearly deterministic.

Table 2 shows the case-study count of flagged cells across a sensitivity sweep on the threshold. The case-study sweep separates flagged cells (largest observed \\(p_{w,\min}\\) across scopes of \\(0.98\\) and \\(1.00\\)) from non-flagged cells (largest observed \\(p_{w,\min}\\) across scopes in \\([0.52, 0.84]\\)) by a wide gap, so the verdict is stable across thresholds in \\([0.85, 0.95]\\).

**Table 2.** Threshold sensitivity on the case-study sweep (Table 4). "Flagged" counts cells where \\(p_{w,\min} \geq \tau\\) at any of the four scopes. The verdict is flat in \\([0.85, 0.95]\\); the recommended \\(\tau = 0.95\\) sits in the middle. The \\(0.99\\) threshold drops one borderline cell; \\(0.80\\) admits a cell whose routing is more spread (largest observed \\(p_{w,\min} = 0.84\\) across scopes).

| Threshold \\(\tau\\) | Cells flagged | Cells added vs. \\(\tau = 0.95\\) |
|:-:|:-:|:--|
| 0.99 | 1 | — (drops Pythia `sente→close L10` at 0.98) |
| 0.95 | 2 | — (recommended) |
| 0.90 | 2 | — |
| 0.85 | 2 | — |
| 0.80 | 3 | adds OLMo `opera→close L24` at 0.84 |
| 0.70 | 3 | same |
| 0.60 | 4 | adds Pythia `opera→close L7` at 0.64 |

The \\(0.95\\) threshold is therefore robust to noise on this data but should be re-evaluated on data where the gap between collapsed and non-collapsed items is narrower.

### 3.3 Relation to existing concentration measures

Two natural alternatives operate on the *marginal* per-class prediction distribution \\(p_c = \tfrac{1}{NS} \sum_{i,s} \mathbf{1}[f(x_{i,s}) = c]\\) rather than on per-item distributions:

- **Maximum share** \\(\max_c p_c \in [1/K, 1]\\). Fires only when the entire marginal mass is on a single class and so misses the typical lucky-default case where \\(|\mathcal{C}^\star| \geq 2\\).
- **Herfindahl–Hirschman index** \\(\mathrm{HHI} = \sum_c p_c^2 \in [1/K, 1]\\) [Hirschman 1964]. Fires later than the max share for the same reason: the index drops as soon as the marginal is spread across two or more dominant classes.

The case-study anchor cell in Table 1 illustrates the gap: at \\(K = 5\\), the cell has \\(\max_c p_c = 0.70\\) and \\(\mathrm{HHI} = 0.57\\), either of which would be reported as "distributed" under standard thresholds (e.g. \\(\mathrm{HHI} < 0.7\\), \\(\max_c p_c \leq 0.85\\)), while \\(p_{w,\min} = 0.50\\) correctly reflects the genuine within-item split. At \\(K \geq 10\\), the marginal indices and \\(p_{w,\min}\\) agree only because \\(|\mathcal{C}^\star| = 1\\). In other words, marginal indices are necessary but not sufficient: they catch single-class-collapse but miss the two- or three-prototype routing patterns where the lucky-default failure mode typically lives.

### 3.4 Practical considerations

**Cost.** \\(p_{w,\min}\\) requires no additional probe evaluations; it is computed from the same per-stimulus prediction array used to compute \\(A_\pi\\) and any marginal concentration measure. Storage cost is the per-item top-class share, an \\(N\\)-vector.

**Multiple stimuli per item.** The definition assumes \\(S \geq 2\\). When only a single stimulus is available per item, the per-item \\(p_{w,\text{top}}(i)\\) degenerates to \\(1\\) trivially. The diagnostic requires the evaluation set to include multiple stimuli per item so that *within-item* prediction variance is observable. In the case study, each invented word appears in 64 template stimuli; in the toy demonstration of Table 1, \\(S\\) ranges from a few dozen to a few hundred per item depending on the readout scope.

**Joint reporting.** \\(p_{w,\min}\\) is not a substitute for \\(A_\pi\\) or marginal concentration. It is a third axis. We recommend studies report all three: *(a)* the coarse property accuracy \\(A_\pi\\), *(b)* a marginal concentration measure (\\(\max_c p_c\\) or \\(\mathrm{HHI}\\)), and *(c)* the per-item \\(p_{w,\min}\\). A cell PASSES generalisation only if all three clear their respective thresholds; a cell with high \\(A_\pi\\) but high \\(p_{w,\min}\\) is *lucky-default-flagged*, and \\(A_\pi\\) at that cell should not be treated as evidence of property-sensitive generalisation without additional corroboration.

### 3.5 What the diagnostic does and does not show

A \\(p_{w,\min}\\) trigger establishes one fact: every out-of-distribution item is being routed to a small set of in-distribution prototypes, nearly deterministically. It does not establish that this routing is artefactual rather than semantically meaningful — a model that maps every novel logical operator to `nand` could be representing them all as semantically near to `nand`, and a model that maps every novel content word to the nearest familiar synonym could be doing something cognitively reasonable. What the trigger *does* establish is that the coarse property accuracy \\(A_\pi\\) at that cell is no longer evidence of property-sensitive generalisation: under deterministic prototype routing, \\(A_\pi\\) is fully predicted by the routing target and the partition \\(\pi\\), with no within-item information involved. Adjudicating whether a flagged routing is meaningful or artefactual requires evidence *outside* the property-coarsened metric — causal interventions, \\(K\\)-way accuracy, behavioural probes, or independent representational tests.

---

## 4. Case study: detector development and pre-registered application

The case study draws on the substrate-invariance experiment of [Budworth 2026]. The diagnostic was developed during methodology iteration on a two-model exploratory sweep (OLMo 2 7B and Gemma 2 9B; §4.1), then pre-registered and applied to a held-out four-scope sweep that added a previously-unseen model family (Pythia 6.9B-deduped; §4.2). The two sweeps provide complementary evidence: development-set diagnostic value (how many "passing" cells the new statistic catches when applied post hoc on the data that motivated it) and held-out diagnostic value (how many additional cells it catches on data it was not developed against, under a pre-registered threshold).

### 4.1 Development sweep (4 of 8)

The exploratory cell sweep over OLMo 2 7B and Gemma 2 9B used a small invented-operator set (\\(N = 5\\)) and a 5-class readout (\\(K = 5\\)), with the headline PASS criterion \\(A_\pi \geq 0.65 \wedge \mathrm{HHI} < 0.70\\). Eight cells across the two models cleared this criterion. Inspecting the per-item prediction distributions revealed a recurring pattern: within an apparently "passing" cell, four of the five invented words would route to one canonical readout class and the fifth would route to another, with both within-item concentrations near 100%. The marginal HHI statistic was below threshold (e.g. \\(\mathrm{HHI} = 0.8^2 + 0.2^2 = 0.68\\) for a 4+1 split), and the marginal max-share was likewise below its \\(0.85\\) threshold. The coarse property accuracy clearing \\(0.65\\) was therefore a coincidence between the property labels of the two prototype classes and the intended-property distribution of the five invented items.

Adding the per-item statistic \\(p_{w,\min} = \min_i p_{w,\text{top}}(i)\\) as a fifth PASS conjunct, with threshold \\(p_{w,\min} < 0.95\\), flagged four of the eight cells as LUCKY-NEG (three OLMo cells, one Gemma cell; Table 3). The remaining four cells either had genuine within-item prediction spread (\\(p_{w,\min} \leq 0.7\\)) or retracted on \\(A_\pi\\) alone once the invented set was expanded.

**Table 3.** The four development-sweep cells flagged by \\(p_{w,\min}\\) but not by the marginal indices. Intended-arity partition over the five invented words (\\(N = 5\\)): three binary (`bliq`, `dren`, `vusp`) and two unary (`molex`, `perph`); standard \\(\max_c p_c \leq 0.85\\) and \\(\mathrm{HHI} < 0.70\\) thresholds. Every cell has \\(A_\pi = 0.80\\), well above the \\(0.65\\) PASS threshold; \\(\max_c p_c \leq 0.85\\) in all four (clearing the marginal max-share threshold) and \\(\mathrm{HHI} < 0.70\\) in all four (clearing the marginal Herfindahl threshold). The per-item \\(p_{w,\min}\\) is \\(\geq 0.98\\) in every cell because each individual invented word is routed deterministically to a single canonical; the marginal measures stay below threshold because the routing is distributed across two prototypes. Numbers are from `outputs/22b_20260520_083957.log`. This is the empirical demonstration of \\(p_{w,\min}\\)'s independent value: the four cells would have passed every aggregate diagnostic in standard use, and only the per-item floor flags them.

| Cell | \\(A_\pi\\) | HHI | \\(\max_c p_c\\) | \\(p_{w,\min}\\) | Dominant prototypes | Routing pattern |
|:--|:-:|:-:|:-:|:-:|:--|:-:|
| OLMo `N→F opera→opera L7`  | 0.80 | 0.68 | 0.80 | 1.00 | `and` (4/5), `necessarily` (1/5) | 4 + 1 |
| OLMo `N→F sente→opera L10` | 0.80 | 0.68 | 0.80 | 1.00 | `and` (4/5), `necessarily` (1/5) | 4 + 1 |
| OLMo `N→F opera→close L24` | 0.80 | 0.68 | 0.80 | 0.98 | `and` (4/5), `necessarily` (1/5) | 4 + 1 |
| Gemma `N→F sente→opera L17` | 0.80 | 0.52 | 0.60 | 1.00 | `and` (2/5), `necessarily` (3/5) | 2 + 3 |

### 4.2 Pre-registered sweep (2 of 8)

After detector development, the substrate-invariance project's pre-registered four-scope sweep ([Budworth 2026], §3.5) added Pythia 6.9B-deduped as a third model family and re-ran the PASS-arity adjudication across an expanded canonical readout (5, then 10, then 15 readout classes) and an expanded invented set (\\(N = 16\\)). The PASS criterion included \\(p_{w,\min} < 0.95\\) as a frozen, pre-extraction conjunct. Eight cells cleared the criterion at the smallest readout (\\(5 \times 5\\)); Table 4 tracks each cell's \\(p_{w,\min}\\) trajectory across the four pre-registered scopes.

**Table 4.** Eight cells that PASS the pre-registered numerical criterion at the smallest readout (\\(5 \times 5\\)) in the v6 sweep of [Budworth 2026]. \\(p_{w,\min}\\) trajectory across the four pre-registered scopes; threshold \\(p_{w,\min} < 0.95\\) for PASS. Two cells trigger the diagnostic at \\(K = 10\\) (both reach \\(p_{w,\min} \geq 0.95\\) as the canonical-set expansion exposes the single-attractor routing pattern). The remaining six cells retract because \\(A_\pi\\) drops below \\(0.65\\) under invented-set or canonical-set expansion. By the largest scope (\\(15 \times 16\\)), no cell remains PASS under any PASS criterion. The 2-of-8 pre-registered count is independent of the 4-of-8 development count in §4.1, and the lucky-default-triggering cells appear in the OLMo and Pythia model families — with Pythia having been added after detector development, the Pythia trigger is an out-of-sample replication.

| Cell | \\(5 \times 5\\) | \\(5 \times 16\\) | \\(10 \times 16\\) | \\(15 \times 16\\) | Trigger? | Final verdict (\\(15 \times 16\\)) |
|:--|:-:|:-:|:-:|:-:|:-:|:--|
| Gemma `N→F opera→first L4`  | 0.46 | 0.36 | 0.48 | 0.52 | no  | M2A-only |
| Gemma `N→F sente→first L8`  | 0.54 | 0.48 | 0.32 | 0.30 | no  | M2A-only |
| OLMo  `F→N first→opera L7`  | 0.52 | 0.50 | 0.32 | 0.28 | no  | M2A-only |
| **OLMo `N→F sente→close L10`** | 0.56 | 0.52 | **1.00** | **1.00** | **yes** | **LUCKY-NEG** |
| OLMo  `N→F opera→close L24` | 0.84 | 0.70 | 0.56 | 0.46 | no  | M2A-only |
| Pythia `N→F opera→close L4` | 0.48 | 0.48 | 0.52 | 0.52 | no  | M2A-only |
| Pythia `N→F opera→close L7` | 0.64 | 0.60 | 0.38 | 0.48 | no  | M2A-only |
| **Pythia `N→F sente→close L10`** | 0.60 | 0.50 | **0.98** | 0.60 | **yes** | **LUCKY-NEG** |

### 4.3 Reading both passes together

The development sweep provides direct evidence of the failure mode: four of eight apparent generalisation results were few-prototype routings whose coarse property accuracy was tracking the routing target rather than per-item property-sensitive structure. The pre-registered sweep provides the out-of-sample test: with \\(N\\) tripled, \\(K\\) tripled, and a new model family added, the diagnostic flags two additional cells as lucky-default routings.

**What the detector catches at a fixed readout versus what readout expansion exposes.** Table 4 is clear on a subtlety that needs naming. Neither of the two pre-registered-sweep trigger cells fires \\(p_{w,\min}\\) at the smallest readout (\\(5 \times 5\\), where \\(p_{w,\min}\\) is \\(0.56\\) and \\(0.60\\)). Both fire at \\(10 \times 16\\); the OLMo cell stays fired at \\(15 \times 16\\), while the Pythia cell relaxes back to \\(p_{w,\min} = 0.60\\) at \\(15 \times 16\\) as the routing target spreads across the wider readout vocabulary. The detector at a fixed small readout does not flag these cells. What the detector flags is the routing pattern that becomes visible at the readout scope where prototypes have been added. The implication is two-fold. First, the diagnostic's full value emerges when it is applied across multiple readout scopes, not just at the original training-scope readout: a single-scope report can miss the collapse. Second, the diagnostic is complementary to readout expansion as a sanity check, not a replacement for it. A study that runs only at a single readout, with no expansion, will not catch this failure mode through \\(p_{w,\min}\\) alone — the diagnostic gives a sharper, mechanistically interpretable signature than \\(A_\pi\\) does once expansion is applied, but it does not eliminate the need for expansion.

The two cells that trigger \\(p_{w,\min}\\) in the pre-registered sweep also coincidentally trigger single-class marginal collapse at the same readout (\\(\max_c p_c = 1.00\\) at \\(K = 10\\) in both cases); a stricter \\(\max_c p_c \leq 0.85\\) marginal threshold would have caught them too. \\(p_{w,\min}\\)'s independent contribution is on cells where the routing target is a small set \\(\mathcal{C}^\star\\) with two or three classes rather than one: there, marginal concentration measures diffuse and miss the failure mode, while the per-item statistic remains close to \\(1\\). Table 3 shows the empirical case: all four development-sweep LUCKY-NEG cells have \\(\max_c p_c \leq 0.80\\) and \\(\mathrm{HHI} \leq 0.68\\), both *within* the standard marginal-PASS region (\\(\max_c p_c \leq 0.85\\) and \\(\mathrm{HHI} < 0.70\\)), while \\(p_{w,\min} \geq 0.98\\) in every one of them. The Gemma cell in particular has \\(\mathrm{HHI} = 0.52\\) — well below the \\(0.70\\) threshold — and would have read as a clean "distributed-routing" positive on marginal indices alone. The pre-registered-sweep cells sharpen further into single-class collapse, in which case the marginal measures agree; the development-sweep cells are the case the diagnostic is uniquely sensitive to (§4.1).

A note on the development–pre-registration distinction: the combined 6-of-16 reclassification rate across the two sweeps should not be read as "the diagnostic catches roughly one cell in three." The base rate of lucky-default routing depends on the readout vocabulary, the invented-item distribution, and the training-data prior of the model family. The development sweep was selected for having "surprising" high-\\(A_\pi\\) cells; the pre-registered sweep was applied across a larger candidate cell space. The diagnostic's value is qualitative (it identifies a specific failure mode) rather than a calibrated false-positive rate.

---

## 5. When to apply the diagnostic

The diagnostic is most useful when three conditions hold:

1. the probe has \\(K \geq 3\\) classes and multiple stimuli per evaluation item (\\(S \geq 2\\));
2. the headline metric is a property-level coarsening \\(A_\pi\\), not the full \\(K\\)-class accuracy; and
3. the evaluation items lie in a different distributional regime from the training items (out-of-distribution items, novel members of a category, cross-lingual transfer items, held-out templates, etc.).

Concretely:

- **Logical and grammatical category transfer.** The case study (Section 4) is one instance. Any probe study claiming generalisation of a logic-inspired or grammar-inspired category — arity, mood, case, polarity, agreement features — through a many-class readout and a property-level coarsening can apply the diagnostic directly.
- **Cross-lingual concept transfer.** Activation patching and probing studies that report "language A representations contain language-agnostic concept C" [Dumas et al. 2025] typically train on language-A items and evaluate on language-B items via a concept-level coarsening. If language-B items collapse to a small set of language-A prototype vectors, the concept-level metric is vulnerable to lucky-default inflation.
- **Sentiment, syntax, and other coarsened many-class probes.** A 5-star sentiment probe coarsened to positive vs. negative; a 30-class syntactic-function probe coarsened to subject vs. non-subject; a multi-class biological-process probe coarsened to a binary functional partition — all produce property metrics that can mask deterministic routing.
- **Few-shot and prototypical-network evaluations.** Prototype-based classifiers [Snell et al. 2017; Mensink et al. 2013] route each query to its nearest support prototype. The lucky-default failure mode is essentially the prototype-routing reading of the probe: each out-of-distribution item is nearest-prototype-routed, and the coarse-property metric rides on the prototype labels.

The diagnostic is not useful (or trivially passes) when the probe reports the full \\(K\\)-class accuracy (no coarsening), or when only a single stimulus is available per item (no within-item variance to measure). It is also weak when the evaluation set is small and balanced enough that the lucky-default coincidence is hard to distinguish from genuine routing — the case study used 16 invented items spanning 8 binary and 8 unary intended labels, which is at the small end of what permits clean detection.

---

## 6. Related work

The probing literature has long acknowledged that probe accuracy is not the same as task-relevance of the probed information [Belinkov 2022; Hewitt & Liang 2019; Ravichander et al. 2021; Pimentel et al. 2020; Alain & Bengio 2017]. Control tasks [Hewitt & Liang 2019] and minimum-description-length probing [Pimentel et al. 2020] address the related question of whether the probe is reading out information that is already present in the representation or memorising the task. Our concern is downstream of that: even when the probe accuracy is genuine, the coarsening to a property metric can be inflated by deterministic routing. The two concerns are independent: a probe can pass a control-task check (the information is in the representation) and still fail a lucky-default check (the property-level generalisation metric is inflated by routing collapse on out-of-distribution items).

The closest neighbours by spirit are in calibration [Guo et al. 2017] and out-of-distribution detection [Hendrycks & Gimpel 2017; Liang et al. 2018; Lee et al. 2018], but the diagnostic question is different. Calibration asks whether the classifier's confidence matches its accuracy; out-of-distribution detection asks whether an input is from the training distribution. Both target the case where the classifier is *wrong* on out-of-distribution inputs. Lucky-default routing targets the case where the classifier is *right by coincidence*: confidently mapping out-of-distribution items to fixed in-distribution prototypes whose coarse-property labels happen to align with the intended labels.

The mechanism is most directly analogous to prototype-based classification [Snell et al. 2017; Mensink et al. 2013]: each out-of-distribution item is nearest-prototype-routed, and the coarse-property metric inherits the prototype labels. The lucky-default detector can be read as: "check whether the probe is silently behaving like a fixed-prototype nearest-neighbour classifier on out-of-distribution items."

In mechanistic interpretability, the activation-patching literature [Heimersheim & Nanda 2024; Zhang & Nanda 2024] has converged on similar reporting hygiene for a related failure mode: a patched intervention can show a behavioural effect without being specific to the targeted feature, and reporting a single metric on a single intervention pair is insufficient to support a causal claim about the targeted feature. The probe-based analogue of that hygiene is the joint reporting of \\(A_\pi\\), marginal concentration, and \\(p_{w,\min}\\) that we recommend in Section 3.

---

## 7. Conclusion

Many-class probes whose predictions are coarsened to a property-level accuracy can pass that property-level threshold while performing only two- or three-prototype deterministic routing of out-of-distribution items. The per-item top-class concentration metric \\(p_{w,\min}\\) catches the routing collapse cheaply and is not redundant with the marginal concentration measures (\\(\max_c p_c\\), \\(\mathrm{HHI}\\)) that are already commonly reported. In a recent substrate-invariance experiment, the diagnostic was developed on a two-model sweep where it flagged four of eight "passing" generalisation cells as lucky-default routings, and then re-applied as a pre-registered criterion to a held-out four-scope sweep including a previously-unseen model family, where it flagged two of eight additional cells. A \\(p_{w,\min}\\) trigger does not by itself prove a result is mistaken; it shows that the property-level metric \\(A_\pi\\) is no longer sufficient evidence of property-sensitive generalisation without additional corroboration. We recommend that probe studies reporting coarsened-label generalisation include \\(p_{w,\min}\\) as a default diagnostic alongside their headline property metric.

---

## References

- **Alain & Bengio 2017.** Understanding intermediate layers using linear classifier probes. *ICLR Workshop.* [arXiv:1610.01644](https://arxiv.org/abs/1610.01644)
- **Belinkov 2022.** Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics 48(1).* [doi:10.1162/coli_a_00422](https://doi.org/10.1162/coli_a_00422)
- **Budworth 2026.** Trained-vocabulary substrate-invariance in mid-scale language models. arXiv preprint. Source repository: [github.com/brendanbudworth-research/trained-vocab-substrate-invariance](https://github.com/brendanbudworth-research/trained-vocab-substrate-invariance)
- **Dumas et al. 2025.** Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers. *ACL.* [arXiv:2411.08745](https://arxiv.org/abs/2411.08745)
- **Guo et al. 2017.** On Calibration of Modern Neural Networks. *ICML.* [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- **Heimersheim & Nanda 2024.** How to use and interpret activation patching. [arXiv:2404.15255](https://arxiv.org/abs/2404.15255)
- **Hendrycks & Gimpel 2017.** A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. *ICLR.* [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
- **Hewitt & Liang 2019.** Designing and Interpreting Probes with Control Tasks. *EMNLP-IJCNLP.* [doi:10.18653/v1/D19-1275](https://doi.org/10.18653/v1/D19-1275)
- **Hirschman 1964.** The Paternity of an Index. *American Economic Review 54(5).*
- **Lee et al. 2018.** A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *NeurIPS.* [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
- **Liang et al. 2018.** Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. *ICLR.* [arXiv:1706.02690](https://arxiv.org/abs/1706.02690)
- **Mensink et al. 2013.** Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost. *IEEE TPAMI 35(11).* [doi:10.1109/TPAMI.2013.83](https://doi.org/10.1109/TPAMI.2013.83)
- **Pimentel et al. 2020.** Information-Theoretic Probing for Linguistic Structure. *ACL.* [doi:10.18653/v1/2020.acl-main.420](https://doi.org/10.18653/v1/2020.acl-main.420)
- **Ravichander et al. 2021.** Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance? *EACL.* [doi:10.18653/v1/2021.eacl-main.295](https://doi.org/10.18653/v1/2021.eacl-main.295)
- **Snell et al. 2017.** Prototypical Networks for Few-shot Learning. *NeurIPS.* [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
- **Zhang & Nanda 2024.** Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. *ICLR.* [arXiv:2309.16042](https://arxiv.org/abs/2309.16042)
