# Paper Outline (working draft)

> **Target venue: IEEE Access** (open access, rapid review). Fallbacks if desk-rejected
> or a better fit is wanted: *IEEE Trans. on Quantum Engineering (TQE)* or *IEEE QCE*.
> Template: **IEEEtran** journal class, `\documentclass[journal]{IEEEtran}` (IEEE Access
> uses the standard journal template; follow their "Preparation of Papers for IEEE
> Access" author instructions).
>
> IEEE Access specifics that shape the paper:
> - **Binary review** against explicit criteria — must be *technically sound, novel,
>   well-presented, and reproducible*. Impact/importance is weighed less than in TQE,
>   but rigor and clarity are non-negotiable.
> - **APC (article processing charge)** applies on acceptance (open access) — budget
>   for it before submitting. Confirm the current fee on the IEEE Access site.
> - Requires an explicit **"contributions of this article"** bullet list in the intro
>   and typically a **related-work comparison table**. We already plan both.
> - Reproducibility (live URL + open code + seeds) is a genuine plus here.
> - arXiv cross-post is fine (open access).
>
> Topic: Quantum Machine Learning (QML) for supervised prediction, built and trained
> entirely in a lightweight **in-browser statevector simulator** (Eshyana). Data is
> generated/encoded from the simulator.
>
> This skeleton is drafted **before** the literature-research report lands. The
> `[RESEARCH]` tags mark places where the deep-research findings (related work,
> the precise gap, competing tools, citations) will be slotted in.
>
> IEEE format notes: two-column IEEEtran; Abstract + Index Terms (keywords);
> references in IEEE numeric style `[1]`; figures as vector where possible; a
> reproducibility statement fits IEEE's open-science push. arXiv cross-post is
> allowed and encouraged (category cs.LG / quant-ph) — check the specific IEEE
> venue's preprint policy before submission.

---

## Candidate titles
- *Eshyana-QML: A Browser-Native, Zero-Install Framework for Reproducible Quantum Machine Learning*
- *Democratizing Quantum Machine Learning: In-Browser Variational Classifiers on a Statevector Simulator*
- *Teaching and Reproducing QML in the Browser: A Statevector Approach to Variational Quantum Classification*

## The contribution (what makes this publishable) — to be sharpened by `[RESEARCH]`
A journal/arXiv paper needs a defensible delta. Our working claims, in priority order:
1. **A system contribution** — the first (to confirm via `[RESEARCH]`) fully
   browser-native, zero-install QML framework: a JS statevector simulator +
   variational training loop that runs client-side, no Python/cloud needed.
2. **A reproducibility/accessibility contribution** — every experiment in the
   paper reproduces in a browser tab with a shareable URL; lowers the barrier to
   QML education and to checking QML claims.
3. **A focused empirical study** — a controlled comparison of **data-encoding
   strategies** (angle vs. amplitude vs. basis / feature maps) for small-qubit
   variational classification, plus honest classical baselines. This is where we
   avoid the common "QML advantage" overclaim (see Challenges).

> Decision to make after research: lead with (1) the tool, or (2) the empirical
> study using the tool. For a journal, a **tool + rigorous reproducible study**
> combined is the strongest framing.

---

## Abstract (fill last)
- 1 sentence: QML promise + the access/reproducibility problem.
- 1–2 sentences: what we built (in-browser statevector QML) and why it matters.
- 1–2 sentences: what we evaluated (encodings, datasets, classical baselines).
- 1 sentence: headline result, stated honestly.
- 1 sentence: availability (live URL + open source).

## 1. Introduction
- Motivation: QML interest is high, but tooling is Python/cloud-heavy → barriers
  for learners and for reproducibility. `[RESEARCH: cite the interest + barriers]`
- Gap: `[RESEARCH: confirm no/limited browser-native QML tool exists]`
- Contributions (bulleted, mirror the list above).
- Paper roadmap.

## 2. Background
- Qubits, gates, statevector simulation (tie to the existing Eshyana `qsim`).
- Supervised learning framing (classification) in one paragraph.
- Variational quantum circuits: encode → parameterized ansatz → measure → loss →
  classical optimizer loop. `[RESEARCH: foundational VQC/QNN citations]`

## 3. Related work `[RESEARCH — this section is mostly filled from the report]`
- QML model families: VQC, QNN, quantum kernels / QSVM. `[RESEARCH]`
- Frameworks & simulators: PennyLane, Qiskit ML, TensorFlow Quantum, Cirq —
  and any browser/educational tools. `[RESEARCH: the competitive landscape + gap]`
- Data encoding strategies and feature maps. `[RESEARCH]`
- Known criticisms: barren plateaus, NISQ noise, contested "advantage". `[RESEARCH]`
- Positioning paragraph: how our work differs.

## 4. System: in-browser QML on a statevector simulator
- Architecture: `qsim` statevector core (already built) + a training layer.
- Encodings implemented: angle, amplitude, basis, (feature map). `[RESEARCH: which]`
- Ansatz family (hardware-efficient layers), measurement → prediction.
- Optimizer: parameter-shift gradients vs. finite-difference / SPSA. `[RESEARCH]`
- Runs client-side; reproducible via URL. Note limits (qubit count, no real noise).

## 5. Experimental setup
- Datasets (simulator-friendly, small-qubit): synthetic parity/linearly-separable,
  Iris (2–4 features), reduced MNIST subset. `[RESEARCH: standard QML benchmarks]`
- Data generation from the simulator (how we make (x, y) pairs; encoding pipeline).
- Metrics: accuracy, loss curves, trainability (gradient variance vs. qubits/layers).
- **Classical baselines** (logistic regression, SVM, small MLP) — mandatory for honesty.
- Protocol: train/test split, seeds, repetitions, hyperparameters.

## 6. Results
- Encoding comparison (the empirical core).
- QML vs. classical baselines — reported without overclaiming.
- Trainability / barren-plateau observations at our scale. `[RESEARCH: what to expect]`
- Reproducibility demonstration (same result in a fresh browser).

## 7. Discussion
- What worked, what didn't, honest limitations (small scale, ideal statevector,
  no hardware noise).
- Educational value / accessibility angle.
- Threats to validity.

## 8. Conclusion & future work
- Recap contributions. Future: noise model, more encodings, hardware backend.

## Reproducibility / Availability
- Live URL, source repo, exact configs, seeds.

---

## Immediate to-dos (once research lands)
- [ ] Fill Related Work + all `[RESEARCH]` tags from the report, with citations.
- [ ] Lock the novelty framing (tool-led vs. study-led) against what exists.
- [ ] Freeze dataset + encoding shortlist.
- [ ] Decide optimizer (parameter-shift feasible in JS? else SPSA/finite-diff).
- [x] Venue: **IEEE Access** (fallback TQE / IEEE QCE). Use IEEEtran journal class.
- [ ] Download the IEEE Access template + "Preparation of Papers" author kit; set up LaTeX.
- [ ] Confirm the current IEEE Access APC (open-access fee) and budget for it.
- [ ] Draft the explicit "Contributions of this article" list + related-work table.
