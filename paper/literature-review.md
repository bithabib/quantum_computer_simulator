# Literature Review — Quantum Machine Learning for the Eshyana in-browser paper

> Synthesized from a multi-source, adversarially-verified search (28 sources,
> 95 verified claims). The automated synthesis step of the research workflow
> hung, so this write-up was assembled by hand from the verified claim set.
> Dates/venues are from the sources; where an exact paper title is inferred it is
> marked "(≈title)". Full source URLs remain in the research journal and can be
> pulled on request for the BibTeX.

## TL;DR for our paper
- **The gap is real but narrow.** In-browser quantum circuit simulators already
  exist (Quirk, Quirk-E, `quantum-circuit`), and all are JavaScript. **None ships
  a variational-circuit / ML-training layer.** So our novelty is *not* "a browser
  quantum simulator" (that exists) but **"a browser-native QML *training* framework
  + a reproducible encoding study,"** which is defensible.
- **Do NOT claim quantum advantage.** The strongest, most recent literature is
  actively skeptical (benchmarks where classical wins; barren-plateau ↔
  simulability results). Reviewers will reward an honest, reproducibility- and
  education-framed contribution and will punish an advantage overclaim.
- **Encoding-as-hyperparameter is a live, publishable angle** and maps perfectly
  onto a browser tool where users can flip encodings and see the effect.

---

## 1. QML model families for prediction (foundations)
- **Quantum Neural Network (QNN) / parameterized classifier** — Farhi & Neven,
  2018 (arXiv:1802.06002). A sequence of parameter-dependent unitaries on an input
  state; a single Pauli measurement on a readout qubit gives the binary label;
  parameters trained by supervised learning. Foundational VQC/QNN formulation.
  Demonstrated (in simulation) on downsampled two-digit MNIST — an early
  "reduced-MNIST" benchmark.
- **Variational Quantum Classifier + Quantum Kernel** — Havlíček et al., *Nature*
  567:209 (2019). Introduces both a variational classifier and a quantum kernel
  estimator; represents the feature space as a quantum state to exploit
  high-dimensional Hilbert space. Advantage rests on a **feature map conjectured
  classically hard to simulate**. Run on real superconducting hardware. This is
  the canonical citation for *both* the VQC and QSVM families.
- **Circuit-centric quantum classifiers** — Schuld, Bocharov, Svore, Wiebe, *PRA*
  101:032308 (2020; arXiv:1804.00633). Low-depth variational algorithm; amplitude
  encoding; number of learnable parameters scales **poly-logarithmically** in input
  dimension; analytic gradients via shifted circuit evaluations (the parameter-shift
  idea); introduces a quantum "dropout" regularizer.
- **Expressive power** — (≈2023-02-09) There exist feature maps / quantum kernels
  making VQC and QSVM efficient solvers for **any BQP** problem (via the
  k-Forrelation / PromiseBQP-complete problem), establishing universal
  expressiveness — useful for the "why quantum could matter" paragraph.

## 2. Frameworks, simulators, and existing browser tools (the competitive map)
- **PennyLane** (Xanadu) — hardware-agnostic; swaps backends in one line; supports
  PyTorch and TensorFlow front-ends. Its **VQC tutorial** uses the parity task and
  Iris (2D→4D padded), trained by Nesterov-momentum gradient descent, reaching 100%
  on those toy tasks — a natural baseline for us to reproduce.
- **TensorFlow Quantum (TFQ)** (Google) — embeds Cirq circuit nodes into TF graphs;
  tied to TensorFlow; builds datasets on MNIST / Fashion-MNIST.
- **Qiskit Machine Learning** (IBM) — provides the standard VQC/feature-map stack.
- **Qiskit vs PennyLane** (CAEPIA 2024, Springer LNCS 14640): for quantum-kernel
  SVMs, **Qiskit → higher accuracy**, **PennyLane → faster**; both stable to ~20
  qubits; Qiskit judged easier for beginners. (A citable "framework trade-off" fact.)
- **Browser / educational tools — the key prior art:**
  - **Quirk** (Craig Gidney) — drag-and-drop, runs fully in-browser, **JS (98%)**,
    up to **16 qubits**, real-time Bloch/state display. Explicitly **educational**.
    But it's a manual circuit toy — **no variational/ML training**.
  - **Quirk-E** (DEQSE Project, v3.0.0) — extended Quirk; statevector sim; exports
    to OpenQASM/Qiskit/Cirq/TFQ; ships teaching circuits (teleportation, Bell test,
    QFT). Still **no QML training layer**.
  - **`quantum-circuit`** (npm, MIT) — JS statevector engine, **20+ qubits** in
    browser and Node, **sparse statevector** representation; exports to Qiskit/Cirq/
    TFQ. Explicitly noted to have **no built-in variational-circuit or ML support**
    — "a concrete gap for a novel in-browser QML contribution."
  > **Implication:** frame Eshyana-QML as the *training/QML layer* that these
  > simulators lack, with reproducibility + education as the wedge. Consider using
  > or benchmarking against `quantum-circuit` so we're not accused of reinventing
  > the statevector core.

## 3. Data encoding & benchmark datasets
- **Encoding families** (survey, 2026-06-03, 66 works 2017–2026): basis, angle,
  dense-angle, amplitude, **data re-uploading**, and IQP encodings — compared on
  cost, expressivity, robustness; includes a joint trainability analysis (Fourier
  expressivity, barren plateaus, kernel concentration) and a **five-regime decision
  framework** (feature dim × qubit budget × error rate × task → recommended encoding).
- **The three staples**: angle (Pauli-rotation), amplitude (data→amplitudes,
  exponential qubit efficiency), basis (bitstrings→basis states).
- **Encoding is a hyperparameter** (arXiv:2508.00768, 2025): on **Wine** (178×13, 3
  classes) the best-vs-worst VQC differed by **~33% avg, 41% max** just from
  encoding choice (layers fixed, re-uploading on); on **Diabetes** (168×8) recall
  swung ~32%. **No universally dominant encoding.** → Strong support for making our
  paper's core an encoding comparison.
- **NISQ reality check** (encoding survey): below a gate-error threshold **p\* ≈
  10⁻³**, amplitude encoding becomes viable; above it, **shallow angle encodings
  win in practice** despite amplitude's qubit advantage. (We run ideal statevector,
  so we can *state* this and note it as a limitation / future noise study.)
- **"Quantum-inspired encodings alone don't help"** (2026-05-23): evaluated as
  deterministic feature maps against strong classical controls (RFF, poly, PCA, RBF
  SVM, NN), **fixed encoding geometry gives no reliable ML advantage** — each has a
  geometric flaw (amplitude normalization destroys magnitude; angle is redundant
  with linear features; basis imposes Hamming geometry). Cite this to justify our
  **classical baselines**.
- **Common benchmarks in QML**: Iris, parity/synthetic, reduced MNIST /
  Fashion-MNIST, Wine, Diabetes. All small-qubit-friendly → good fit for us.

## 4. Open challenges & the "advantage" debate (frame honestly)
- **Barren plateaus** — Larocca et al., *Nature Reviews Physics* 7:174 (2025;
  arXiv:2405.00781). Loss landscape becomes **exponentially flat** as system size
  grows; triggers include global cost functions, deep/expressive ansätze, poor init,
  and **hardware noise**. Major obstacle to trainability at scale; large active
  mitigation literature.
- **Barren plateau ↔ classical simulability** — Cerezo et al., *Nature
  Communications* 16:7907 (2025; arXiv:2312.09121). The structural properties that
  **guarantee no barren plateaus tend to confine the model to small, classically
  simulable subspaces**; many provably-trainable models are also classically
  simulable (given a data-collection phase). *Not universal*, but a serious caution.
- **Trainability vs dequantization not mutually exclusive** — (ICLR 2025;
  arXiv:2406.xxxxx). Counter-result: one *can* construct PQC models that are **both
  trainable and non-dequantizable** — genuine value needs *both*. Good nuance to cite
  so we're not one-sided.
- **Dequantization** — Tang, "A quantum-inspired classical algorithm for
  recommendation systems," STOC 2019 (arXiv:1807.04271). Kills the exponential
  speedup of the Kerenidis–Prakash recommendation algorithm; shows the "advantage"
  came from **state-preparation/input assumptions**, not quantum computation.
- **Empirical benchmarks favor classical** — Bowles, Ahmed, Schuld,
  "Better than classical? …" (arXiv:2403.07059, 2024). 12 QML models × 6 tasks (160
  datasets): **out-of-the-box classical models outperform** the quantum ones; and
  **removing entanglement often matches or improves** performance — i.e.
  "quantumness" wasn't the decisive ingredient. Open-source benchmark on PennyLane.
- **Perspective** — Schuld & Killoran, *PRX Quantum* 3:030101 (2022). Argues the
  field should **stop letting "beat classical ML" dominate**; at realistic scales
  theory (not benchmarks) is the main tool; classical ML is empirically strong but
  theoretically opaque, quantum is the reverse.

## 5. Recent directions (2023–2025) & our defensible contribution
Recent momentum is in: (a) **encoding as a first-class design choice** (surveys,
hybrid encodings, hardware-grounded decision frameworks); (b) **honest benchmarking
and de-hyping**; (c) **theory of trainability vs simulability**. Notably absent:
an **accessible, zero-install, browser-native environment for actually *training*
QML models** and for **reproducibly** comparing encodings.

**Proposed contribution (Access-defensible):**
1. **Eshyana-QML** — a browser-native VQC/QNN *training* layer on a JS statevector
   core (extending our `qsim`; optionally benchmarked against `quantum-circuit`),
   with parameter-shift or SPSA gradients, running fully client-side.
2. **A reproducible encoding study** — angle vs amplitude vs basis (± data
   re-uploading) on small standard datasets (Iris, parity, Wine), **with honest
   classical baselines** (LogReg, SVM, small MLP), every run reproducible from a
   shareable URL + fixed seeds.
3. **Accessibility/education contribution** — zero-install QML that lowers the
   barrier for learners and for *checking* QML claims, positioned against Quirk /
   Quirk-E / `quantum-circuit` (which lack training).

**Honesty guardrails (write these into the paper):** ideal statevector only (no
hardware noise); small scale; explicitly *not* claiming quantum advantage — instead
claiming accessibility, reproducibility, and a controlled encoding comparison.

---

## Candidate reference list (verify titles/DOIs before submission)
1. Farhi & Neven, *Classification with Quantum Neural Networks on Near Term Processors*, arXiv:1802.06002 (2018).
2. Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567:209 (2019).
3. Schuld, Bocharov, Svore, Wiebe, *Circuit-centric quantum classifiers*, PRA 101:032308 (2020).
4. Schuld & Killoran, *Is quantum advantage the right goal for QML?*, PRX Quantum 3:030101 (2022).
5. Larocca et al., *Barren plateaus in variational quantum computing*, Nature Rev. Phys. 7:174 (2025).
6. Cerezo et al., *Does provable absence of barren plateaus imply classical simulability?*, Nature Comms 16:7907 (2025).
7. Tang, *A quantum-inspired classical algorithm for recommendation systems*, STOC (2019).
8. Bowles, Ahmed, Schuld, *Better than classical? The subtle art of benchmarking QML models*, arXiv:2403.07059 (2024).
9. (ICLR 2025) *Trainability vs dequantization of variational QML* — confirm exact title/authors.
10. Encoding survey (2026) — 66 works, six encoding families — confirm citation.
11. *Encoding as a hyperparameter for VQC*, arXiv:2508.00768 (2025).
12. Quirk (C. Gidney); Quirk-E (DEQSE Project); `quantum-circuit` (Quantastica) — tool citations.
13. PennyLane VQC tutorial (Xanadu); Qiskit Machine Learning docs; TensorFlow Quantum.
14. Qiskit vs PennyLane QSVM comparison, CAEPIA 2024 (Springer LNCS 14640).
