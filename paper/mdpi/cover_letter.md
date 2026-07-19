# Cover Letter

**To:** The Editorial Office, *Quantum Reports* (MDPI)

**Manuscript title:** Predicting the Trainability of Variational Quantum Circuits: A Data-Driven Model for Barren Plateaus

**Author:** Md Habibur Rahman (sole and corresponding author)

**Affiliation:** Department of AI Convergence Engineering, Gyeongsang National University, Jinju, South Korea

---

Dear Editors,

I am pleased to submit my manuscript, *"Predicting the Trainability of Variational Quantum Circuits: A Data-Driven Model for Barren Plateaus,"* for consideration as an original research article in *Quantum Reports*.

**Significance.** Barren plateaus—the exponential vanishing of cost-function gradients as a variational quantum circuit is scaled—are widely regarded as the central obstacle to training quantum machine-learning models. At present they are diagnosed *reactively*, by sampling many gradients on each candidate circuit, an expensive procedure that becomes infeasible precisely in the large-qubit regime where the problem matters most. This manuscript reframes that diagnosis as a *supervised prediction* problem: I ask whether a purely classical model can predict a circuit's trainability from its architecture alone—qubit count, depth, ansatz type, entanglement pattern, entangler gate, and cost-observable locality—without evaluating a single gradient. Using an exact statevector simulator I generate a labeled dataset of 20,000 random circuits and show that a gradient-boosted model predicts the gradient-variance trainability with high accuracy (AUC 0.99 for the barren/trainable decision), and—most notably—*extrapolates* from small circuits to the largest sizes that can still be simulated. This yields a cheap ansatz-screening heuristic and a data-driven, interpretable account of which design choices drive untrainability.

**Context and relationship to existing work.** The barren-plateau literature is overwhelmingly analytical, deriving variance scaling laws for particular ansatz families (e.g., McClean et al. 2018; Cerezo et al. 2021; Holmes et al. 2022; Larocca et al. 2025). This contribution is complementary and empirical: rather than proving a bound for one family, I learn a predictive map across a heterogeneous design space, and show that the learned model recovers the established theory from data (cost locality and entanglement structure dominate). Consistent with recent, appropriately skeptical benchmarking work (Bowles et al. 2024; Schuld & Killoran 2022) and the barren-plateau/simulability results of Cerezo et al. (2025), I make **no** claim of quantum advantage: the model, the data, and the value proposition are entirely classical. The object predicted is a property *of* quantum circuits; the predictor is a classical diagnostic tool.

**Fit to the scope of *Quantum Reports*.** The manuscript sits squarely within the journal's remit of quantum computing, quantum information, and quantum machine learning. It addresses a recognized, high-impact problem in variational quantum algorithms and offers a practically useful, reproducible methodology of interest to both the quantum-computing and machine-learning readerships the journal serves.

**Reproducibility.** In keeping with the journal's open-science standards, all code (simulator, dataset generator, training and model-comparison scripts), the exact command lines, fixed random seeds, and the full dataset are openly available at https://github.com/bithabib/quantum_computer_simulator, and interactive demonstrations of the underlying simulator run in the browser at https://quantum.qbithabib.com. Every reported number regenerates from a single command.

**Required statements.**

- I confirm that neither the manuscript nor any parts of its content are currently under consideration for publication with or published in another journal.
- I am the sole author of this manuscript and approve its submission to *Quantum Reports*.
- This manuscript has not been previously submitted to any MDPI journal.
- Generative AI tools (ChatGPT, OpenAI; and Claude, Anthropic) were used for language editing and grammar correction only; this is disclosed in the Acknowledgments of the manuscript.
- The author declares no conflicts of interest.

I believe this work will be of interest to the readership of *Quantum Reports* and thank you for considering it. I look forward to the reviewers' feedback.

Sincerely,

Md Habibur Rahman
Sole and corresponding author — habib@gnu.ac.kr
Department of AI Convergence Engineering, Gyeongsang National University, Jinju, South Korea
