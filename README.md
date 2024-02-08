<h1 align="center">
  <strong><a href="https://bithabib.github.io/quantum_computer_simulator/templates/quantum_gate_simulator.html">Eshyana: The Quantum Computer Simulator</a></strong>
</h1>
Eshyana: The Quantum Computer Simulator is an immersive and interactive quantum computing simulator designed to demystify the complexities of quantum mechanics. Embark on a journey through the fascinating realm of quantum computing, where users can experiment with a comprehensive set of quantum gates, manipulate qubits, and witness the intriguing principles of superposition, entanglement, and quantum parallelism in action. Whether you are a novice curious about the quantum world or a seasoned enthusiast eager to deepen your understanding, EshyQuantum Explorer provides a user-friendly interface for exploring the limitless possibilities of quantum information processing. Dive into the quantum frontier, unravel the mysteries of quantum gates, and unlock the potential of quantum computing in this educational and engaging simulation experience.

## All Quantum Gates: 
 - Pauli X Gate (NOT Gate): Flips the state of a qubit from |0⟩ to |1⟩ and vice versa.
 - Pauli Y Gate: Similar to the Pauli X gate but includes a phase flip.
 - Pauli Z Gate: Introduces a phase flip without changing the state of the qubit.
 - Hadamard Gate: Creates a superposition by putting a qubit in an equal probability of |0⟩ and |1⟩ states.
 - CNOT Gate (Controlled-NOT Gate): Performs a NOT operation on the target qubit if the control qubit is |1⟩.
 - SWAP Gate: Swaps the states of two qubits.
 - Toffoli Gate (CCNOT Gate): Performs a NOT operation on the target qubit if both control qubits are |1⟩.
 - Fredkin Gate (CSWAP Gate): Swaps the states of two qubits based on the state of a third qubit.
 - Phase Gate: Introduces a phase shift without changing the state of the qubit.
 - CPhase Gate (Controlled-Phase Gate): Applies a phase shift to the target qubit depending on the state of the control qubit.
 - Rx, Ry, Rz Gates: Rotation gates around the x, y, and z axes, respectively.
 - U Gate (Arbitrary Single-Qubit Gate): Represents a general single-qubit unitary operation.
 - CCZ Gate (Controlled-Controlled-Z Gate): Applies a phase shift to the target qubit if both control qubits are |1⟩.
 - CS Gate (Controlled-S Gate): Applies a phase shift to the target qubit if the control qubit is |1⟩.
 - CRx, CRy, CRz Gates (Controlled-Rotation Gates): Controlled versions of the single-qubit rotation gates.



## Probability Amplitude:
The probability amplitude is a complex number that represents the probability of finding a qubit in a particular state. The probability amplitude is a combination of the amplitude and phase of the qubit state. The amplitude is the magnitude of the probability amplitude, and the phase is the angle of the probability amplitude. The probability amplitude is used to calculate the probability of measuring a qubit in a particular state. The probability of measuring a qubit in a particular state is the square of the magnitude of the probability amplitude.
# Derivation of Probability Formula for Quantum Mechanics

## Introduction

In quantum mechanics, the probability of a particular outcome is related to the square of the magnitude of the probability amplitude. When considering the combination of two probability amplitudes, alpha1 and alpha2, the probability formula involves the interference between them. This README explains the derivation of the probability formula including the interference term.

## Complex Conjugates

$$ \text{In the context of complex numbers, the complex conjugate of a complex number } z, \text{denoted as } z^* \text{ or } \bar{z} \text{, is formed by changing the sign of the imaginary part of } z .$$

If $ z = a + bi $ where $ a $ and $ b $ are real numbers and $ i $ is the imaginary unit, then the complex conjugate $ z^* $ is given by:

$$ z^* = a - bi $$

In other words, the complex conjugate of a complex number simply reflects the point representing that number across the real axis on the complex plane. Geometrically, if you have a point representing a complex number in the complex plane, its complex conjugate is the reflection of that point across the real axis.

The complex conjugate is particularly useful when dealing with operations involving complex numbers, such as multiplication, division, and taking the modulus. For example, when multiplying a complex number by its conjugate, the imaginary parts cancel out, resulting in a real number:

$$ z \cdot z^* = (a + bi)(a - bi) = a^2 + b^2 $$

This property is often used in various mathematical and physical contexts, including quantum mechanics, where complex numbers and their conjugates play a crucial role in describing the behavior of quantum systems.


### Step 1: Combined Probability Amplitude

Let alpha be the combined probability amplitude:

$$ \alpha = \alpha_1 + \alpha_2 $$

### Step 2: Probability Calculation

The probability of the outcome is given by the square of the magnitude of alpha:

$$ \text{Probability} = |\alpha|^2 = |\alpha_1 + \alpha_2|^2 $$


### Step 3: Expansion

Expanding the expression using complex conjugates:

$$ |\alpha_1 + \alpha_2|^2 = (\alpha_1 + \alpha_2) \cdot (\alpha_1^* + \alpha_2^*) $$

$$ = \alpha_1 \alpha_1^* + \alpha_2 \alpha_2^* + \alpha_1 \alpha_2^* + \alpha_2 \alpha_1^* $$

### Step 4: Interference Term

The interference term alpha1 alpha2^* contains the phase difference between alpha1 and alpha2, denoted by phi2 - phi1, where phi1 and phi2 are the phases of alpha1 and alpha2 respectively.

$$ \alpha_1 \alpha_2^* = |\alpha_1||\alpha_2|\cos(\phi_2 - \phi_1) $$

### Step 5: Final Probability Formula

Substituting the interference term back into the expression:

$$ |\alpha_1 + \alpha_2|^2 = |\alpha_1|^2 + |\alpha_2|^2 + 2|\alpha_1||\alpha_2|\cos(\phi_2 - \phi_1) $$

This formula accounts for the interference between the probability amplitudes and determines the probability of the outcome.




 ## References:
   - [Quantum States And The Bloch Sphere](https://medium.com/quantum-untangled/quantum-states-and-the-bloch-sphere-9f3c0c445ea3)
   - [Visualizing Single Qubit Quantum Logic Gates](https://medium.com/quantum-untangled/visualizing-quantum-logic-gates-part-1-515bb7b58916)
   - [The basics of Quantum Computing](https://www.quantum-inspire.com/kbase/introduction-to-quantum-computing)
   - [Quantum Computing Book under writing](https://qubit.guide/)
   - [Quantum Computing for Computer Scientists](https://www.cambridge.org/9781108481976)
   - [Quantum Computing Course based on a book](https://www.youtube.com/playlist?list=PLkespgaZN4gmu0nWNmfMflVRqw0VPkCGH)

