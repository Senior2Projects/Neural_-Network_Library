# Neural Network Library (From Scratch)

**Course:** CSE473s: Computational Intelligence - Fall 2025  
**Institution:** Faculty of Engineering, Ain Shams University  
**Project Title:** Build Your Own Neural Network Library & Advanced Applications

## Overview

This project involves developing a modular Deep Learning framework entirely from scratch using **Python** and **NumPy**. The goal is to demystify the "black box" of machine learning by implementing the core mathematics of forward and backward propagation manually.

The project is divided into two distinct milestones:
* **Milestone 1:** Core Library Implementation & XOR Validation.
* **Milestone 2:** Unsupervised Learning (Autoencoders) & Transfer Learning (SVM).

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚩 Milestone 1: Core Library & Validation

This milestone focuses on building the foundational building blocks of the neural network and verifying their correctness.

### 1. Library Architecture
We have implemented a modular architecture where every component is a separate class:

* **`Layer` Class**: A base abstract class defining the `forward` and `backward` interfaces.
* **`Dense` Layer**: A fully connected layer that manages Weights ($W$), Biases ($b$), and their respective gradients ($\partial L/\partial W$, $\partial L/\partial b$).
* **Activations**: Implemented as layers with their own derivatives:
    * `ReLU`: $f(x) = \max(0, x)$
    * `Sigmoid`: $\sigma(x) = \frac{1}{1+e^{-x}}$
    * `Tanh`: $\tanh(x)$
    * `Softmax`: For probability distribution outputs.
* **Loss Function**: `MSE` (Mean Squared Error) for calculating error and initial gradients.
* **Optimizer**: `SGD` (Stochastic Gradient Descent) to update parameters using the rule $W_{new} = W_{old} - \eta(\partial L/\partial W)$.

### 2. Unit Testing: Gradient Checking
Before training, we verify the backpropagation math using **Numerical Gradient Checking**. We compare the analytical gradients (computed by our library) against numerical approximations using the finite difference formula:

$$ \frac{\partial L}{\partial W} \approx \frac{L(W+\epsilon) - L(W-\epsilon)}{2\epsilon} $$

* **Location**: `notebooks/project_demo.ipynb` (Section 1).
* **Success Criteria**: The difference between analytical and numerical gradients must be negligible ($< 10^{-7}$).

### 3. Validation: The XOR Problem
To prove the library's capability to learn non-linear boundaries, we solve the classic **XOR problem**.

* **Architecture**: A simple Multilayer Perceptron (e.g., 2-Input $\to$ Hidden Tanh $\to$ 1-Output Sigmoid).
* **Training**: We optimize the network using our `SGD` and `MSE` classes.
* **Goal**: Achieve near-perfect classification (0 or 1) for the four XOR states.
* **Location**: `notebooks/project_demo.ipynb` (Section 2).

---

## 🚩 Milestone 2: Advanced Applications

*(This section is currently under development. It will include Autoencoder implementation for MNIST reconstruction, Latent Space visualization, and SVM Classification comparison with TensorFlow.)*

---

## 📂 Repository Structure

```text
.
├── .gitignore
├── README.md               # Project Documentation
├── requirements.txt        # Project Dependencies
├── lib/                    # Core Neural Network Library
│   ├── __init__.py
│   ├── layers.py           # Dense Layer implementation
│   ├── activations.py      # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py           # MSE Loss
│   ├── optimizers.py       # SGD Optimizer
│   └── network.py          # Network container for training
├── notebooks/
│   └── project_demo.ipynb  # Primary Demo (Gradient Check & XOR)
└── report/
    └── ms1_report.pdf      # Milestone 1 Report
    └── project_report.pdf  # Final Technical Report
