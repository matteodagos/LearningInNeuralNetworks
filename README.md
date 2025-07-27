# Learning in Neural Networks

## 🔍 Overview

Welcome to the **Learning in Neural Networks** repository! This collection encompasses projects, code, analyses, and presentations developed during the EPFL Master course "Learning in Neural Networks". Here, we explore some of the main concepts in biologically plausible learning methods and neural network architectures beyond traditional Backpropagation.

---

## 🧠 Key Concepts Explored

* **Hebbian Learning**: Explore biologically plausible two-factor rules for PCA, ICA, and dictionary learning.
* **Neuromodulation and Reinforcement Learning**: Deep dive into three-factor learning rules, TD reinforcement learning, actor-critic networks, and reward-based systems.
* **Surprise and Novelty**: Understand how biological systems learn by surprise, facilitating exploration and adaptation in changing environments.
* **Multi-layer Representation Learning**: Investigate algorithms that achieve multi-layer learning without traditional backpropagation.
* **Neuromorphic Computing**: Examine bio-inspired hardware and in-memory computing for more energy-efficient AI.

---

## 📁 Contents

### 1. Visual Receptive Field Development (Mini Project)

* **Objective**: Study how Hebbian learning explains receptive field (RF) formation in visual signals, specifically within the primary visual cortex V1.
* **Methodology**:

  * Single and multi-neuron RF development with various Hebbian non-linearities (e.g., BCM rule).
  * Competitive and non-competitive learning environments.
  * Parametric extraction and analysis of receptive fields.
* **Tools**: Python, NumPy, Matplotlib, PIL (Pillow), custom network architecture (provided).

### 2. Paper Presentation: Duality in Self-Supervised Learning

* **Title**: "On the Duality between Contrastive and Non-Contrastive Self-Supervised Learning" (ICLR 2023)
* **Authors**: Quentin Garrido, Yubei Chen, Adrien Bardes, Laurent Najman, Yann LeCun
* **Insights**:

  * Demonstrates the theoretical equivalence between contrastive (SimCLR, DCL) and non-contrastive methods (VICReg, Barlow Twins).
  * Shows practical strategies for bridging performance gaps through careful hyperparameter tuning.
  * Highlights misconceptions around embedding dimensions and batch sizes in SSL.

---

## 🚀 Getting Started

Clone this repository and explore the detailed Jupyter Notebooks (`.ipynb` files) and associated Python scripts provided. Each notebook is self-contained, showcasing experiments, results, visualizations, and analysis clearly documented for easy reproducibility.

```bash
git clone https://github.com/yourusername/LearningInNeuralNetworks.git
cd LearningInNeuralNetworks
```

### Installation

Set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate your_environment_name
```

Replace `your_environment_name` with the environment name defined within the `environment.yml` file.

---

## 🌟 Highlights

* **Biological Plausibility**: Focuses on neural algorithms feasible in biological neural systems.
* **Interdisciplinary Approach**: Bridges neuroscience, machine learning, and computational biology.
* **Cutting-Edge Research**: Engages with contemporary research like the duality in SSL and emerging neuromorphic technologies.

---

## 📖 Resources & References

* Course Website: [EPFL Learning in Neural Networks](https://edu.epfl.ch/coursebook/en/learning-in-neural-networks-CS-479)
* Main References:

  * Garrido et al., "On the Duality between Contrastive and Non-Contrastive Self-Supervised Learning," ICLR 2023.
  * Brito & Gerstner, "Nonlinear Hebbian Learning as a Unifying Principle in Receptive Field Formation," PLOS Computational Biology.
