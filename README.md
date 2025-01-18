# micrograd- A Tiny Autograd Engine
This project implements a tiny autograd engine (`micrograd`) and a simple neural network library built on top of it. The engine supports automatic differentiation, and the neural network library can be used to train models on small datasets.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [License](#license)
---

## Overview

`micrograd` is a minimalistic autograd engine that allows you to compute gradients of scalar-valued functions. It is inspired by PyTorch's autograd system but is much simpler and designed for educational purposes.

The project also includes a neural network library (`nn.py`) that uses the `micrograd` engine to define and train neural networks. The library supports fully connected layers, activation functions (e.g., `tanh`), and basic optimization techniques like stochastic gradient descent (SGD).

---
## Features

- **Autograd Engine**:
  - Supports basic operations like addition, multiplication, and activation functions.
  - Computes gradients using reverse-mode automatic differentiation (backpropagation).

- **Neural Network Library**:
  - Defines neurons, layers, and multi-layer perceptrons (MLPs).
  - Supports customizable architectures (e.g., number of layers, neurons per layer).
  - Includes a simple training loop with SGD.

- **Visualization**:
  - Visualizes the decision boundary of trained models on 2D datasets.
---
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ahtesham15/micrograd-.git
   cd micrograd
---
## Acknowledgments

- **Andrej Karpathy**: This project is heavily inspired by Andrej Karpathy's work. His work and educational content have been a great source of inspiration.
- **PyTorch**: The design of this project is influenced by the [PyTorch](https://pytorch.org/) library, particularly its autograd system and neural network module.
