# micrograd- A Tiny Autograd Engine
This module builds a tiny **autograd engine** (short for **automatic gradient**), which implements the **backpropagation algorithm**. This algorithm was prominently popularized for training neural networks in the seminal 1986 paper, *"Learning Internal Representations by Error Propagation"* by Rumelhart, Hinton, and Williams.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
---

## Overview
The code we develop here is the **core of neural network training**—it enables us to calculate how to update the parameters of a neural network to improve its performance on a given task, such as **next-token prediction** in autoregressive language models.

The same algorithm is used in all modern deep learning libraries, such as **PyTorch**, **TensorFlow**, and **JAX**, except that those libraries are far more optimized and feature-rich.
The project also includes a neural network library (`nn.py`) that uses the `micrograd` engine to define and train neural networks. The library supports fully connected layers, activation functions (e.g., `tanh`), and basic optimization techniques like stochastic gradient descent (SGD).

---
## Features

1. **Autograd Engine (micrograd)**:
   - A minimalistic implementation of automatic differentiation, which computes gradients for scalar-valued functions.
   - The backbone of neural network training, enabling efficient parameter updates via backpropagation.

2. **Neural Network (NN) with 1 Hidden Layer (MLP)**:
   - A simple **Multi-Layer Perceptron (MLP)** built on top of the autograd engine.
   - Demonstrates how to define and train a neural network for a classification task.

3. **Visual and Intuitive**:
   - The 2D nature of the training data allows for easy visualization of decision boundaries and model behavior.
   - Perfect for building intuition about how neural networks learn and generalize.

this repository contains 2 notebooks (poart1.ipynt and part2.ipynb), which i have created during my learning process as notes, which explain the building process quit briefly.
  
---
## Acknowledgments

- **Andrej Karpathy**: This project is heavily inspired by Andrej Karpathy's work. His work and educational content have been a great source of inspiration.
- **PyTorch**: The design of this project is influenced by the [PyTorch](https://pytorch.org/) library, particularly its autograd system and neural network module.
