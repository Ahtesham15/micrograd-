import random
import numpy as np
import matplotlib.pyplot as plt
from micrograd.engine import Value
from micrograd.nn import MLP

# Set random seeds for reproducibility
np.random.seed(1337)
random.seed(1337)

# Generate a toy dataset using sklearn's make_moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.1)

# Transform y to be -1 or 1 (for binary classification)
y = y * 2 - 1

# Visualize the dataset
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
plt.title("Toy Dataset (make_moons)")
plt.show()

# Initialize a 2-layer neural network with 16 hidden neurons in each layer
model = MLP(2, [16, 16, 1])  # Input size: 2, Hidden layers: [16, 16], Output size: 1
print(model)  # Print the model architecture
print("Number of parameters:", len(model.parameters()))  # Print the total number of parameters

# Define the loss function
def loss(batch_size=None):
    # If batch_size is None, use the entire dataset; otherwise, sample a mini-batch
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]  # Randomly sample indices
        Xb, yb = X[ri], y[ri]  # Select the corresponding data points

    # Convert input data to Value objects for automatic differentiation
    inputs = [list(map(Value, xrow.tolist())) for xrow in Xb]

    # Forward pass: compute model predictions (scores)
    scores = list(map(model, inputs))

    # Compute SVM "max-margin" loss
    losses = [(1 + -yi * scorei).tanh() for yi, scorei in zip(yb, scores)]  # Use tanh instead of relu
    data_loss = sum(losses) * (1.0 / len(losses))  # Average loss over the batch

    # Add L2 regularization to the loss
    alpha = 1e-4  # Regularization strength
    reg_loss = alpha * sum((p * p for p in model.parameters()))  # Sum of squared parameters
    total_loss = data_loss + reg_loss  # Total loss = data loss + regularization loss

    # Compute accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)  # Return loss and accuracy

# Initial loss and accuracy
total_loss, acc = loss()
print("Initial loss:", total_loss.data, "Accuracy:", acc * 100, "%")

# Training loop
for k in range(100):
    # Forward pass: compute loss and accuracy
    total_loss, acc = loss()

    # Backward pass: compute gradients
    model.zero_grad()  # Reset gradients to zero
    total_loss.backward()  # Perform backpropagation

    # Update model parameters using stochastic gradient descent (SGD)
    learning_rate = 1.0 * (0.9 ** k)  # Exponential decay
    for p in model.parameters():
        p.data -= learning_rate * p.grad  # Update parameter values

    # Print progress
    if k % 1 == 0:
        print(f"Step {k}: Loss {total_loss.data}, Accuracy {acc * 100}%")

# Visualize the decision boundary
h = 0.25  # Step size for the grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # X-axis range
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Y-axis range
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Create grid
Xmesh = np.c_[xx.ravel(), yy.ravel()]  # Flatten grid and create input points

# Compute model predictions for each point in the grid
inputs = [list(map(Value, xrow.tolist())) for xrow in Xmesh]  # Convert NumPy array to list of Value objects
scores = list(map(model, inputs))
Z = np.array([s.data for s in scores])  # Use raw scores for smoother decision boundary
Z = Z.reshape(xx.shape)  # Reshape predictions to match grid shape

# Plot the decision boundary
fig = plt.figure()
plt.contourf(xx, yy, Z, levels=50, cmap=plt.cm.Spectral, alpha=0.8)  # Fill decision regions
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)  # Plot data points
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision Boundary")
plt.show()