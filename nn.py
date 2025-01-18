import random
from engine import Value

class Neuron:
    """
    Represents a single neuron in a neural network.
    """

    def __init__(self, nin, nonlin=True):
        """
        Initializes a neuron.

        Args:
            nin (int): Number of inputs to the neuron.
            nonlin (bool): Whether to apply a non-linear activation function (default: True).
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # Weights (randomly initialized)
        self.b = Value(0)  # Bias (initialized to 0)
        self.nonlin = nonlin  # Whether to apply a non-linear activation

    def __call__(self, x):
        """
        Performs the forward pass of the neuron.

        Args:
            x (list of Value): Input values to the neuron.

        Returns:
            Value: The output of the neuron after applying the activation function.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # Weighted sum of inputs + bias
        return act.tanh() if self.nonlin else act  # Apply tanh activation if nonlin=True

    def parameters(self):
        """
        Returns a list of all trainable parameters in the neuron (weights and bias).
        """
        return self.w + [self.b]

    def __repr__(self):
        """
        Returns a string representation of the neuron.
        """
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer:
    """
    Represents a layer of neurons in a neural network.
    """

    def __init__(self, nin, nout, **kwargs):
        """
        Initializes a layer.

        Args:
            nin (int): Number of inputs to the layer.
            nout (int): Number of neurons in the layer.
            **kwargs: Additional arguments passed to the Neuron constructor.
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]  # List of neurons in the layer

    def __call__(self, x):
        """
        Performs the forward pass of the layer.

        Args:
            x (list of Value): Input values to the layer.

        Returns:
            Value or list of Value: Outputs of the neurons in the layer.
        """
        out = [n(x) for n in self.neurons]  # Compute outputs of all neurons
        return out[0] if len(out) == 1 else out  # Return a single value if there's only one neuron

    def parameters(self):
        """
        Returns a list of all trainable parameters in the layer (weights and biases of all neurons).
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """
        Returns a string representation of the layer.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
    """
    Represents a Multi-Layer Perceptron (MLP), a feedforward neural network with multiple layers.
    """

    def __init__(self, nin, nouts):
        """
        Initializes an MLP.

        Args:
            nin (int): Number of inputs to the MLP.
            nouts (list of int): Number of neurons in each layer.
        """
        sz = [nin] + nouts  # List of layer sizes (input + hidden + output layers)
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))
        ]  # Create layers

    def __call__(self, x):
        """
        Performs the forward pass of the MLP.

        Args:
            x (list of Value): Input values to the MLP.

        Returns:
            Value or list of Value: Output of the MLP.
        """
        for layer in self.layers:
            x = layer(x)  # Pass input through each layer
        return x

    def parameters(self):
        """
        Returns a list of all trainable parameters in the MLP (weights and biases of all layers).
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """
        Returns a string representation of the MLP.
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"