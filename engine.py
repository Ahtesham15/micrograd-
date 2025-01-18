import math

class Value:
    """
    Represents a scalar value and its gradient, used for automatic differentiation.
    """

    def __init__(self, data, _children=(), _op=''):
        """
        Initializes a Value object.

        Args:
            data (float): The scalar value.
            _children (tuple): Child nodes that produced this value (used for graph construction).
            _op (str): The operation that produced this value (e.g., '+', '*', 'tanh').
        """
        self.data = data  # The scalar value
        self.grad = 0  # Gradient of the loss with respect to this value (initialized to 0)
        self._backward = lambda: None  # Function to compute gradients during backpropagation
        self._prev = set(_children)  # Child nodes that produced this value
        self._op = _op  # Operation that produced this value (for debugging/graph visualization)

    def tanh(self):
        """
        Applies the hyperbolic tangent (tanh) activation function.

        Returns:
            Value: A new Value object representing the tanh output.
        """
        x = self.data  # Use the data attribute (a real number)
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)  # Compute tanh
        out = Value(t, (self,), 'tanh')  # Create a new Value for the tanh output

        def _backward():
            """
            Computes gradients for the tanh operation during backpropagation.
            """
            self.grad += (1 - t**2) * out.grad  # Gradient flows to self
        out._backward = _backward  # Attach the backward function to the output

        return out

    def __add__(self, other):
        """
        Adds two Value objects or a Value object and a scalar.

        Args:
            other (Value or float): The other value to add.

        Returns:
            Value: A new Value object representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)  # Convert scalar to Value if necessary
        out = Value(self.data + other.data, (self, other), '+')  # Create a new Value for the sum

        def _backward():
            """
            Computes gradients for the addition operation during backpropagation.
            """
            self.grad += out.grad  # Gradient flows to self
            other.grad += out.grad  # Gradient flows to other
        out._backward = _backward  # Attach the backward function to the output

        return out

    def __mul__(self, other):
        """
        Multiplies two Value objects or a Value object and a scalar.

        Args:
            other (Value or float): The other value to multiply.

        Returns:
            Value: A new Value object representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)  # Convert scalar to Value if necessary
        out = Value(self.data * other.data, (self, other), '*')  # Create a new Value for the product

        def _backward():
            """
            Computes gradients for the multiplication operation during backpropagation.
            """
            self.grad += other.data * out.grad  # Gradient flows to self
            other.grad += self.data * out.grad  # Gradient flows to other
        out._backward = _backward  # Attach the backward function to the output

        return out

    def __repr__(self):
        """
        Returns a string representation of the Value object.

        Returns:
            str: A string showing the value and its gradient.
        """
        return f"Value(data={self.data}, grad={self.grad})"