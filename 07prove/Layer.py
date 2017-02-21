import numpy as np
from Neuron import Neuron


class Layer:
    """Creates and handles operations in a neural network."""
    # Initialization
    # Set all weights to small (positive and negative) random numbers.
    def __init__(self, num, weights, bias=1):
        self.nodes = [Neuron((np.random.ranf(-1, 1, weights + 1).tolist()), bias) for _ in range(num)]

    def __iter__(self):
        return iter(self.nodes)

