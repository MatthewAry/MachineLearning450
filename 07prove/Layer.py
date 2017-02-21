import numpy as np
from Neuron import Neuron


class Layer:
    """Creates and handles operations in a neural network."""
    # Initialization
    # Set all weights to small (positive and negative) random numbers.
    def __init__(self, num, weights):
        self.nodes = [Neuron(weights) for _ in range(num)]

    def __iter__(self):
        return iter(self.nodes)

