import numpy as np
from Neuron import Neuron


class NeuralNetwork:
    """Creates and handles operations in a neural network."""
    # Initialization
    # Set all weights to small (positive and negative) random numbers.
    def __init__(self, num, weights, bias=1):
        self.nodes = [Neuron((np.random.uniform(-1, 1, weights + 1).tolist()), bias) for i in range(num)]

    def train(self, train_data, target_data):
        """Training Function Slug"""
        result = []
        for instance in train_data:
            result = [[node.activation(instance)] for node in self.nodes]

        return result
