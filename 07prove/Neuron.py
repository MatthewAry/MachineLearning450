import numpy as np
import math


class Neuron:
    """Simple Neuron"""
    def __init__(self, weights):
        self.weights = np.random.ranf(weights) - .5
        self.weights = self.weights.tolist()
        self.weights.append(-1)
        self.threshold = 0
        self.error = 0
        self.output = 0
        return

    def activation(self, instance):
        instance = list(instance)
        instance.append(-1)
        activations = np.dot(self.weights, instance)
        self.output = 1.0 / (1.0 + math.exp(-activations))
        return self.output
