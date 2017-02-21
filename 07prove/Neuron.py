import numpy as np
import math


class Neuron:
    """Simple Neuron"""
    def __init__(self, weights, bias=1):
        self.bias_input = bias
        self.weights = weights
        self.threshold = 0
        self.error = 0
        self.output = 0
        return

    def activation(self, instance):
        instance = list(instance)
        instance.append(self.bias_input)
        activations = np.dot(self.weights, instance)
        self.output = 1.0 / (1.0 + math.exp(-activations))
        return self.output
