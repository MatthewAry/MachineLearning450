import numpy as np


class Neuron:
    """Simple Neuron"""
    def __init__(self, weights, bias=1):
        self.bias_input = bias
        self.weights = weights
        self.threshold = 0
        return

    def activation(self, instance):
        instance = instance.tolist()
        instance.append(self.bias_input)
        activations = np.dot(self.weights, instance)
        return True if (activations > self.threshold) else False
