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
        print(instance)
        activations = np.dot(self.weights, instance)
        return np.where(activations > self.threshold)
