import Layer
import numpy as np


class NeuralNetwork:
    def __init__(self, edges, dimensions, num_targets):
        # For each value, create a layer of that value's size
        self.layers = []
        for i in dimensions:
            # If we have layers...
            if len(self.layers) > 0:
                self.layers.append(Layer.Layer(i, len(self.layers[-1].nodes)))
            # If we are just starting, we need to have the right number of edges
            else:
                self.layers.append(Layer.Layer(i, edges))

        # The last layer needs to equal the number of targets.
        self.layers.append(Layer.Layer(num_targets, len(self.layers[-1].nodes)))
        self.epochs = 1
        self.learning_rate = .3
        self.target_list = []

    def predict(self, instance):
        result = self.feed_forward(instance.tolist())
        target = self.target_list[np.argmax(result)]
        return target

    def feed_forward(self, instance):
        outputs = []
        for layer_index, layer in enumerate(self.layers):
            layer_output = []
            # If at input layer
            if layer_index == 0:
                for neuron_index, neuron in enumerate(layer):
                    layer_output.append(neuron.activation(instance))
            else:
                for neuron_index, neuron in enumerate(layer):
                    layer_output.append(neuron.activation(outputs))
            outputs = layer_output
        return outputs

    def train(self, instances, targets):
        # We need to know what the allowed targets are for our predictions
        self.target_list = list(set(targets))
        for epoch in range(self.epochs):
            for index, instance in enumerate(instances):
                self.back_propagate(instance.tolist(), targets[index])

    def back_propagate(self, instance, target):
        np.argmax(self.feed_forward(instance))

        # Compute the error in the network
        for layer_index, layer in reversed(list(enumerate(self.layers))):
            if layer_index == len(self.layers) - 1:
                # Output layer
                for neuron_index, neuron in enumerate(layer):
                    neuron_target = 1 if neuron_index == target else 0
                    # Compute the error
                    neuron.error = neuron.output * (1 - neuron.output) * (neuron.output - neuron_target)
            else:
                # Inner layers
                for neuron_index, neuron in enumerate(layer):
                    sum = 0
                    for neuron_k in self.layers[layer_index + 1]:
                        sum += neuron_k.error * neuron_k.weights[neuron_index]
                    neuron.error = neuron.output * (1 - neuron.output) * sum

        # Update the network weights
        for layer_index, layer in enumerate(self.layers):
            outputs = []
            # If we are at the input nodes then our instance values are our output values
            if layer_index == 0:
                outputs = instance

            # Otherwise, get the values from the i th layer.
            else:
                for neuron_index, neuron in enumerate(self.layers[layer_index - 1]):
                    outputs = np.append(outputs, np.array([neuron.output])).tolist()

            outputs.append(-1)
            for neuron_index, neuron in enumerate(layer):
                for weight_index, weight in enumerate(neuron.weights):
                    neuron.weights[weight_index] -= self.learning_rate * neuron.error * outputs[weight_index]
