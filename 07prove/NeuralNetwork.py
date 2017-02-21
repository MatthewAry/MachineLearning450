import Layer
import numpy as np


class NeuralNetwork:
    def __init__(self, input_weights, dimensions, num_targets):
        # For each value, create a layer of that value's size
        self.layers = []
        for i in dimensions:
            if len(self.layers) > 0:
                self.layers.append(Layer.Layer(i, len(self.layers[-1].nodes)))
            else:
                self.layers.append(Layer.Layer(i, input_weights))
        # The last layer needs to equal the number of targets.
        self.layers.append(Layer.Layer(num_targets, len(self.layers[-1].nodes)))
        self.epochs = 8
        self.learning_rate = .3
        self.target_list = []

    def predict(self, instance):
        #print(self.feed_forward(instance))
        return self.target_list[np.argmax(self.feed_forward(instance))]

    def feed_forward(self, instance):
        outputs = []
        instance += [-1]
        for layer_index, layer in enumerate(self.layers):
            layer_output = []
            # If at input layer
            if layer_index == 0:
                for neuron in layer:
                    layer_output.append(neuron.activation(instance))
            else:
                for neuron in layer:
                    layer_output.append(neuron.activation(outputs))
            outputs = layer_output
        # print(outputs)
        return outputs

    def train(self, instances, targets):
        # We need to know what the allowed targets are for our predictions
        self.target_list = list(set(targets))
        for epoch in range(self.epochs):
            for index, instance in enumerate(instances):
                self.back_propagate(instance, targets[index])

    def back_propagate(self, instance, target):
        self.feed_forward(instance)
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
                    outputs = np.append(outputs, np.array([neuron.output]))

            # Add the bias value to outputs
            outputs = np.append(outputs, np.array([-1]))

            for neuron_index, neuron in enumerate(layer):
                # print("pre", neuron.weights)
                for weight_index, weight in enumerate(neuron.weights):
                    neuron.weights[weight_index] -= self.learning_rate * neuron.error * outputs[weight_index]
                # print("post", neuron.weights)