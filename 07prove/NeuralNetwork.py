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
        self.epochs = 1
        self.correct = 0
        self.total = 0
        self.learning_rate = .3
        self.target_list = []

    def predict(self, instance):
        result = self.feed_forward(instance)
        print("result: ", result)
        print("argmax: ", np.argmax(result))
        return self.target_list[np.argmax(result)]

    def feed_forward(self, instance):
        outputs = []
        instance += [-1]
        for layer_index, layer in enumerate(self.layers):
            layer_output = []
            # If at input layer
            if layer_index == 0:
                for neuron_index, neuron in enumerate(layer):
                    # print('For neuron ', layer_index, ',', neuron_index)
                    # print('Activating on instance: ', instance)
                    layer_output.append(neuron.activation(instance))
            else:
                for neuron_index, neuron in enumerate(layer):
                    # print('For neuron ', layer_index, ',', neuron_index)
                    # print('Activating on instance: ', instance)
                    layer_output.append(neuron.activation(outputs))
            outputs = layer_output
        # print(outputs)
        return outputs

    def train(self, instances, targets):
        # We need to know what the allowed targets are for our predictions
        self.target_list = list(set(targets))
        for epoch in range(self.epochs):
            # print('epoch: ', epoch)
            for index, instance in enumerate(instances):
                # print('index: ', index)
                # print('instance: ', instance)
                self.back_propagate(instance, targets[index])

    def back_propagate(self, instance, target):
        if target == np.argmax(self.feed_forward(instance)):
            print("CORRECT")
            self.correct += 1
        else:
            print("WRONG")
        self.total += 1
        # Compute the error in the network
        for layer_index, layer in reversed(list(enumerate(self.layers))):
            # print('layer_index: ', layer_index)
            # print('layer', layer)
            if layer_index == len(self.layers) - 1:
                # Output layer
                for neuron_index, neuron in enumerate(layer):
                    neuron_target = 1 if neuron_index == target else 0
                    # print('neuron: ', neuron)
                    # print('neuron weights: ', neuron.weights)
                    # print('neuron output: ', neuron.output)
                    # print('neuron target: ', neuron_target)
                    # Compute the error
                    neuron.error = neuron.output * (1 - neuron.output) * (neuron.output - neuron_target)
                    # print('neuron ', layer_index, ',', neuron_index, ' error: ', neuron.error)
            else:
                # Inner layers
                for neuron_index, neuron in enumerate(layer):
                    sum = 0
                    for neuron_k in self.layers[layer_index + 1]:
                        # print('neuron weights: ', neuron.weights)
                        # print('neuron output: ', neuron.output)
                        sum += neuron_k.error * neuron_k.weights[neuron_index]
                    neuron.error = neuron.output * (1 - neuron.output) * sum
                    # print('neuron ', layer_index, ',', neuron_index, ' error: ', neuron.error)

        # Update the network weights
        for layer_index, layer in enumerate(self.layers):
            outputs = []
            # print('layer_index: ', layer_index)
            # If we are at the input nodes then our instance values are our output values
            if layer_index == 0:
                outputs = instance
                # print('layer ', layer_index, ' outputs: ', outputs)
            # Otherwise, get the values from the i th layer.
            else:
                for neuron_index, neuron in enumerate(self.layers[layer_index - 1]):
                    # print('Neuron ', layer_index - 1, ',', neuron_index, 'output: ', neuron.output)
                    outputs = np.append(outputs, np.array([neuron.output]))
                    # print('Layer ', layer_index - 1, ' outputs: ', outputs)

            # Add the bias value to outputs
            outputs = np.append(outputs, np.array([-1]))
            # print('layer ', layer_index, ' outputs: ', outputs)

            for neuron_index, neuron in enumerate(layer):
                # print('neuron ', layer_index, ',', neuron_index, ' output: ', neuron.output)
                # print('neuron ', layer_index, ',', neuron_index, ' error: ', neuron.error)
                # print('neuron ', layer_index, ',', neuron_index, ' weights: ', neuron.weights)
                for weight_index, weight in enumerate(neuron.weights):
                    # print('weight_index: ', weight_index)
                    # print('weight before update: ', weight)
                    # print('weight before update: ', neuron.weights[weight_index])
                    # print('.1 * neuron.error * outputs[weight_index]')
                    # print(.1, ' * ', neuron.error, ' * ', outputs[weight_index])
                    # print(.1 * neuron.error * outputs[weight_index])
                    # print("neuron error", neuron.error)
                    neuron.weights[weight_index] -= self.learning_rate * neuron.error * outputs[weight_index]
                    # print('weight after update: ', neuron.weights[weight_index])
        # print("Correct: ", self.correct)
        # print("Total: ", self.total)
