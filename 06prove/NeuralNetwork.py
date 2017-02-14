import Layer


class NeuralNetwork:
    def __init__(self, input_weights, dimensions):
        # For each value, create a layer of that value's size
        self.layers = []
        for i in dimensions:
            if len(self.layers) > 0:
                self.layers.append(Layer.Layer(i, len(self.layers[-1].nodes)))
            else:
                self.layers.append(Layer.Layer(i, input_weights))

    def predict(self, row):
        x = row
        for layer in self.layers:
            outputs = []
            for neuron in layer.nodes:
                outputs.append(neuron.activation(x))
            x = outputs
        print(outputs)
        return outputs.index(max(outputs))
