import numpy as np
import sys
from sklearn import datasets
from normalize import normalize
from sklearn import preprocessing
from NeuralNetwork import NeuralNetwork as Network
from graphviz import Digraph
import matplotlib.pyplot as plt
import argparse


def main(argv):
    description = "Create a neural network to classify data."
    parser = argparse.ArgumentParser(description=description)
    file_description = "Path to the data file that will be used to train and test the network."
    parser.add_argument('-f', '--file', type=str, help=file_description, required=False)
    network_description = "The configuration of the network you want. " \
                          "Specify the number of nodes in each layer by " \
                          "providing comma separated " \
                          "values. (no spaces) e.g.: 2,3,2"
    parser.add_argument('-n', '--network', type=str, help=network_description, required=True)
    draw_description = "Generate a diagram of the neural network using GraphViz."
    parser.add_argument('-d', '--draw', dest='draw_network', action='store_true', help=draw_description)
    error_description = "Generate a chart showing the accuracy rate over iterations."
    parser.add_argument('-e', '--error', dest='draw_error', action='store_true', help=error_description)
    epoch_description = "Set the number of epochs performed during training."
    parser.add_argument('-ep', '--epochs', help=epoch_description, type=int, required=True)

    args = parser.parse_args()

    if args.file is not None:
        csv = np.genfromtxt(args.file, delimiter=",", dtype=str)

        response = ""
        while response.lower() != "yes" and response.lower() != "no":
            response = input("Is the key on the right? (yes/no): ")
        csv = normalize(csv)

        if response == "yes":
            data = csv[:, :-1]
            targets = csv[:, -1]
        else:
            data = csv[:, 1:]
            targets = csv[:, 0]

    else:
        iris = datasets.load_iris()
        data = iris.data
        targets = iris.target

    network_configuration = [int(x) for x in args.network.split(',')]

    # Randomize the data
    perm = np.random.permutation(len(data))

    data = preprocessing.scale(np.array(data[perm]))
    targets = targets[perm]

    index = int(round(perm.size*.3))
    test = perm[:index]
    train = perm[index:]

    network = Network(len(data[0]), network_configuration, len(set(targets)))

    if not args.draw_error:
        network.epochs = args.epochs
        network.train(data[train], targets[train])
        print("Accuracy of network: ", accuracy(data[test], targets[test], network), "%")
    else:
        result = []
        for i in range(args.epochs):
            network.train(data[train], targets[train])
            result.append(accuracy(data[test], targets[test], network))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.plot(result)
        plt.show()

    if args.draw_network:
        visualize(network, data)


def accuracy(data, targets, network):
    correct_guesses = 0
    test_targets = targets
    for i, val in enumerate(data):
        r = network.predict(val)
        # print(r, targets[i])
        if test_targets[i] == r:
            correct_guesses += 1
    return correct_guesses / len(targets) * 100



def visualize(network, data):
    # Visualize the neural network
    n_graph = Digraph('Network', 'network.gv')
    n_graph.body.extend(['rankdir=LR'])
    n_graph.attr('node', shape='circle')

    for i, layer in enumerate(network.layers):
        for j, node in enumerate(layer):
            n_graph.node(str(i) + '_' + str(j))

    for i, layer in enumerate(network.layers):
        if i > 0:
            n_graph.attr('node', shape='circle')
            if i < len(network.layers):
                for k, k_node in enumerate(layer):
                    # Create Bias Node
                    n_graph.edge('bias_node_' + str(i), str(i) + '_' + str(k), label='-1')
                    for j, node in enumerate(network.layers[i - 1]):
                        # Node_j to Node_K
                        n_graph.edge(str(i - 1) + '_' + str(j), str(i) + '_' + str(k), label=str(k_node.weights[j]))

        else:
            # Create Bias Node
            n_graph.attr('node', shape='doublecircle')
            # Input nodes

            for k, k_node in enumerate(network.layers[0]):
                n_graph.edge('bias_node_0', str(i) + '_' + str(k), label='-1')
                for j in range(len(data[0])):
                    # Create input source nodes
                    n_graph.edge('attribute_' + str(j), str(i) + '_' + str(k), label=str(k_node.weights[j]))
    n_graph.view()

if __name__ == "__main__":
    main(sys.argv)