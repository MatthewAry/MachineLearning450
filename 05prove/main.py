import numpy as np
import sys
from sklearn import datasets
from normalize import normalize
from NeuralNetwork import NeuralNetwork


def main(argv):
    inputs = {'-file': None, '-n': None}
    for i, input in enumerate(argv):
        if input in inputs:
            if (i + 1) < len(argv):
                if input == '-file':
                    inputs[input] = argv[i + 1]
                elif input == '-n':
                    inputs[input] = int(argv[i + 1])
    if inputs['-n'] is None:
        inputs['-n'] = 5

    # Load data
    if inputs['-file'] is not None:
        csv = np.genfromtxt(inputs['-file'], delimiter=",", dtype=str)
        # We might have to delimit by spaces!
        # The first column would be a row count then.
        if len(csv.shape) < 2:
            csv = np.delete(np.genfromtxt(inputs['-file'], dtype=str), 0, 1)

        response = ""
        while response.lower() != "yes" and response.lower() != "no":
            response = input("Is the key on the left? (yes/no): ")

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

    # Randomize the data
    perm = np.random.permutation(len(data))

    data = data[perm]
    targets = targets[perm]

    index = int(round(perm.size*.3))
    test = perm[:index]
    train = perm[index:]

    network = NeuralNetwork(inputs['-n'], len(data[0]))
    print(network.train(data[train], targets[train]))


if __name__ == "__main__":
    main(sys.argv)