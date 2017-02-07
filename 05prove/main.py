import numpy as np
import sys
from sklearn import datasets
from normalize import normalize
from NeuralNetwork import NeuralNetwork


def main(argv):
    # Load data
    if len(argv) >= 2:
        csv = np.genfromtxt(argv[1], delimiter=",", dtype=str)
        # We might have to delimit by spaces!
        # The first column would be a row count then.
        if len(csv.shape) < 2:
            csv = np.delete(np.genfromtxt(argv[1], dtype=str), 0, 1)

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

    network = NeuralNetwork(len(data), len(data[0]))
    network.train(data[train], targets[train])


if __name__ == "__main__":
    main(sys.argv)