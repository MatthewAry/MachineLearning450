import numpy as np
import sys
from sklearn import datasets
from normalize import normalize
from sklearn import preprocessing
from NeuralNetwork import NeuralNetwork as Network


def main(argv):
    inputs = {'-file': None, '-n': None}
    for i, input in enumerate(argv):
        if input in inputs:
            if (i + 1) < len(argv):
                if input == '-file':
                    inputs[input] = argv[i + 1]
                elif input == '-n':
                    inputs[input] = [int(x) for x in argv[i + 1].split(',')]
    if inputs['-n'] is None:
        print("You need to specify your network dimensions")
        return

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

    data = preprocessing.scale(np.array(data[perm]))
    targets = targets[perm]

    index = int(round(perm.size*.3))
    test = perm[:index]
    train = perm[index:]

    result = []

    network = Network(len(data[0]), inputs['-n'], len(set(targets)))
    network.train(data[train], targets[train])

    correct_guesses = 0

    test_targets = targets[test]
    for i, val in enumerate(data[test]):
        r = network.predict(val)
        # print(r, targets[i])
        if test_targets[i] == r:
            correct_guesses += 1
        result.append(r)
    print(result)
    print(list(test_targets))
    print(correct_guesses / len(targets) * 100)

    for j, layer in enumerate(network.layers):
        for i, node in enumerate(layer):
            print(j, i, node.weights)



if __name__ == "__main__":
    main(sys.argv)