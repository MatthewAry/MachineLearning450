import sys
import numpy as np
from sklearn import datasets
import csv
# My Tree Classifier
from dTreeClassifer import TreeClassifier
# For comparison
from sklearn import tree
from csvProcessor import process_csv

def calculateAccuracy(predictions, test, targets):
    correct = 0
    for (prediction, actual) in zip(predictions, test):

        if prediction == targets[actual]:
            correct += 1
    # Display accuracy
    print(correct / test.size * 100)

def main(argv):
    # Load data
    if len(argv) >= 2:
        csv = np.genfromtxt(argv[1], delimiter=",", dtype=str)
        # We might have to delimit by spaces!
        if len(csv.shape) < 2:
            csv = np.genfromtxt(argv[1], dtype=str)

        data = csv[:, :-1]
        targets = csv[:, -1]
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

    classifier = TreeClassifier()

    if argv[-1] is not 'True':
        # If we are reading in data we will need to clean it up
        data = process_csv(data)

    # Create the tree
    classifier.train(data[train], targets[train])
    # Make predictions
    predictions = classifier.predict(data[test])

    sci_d_tree = tree.DecisionTreeClassifier()
    sci_d_tree = sci_d_tree.fit(data[train], targets[train])
    sci_predictions = sci_d_tree.predict(data[test])

    calculateAccuracy(predictions, test, targets)
    calculateAccuracy(sci_predictions, test, targets)


# To make sure main is ran only when ran, not loaded as well
if __name__ == "__main__":
    main(sys.argv)