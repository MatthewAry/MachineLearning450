from sklearn import datasets
from sklearn import preprocessing as pp
from sklearn.neighbors import KNeighborsClassifier
import numpy
from kNNClassifier import KNearestNeighbor
import csv
import sys

def compute_accuracy(predictions, key):
    matches = 0
    predictions = list(predictions)
    key = list(key)

    for i in range(0, len(predictions)):
        if predictions[i] == key[i]:
            matches += 1
    return (matches / len(predictions)) * 100


def main(argv):
    # Check for arguments
    if len(argv) >= 2:
        file = argv[1]
        with open(file, 'r') as myfile:
           f = myfile.read()
        # That pesky last line...
        file_data = list(csv.reader(f.split('\n'), delimiter=','))[:-1]
        data = []
        targets = []
        for sub_list in file_data:
            # The last column is our key!
            # Convert strings to floats except for the key.
            data.append(sub_list[:-1])
            # Get the target data but keep the string
            targets.append(sub_list[-1])


        data = numpy.transpose(data)
        for i in range(0, len(data)):
            le = pp.LabelEncoder()
            le.fit(data[i])
            data[i] = le.transform(data[i])
        data = numpy.transpose(data)
        data = numpy.array(data)
        targets = numpy.array(targets)
    else:
        print("Using Iris. This script can take an argument. Provide a the name of a csv file relative to this script "
              "and it can also be used for kNN.")
        # Use Iris
        iris = datasets.load_iris()
        data = iris.data
        targets = iris.target

    # Get a permutation seed for the data
    permutation = numpy.random.permutation(len(data))

    # Randomize the data using the seed
    data = data[permutation]
    targets = targets[permutation]

    # Split the data into their training and test sets
    while True:
        size = input('How much of the data should be used for training? (1 - 100): ')
        if size.isdigit():
            size = int(size) * .01
            break
        else:
            print("You either did not enter a number or your response was invalid.")
            continue

    split_point = int(round(permutation.size * size))

    test = permutation[split_point:]
    train = permutation[:split_point]

    # Create classifier
    my_kNNClassifier = KNearestNeighbor(3)

    # Train it by giving it data
    my_kNNClassifier.train(data[train], targets[train])

    # Make predictions
    predictions = my_kNNClassifier.predict(data[test])

    # use library version to compare to your solution
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(data[train], targets[train])
    scikit_predictions = classifier.predict(data[test])

    print("My kNN classifier got " + str(compute_accuracy(predictions, targets[test])) + "% accuracy.")
    print("scikit-learn's KNeighborsClassifier got " + str(compute_accuracy(scikit_predictions, targets[test])) +
               "% accuracy.")

    return

if __name__ == '__main__':
    main(sys.argv)