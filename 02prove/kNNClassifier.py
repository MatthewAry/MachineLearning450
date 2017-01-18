from sklearn import datasets
from numpy.random import permutation
import numpy as np
from math import sqrt
from numbers import Number

class HardCoded(object):

    def fit(self, training_data, training_target):
        # This method does not need to do anything.
        pass

    def predict(self, testing_data):
        predictions = []
        for i in range(len(testing_data)):
            # The hardcoded classification
            predictions.append(1)

        return predictions

class KNearestNeighbor(object):
    def __init__(self):
        self.data = []
        self.target = []

    def get_difference(self, left, right):
        # Ensure that left and right are numbers
        if isinstance(left, Number) and isinstance(right, Number):
            # return difference if they are
            return left - right
        else:
            # Otherwise return 1 if they are different
            return 0 if left == right else 1

    def train(self, train_data, train_target):
        self.data = train_data
        self.target = train_target

    def find_nearest(self, item, n_neighbors):
        # Obtain the distance of this item from items in the training data
        distances = []
        for element in self.data:
            distance = 0
            for i in range(0, len(item)):
                distance += pow(self.get_difference(item[i], element[i]), 2)
            distances.append(distance)

        # Assign a distance to each target, We are merging the two lists together so we can sort them
        distance_list = sorted(list(zip(distances, self.target)), key=lambda element: element[0])
        # We unzip the list because we want to throw away the distances and only get back the sorted target_list
        throw_away, target_list = zip(*distance_list)
        # print(target_list)
        return target_list[:n_neighbors]

    def predict(self, predict_data, n_neighbors=3):
        results = []
        for item in predict_data:
            # find nearest targets and take max occurrence
            nearest = self.find_nearest(item, n_neighbors)
            # print(nearest)
            results.append(max(set(nearest), key=nearest.count))
        return results

def main():
    size = input('How much of the data should be used for training? (1 - 100): ')
    if size.isdigit():
        size = int(size) * .01
    else:
        print("You either did not enter a number or your response was invalid.")
        print("Setting data set size to 70%.\n")
        size = .7

    # Load in data
    iris = datasets.load_iris()

    # Randomize my data
    indices = permutation(len(iris.data))


    length = len(iris.data)
    trainingSize = int(size * length)


    training_data = iris.data[indices[:trainingSize]]
    training_target = iris.target[indices[:trainingSize]]

    testing_data = iris.data[indices[trainingSize:length]]
    testing_target = iris.target[indices[trainingSize:length]]

    kNN = KNearestNeighbor()
    kNN.train(training_data, training_target)

    prediction_result = kNN.predict(testing_data, 3)
    # prediction_result = kNN.knn(3, testing_data)

    # hard_coded_classifier = HardCoded()
    # prediction_result = hard_coded_classifier.predict(testing_data)

    # Calculate accuracy
    result = 0.0
    for i in range(len(prediction_result)):
        if prediction_result[i] == testing_target[i]:
            result += 1

    result = (result/len(testing_target)) * 100
    print("The predictive accuracy of this program is " + str(result) + "%.")

main()