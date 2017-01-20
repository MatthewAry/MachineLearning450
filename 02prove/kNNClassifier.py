

import pandas as pd
from pathlib import Path
import numpy as np
from math import sqrt
from numbers import Number

class KNearestNeighbor(object):
    def __init__(self, k=1):
        self.k = k
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

    def find_nearest(self, item):
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
        # Return the classifications of the nearest k neighbors
        return target_list[:self.k]

    def predict(self, predict_data):
        results = []
        for item in predict_data:
            # find nearest targets and take max occurrence
            nearest = self.find_nearest(item)
            # print(nearest)
            results.append(max(set(nearest), key=nearest.count))
        return results
