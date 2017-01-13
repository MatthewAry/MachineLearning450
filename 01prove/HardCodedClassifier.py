from sklearn import datasets
from numpy.random import permutation

# Load in data
iris = datasets.load_iris()

# Randomize my data
indices = permutation(len(iris.data))


length = len(iris.data)
trainingSize = int(.7 * length)


training_data = iris.data[indices[:trainingSize]]
training_target = iris.target[indices[:trainingSize]]

testing_data = iris.data[indices[trainingSize:length]]
testing_target = iris.target[indices[trainingSize:length]]

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

hard_coded_classifier = HardCoded()
prediction_result = hard_coded_classifier.predict(testing_data)

# Calculate accuracy
result = 0.0
for i in range(len(prediction_result)):
    if prediction_result[i] == testing_target[i]:
        result += 1

result = (result/len(testing_target)) * 100
print("The predictive accuracy of this program is " + str(result) + "%.")