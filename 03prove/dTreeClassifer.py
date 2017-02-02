import numpy as np
import copy
from collections import OrderedDict
from decimal import *
import operator


def tree_classifier_debugger(data, attribute, targets):
    print("DEBUG OUTPUT")
    print("Data: ", data)
    print("Attribute: ", attribute)
    print("Targets: ", [i[-1] for i in data])
    print("Unique Targets: ", np.unique(targets))


class Node:
    def __init__(self, value = 0):
        self.attribute = value
        self.branches = {}


class TreeClassifier:
    def __init__(self):
        self.root = Node()
        self.default = 0

    @staticmethod
    def calc_entropy(p):
        if p != 0:
            return -p * np.log2(p)
        return 0

    def calc_total_entropy(self, target_counts, n_data):
        total_entropy = 0
        for target in target_counts:
            total_entropy += self.calc_entropy(Decimal(target[1]) / n_data)
        return total_entropy

    @staticmethod
    def calculate_data_loss(data, attribute):
        information_loss = 0
        # Get all unique values in column
        values = list(set(np.transpose(data)[attribute]))

        for value in values:
            # Group rows by matching values
            groups = [instance for instance in data if instance[attribute] == value]
            # Calculate the frequency of each target in each group
            target_frequencies = OrderedDict()
            for group in groups:
                target_frequencies[group[-1]] = target_frequencies.get(group[-1], 0) + 1

            # Convert values to percentages
            groups_len = len(groups)
            for key in target_frequencies.keys():
                target_frequencies[key] /= groups_len

            entropy = 0
            for probability in target_frequencies.values():
                if probability != 0:
                    entropy += -probability * np.log2(probability)

            information_loss += (groups_len / len(data)) * entropy
        return information_loss

    def train(self, data, targets):
        # Enumerate columns in data
        attributes = list(range(len(data[0])))

        self.default = targets[0]
        # convert data and targets toList
        t_data = data.tolist()
        t_targets = targets.tolist()

        # Join targets with data
        for i in range(len(t_data)):
            t_data[i].append(t_targets[i])

        # Build tree with training data
        self.root = self.make_tree(t_data, attributes)

        # Display the tree
        self.display_tree(self.root, 0)

    def traverse_tree(self, element, node):
        if element[node.attribute] not in node.branches.keys():
            return 0
        branch = node.branches[element[node.attribute]]
        if isinstance(branch, Node):
            return self.traverse_tree(element, branch)
        else:
            return node.branches[element[node.attribute]]

    def display_tree(self, node, level):
        indent = "    "
        i = 0
        while i < level:
            indent += "    "
            i += 1

        if isinstance(node, Node):
            print(indent, "Column: ", node.attribute)
            for key, val in node.branches.items():
                print(indent, "    Branch: ", key, ": ")
                self.display_tree(val, level + 1)
        else:
            print(indent, node)

    def predict(self, data):
        result = []
        for element in data:
            result.append(self.traverse_tree(element, self.root))
        return result

    def make_tree(self, data, attributes):
        # Create a Root Node for the tree. (This is a recursive function BTW)
        node = Node()

        # If data is empty
        if len(data) == 0:
            return self.default

        # Count the occurrences of each target.
        target_counts = OrderedDict()
        trans_data = np.transpose(data)
        for item in trans_data[-1]:
            target_counts[item] = target_counts.get(item, 0) + 1

        # If all targets are the same
        if len(target_counts) == 1:
            # Return the target
            return list(target_counts.keys())[0]

        elif len(attributes) == 0:
            # Return default target
            #tree_classifier_debugger(data, attributes, target_counts.keys)
            return max(target_counts.items(), key=operator.itemgetter(1))[0]

        # Implicit Else
        entropy_dict = {}
        for attribute in attributes:
            # Calculate the data loss
            entropy_dict[attribute] = self.calculate_data_loss(data, attribute)

        min_entropy = min(iter(entropy_dict.values()))
        # Select column with the least entropy
        node.attribute = [i for i, v in iter(entropy_dict.items()) if v == min_entropy][0]

        values = np.transpose(data)[node.attribute]

        # Isolate the values for each branch for recursive tree generation
        for value in values:
            t_data = [row for row in data if row[node.attribute] == value]

            t_labels = copy.deepcopy(attributes)
            t_labels.remove(node.attribute)
            # Recurse
            node.branches[value] = self.make_tree(t_data, t_labels)
        return node




