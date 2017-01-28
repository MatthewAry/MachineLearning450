"""
Translation of Marsland Code of dTree
    MINE          Marsland
    attributes =  featureNames (The column name)
    data       =  the attribute values
    target     =  class (The target values)


Pre Processing:
    Create a list of possible classes


ID3 Classifier (Examples, Target_Attribute, Candidate_Attributes)
    Create a Root node for the tree. (This function is recursive)
    If all examples have the same value of the Target_Attribute,
        Return the Root node with the label = Target_Attribute
    elsIf the list of Candidate_Attributes is empty,
        Return the Root node with the label = most common Target_Attribute in Examples
    else

"""

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
    print("Length of Unique Targets: ", len(np.unique(targets)))


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
                target_frequencies[group[-1]] = target_frequencies.get(group, 0) + 1

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

        # convert data and targets toList
        t_data = data.toList()
        t_targets = targets.toList()

        # Join targets with data
        for i in range(len(t_data)):
            t_data[i].append(t_targets[i])

        # Build tree with training data
        self.root = self.make_tree(t_data, attributes)

        # Display the tree
        self.display_tree(self.root, 0)

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

    def make_tree(self, data, attributes, targets):
        # Create a Root Node for the tree. (This is a recursive function BTW)
        node = Node()

        # Get size of data
        n_data = len(data)
        # Get size of attributes
        n_attributes = len(attributes)

        # Count the occurrences of each target.
        target_counts = OrderedDict()
        for item in targets:
            target_counts[item] = target_counts.get(item, 0) + 1

        # If all targets are the same
        if len(target_counts) == 1:
            # Return the target
            return list(target_counts.keys())[0]
        # If data is empty
        elif n_data == 0 or n_attributes == 0:
            # Return default target
            # print("Data is or Attributes is Empty")
            # tree_classifier_debugger(data, attributes, targets)
            return max(target_counts.iteritems, key=operator.itemgetter(1))[0]

        # Implicit Else
        entropy_dict = {}
        for attribute in attributes:
            # Calculate the data loss
            entropy_dict[attribute] = self.calculate_data_loss(data, attribute)

        min_entropy = min(iter(entropy_dict.values()))
        # Select column with the least entropy
        node.attribute = [i for i, v in iter(entropy_dict.items()) if v == min_entropy]

        values = np.transpose(data)[node.attribute]

        # Isolate the values for each branch for recursive tree generation
        for value in values:
            t_data = [row for row in range(len(data)) if row[node.attribute] == value]

            t_labels = copy.deepcopy(attributes).remove(node.attribute)
            # Recurse
            node.branches[value] = self.make_tree(t_data, t_labels)
        return node




