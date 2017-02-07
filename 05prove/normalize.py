from sklearn import preprocessing as pp
import numpy as np


def is_number(s):
    """Evaluates if the string is comprised of numbers.

    Args:
        s (str): The string being evaluated.

    Returns:
        bool: True if it is a number. False if any character is not a number.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize(data):
    """Normalizes data consisting of strings to numeric values

    Args:
        data (collection of strings)

    Returns:
        A collection of normalized number values, each value representing a distinct string.
    """
    transposed_data = np.transpose(data)
    for i in range(len(transposed_data)):
        if all((not is_number(x)) for x in transposed_data[i]):
            le = pp.LabelEncoder()
            le.fit(transposed_data[i])
            transposed_data[i] = le.transform(transposed_data[i])
    data = np.transpose(transposed_data)
    return np.array(data)
