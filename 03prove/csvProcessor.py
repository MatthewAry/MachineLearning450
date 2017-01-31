import sys


def process_csv(csv):
    # Figure out what the Columns Are
    columns = csv[0]

    # Get a list of number Columns
    number_column_list = []
    for index, attribute in enumerate(columns):
        number_column_list.append(index)

    # List of sets needed to change the numbers
    sets = []
    # Each index will need one of these sets
    for index in number_column_list:
        sets.append(set())

    # Add all the possibilities (from every instance) for each column
    for instance in csv:
        for i, column in enumerate(number_column_list):
            sets[i].add(instance[column])

    # Simplify the sets
    for i, column in enumerate(number_column_list):
        # We assume this is a number list, let's check
        number_col = True

        # If any value in the set is not a number then it is not a number column
        for value in sets[i]:
            if not is_number(value):
                number_col = False

        # If not a number column then we should not simplify (we do not know how)
        if number_col:
            print("Number of bins of column ", i, ": ")
            num_bins = input()

            # be sure to get a positive number
            while int(num_bins) < 1:
                print("Number must be positive.\nNumber of bins of column ", i, ": ")
                num_bins = input()

            cutoffs = []

            # The min and max for column
            # print (csv)
            max_num = max(float(row[i]) for row in csv)
            min_num = min(float(row[i]) for row in csv)

            # calculate the cutoff points
            for j in range(int(num_bins) - 1):
                cutoffs.append((max_num - min_num) * ((j + 1) / int(num_bins)) + min_num)

            # replace the values with bin value
            for j, instance in enumerate(csv):

                bin_num = 0
                if float(instance[i]) > cutoffs[-1]:
                    bin_num = len(cutoffs)

                else:
                    for k in range(len(cutoffs)):
                        if k != 0:
                            if cutoffs[k - 1] < float(instance[i]) <= cutoffs[k]:
                                bin_num = k
                        else:
                            if float(instance[i]) <= cutoffs[k]:
                                bin_num = k

                # replace it with the correct bin
                csv[j][i] = bin_num

    # return the file
    return csv


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)