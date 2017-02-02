import numpy as np
from sklearn import preprocessing as pp


def process_csv(csv):
    transposed_csv = np.transpose(csv)
    for i in range(len(transposed_csv)):
        if any(x.isalpha() for x in transposed_csv[i]):
            le = pp.LabelEncoder()
            le.fit(transposed_csv[i])
            transposed_csv[i] = le.transform(transposed_csv[i])

    csv = np.transpose(transposed_csv)
    csv = np.array(csv)
    response = ""
    while response.lower() != "yes" and response.lower() != "no":
        response = input("Would you like to bin the data? (yes/no): ")

    if response == "yes":
        # Bin data
        for i in range(len(csv[0])):
            num_bins = 0
            while int(num_bins) < 1:
                num_bins = input("How many bins do you want for column " + str(i) + "?: ")
                if int(num_bins) < 1:
                    print("The number you provide must be positive.")

            max_num = max(float(row[i]) for row in csv)
            min_num = min(float(row[i]) for row in csv)

            cutoffs = []

            for j in range(int(num_bins) - 1):
                cutoffs.append((max_num - min_num) * ((j + 1) / int(num_bins)) + min_num)

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

                csv[j][i] = bin_num

    return csv
