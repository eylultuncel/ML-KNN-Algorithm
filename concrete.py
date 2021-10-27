import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def k_fold(x, is_weighted):
    # start and end points of each fold
    arr = [0, 206, 412, 618, 824, 1030]
    # for each fold, we create our test and train set and then call KNN classification function
    for i in range(5):
        # 1/5 part of the data set as test data
        x_test = x[arr[i]:arr[i + 1]]

        # rest of the data set as train data
        a = x[0:arr[i]]
        b = x[arr[i + 1]:]
        x_train = np.concatenate((a, b), axis=0)

        print()
        print("--------------------------FOLD", i+1, "--------------------------------------------")

        # for every fold use knn classification
        knn_maes = knn_classification(x_train, x_test, is_weighted)

        list_mae = []

        for e in range(0, 9, 2):
            mae = knn_maes[e] / len(x_test)
            print("MAE for (KNN) k=", (e + 1), " : ", mae)
            list_mae.append(mae)

        plt.plot([1, 3, 5, 7, 9], list_mae)
        plt.axis([0, 9, 0, 10])
    return


def normalize(x):
    # for each column
    for i in range(0, x.shape[1]-1):
        col = []
        # for each row of that specific column
        for k in range(x.shape[0]):
            # get all the values of the specific column
            col.append(x[k, i])
        # sort the column array so the first index contains min value, last index contains max value for that column
        col.sort()
        min_of_col = col[0]
        max_of_col = col[x.shape[0]-1]
        # for each element in that column normalize one by one
        for j in range(x.shape[0]):
            x[j, i] = (x[j, i] - min_of_col) / (max_of_col - min_of_col)
    return x


# for each element in that column normalize one by one
def calculate_predictions(x_train, sorted_keys, test, maes):
    closest_points = []
    for i in range(9):
        # in every loop, add one more nearest neighbor to the closest_points array
        closest_points.append(x_train[sorted_keys[i]][8])

        # when k=1 -> i=0  (length of closest_points array=1)
        #      k=3 -> i=2  (length of closest_points array=3)
        #      k=5 -> i=4  (length of closest_points array=5)
        #      k=7 -> i=6  (length of closest_points array=7)
        #      k=9 -> i=8  (length of closest_points array=9)
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
            # get mean value of the closest points
            estimated = (sum(closest_points)) / (len(closest_points))
            # get sum of the all closest points, later it will used to get calculate mae
            maes[i] += abs(test[8] - estimated)

    return maes


# for weighted KNN cases, we use this function to predict test data's classes
def calculate_weighted_predictions(x_train, sorted_keys, test, maes, euclidean_distances):
    # in closest_points_and_weights dictionary there are csMPa values and their weights
    # key=csMPa , value=weight
    closest_points_and_weights = {}
    for i in range(9):
        key = x_train[sorted_keys[i]][8]

        if euclidean_distances.get(sorted_keys[i]) == 0:
            euc_dist_of_point = math.inf
        else:
            euc_dist_of_point = euclidean_distances.get(sorted_keys[i])

        # for each neighbor calculate weight as (1/distance) and add it to the dictionary
        if key in closest_points_and_weights.keys():
            closest_points_and_weights[x_train[sorted_keys[i]][8]] += 1 / euc_dist_of_point
        else:
            closest_points_and_weights[x_train[sorted_keys[i]][8]] = 1 / euc_dist_of_point

        # when k=1 -> i=0  (length of closest_points array=1)
        #      k=3 -> i=2  (length of closest_points array=3)
        #      k=5 -> i=4  (length of closest_points array=5)
        #      k=7 -> i=6  (length of closest_points array=7)
        #      k=9 -> i=8  (length of closest_points array=9)
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:

            # get weighted mean value of the closest csMPa values
            estimated = 0
            total_weight = 0
            for k in closest_points_and_weights.keys():
                estimated += k * closest_points_and_weights.get(k)
                total_weight += closest_points_and_weights.get(k)

            if estimated < 0.0001 or total_weight == math.inf:
                estimated = test[8]
            else:
                estimated = estimated / total_weight

            # get sum of the all closest points difference between real value, later it will used to get calculate mae
            maes[i] += abs(test[8] - estimated)

    return maes


def knn_classification(x_train, x_test, is_weighted):
    # indices of the array represents k in kNN (we only use 1-3-5-7-9)
    maes = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # for each row in the test set, calculate euclidean distance
    for k in range(x_test.shape[0]):
        test = x_test[k]
        euclidean_distances = {}
        for j in range(0, x_train.shape[0]):
            cement = x_train[j][0] - test[0]
            slag = x_train[j][1] - test[1]
            flyash = x_train[j][2] - test[2]
            water = x_train[j][3] - test[3]
            superplasticizer = x_train[j][4] - test[4]
            coarseaggregate = x_train[j][5] - test[5]
            fineaggregate = x_train[j][6] - test[6]
            age = x_train[j][7] - test[7]
            euc_dist = math.sqrt(cement**2 + slag**2 + flyash**2 + water**2 + superplasticizer**2 + coarseaggregate**2 + fineaggregate**2 + age**2)

            euclidean_distances[j] = euc_dist

        # sort by the value of euclidean distance, first element will be the nearest point
        sorted_keys = sorted(euclidean_distances, key=euclidean_distances.get)

        # get the only first 9 nearest points because we dont need more for prediction
        sorted_keys = sorted_keys[:9]

        if not is_weighted:
            # calculate mae by looking first 1,3,5,7,9 neighbors
            mae = calculate_predictions(x_train, sorted_keys, test, maes)
        else:
            # calculate mae by looking first 1,3,5,7,9 neighbors
            mae = calculate_weighted_predictions(x_train, sorted_keys, test, maes, euclidean_distances)

    return mae


def knn(x):
    print("KNN")
    k_fold(x, False)
    plt.ylabel("KNN")
    plt.show()


def knn_with_normalization(x):
    print("KNN WITH NORMALIZATION")
    x = normalize(x)
    k_fold(x, False)
    plt.ylabel("KNN-normalization")
    plt.show()


def weighted_knn(x):
    print("WEIGHTED KNN")
    k_fold(x, True)
    plt.ylabel("Weighted KNN")
    plt.show()


def weighted_knn_with_normalization(x):
    print("WEIGHTED KNN WITH NORMALIZATION")
    x = normalize(x)
    k_fold(x, True)
    plt.ylabel("Weighted KNN-normalization")
    plt.show()


def main():
    # reading data's in the csv file to the numpy array
    df = pd.read_csv('./concrete.csv')
    x = np.array(df.iloc[:, :])

    # shuffle the data
    np.random.seed(101)
    np.random.shuffle(x)

    # KNN function
    knn(x.copy())

    # KNN with normalization function
    knn_with_normalization(x.copy())

    # Weighted KNN function
    weighted_knn(x.copy())

    # Weighted KNN with normalization function
    weighted_knn_with_normalization(x.copy())


if __name__ == "__main__":
    main()
