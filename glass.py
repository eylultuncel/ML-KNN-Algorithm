import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


np.set_printoptions(suppress=True)


def k_fold(x, is_weighted):
    # start and end points of each fold (for 214 rows of data 43+43+43+43+42)
    arr = [0, 43, 86, 129, 172, 215]
    for i in range(5):
        # 1/5 part of the data set as test data
        x_test = x[arr[i]:arr[i + 1]]

        # rest of the data set as train data
        a = x[0:arr[i]]
        b = x[arr[i + 1]:]
        x_train = np.concatenate((a, b), axis=0)

        print("--------------------------FOLD", i, "--------------------------------------------")

        # for every fold use knn classification
        knn_true_prediction_count, knn_false_prediction_count = knn_classification(x_train, x_test, is_weighted)

        print(knn_true_prediction_count)
        print(knn_false_prediction_count)
        print(".................................")

        list_accuracy = []

        for e in range(0, 9, 2):
            accuracy = \
                (100 * knn_true_prediction_count[e]) / (knn_true_prediction_count[e] + knn_false_prediction_count[e])
            print("Accuracy for (KNN) k=", (e + 1), " : ", accuracy)
            list_accuracy.append(accuracy)

        plt.plot([1, 3, 5, 7, 9], list_accuracy)
        plt.axis([0, 9, 0, 100])
    return


def normalize(x):
    for i in range(0, x.shape[1] - 2):
        col = []
        for k in range(x.shape[0]):
            col.append(x[k, i])
        col.sort()
        min_of_col = col[0]
        max_of_col = col[x.shape[0] - 1]

        for j in range(x.shape[0]):
            x[j, i] = (x[j, i] - min_of_col) / (max_of_col - min_of_col)
    return x


def calculate_predictions(x_train, sorted_keys, test, true_prediction, false_prediction):
    closest_points = []
    for i in range(9):
        closest_points.append(x_train[sorted_keys[i]][9])
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
            # get the most frequent output in the closest points array
            frequent = max(set(closest_points), key=closest_points.count)
            if frequent == test[9]:
                true_prediction[i] += 1
            else:
                false_prediction[i] += 1

    return true_prediction, false_prediction


def calculate_weighted_predictions(x_train, sorted_keys, euclidean_distances, test, true_prediction, false_prediction):
    # key=index , value=weight
    closest_points_and_weights = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
    for i in range(9):
        index = sorted_keys[i]
        if euclidean_distances.get(index) != 0:
            closest_points_and_weights[x_train[sorted_keys[i]][9]] += 1 / euclidean_distances.get(index)

        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
            frequent = max(closest_points_and_weights, key=closest_points_and_weights.get)
            if frequent == test[9]:
                true_prediction[i] += 1
            else:
                false_prediction[i] += 1

    return true_prediction, false_prediction
    return


def knn_classification(x_train, x_test, is_weighted):
    # array index represent k in kNN (we only use 1-3-5-7-9)
    true_prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    false_prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(x_test.shape[0]):
        test = x_test[i]
        euclidean_distances = {}
        for j in range(0, x_train.shape[0]):
            ri = x_train[j][0] - test[0]
            na = x_train[j][1] - test[1]
            mg = x_train[j][2] - test[2]
            al = x_train[j][3] - test[3]
            si = x_train[j][4] - test[4]
            k = x_train[j][5] - test[5]
            ca = x_train[j][6] - test[6]
            ba = x_train[j][7] - test[7]
            fe = x_train[j][8] - test[8]
            euc_dist = math.sqrt(
                # (ri * ri) + (ca * ca) + (mg * mg) + (na * na))
                # (al * al) + (mg * mg) )
                (ri * ri) + (ca * ca) + (mg * mg) + (na * na) + (al * al) + (si * si) + (k * k) + (ba * ba) + (fe * fe))

            euclidean_distances[j] = euc_dist

        # sort by the value of euclidean distance, first element will be the nearest point
        sorted_keys = sorted(euclidean_distances, key=euclidean_distances.get)

        sorted_keys = sorted_keys[:9]

        if is_weighted == False:
            # classify predictions as true or false by looking first 1,3,5,7,9 neighbors
            true_prediction, false_prediction = calculate_predictions(x_train, sorted_keys, test,
                                                                      true_prediction, false_prediction)
        else:
            true_prediction, false_prediction = calculate_weighted_predictions(x_train, sorted_keys,
                                                                               euclidean_distances,
                                                                               test, true_prediction, false_prediction)

    return true_prediction, false_prediction


def knn(x):
    k_fold(x, False)
    plt.ylabel("KNN")
    plt.show()


def knn_with_normalization(x):
    x = normalize(x)
    k_fold(x, False)
    plt.ylabel("KNN-NORM")
    plt.show()


def weighted_knn(x):
    k_fold(x, True)
    plt.ylabel("W-KNN")
    plt.show()


def weighted_knn_with_normalization(x):
    x = normalize(x)
    k_fold(x, True)
    plt.ylabel("W-KNN-NORM")
    plt.show()


def main():
    df = pd.read_csv('~/Desktop/BBM406&409/First Assignment/glass.csv')
    x = np.array(df.iloc[:, :])
    print(x.shape)

    np.random.seed(101)
    np.random.shuffle(x)

    print("KNN")
    knn(x.copy())
    print("\n\n\n\n\nKNN WITH NORMALIZATION")
    knn_with_normalization(x.copy())
    print("\n\n\n\n\nWEIGHTED KNN")
    weighted_knn(x.copy())
    print("\n\n\n\n\nWEIGHTED KNN WITH NORMALIZATION")
    weighted_knn_with_normalization(x.copy())


if __name__ == "__main__":
    main()


# (ri * ri) + (ca * ca) + (mg * mg) + (na * na))
