import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def k_fold(x, is_weighted):
    # start and end points of each fold (for 214 rows of data 43+43+43+43+42)
    arr = [0, 43, 86, 129, 172, 215]
    # for each fold, we create our test and train set and then call KNN classification function
    for i in range(5):
        # 1/5 part of the data set as test data
        x_test = x[arr[i]:arr[i + 1]]

        # rest of the data set as train data
        a = x[0:arr[i]]
        b = x[arr[i + 1]:]
        x_train = np.concatenate((a, b), axis=0)

        print("--------------------------FOLD", i+1, "--------------------------------------------")

        # for every fold use knn classification
        knn_true_prediction_count, knn_false_prediction_count = knn_classification(x_train, x_test, is_weighted)

        list_accuracy = []

        # for each k values {1,3,5,7,9}, we calculate accuracy seperately
        for e in range(0, 9, 2):
            accuracy = \
                (100 * knn_true_prediction_count[e]) / (knn_true_prediction_count[e] + knn_false_prediction_count[e])
            print("Accuracy for (KNN) k=", (e + 1), " : ", accuracy)
            list_accuracy.append(accuracy)

        plt.plot([1, 3, 5, 7, 9], list_accuracy)
        plt.axis([0, 9, 0, 100])
    return


def normalize(x):
    # for each column
    for i in range(0, x.shape[1] - 2):
        col = []
        # for each row of that specific column
        for k in range(x.shape[0]):
            # get all the values of the specific column
            col.append(x[k, i])
        # sort the column array so the first index contains min value, last index contains max value for that cloumn
        col.sort()
        min_of_col = col[0]
        max_of_col = col[x.shape[0] - 1]
        # for each element in that column normalize one by one
        for j in range(x.shape[0]):
            x[j, i] = (x[j, i] - min_of_col) / (max_of_col - min_of_col)

    return x


# for unweighted KNN cases, we use this function to predict test datas classes
def calculate_predictions(x_train, sorted_keys, test, true_prediction, false_prediction):
    closest_points = []
    for i in range(9):
        # in every loop, add one more nearest neighbor to the closest_points array
        closest_points.append(x_train[sorted_keys[i]][9])

        # when k=1 -> i=0  (length of closest_points array=1)
        #      k=3 -> i=2  (length of closest_points array=3)
        #      k=5 -> i=4  (length of closest_points array=5)
        #      k=7 -> i=6  (length of closest_points array=7)
        #      k=9 -> i=8  (length of closest_points array=9)
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
            # get the most frequent output in the closest_points array
            frequent = max(set(closest_points), key=closest_points.count)

            # if the predicted class equals to the actual class, increment true prediction count
            # or else increment false prediction count
            if frequent == test[9]:
                true_prediction[i] += 1
            else:
                false_prediction[i] += 1

    return true_prediction, false_prediction


# for weighted KNN cases, we use this function to predict test datas classes
def calculate_weighted_predictions(x_train, sorted_keys, euclidean_distances, test, true_prediction, false_prediction):
    # in closest_points_and_weights dictionary there are class numbers and their weights equals to 0 at beginning
    # key=class number , value=weight
    closest_points_and_weights = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
    for i in range(9):
        index = sorted_keys[i]
        # for each neighbor calculate weight as (1/distance) and add it to the dictionary
        if euclidean_distances.get(index) != 0:
            closest_points_and_weights[x_train[sorted_keys[i]][9]] += 1 / euclidean_distances.get(index)
        else:
            closest_points_and_weights[x_train[sorted_keys[i]][9]] += math.inf

        # find the maximum weighted class to predict
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
            frequent = max(closest_points_and_weights, key=closest_points_and_weights.get)

            # if the predicted class equals to the actual class, increment true prediction count
            # or else increment false prediction count
            if frequent == test[9]:
                true_prediction[i] += 1
            else:
                false_prediction[i] += 1

    return true_prediction, false_prediction


def knn_classification(x_train, x_test, is_weighted):
    # indices of the array represents k in kNN (we only use 1-3-5-7-9)
    true_prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    false_prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # for each row in the test set, calculate euclidean distance
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
                (ri * ri) + (ca * ca) + (mg * mg) + (na * na) + (al * al) + (si * si) + (k * k) + (ba * ba) + (fe * fe))

            euclidean_distances[j] = euc_dist

        # sort by the value of euclidean distance, first element will be the nearest point
        sorted_keys = sorted(euclidean_distances, key=euclidean_distances.get)

        # get the only first 9 nearest points because we dont need more for prediction
        sorted_keys = sorted_keys[:9]

        if is_weighted == False:
            # classify predictions as true or false by looking first 1,3,5,7,9 neighbors
            true_prediction, false_prediction = calculate_predictions(x_train, sorted_keys, test,
                                                                      true_prediction, false_prediction)
        else:
            # classify predictions with weights as true or false by looking first 1,3,5,7,9 neighbors
            true_prediction, false_prediction = calculate_weighted_predictions(x_train, sorted_keys,
                                                                               euclidean_distances,
                                                                               test, true_prediction, false_prediction)

    return true_prediction, false_prediction


def knn(x):
    print("KNN\n")
    k_fold(x, False)
    plt.ylabel("KNN")
    plt.show()


def knn_with_normalization(x):
    print("KNN WITH NORMALIZATION\n")
    x = normalize(x)
    k_fold(x, False)
    plt.ylabel("KNN-normalization")
    plt.show()


def weighted_knn(x):
    print("WEIGHTED KNN\n")
    k_fold(x, True)
    plt.ylabel("Weighted KNN")
    plt.show()


def weighted_knn_with_normalization(x):
    print("WEIGHTED KNN WITH NORMALIZATION\n")
    x = normalize(x)
    k_fold(x, True)
    plt.ylabel("Weighted KNN-normalization")
    plt.show()


def main():
    # reading data's in the csv file to the numpy array
    df = pd.read_csv('glass.csv')
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
