import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import math
import svm
import svmpredict

# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column

def run(k,X,y):
    # Your code goes here return z
    n = np.shape(X)[0]
    d = np.shape(X)[1]

    z = np.zeros(shape=(k,1))

    for i in range(k):
        # boundaries for each fold
        lower = math.floor(n * i / k)
        upper = math.floor((n * (i+1) / k) - 1)

        # testing set
        T = list(range(lower, upper + 1))

        # training set
        S = [*range(lower), *range(upper + 1, n)]

        X_train = np.zeros(shape=(len(S), d))
        Y_train = np.zeros(shape=(len(S), 1))

        # fill up X_train and Y_train according to training set
        for j in range(len(S)):
            row = S[j]
            X_train[j] = X[row]
            Y_train[j] = y[row]

        clf = svm.run(X_train, Y_train)

        z_i = 0

        for t in T:
            x_t = X[t].reshape(d,1)
            if y[t][0] != svmpredict.run(clf, x_t):
                z_i += 1

        z_i = z_i / len(T)
        z[i] = z_i

    return z
