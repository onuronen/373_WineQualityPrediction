import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import math
import kerperceptron
import kerpredict

# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# number of iterations L for kernel perceptron
# Output: error rate with given number of iterations as the hyperparameter

def run(k,X,y,L):
    # Your code goes here return z
    n = np.shape(X)[0]
    d = np.shape(X)[1]

    y_pred = np.zeros(shape=(n,1))

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

        # learning for kernel perceptron
        alpha, iterations = kerperceptron.run(L,X_train, Y_train)

        # predicting labels
        for t in T:
            test_point = X[t]
            y_pred[t] = kerpredict.run(alpha,X_train,Y_train, test_point)
    
    error = np.mean(y != y_pred)
    return error