import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import K

# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column
# number of iterations that were actually executed (iter+1)

def run(L,X,y):
    # Your code goes here

    (n,d) = np.shape(X)
    alpha = np.zeros((n,1))

    for iterr in range(L):
        all_points_classified_correctly = True

        for t in range(n):
            summ = 0

            x_t = X[t].reshape((d,1))
            for i in range(n):
                x_j = X[i].reshape((d,1))
                summ += (alpha[i][0] * y[i][0] * K.run(x_j, x_t))

            if y[t][0] * summ <= 0:
                alpha[t][0] += 1
                all_points_classified_correctly = False

        if all_points_classified_correctly:
            break

    return alpha, iterr+1