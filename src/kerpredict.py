import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import K

# Input: numpy vector alpha of n rows, 1 column
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)


def run(alpha,X,y,z):
    # Your code goes here
    (n,d) = np.shape(X)
    total = 0

    for i in range(n):
        x_i = X[i].reshape((d,1))
        total += (alpha[i][0] * y[i][0] * K.run(x_i, z))

    if total > 0:
        label = 1.0
    else:
        label = -1.0

    return label