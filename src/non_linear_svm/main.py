import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import kfoldcv_svm
import matplotlib.pyplot as plt

X = np.loadtxt('X.txt')

# Note: no need to reshape X, need to reshape Y to n,1
Y = np.loadtxt('labels.txt')

X = X[0:800]
Y = Y[0:800]
(n,d) = np.shape(X)
Y = Y.reshape((n,1))

gammas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
err = []
for i in gammas:
    err.append(kfoldcv_svm.run(5, X, Y, i))

tuning_result = list(zip(gammas, err))

print("SVM hyperparameter errors", tuning_result)

# print graph of gamma vs accuracy
plt.plot(gammas, err)
plt.ylabel("Prediction Error")
plt.xlabel("Gamma value")
plt.xscale('log')
plt.xticks(gammas)
plt.title("Gamma hyperparameter vs Prediction Error")
plt.show()