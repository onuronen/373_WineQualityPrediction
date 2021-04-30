import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import kfoldcv_perceptron
import matplotlib.pyplot as plt

X = np.loadtxt('X.txt')

# Note: no need to reshape X, need to reshape Y to n,1
Y = np.loadtxt('labels.txt')

X = X[0:800]
Y = Y[0:800]
(n,d) = np.shape(X)
Y = Y.reshape((n,1))

number_of_iterations = [1,2,3,4,5,6]
err = []
for i in number_of_iterations:
    print(i)
    err.append(kfoldcv_perceptron.run(5,X,Y,i))

tuning_result = list(zip(number_of_iterations, err))

print("Perceptron hyperparameter errors", tuning_result)

# print graph of number_of_iterations vs accuracy
plt.plot(number_of_iterations, err)
plt.ylabel("Error Rate")
plt.xlabel("Iterations")
plt.xticks(number_of_iterations)
plt.title("# of Iterations vs Prediction Error")
plt.show()
