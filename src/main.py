import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import kfoldcv_svm
import kfoldcv_perceptron
import matplotlib.pyplot as plt
import random

X = np.loadtxt('X.txt')

# Note: no need to reshape X, need to reshape Y to n,1
Y = np.loadtxt('labels.txt')
print(len(X))


# number of total samples
n = len(X)

# subset sizes that we are using
subset = [100, 400, 700, 1000, 1300, 1599]
svm_error=[]
perceptron_error = []

# run kfold svm and perceptron with each size
for i in subset:
    print(i)
    index = random.sample(list(range(n)), i)
    X_sub = []
    Y_sub = []
    for j in range(len(index)):
        X_sub.append(X[index[j]])
        Y_sub.append(Y[index[j]])
    svm_error.append(kfoldcv_svm.run(5,X_sub,Y_sub,0.001));
    perceptron_error.append(kfoldcv_perceptron.run(5,X_sub,Y_sub,5));
    
# plotting data
fig, axs = plt.subplots(2)
fig.suptitle('Error Rate vs. Sample Size')
axs[0].plot(subset,svm_error)
axs[0].set_ylabel("SVM Error Rate")
axs[1].plot(subset,perceptron_error)
axs[1].set_ylabel("Kernel Perceptron Error Rate")
plt.xlabel("Sample Size")


