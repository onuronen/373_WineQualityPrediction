import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import kerperceptron, kerpredict, svm, svmpredict
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

X = np.loadtxt('X.txt')

# Note: no need to reshape X, need to reshape Y to n,1
Y = np.loadtxt('labels.txt')

# number of total samples
n = len(X)

# subset sizes that we are using
subset = [100, 400, 700, 1000, 1300, 1599]
svm_error=[]
perceptron_error = []

# run kfold svm and perceptron with each size
for i in subset:
    print(i)
    # subset the data
    X_sub = X[0:i]
    Y_sub = Y[0:i]

    #split training and testing half-half
    X_train, X_test, y_train, y_test = train_test_split(X_sub, Y_sub, test_size = 0.5)
    (training_length, dd) = np.shape(X_train)
    (test_length, features) = np.shape(X_test)

    #reshape for algorthm inputs
    y_train = y_train.reshape((training_length,1))
    y_test = y_test.reshape((test_length,1))

    y_pred_perceptron = np.zeros((test_length,1))

    #train perceptron
    alpha, iterr = kerperceptron.run(5, X_train, y_train)

    #make prediction for each test point
    index = 0
    for t in X_test:
        y_pred_perceptron[index] = kerpredict.run(alpha,X_train,y_train, t)
        index += 1
    
    perceptron_err = np.mean(y_test != y_pred_perceptron)
    perceptron_error.append(perceptron_err)

    #svm
    (training_length, features) = np.shape(X_train)

    #training svm 
    clf = svm.run(X_train, y_train.reshape((training_length)), 0.001)
    #testing svm
    y_pred_svm = svmpredict.run(clf, X_test)
    y_pred_svm = y_pred_svm.reshape((test_length,1))

    svm_err = np.mean(y_test != y_pred_svm)
    svm_error.append(svm_err)

    
# plotting data
fig, axs = plt.subplots(2)
fig.suptitle('Error Rate vs. Sample Size')
axs[0].plot(subset,svm_error)
axs[0].set_ylabel("SVM Error Rate")
axs[1].plot(subset,perceptron_error)
axs[1].set_ylabel("Kernel Perceptron Error Rate")
plt.xlabel("Sample Size")
plt.xticks(subset)
plt.show()


