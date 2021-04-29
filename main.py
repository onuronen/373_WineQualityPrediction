import os
import sys
sys.path.append(os.getcwd())

import numpy as np

X = np.loadtxt('X.txt')

# Note: no need to reshape X, need to reshape Y to n,1
Y = np.loadtxt('labels.txt')

# svm
import random
import svm
import svmpredict


# randomly choose 250 negativ sample and 250 positive sample
pos = np.where(Y==1)[0]
neg = np.where(Y==-1)[0]
pos = np.random.choice(pos,250)
neg = np.random.choice(neg,250)
join = np.concatenate((neg, pos))
X_reduced = [X[i] for i in join]
Y_reduced = [Y[i] for i in join]

# number of total samples
n = len(X)

# subset sizes that we are using
subset = [100, 300, 500, 700, 900, 1100]
err = []

# train and test the modle on each subset
for i in range(len(subset)):

    # randomly sample from dataset
    index = random.sample(list(range(n)), subset[i])
    X_sub = []
    Y_sub = []
    for j in range(len(index)):
        X_sub.append(X[index[j]])
        Y_sub.append(Y[index[j]])

    # split subset into equal halves for training and testing
    X_train = X_sub[len(X_sub)//2:]
    Y_train = Y_sub[len(Y_sub)//2:]

    X_test = X_sub[:len(X_sub)//2]
    Y_test = Y_sub[:len(Y_sub)//2]

    # train the model and test the model
    clf = svm.run(X_train,Y_train,1)
    Y_predict = clf.predict(X_test)

    s = 0

    # calculate the prediction error
    for i in range(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            s += 1

    err.append(s / len(Y_predict))

# final array of prediction errors
print(err)


import kfoldcv_svm
import kfoldcv_perceptron
import matplotlib.pyplot as plt

#plot literation fro perceptron
error = []
iteration = [1, 3, 5, 7, 9, 11]
for i in iteration:
    print(i)
    error.append(kfoldcv_perceptron.run(2,X_reduced,Y_reduced,i));
    print(error)
plt.plot(iteration, error)
plt.ylabel("Error Rate")
plt.xlabel("Iterations")
plt.title("Iteration Hyperparameter vs. Error Rate")




#plot different size
svm_error=[]
perceptron_error = []
for i in subset:
    print(i)
    index = random.sample(list(range(n)), i)
    X_sub = []
    Y_sub = []
    for j in range(len(index)):
        X_sub.append(X[index[j]])
        Y_sub.append(Y[index[j]])
    svm_error.append(kfoldcv_svm.run(5,X_sub,Y_sub,1));
    perceptron_error.append(kfoldcv_perceptron.run(5,X_sub,Y_sub,3));
fig, axs = plt.subplots(2)
fig.suptitle('Error Rate vs. Sample Size')
axs[0].plot(subset,svm_error)
axs[0].set_ylabel("SVM Error Rate")
axs[1].plot(subset,perceptron_error)
axs[1].set_ylabel("Kernel Perceptron Error Rate")
plt.xlabel("Sample Size")


