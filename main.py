import os
import sys
sys.path.append(os.getcwd())

import numpy as np

X = np.loadtxt('X.txt')

# Note: no need to reshape X, need to reshape Y to n,1
Y = np.loadtxt('labels.txt')

# We are using the first 500 data points - there is already 244 positive and 256 negative labels in a shuffled way
X = X[:500]
Y = Y[:500]

#positive_samples = list(np.where(Y==1)[0])
#print(len(positive_samples)) 
#negative_samples = list(np.where(Y==-1)[0])
#print(len(negative_samples))


# svm
import random
import svm
import svmpredict

# number of total samples
n = len(X)

# subset sizes that we are using
subset = [500, 600, 700, 800, 900, 1000]
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
    clf = svm.run(X_train,Y_train)
    Y_predict = clf.predict(X_test)

    s = 0

    # calculate the prediction error
    for i in range(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            s += 1

    err.append(s / len(Y_predict))

# final array of prediction errors
print(err)
