import os
import sys
sys.path.append(os.getcwd())

import numpy as np

X = np.loadtxt('X.txt')
print(X)
print(X.shape)

# Note: no need to reshape X, need to reshape Y to n,1
Y = np.loadtxt('labels.txt')
print(Y)
print(Y.shape)


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


#perceptron using sklearn
import kerperceptron
import kerpredict
from sklearn.model_selection import train_test_split, cross_val_score
# import sk
# from sklearn.linear_model import Perceptron

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                     test_size=0.5,
                                                     random_state = 3)
# transpose y traininng data to fit kerperceptron
y_train = np.asmatrix(y_train).transpose()

# fit the model
alpha, iterr = kerperceptron.run(5, X_train, y_train)

# predict with testing data
y_predict = np.empty(len(y_test))
error = 0
for i in range(len(y_test)): #looping through each testing data point (z)
    z = X[i]
    # predict y label given an X test data point
    label = kerpredict.run(alpha, X_train, y_train, z)
    if label != y_test[i]: # if wrong prediciton
        error += 1
print("error rate: ", error/len(y_test))



# #making the model + fitting the model with sklearn
# perceptron_model = Perceptron(tol=1e-3, random_state=0)
# perceptron_model.fit(X_train,y_train)
# print('perceptron_model Training score:', perceptron_model.score(X_train,y_train))
# print('perceptron_model Test score:    ', perceptron_model.score(X_test,y_test),'\n')
#
# #cross validation for perceptron
# scores = cross_val_score(estimator=perceptron_model, X=X, y=Y, cv=10)
# print('Cross validation scores: ', scores, '\nmean:', scores.mean(),
#       '\nStandard deviation: ', scores.std(), "\n")
