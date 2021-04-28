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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

#reduce size
#X=X[100:700,:]
#Y=Y[100:700]

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                     test_size=0.2,
                                                     random_state = 3)

#trying with decision tree
decision_tree_model = DecisionTreeClassifier(random_state=0, max_depth = 5)
decision_tree_model.fit(X_train, y_train)
print('Decision Tree Training score:', decision_tree_model.score(X_train,y_train))
print('Decision Tree Test score:    ', decision_tree_model.score(X_test,y_test),'\n')
random_forest_model = RandomForestClassifier(random_state=0, max_depth = 5)

#trying with random forest
random_forest_model.fit(X_train, y_train)
print('Random Forest Training score:', random_forest_model.score(X_train,y_train))
print('Random Forest Test score:    ', random_forest_model.score(X_test,y_test),'\n')

# trying with linear perceptron
perceptron_model = Perceptron(tol=1e-3, random_state=0)
perceptron_model.fit(X_train,y_train)
print('perceptron_model Training score:', perceptron_model.score(X_train,y_train))
print('perceptron_model Test score:    ', perceptron_model.score(X_test,y_test),'\n')

# cross validation for perceptron
scores = cross_val_score(estimator=perceptron_model, X=X, y=Y, cv=10)
print('Cross validation scores: ', scores, '\nmean:', scores.mean(),
       '\nStandard deviation: ', scores.std(), "\n")



# transpose y traininng data to fit kerperceptron
y_train = np.asmatrix(y_train).transpose()
# fit the kernel perceptron model we made for HW6
alpha, iterr = kerperceptron.run(10, X_train, y_train)

# predict with testing data
y_predict = np.empty(len(y_test))
error = 0
for i in range(len(y_test)): #looping through each testing data point (z)
    z = X[i]
    # predict y label given an X test data point
    label = kerpredict.run(alpha, X_train, y_train, z)
    if label == y_test[i]: # if wrong prediciton
        error += 1
print("error rate: ", error/len(y_test))


# we want to draw graph of number of iterations in perceptron vs accuracy.
# we can test perceptron with same training and testing sets with different iterations.
# we will be using cross validation, where number of folds will be the same for each number of iterations
# The output from each cross validation will give a vector containing errror rate in each fold. We can
# take average of error rate from the veector to determine final error rate for the k fold with given number of iterations.
# Repeat this process for different Ls. 
# the way its done in case study cv is actually seems different.we have 10 folds, (k = 10). For the first time
# the test set will contain first 160 samples if we have 1600 data. We train with rest and then we have a global variable
# y_pred. With the theta we find, we make prediction from the X data of first 160 samples and put the labels we find in first 160 
# rows of y_pred. Then we repeat this process with second 160 data as test points, fill in y predd. In the end, after 10 folds,
# we do err = np.mean(y!=y_pred) and get a single error value for number of iterations with 10 fold cross val. 


