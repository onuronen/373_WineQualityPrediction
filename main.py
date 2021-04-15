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


#svm
import random
import svm
import svmpredict

n = len(X)

subset = [500, 600, 700, 800, 900, 1000]
err = []

for i in range(len(subset)):

    index = random.sample(list(range(n)), subset[i])
    X_sub = []
    Y_sub = []
    for j in range(len(index)):
        X_sub.append(X[index[j]])
        Y_sub.append(Y[index[j]])

    X_train = X_sub[len(X_sub)//2:]
    Y_train = Y_sub[len(Y_sub)//2:]

    X_test = X_sub[:len(X_sub)//2]
    Y_test = Y_sub[:len(Y_sub)//2]

    clf = svm.run(X_train,Y_train)
    Y_predict = clf.predict(X_test)

    s = 0

    for i in range(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            s += 1

    err.append(s / len(Y_predict))

print(err)


#perceptron using sklearn
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, cross_val_score
#split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=4)
#making the model + fitting the model
perceptron_model = Perceptron(tol=1e-3, random_state=0)
perceptron_model.fit(X_train,y_train)
print('perceptron_model Training score:', perceptron_model.score(X_train,y_train))
print('perceptron_model Test score:    ', perceptron_model.score(X_test,y_test),'\n')

#cross validation for perceptron
scores = cross_val_score(estimator=perceptron_model, X=X, y=Y, cv=10)
print('Cross validation scores: ', scores, '\nmean:', scores.mean(),
      '\nStandard deviation: ', scores.std(), "\n")
