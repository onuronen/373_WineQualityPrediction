from sklearn import svm
import numpy as np

# Input: sklearn svm model object clf
# matrix X of features, with n rows, d columns
# Output: array y of prediction labels, with n rows
def run(clf, X):
    return clf.predict(X)
