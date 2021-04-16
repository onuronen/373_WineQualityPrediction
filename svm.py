from sklearn import svm
import numpy as np

# Input: matrix X of features, with n rows, d columns
# vector y of labels, with n rows
# Output: Trained svm model object
def run(X,y):
    clf = svm.NuSVC(gamma="auto")
    clf.fit(X,y)
    return clf
