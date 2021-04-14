from sklearn import svm
import numpy as np

def run(X,y):
    clf = svm.NuSVC(gamma="auto")
    clf.fit(X,y)
    return clf
