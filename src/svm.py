from sklearn import svm

# Input: matrix X of features, with n rows, d columns
# vector y of labels, with n rows
# Output: Trained svm model object
def run(X,y,g):
    clf = svm.NuSVC(gamma=g)
    clf.fit(X,y)
    return clf
