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

