# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
import math
import numpy as np
import probclearn
import probcpredict

def run(k,X,y):
  y = y.flatten()
  n = len(y)
  z = np.zeros(k)
  for i in range(k):
    T = list(range(math.floor(n * i / k), math.floor(n * (i + 1) / k)))
    S = list(range(0, T[0])) + list(range(T[-1] + 1, n))
    X_t = np.empty([len(S), len(X[0])])
    y_t = np.empty(len(S))
    for j in range(len(X_t)):
      X_t[j] = np.array(X[S[j]]);

    for j in range(len(X_t)):
      y_t[j] = y[S[j]]

    q, mu_p, mu_n, sigma2_p, sigma2_n = probclearn.run(X_t, y_t)

    for t in range(len(T)):
      if y[T[t]] != probcpredict.run(q, mu_p, mu_n, sigma2_p, sigma2_n, np.asmatrix(X[T[t]]).T):
        z[i] += 1

    z[i] = z[i] / len(T)
  return np.asmatrix(z).T
