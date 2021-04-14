import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from numpy import genfromtxt

df = pd.read_csv("wine_data.csv")

# updating the column value/data
# good  = 1, bad = -1
df['quality'] = df['quality'].replace({'good': 1})
df['quality'] = df['quality'].replace({'bad': -1})

# writing into the file
df.to_csv("wine_data_preprocessed.csv", index=False)

# turn data into numpy matrix
data = genfromtxt('wine_data_preprocessed.csv', delimiter=',', skip_header=1)
(n,d) = np.shape(data)

# quality column
labels = data[:, -1]

# reshape to n x 1 matrix
Y = labels.reshape((n,1))


# exclude quality column from data
X = data[:, :-1]

# save X and Y to text files
np.savetxt("X.txt", X)
np.savetxt("labels.txt", Y)









