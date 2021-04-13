import os
import sys
sys.path.append(os.getcwd())
import pandas as pd

df = pd.read_csv("wine_data.csv")

# updating the column value/data
# good  = 1, bad = -1
df['quality'] = df['quality'].replace({'good': 1})
df['quality'] = df['quality'].replace({'bad': -1})

# writing into the file
df.to_csv("wine_data_preprocessed.csv", index=False)

