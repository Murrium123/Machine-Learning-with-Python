import numpy as np
import pandas as pd
import sklearn.datasets as ski

# Load a classic dataset
iris = ski.load_iris
df = pd.DataFrame(
    ski.load_iris().data,
    columns=ski.load_iris().feature_names
)
df['species']=ski.load_iris().target

print(df.shape)
print(df.head())
print(df.describe())
print(df.hist())