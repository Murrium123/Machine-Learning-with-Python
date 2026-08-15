import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load a classic dataset
iris = load_iris
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
df['species']=iris.target

print(df.shape)
print(df.head())

