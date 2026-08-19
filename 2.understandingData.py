import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target
print('Shape: ', df.shape)
print(df.describe())

print('Missing values per column: ')
print(df.isnull().sum())
print()
print('Percentage missing: ')
print((df.isnull().mean() * 100).round(2))

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True,
cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3,3,figsize=(12,10))
axes = axes.flatten()

for i, col in enumerate(df.columns):
	df[col].hist(bins=50, ax=axes[i])
	axes[i].set_title(col)
	
plt.tight_layout()
plt.show()	

