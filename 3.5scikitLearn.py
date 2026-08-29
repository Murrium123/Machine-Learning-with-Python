from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(42)
X = np.random.randn(200,1)
y = 3 * X.squeeze() + 2 + np.random.randn(200) * 0.5
