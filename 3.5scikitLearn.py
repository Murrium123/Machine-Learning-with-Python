from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(42)
X = np.random.randn(200,1)
y = 3 * X.squeeze() + 2 + np.random.randn(200) * 0.5

X_train, X_test, y_train, y_test = X,y, test_size = 0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Coefficient: {model.coef_[0]:.3f}')
print(f'Intercept: {model.intercept_:.3f}')
print(f'MSE:          {mse: .4f}')



















