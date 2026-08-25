import numpy as np

print(f"\nThe Perceptron\n")
class SimplePerceptron:
    def __init__(self, lr=0.01, n_iter=100):
        self.lr=lr
        self.n_iter = n_iter

    def fit(self,X,y):
        self.weights=np.zeros(X.shape[1])
        self.bias=0
        for _ in range(self.n_iter):
            for xi, yi in zip(X,y):
                pred=self._predict(xi)
                delta = self.lr*(yi-pred)
                self.weights += delta*xi
                self.bias += delta

    def _predict(self,x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

model = SimplePerceptron()
print(model)

print(f"\n9.3 Activation functions\n")

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,200)

sigmoid = 1/(1+np.exp(-x))
relu = np.maximum(0,x)
leaky_relu= np.where( x > 0, x, 0.01 * x)
tanh_fn=np.tanh(x)

fig, axes = plt.subplots(2,2,figsize=(12,8))
for ax, (name, fn) in zip(axes.flatten(), [
    ('Sigmoid', sigmoid), ('ReLU', relu), 
    ('Leaky ReLU', leaky_relu),
    ('Tanh', tanh_fn)]):
        ax.plot
