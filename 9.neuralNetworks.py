import numpy as np

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
