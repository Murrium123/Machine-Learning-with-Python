import numpy as np

# 1D array from a list
a = np.array([1,2,3,4,5])
print(a.shape) # (5,)
print(a.dtype) # int64

# 2D matrix 
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix.shape) # (3,3)

# Useful shortcuts
zeroes = np.zeros((3,4)) # 3x4
# array of zeros
random = np.random.randn(100) # 100
# standard normal samples
linsp = np.linspace(0,1,5) # [0,0.25,0.5,0.75,1.0]
