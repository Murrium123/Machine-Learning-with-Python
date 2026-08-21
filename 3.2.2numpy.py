import time

size = 1_000_000
a = np.random.randn(size)
b = np.random.randn(size)

# Python loop
start = time.time()
result = [a[i] + b[i] for i in range(size)]
print(f'Python loop: {time.time() - start:.3f}s')

# NumPy vectorized
start = time.time()
result = a + b
print(f'Numpy:          {time.time() - start:.4f}s')
