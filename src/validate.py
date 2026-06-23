import numpy as np

matrixA = np.array([[2.0,2.0,2.0], [2.0,2.0,2.0]])
matrixB = np.array([[3.0,3.0], [3.0,3.0], [3.0,3.0]])
bias = np.array([[5.0,5.0], [5.0,5.0]])

relu_out = np.maximum(0,np.add(np.dot(matrixA, matrixB),  bias))

print(relu_out)