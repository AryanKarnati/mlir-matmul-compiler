import numpy as np

matrixA = np.full((512, 512), 2.0)
matrixB = np.full((512, 512), 3.0)
bias = np.full((512, 512), 5.0)

K = 512  # shared dimension
expected_matmul_value = K * 2.0 * 3.0  # = 3072.0
expected_with_bias = expected_matmul_value + 5.0  # = 3077.0
expected_after_relu = max(0, expected_with_bias)  # = 3077.0

print(f"Expected matmul value: {expected_matmul_value}")
print(f"Expected with bias: {expected_with_bias}")
print(f"Expected after ReLU: {expected_after_relu}")
print(f"Matrix shape: (512, 512)")
print("All entries should be 3077.0")