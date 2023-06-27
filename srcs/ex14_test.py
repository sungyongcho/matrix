import numpy as np

# Define the projection matrix
projection_matrix = np.array([[1, 0], [0, 0]])

# Define the vector to be projected
vector = np.array([2, 3])

# Project the vector onto the subspace using the projection matrix
projected_vector = np.dot(projection_matrix, vector)

print("Projected vector:", projected_vector)
