from matrix import Matrix
m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)
# # Output:
# (3, 2)
print(m1.T())
# # Output:
# Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.T().shape)
# # Output
# (2, 3)
m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape)
# # Output:
# (2, 3)
print(m1.T())
# # Output:
# Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.T().shape)
# # Output:
# (3, 2)
