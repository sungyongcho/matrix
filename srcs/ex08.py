from matrix import Matrix


u = Matrix([[1., 0.], [0., 1.]])

# 2.0
print(u.trace())

u = Matrix([[2., 5., 0.], [4., 3., 7.], [-2., 3., 4.]])

# 9.0
print(u.trace())

u = Matrix([[-2., -8., 4.], [1., -23., 4.], [0., 6., 4.]])

# -21.0
print(u.trace())
