from matrix import Matrix

u = Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

# [1.0, 0.0, 0.0]
# [0.0, 1.0, 0.0]
# [0.0, 0.0, 1.0]
print(u.row_echelon())


u = Matrix([[1., 2.], [3., 4.]])
# [1.0, 0.0]
# [0.0, 1.0]
print(u.row_echelon())

u = Matrix([[1., 2.], [2., 4.]])
# [1.0, 2.0]
# [0.0, 0.0]
print(u.row_echelon())
