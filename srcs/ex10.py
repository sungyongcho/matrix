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

u = Matrix([[8., 5., -2., 4., 28.],
            [4., 2.5, 20., 4., -4.],
            [8., 5., 1., 4., 17.]])
# [1.0, 0.625, 0.0, 0.0, -12.1666667]
# [0.0, 0.0, 1.0, 0.0, -3.6666667]
# [0.0, 0.0, 0.0, 1.0, 29.5 ]
print(u.row_echelon())

print("=======eval===========")
u = Matrix([[0., 0.], [0., 0.]])
print(u.row_echelon())

u = Matrix([[1., 0.], [0., 1.]])
print(u.row_echelon())

u = Matrix([[4., 2.], [2., 1.]])
print(u.row_echelon())

u = Matrix([[-7., 2.], [4., 8.]])
print(u.row_echelon())

u = Matrix([[1., 2.], [4., 8.]])
print(u.row_echelon())

print("=======eval===========")
