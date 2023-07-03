from matrix import Matrix

u = Matrix([[1., -1], [-1., 1.]])
# 0.0
print(u.determinant())

u = Matrix([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
# 8.0
print(u.determinant())


u = Matrix([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]])
# -174.0
print(u.determinant())


u = Matrix([[8., 5., -2., 4.],
            [4., 2.5, 20., 4.],
            [8., 5., 1., 4.],
            [28., -4., 17., 1.]])
# 1032.0
print(u.determinant())

print("=======eval===========")
u = Matrix([[0., 0.], [0., 0.]])
print(u.determinant())

u = Matrix([[1., 0.], [0., 1.]])
print(u.determinant())

u = Matrix([[2., 0.], [0., 2.]])
print(u.determinant())

u = Matrix([[1., 1.], [1., 1.]])
print(u.determinant())

u = Matrix([[0., 1.], [1., 0.]])
print(u.determinant())

u = Matrix([[1., 2.], [3., 4.]])
print(u.determinant())

u = Matrix([[-7., 5.], [4., 6.]])
print(u.determinant())

u = Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
print(u.determinant())
print("=======eval===========")
