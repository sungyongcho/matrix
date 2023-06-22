from matrix import Matrix, Vector

u = Matrix([[1., 0.], [0., 1.]])
v = Vector([[4.], [2.]])

# [4.]
# [2.]
print(u*v)

u = Matrix([[2., 0.], [0., 2.]])
v = Vector([[4.], [2.]])

# [8.]
# [4.]
print(u*v)

u = Matrix([[2., -2.], [-2., 2.]])
v = Vector([[4.], [2.]])

# [4.]
# [-4.]
print(u*v)

u = Matrix([[1., 0.], [0., 1.]])
v = Matrix([[1., 0.], [0., 1.]])

# [1., 0.]
# [0., 1.]
print(u*v)

u = Matrix([[1., 0.], [0., 1.]])
v = Matrix([[2., 1.], [4., 2.]])

# [2., 1.]
# [4., 2.]
print(u*v)


u = Matrix([[3., -5.], [6., 8.]])
v = Matrix([[2., 1.], [4., 2.]])

# [-14., 7.]
# [44., 22.]
print(u*v)
