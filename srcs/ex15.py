import numpy as np
from matrix import Vector, cross_product
from matrix import Vector, angle_cos
from matrix import lerp
from matrix import Matrix, Vector
from matrix import linear_combination

print("ex00=========================================")
a = Vector([[2.j], [3.]])
b = Vector([[5.], [7.j]])

print("# ex00")
print("## Vector")
print("### add")
print(a.add(b))
print(a + b)

# a = Vector([[2.], [3.]])
# b = Vector([[5.], [7.]])
print("### sub")
print(a.sub(b))
print(a - b)

# a = Vector([[2.], [3.]])
# b = Vector([[5.], [7.]])
print("### scl")
print(a.scl(2.))

print("## Matrix")
print("### add")
u = Matrix([[1., 2.j], [3.j, 4.]])
v = Matrix([[7.j, 4.], [-2.j, 2.]])
print(u.add(v))
print(u + v)

# u = Matrix([[1., 2.], [3., 4.]])
# v = Matrix([[7., 4.], [-2., 2.]])

print("### sub")
print(u.sub(v))
print(u - v)

print("### scl")
print(u.scl(2.))
print("=============================================")

print("ex01=========================================")
e1 = Vector([[1.j], [0.], [0.]])
e2 = Vector([[0.], [1.], [0.]])
e3 = Vector([[0.], [0.], [1.j]])

v1 = Vector([[1.], [2.], [3.j]])
v2 = Vector([[0.], [10.j], [-100.]])

print(linear_combination([e1, e2, e3], [10., -2., 0.5]))

print("=============================================")

print("ex02=========================================")

print(lerp(0., 1.j, 0.))

print(lerp(0.j, 1., 1.))

# to show exception handling
# print(lerp(0., 1., .5j))

print(lerp(21.j, 42., .3))

print(lerp(Vector([[2.j], [1.j]]), Vector([[4.j], [2.]]), .3))

print(lerp(Matrix([[2., 1.j], [3., 4.]]),
      Matrix([[20.j, 10.], [30.j, 40.]]), .5))

print("=============================================")

print("ex03=========================================")
u = Vector([[0.], [0.j]])
v = Vector([[1.j], [1.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

print(u.dot(v))

u = Vector([[1.j], [1.j]])
v = Vector([[1.j], [1.j]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

print(u.dot(v))


u = Vector([[-1.], [6.j]])
v = Vector([[3.], [2.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

print(u.dot(v))
print("=============================================")

print("ex04=========================================")
print(u.norm_1(), u.norm(), u.norm_inf())

u = Vector([[1.j], [2.], [3.]])
# 6.0, 3.74165738, 3.0
print(u.norm_1(), u.norm(), u.norm_inf())

u = Vector([[-1.j], [-2.j]])
# 6.0, 3.74165738, 3.0
print(u.norm_1(), u.norm(), u.norm_inf())

print("=============================================")

print("ex05=========================================")

u = Vector([[1.], [0.]])
v = Vector([[1.], [0.]])

# 1.0
print(angle_cos(u, v))

u = Vector([[1.j], [0.]])
v = Vector([[0.], [1.]])

# 0.0
print(angle_cos(u, v))

# to show error handling
# u = Vector([[-1.j], [1.j]])
# v = Vector([[1.], [-1.]])

# # -1.0
# print(angle_cos(u, v))

u = Vector([[2.j], [1.]])
v = Vector([[4.j], [2.]])

# 1.0
print(angle_cos(u, v))

u = Vector([[1.j], [2.], [3.]])
v = Vector([[4.j], [5.], [6.]])

# 0.974631846
print(angle_cos(u, v, 0))
print("=============================================")

print("ex06=========================================")

u = Vector([[0.], [0.j], [1.]])
v = Vector([[1.], [0.j], [0.]])

# [0.]
# [1.]
# [0.]
print(cross_product(u, v))

u = Vector([[1.j], [2.], [3.]])
v = Vector([[4.], [5.], [6.j]])

# [-3.]
# [6.]
# [-3.]
print(cross_product(u, v))


u = Vector([[4.], [2.j], [-3.]])
v = Vector([[-2.j], [-5.], [16.]])

# [17.]
# [-58.]
# [-16.]
print(cross_product(u, v))

print("=============================================")

print("ex07=========================================")
u = Matrix([[1.j, 0.j], [0., 1.]])
v = Vector([[4.j], [2.j]])

# [4.]
# [2.]
print(u*v)

u = Matrix([[2., 0.j], [0., 2.]])
v = Vector([[4.], [2.j]])

# [8.]
# [4.]
print(u*v)

u = Matrix([[2., -2.j], [-2., 2.]])
v = Vector([[4.j], [2.]])

# [4.]
# [-4.]
print(u*v)

u = Matrix([[1., 0.j], [0.j, 1.]])
v = Matrix([[1., 0.j], [0.j, 1.]])

# [1., 0.]
# [0., 1.]
print(u*v)

u = Matrix([[1.j, 0.], [0., 1.]])
v = Matrix([[2., 1.], [4., 2.]])

# [2., 1.]
# [4., 2.]
print(u*v)


u = Matrix([[3., -5.j], [6., 8.]])
v = Matrix([[2., 1.], [4., 2.]])

# [-14., 7.]
# [44., 22.]
print(u*v)

print("=============================================")

print("ex08=========================================")
u = Matrix([[1.j, 0.], [0., 1.]])

# 2.0
print(u.trace())

u = Matrix([[2., 5., 0.j], [4., 3., 7.], [-2., 3., 4.]])

# 9.0
print(u.trace())

u = Matrix([[-2.j, -8.j, 4.j], [1., -23., 4.], [0., 6., 4.]])

# -21.0
print(u.trace())

print("=============================================")

print("ex09=========================================")
m1 = Matrix([[0.0j, 1.0j], [2.0j, 3.0j], [4.0, 5.0]])
print(m1.shape)
# # Output:
# (3, 2)
print(m1.T())
# # Output:
# Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.T().shape)
# # Output
# (2, 3)
m1 = Matrix([[0.j, 2.j, 4.], [1.j, 3.j, 5.j]])
print(m1.shape)
# # Output:
# (2, 3)
print(m1.T())
# # Output:
# Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.T().shape)
# # Output:
# (3, 2)

print("=============================================")

print("ex10=========================================")
u = Matrix([[1., 0.j, 0.j], [0., 1., 0.], [0., 0., 1.]])

# [1.0, 0.0, 0.0]
# [0.0, 1.0, 0.0]
# [0.0, 0.0, 1.0]
print(u.row_echelon())


u = Matrix([[1.j, 2.j], [3.j, 4.j]])
# [1.0, 0.0]
# [0.0, 1.0]
print(u.row_echelon())

u = Matrix([[1.j, 2.j], [2., 4.]])
# [1.0, 2.0]
# [0.0, 0.0]
print(u.row_echelon())

u = Matrix([[8., 5., -2.j, 4., 28.],
            [4., 2.5j, 20., 4.j, -4.],
            [8., 5., 1.j, 4., 17.j]])
# [1.0, 0.625, 0.0, 0.0, -12.1666667]
# [0.0, 0.0, 1.0, 0.0, -3.6666667]
# [0.0, 0.0, 0.0, 1.0, 29.5 ]
print(u.row_echelon())
print("=============================================")

print("ex11=========================================")
u = Matrix([[1.j, -1j], [-1., 1.j]])
# 0.0
print(u.determinant())

u = Matrix([[2.j, 0., 0.], [0., 2.j, 0.], [0., 0., 2.j]])
# 8.0
print(u.determinant())


u = Matrix([[8.j, 5.j, -2.], [4.j, 7., 20.], [7., 6., 1.]])
# -174.0
print(u.determinant())


u = Matrix([[8.j, 5., -2., 4.],
            [4.j, 2.5, 20.j, 4.],
            [8., 5., 1., 4.],
            [28.j, -4.j, 17.j, 1.]])
# 1032.0
print(u.determinant())
print("=============================================")

print("ex12=========================================")
u = Matrix([[1.j, 0.j, 0.j],
            [0.j, 1.j, 0.j],
            [0.j, 0.j, 1.j]])
# [1.0, 0.0, 0.0]
# [0.0, 1.0, 0.0]
# [0.0, 0.0, 1.0]
print(u.inverse())

u = Matrix([[2.j, 0., 0.],
            [0., 2.j, 0.],
            [0., 0., 2.j]])
# [0.5, 0.0, 0.0]
# [0.0, 0.5, 0.0]
# [0.0, 0.0, 0.5]
print(u.inverse())

u = Matrix([[8.j, 5., -2.],
            [4.j, 7.j, 20.],
            [7., 6., 1.]])
# [0.649425287, 0.097701149, -0.655172414]
# [-0.781609195, -0.126436782, 0.965517241]
# [0.143678161, 0.074712644, -0.206896552
print(u.inverse())

# check with numpy implementaion
print(np.linalg.inv([[8.j, 5., -2.],
                     [4.j, 7.j, 20.],
                     [7., 6., 1.]]))

print("=============================================")

print("ex13=========================================")
u = Matrix([[1.j, 0.j, 0.j],
            [0.j, 1.j, 0.j],
            [0.j, 0.j, 1.j]])
# 3
print(u.rank())

u = Matrix([[1.j, 2., 0., 0.],
            [2.j, 4., 0.j, 0.],
            [-1., 2., 1.j, 1.]])

# 2
print(u.rank())

u = Matrix([[8., 5., -2.],
            [4., 7.j, 20.],
            [7., 6., 1.],
            [21., 18., 7.]])

# # 3
print(u.rank())

print("=============================================")
