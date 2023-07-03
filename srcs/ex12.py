from matrix import Matrix
import numpy as np

u = Matrix([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
# [1.0, 0.0, 0.0]
# [0.0, 1.0, 0.0]
# [0.0, 0.0, 1.0]
print(u.inverse())

u = Matrix([[2., 0., 0.],
            [0., 2., 0.],
            [0., 0., 2.]])
# [0.5, 0.0, 0.0]
# [0.0, 0.5, 0.0]
# [0.0, 0.0, 0.5]
print(u.inverse())

u = Matrix([[8., 5., -2.],
            [4., 7., 20.],
            [7., 6., 1.]])
# [0.649425287, 0.097701149, -0.655172414]
# [-0.781609195, -0.126436782, 0.965517241]
# [0.143678161, 0.074712644, -0.206896552
print(u.inverse())

# check with numpy implementaion
print(np.linalg.inv([[8., 5., -2.],
                     [4., 7., 20.],
                     [7., 6., 1.]]))

print("=======eval===========")
u = Matrix([[1., 0.,],
            [0., 1.,]])
print(u.inverse())

u = Matrix([[2., 0.,],
            [0., 2.,]])
print(u.inverse())

u = Matrix([[.5, 0.,],
            [0., .5,]])
print(u.inverse())

u = Matrix([[0., 1.,],
            [1., 0.,]])
print(u.inverse())

u = Matrix([[1., 2.,],
            [3., 4.,]])
print(u.inverse())

u = Matrix([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
print(u.inverse())

print("=======eval===========")
