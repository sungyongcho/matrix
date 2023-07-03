from matrix import Matrix

u = Matrix([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
# 3
print(u.rank())

u = Matrix([[1., 2., 0., 0.],
            [2., 4., 0., 0.],
            [-1., 2., 1., 1.]])

# 2
print(u.rank())

u = Matrix([[8., 5., -2.],
            [4., 7., 20.],
            [7., 6., 1.],
            [21., 18., 7.]])

# # 3
print(u.rank())

print("=======eval===========")

u = Matrix([[0., 0.], [0., 0.]])
print(u.rank())

u = Matrix([[1., 0.], [0., 1.]])
print(u.rank())

u = Matrix([[2., 0.], [0., 2.]])
print(u.rank())

u = Matrix([[1., 1.], [1., 1.]])
print(u.rank())

u = Matrix([[0., 1.], [1., 0.]])
print(u.rank())

u = Matrix([[1., 2.], [3., 4.]])
print(u.rank())

u = Matrix([[-7., 5.], [4., 6.]])
print(u.rank())

u = Matrix([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
print(u.rank())

print("=======eval===========")
