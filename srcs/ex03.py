from matrix import Matrix, Vector


u = Vector([[0.], [0.]])
v = Vector([[1.], [1.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

# 0.0
print(u.dot(v))

u = Vector([[1.], [1.]])
v = Vector([[1.], [1.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

# 2.0
print(u.dot(v))

u = Vector([[-1.], [6.]])
v = Vector([[3.], [2.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

# 9.0
print(u.dot(v))

print("=======eval===========")

u = Vector([[0.], [0.]])
v = Vector([[0.], [0.]])
print(u.dot(v))

u = Vector([[1.], [0.]])
v = Vector([[0.], [0.]])
print(u.dot(v))

u = Vector([[1.], [0.]])
v = Vector([[1.], [0.]])
print(u.dot(v))

u = Vector([[1.], [0.]])
v = Vector([[0.], [1.]])
print(u.dot(v))

u = Vector([[1.], [1.]])
v = Vector([[1.], [1.]])
print(u.dot(v))

u = Vector([[4.], [2.]])
v = Vector([[2.], [1.]])
print(u.dot(v))

print("=======eval===========")
