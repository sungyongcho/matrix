from matrix import Matrix, Vector


u = Vector([[0.], [0.]])
v = Vector([[1.], [1.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

print(u.dot(v))

u = Vector([[1.], [1.]])
v = Vector([[1.], [1.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

print(u.dot(v))


u = Vector([[-1.], [6.]])
v = Vector([[3.], [2.]])

# to show the 'requirement' of the dot product
print(u.shape, v.shape)

print(u.dot(v))
