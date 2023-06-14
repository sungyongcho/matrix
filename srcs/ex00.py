from matrix import Matrix, Vector

a = Vector([[2.], [3.]])
b = Vector([[5.], [7.]])

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
u = Matrix([[1., 2.], [3., 4.]])
v = Matrix([[7., 4.], [-2., 2.]])
print(u.add(v))
print(u + v)

# u = Matrix([[1., 2.], [3., 4.]])
# v = Matrix([[7., 4.], [-2., 2.]])

print("### sub")
print(u.sub(v))
print(u - v)

print("### scl")
print(u.scl(2.))
