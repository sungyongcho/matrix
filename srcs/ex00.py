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


print("=======eval===========")

print("# add")
a = Vector([[0.], [0.]])
b = Vector([[0.], [0.]])
print(a.add(b))

a = Vector([[1.], [0.]])
b = Vector([[0.], [1.]])
print(a.add(b))

a = Vector([[1.], [1.]])
b = Vector([[1.], [1.]])
print(a.add(b))

a = Vector([[21.], [21.]])
b = Vector([[21.], [21.]])
print(a.add(b))

a = Vector([[-21.], [21.]])
b = Vector([[21.], [-21.]])
print(a.add(b))

a = Vector([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]])
b = Vector([[9.], [8.], [7.], [6.], [5.], [4.], [3.], [2.], [1.], [0.]])
print(a.add(b))


u = Matrix([[0., 0.], [0., 0.]])
v = Matrix([[0., 0.], [0., 0.]])

print(u.add(v))

u = Matrix([[1., 0.], [0., 1.]])
v = Matrix([[0., 0.], [0., 0.]])

print(u.add(v))

u = Matrix([[1., 1.], [1., 1.]])
v = Matrix([[1., 1.], [1., 1.]])

print(u.add(v))

u = Matrix([[21., 21.], [21., 21.]])
v = Matrix([[21., 21.], [21., 21.]])

print(u.add(v))

print("# subtract")
a = Vector([[0.], [0.]])
b = Vector([[0.], [0.]])
print(a.sub(b))

a = Vector([[1.], [0.]])
b = Vector([[0.], [1.]])
print(a.sub(b))

a = Vector([[1.], [1.]])
b = Vector([[1.], [1.]])
print(a.sub(b))

a = Vector([[21.], [21.]])
b = Vector([[21.], [21.]])
print(a.sub(b))

a = Vector([[-21.], [21.]])
b = Vector([[21.], [-21.]])
print(a.sub(b))

a = Vector([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]])
b = Vector([[9.], [8.], [7.], [6.], [5.], [4.], [3.], [2.], [1.], [0.]])
print(a.sub(b))


u = Matrix([[0., 0.], [0., 0.]])
v = Matrix([[0., 0.], [0., 0.]])

print(u.sub(v))

u = Matrix([[1., 0.], [0., 1.]])
v = Matrix([[0., 0.], [0., 0.]])

print(u.sub(v))

u = Matrix([[1., 1.], [1., 1.]])
v = Matrix([[1., 1.], [1., 1.]])

print(u.sub(v))

u = Matrix([[21., 21.], [21., 21.]])
v = Matrix([[21., 21.], [21., 21.]])

print(u.sub(v))

print("# multiplication (scalar multiplication)")
a = Vector([[0.], [0.]])
print(a.scl(1.))

a = Vector([[1.], [0.]])
print(a.scl(1.))

a = Vector([[1.], [1.]])
print(a.scl(2.))

a = Vector([[21.], [21.]])
print(a.scl(2.))

a = Vector([[42.], [42.]])
print(a.scl(0.5))

u = Matrix([[0., 0.], [0., 0.]])
print(u.scl(0.))

u = Matrix([[1., 0.], [0., 1.]])
print(u.scl(1.))

u = Matrix([[1., 2.], [3., 4.]])
print(u.scl(2.))

u = Matrix([[21., 21.], [21., 21.]])
print(u.scl(0.5))
print("=======eval===========")
