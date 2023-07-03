from matrix import Vector, cross_product

u = Vector([[0.], [0.], [1.]])
v = Vector([[1.], [0.], [0.]])

# [0.]
# [1.]
# [0.]
print(cross_product(u, v))

u = Vector([[1.], [2.], [3.]])
v = Vector([[4.], [5.], [6.]])

# [-3.]
# [6.]
# [-3.]
print(cross_product(u, v))


u = Vector([[4.], [2.], [-3.]])
v = Vector([[-2.], [-5.], [16.]])

# [17.]
# [-58.]
# [-16.]
print(cross_product(u, v))

print("=======eval===========")
u = Vector([[0.], [0.], [0.]])
v = Vector([[0.], [0.], [0.]])

print(cross_product(u, v))

u = Vector([[1.], [0.], [0.]])
v = Vector([[0.], [0.], [0.]])

print(cross_product(u, v))

u = Vector([[1.], [0.], [0.]])
v = Vector([[0.], [1.], [0.]])

print(cross_product(u, v))

u = Vector([[8.], [7.], [-4.]])
v = Vector([[3.], [2.], [1.]])

print(cross_product(u, v))

u = Vector([[1.], [1.], [1.]])
v = Vector([[0.], [0.], [0.]])

print(cross_product(u, v))

u = Vector([[1.], [1.], [1.]])
v = Vector([[1.], [1.], [1.]])

print(cross_product(u, v))

print("=======eval===========")
