from matrix import Vector, angle_cos

u = Vector([[1.], [0.]])
v = Vector([[1.], [0.]])

# 1.0
print(angle_cos(u, v))

u = Vector([[1.], [0.]])
v = Vector([[0.], [1.]])

# 0.0
print(angle_cos(u, v))

u = Vector([[-1.], [1.]])
v = Vector([[1.], [-1.]])

# -1.0
print(angle_cos(u, v))

u = Vector([[2.], [1.]])
v = Vector([[4.], [2.]])

# 1.0
print(angle_cos(u, v))

u = Vector([[1.], [2.], [3.]])
v = Vector([[4.], [5.], [6.]])

# 0.974631846
print(angle_cos(u, v, 0))

print("=======eval===========")

u = Vector([[1.], [0.]])
v = Vector([[0.], [1.]])

print(angle_cos(u, v))

u = Vector([[8.], [7.]])
v = Vector([[3.], [2.]])

# decimal place rounding disabled
print(angle_cos(u, v, 0))

u = Vector([[1.], [1.]])
v = Vector([[1.], [1.]])

print(angle_cos(u, v))

u = Vector([[4.], [2.]])
v = Vector([[1.], [1.]])

print(angle_cos(u, v, 0))

u = Vector([[-7.], [3.]])
v = Vector([[6.], [4.]])

print(angle_cos(u, v, 0))

print("=======eval===========")
