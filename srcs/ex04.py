from matrix import Matrix, Vector
u = Vector([[0.], [0.], [0.]])
# 0.0, 0.0, 0.0
print(u.norm_1(), u.norm(), u.norm_inf())

u = Vector([[1.], [2.], [3.]])
# 6.0, 3.74165738, 3.0
print(u.norm_1(), u.norm(), u.norm_inf())

u = Vector([[-1.], [-2.]])
# 6.0, 3.74165738, 3.0
print(u.norm_1(), u.norm(), u.norm_inf())

print("=======eval===========")

u = Vector([[0.]])
print(u.norm(), u.norm_1())

u = Vector([[1.]])
print(u.norm(), u.norm_1())

u = Vector([[0.], [0.]])
print(u.norm(), u.norm_1())

u = Vector([[1.], [0.]])
print(u.norm(), u.norm_1())

u = Vector([[2.], [1.]])
print(u.norm(), u.norm_1())

u = Vector([[4.], [2.]])
print(u.norm(), u.norm_1())

u = Vector([[-4.], [-2.]])
print(u.norm(), u.norm_1())

print("=======eval===========")
