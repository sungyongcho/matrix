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
