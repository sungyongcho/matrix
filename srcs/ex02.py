from matrix import Matrix, Vector
from matrix import lerp

print(lerp(0., 1., 0.))

print(lerp(0., 1., 1.))

print(lerp(0., 1., .5))

print(lerp(21., 42., .3))

print(lerp(Vector([[2.], [1.]]), Vector([[4.], [2.]]), .3))

print(lerp(Matrix([[2., 1.], [3., 4.]]), Matrix([[20., 10.], [30., 40.]]), .5))

print("=======eval===========")
print(lerp(0., 1., 0.))

print(lerp(0., 1., 1.))

print(lerp(0., 42., 0.5))

print(lerp(-42., 42., 0.5))

print(lerp(Vector([[-42.], [42.]]), Vector([[42.], [-42.]]), .5))

print("=======eval===========")
